"""
main.py  —  Zarr Clothing Matcher API
"""

import io
import os
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.cluster import KMeans
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

from extract_metadata import (
    TYPE_MAP, COLOUR_MAP, FIT_MAP, STYLE_MAP, GENDER_MAP
)
from search import load_db, search as chroma_search
from product_store import ProductStore

# ── Config ────────────────────────────────────────────────────────────────────

IMAGE_DIR        = "zarr_data/image_resources"
DETECT_CONF      = 0.30
MIN_DETECTION_PX = 30

COLOUR_CSS = {
    "black": "#111", "white": "#f5f5f5", "light grey": "#c0bdb8",
    "dark grey": "#555", "navy blue": "#1a2744", "royal blue": "#2952a3",
    "sky blue": "#6aafd6", "light blue": "#a8d4ed", "dark blue": "#0a1a3a",
    "blue": "#2060c0", "red": "#c0302a", "burgundy": "#6e1c2a",
    "coral": "#e8725a", "salmon": "#e89080", "hot pink": "#e0356e",
    "light pink": "#f0b8c8", "pink": "#e8a0b8", "orange": "#d96020",
    "yellow": "#ddb830", "forest green": "#2d5a2d", "olive green": "#6b7a30",
    "mint green": "#7ecfb4", "green": "#2d8a2d", "brown": "#7a4a28",
    "tan": "#b8915c", "camel": "#c49a50", "beige": "#d8c8a8",
    "cream": "#eee8d8", "purple": "#6a3a8a", "lavender": "#b0a0d0",
    "grey": "#909090", "multicolor": "#aaa",
}

BACKGROUND_COLOURS = {"white", "cream", "light grey", "beige", "grey"}

# ── Startup: load models and DB ───────────────────────────────────────────────

app = FastAPI(title="Zarr Clothing Matcher", version="1.0.0")

device    = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained("patrickjohncyh/fashion-clip")
clip      = AutoModelForZeroShotImageClassification.from_pretrained(
    "patrickjohncyh/fashion-clip"
).to(device)
clip.eval()

weights_path = hf_hub_download(repo_id="Bingsu/adetailer", filename="deepfashion2_yolov8s-seg.pt")
yolo         = YOLO(weights_path)

collection, store = load_db()

models = {"device": device, "processor": processor, "clip_model": clip}

static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Serve local product images
img_dir = Path(IMAGE_DIR)
if img_dir.exists():
    app.mount("/product_images", StaticFiles(directory=str(img_dir)), name="product_images")


# ── Image helpers ─────────────────────────────────────────────────────────────

MAX_INPUT_HEIGHT = 500   # input image is capped at this height before detection

def fix_orientation(img):
    try:
        return ImageOps.exif_transpose(img)
    except Exception:
        return img


def resize_input(img: Image.Image, max_height: int = MAX_INPUT_HEIGHT) -> Image.Image:
    """Downscale image so height <= max_height, preserving aspect ratio.
    Images already smaller than max_height are returned unchanged."""
    w, h = img.size
    if h <= max_height:
        return img
    new_h = max_height
    new_w = int(w * (max_height / h))
    return img.resize((new_w, new_h), Image.LANCZOS)


def pad_square(crop, size=336):
    w, h   = crop.size
    side   = max(w, h)
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(crop, ((side - w) // 2, (side - h) // 2))
    return canvas.resize((size, size), Image.LANCZOS)


def mask_to_crop(image, mask_xy, full_mask=None, padding=10):
    W, H = image.size
    if mask_xy is not None and len(mask_xy) > 0:
        pts = mask_xy.reshape(-1, 2)
        x1, y1 = pts[:, 0].min(), pts[:, 1].min()
        x2, y2 = pts[:, 0].max(), pts[:, 1].max()
    else:
        return image, None
    x1 = max(0, int(x1) - padding)
    y1 = max(0, int(y1) - padding)
    x2 = min(W, int(x2) + padding)
    y2 = min(H, int(y2) + padding)
    if (x2 - x1) < 20 or (y2 - y1) < 20:
        return image, None
    crop       = image.crop((x1, y1, x2, y2))
    pixel_mask = full_mask[y1:y2, x1:x2] if full_mask is not None else None
    return crop, pixel_mask


def nearest_colour(rgb):
    r, g, b = rgb
    COLOUR_RGB = {
        "black": (20,20,20), "white": (245,245,245), "light grey": (200,200,200),
        "dark grey": (90,90,90), "navy blue": (26,39,68), "royal blue": (41,82,163),
        "sky blue": (106,175,214), "light blue": (168,212,237), "dark blue": (10,26,58),
        "blue": (32,96,192), "red": (192,48,42), "burgundy": (110,28,42),
        "coral": (232,114,90), "salmon": (232,144,128), "hot pink": (224,53,110),
        "light pink": (240,184,200), "pink": (232,160,184), "orange": (217,96,32),
        "yellow": (221,184,48), "forest green": (45,90,45), "olive green": (107,122,48),
        "mint green": (126,207,180), "green": (45,138,45), "brown": (122,74,40),
        "tan": (184,145,92), "camel": (196,154,80), "beige": (216,200,168),
        "cream": (238,232,216), "purple": (106,58,138), "lavender": (176,160,208),
        "grey": (144,144,144), "multicolor": (128,128,128),
    }
    best, dist = "multicolor", float("inf")
    for name, (cr, cg, cb) in COLOUR_RGB.items():
        d = (r-cr)**2 + (g-cg)**2 + (b-cb)**2
        if d < dist:
            dist, best = d, name
    return best


def dominant_colour(crop, pixel_mask=None, n_clusters=5):
    arr = np.array(crop.resize((150, 150), Image.LANCZOS)).reshape(-1, 3).astype(float)
    if pixel_mask is not None:
        m = np.array(
            Image.fromarray((pixel_mask.astype(np.uint8) * 255)).resize((150, 150), Image.NEAREST)
        ) > 127
        flat = m.reshape(-1)
        if flat.sum() > 50:
            arr = arr[flat]
    n = min(n_clusters, len(arr))
    if n < 1:
        return "multicolor"
    km     = KMeans(n_clusters=n, n_init=5, random_state=0)
    km.fit(arr)
    counts = np.bincount(km.labels_)
    order  = np.argsort(counts)[::-1]
    for idx in order:
        name = nearest_colour(tuple(km.cluster_centers_[idx].astype(int)))
        if name not in BACKGROUND_COLOURS:
            return name
    return nearest_colour(tuple(km.cluster_centers_[order[0]].astype(int)))


def clip_classify(crop, candidates):
    inputs = processor(
        images=pad_square(crop), text=candidates,
        return_tensors="pt", padding=True,
    ).to(device)
    with torch.no_grad():
        out = clip(**inputs)
    # logits_per_image is always a plain tensor for zero-shot classification
    logits = out.logits_per_image if hasattr(out, "logits_per_image") else out[0]
    probs  = logits.softmax(dim=1)[0].cpu().numpy()
    best   = int(np.argmax(probs))
    return candidates[best], round(float(probs[best]), 4)


# ── Detection ─────────────────────────────────────────────────────────────────

def iou(a: list, b: list) -> float:
    """Intersection over Union for two [x1,y1,x2,y2] boxes."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def deduplicate_regions(regions: list, iou_threshold: float = 0.30) -> list:
    """
    Remove duplicate detections caused by YOLO detecting the same garment
    multiple times with slightly different bounding boxes or class labels.
    Keeps the first detection when two boxes overlap above iou_threshold.
    """
    kept = []
    for r in regions:
        duplicate = False
        for k in kept:
            if iou(r["bbox"], k["bbox"]) > iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(r)
    return kept


def detect_and_classify(image: Image.Image) -> list[dict]:
    W, H    = image.size
    arr     = np.array(image)
    results = yolo(arr, conf=DETECT_CONF, verbose=False)[0]

    regions = []
    if results.boxes is None or len(results.boxes) == 0:
        regions = [{"df2_class": "unknown", "bbox": [0,0,W,H], "crop": image, "mask": None}]
    else:
        for idx, box in enumerate(results.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1, y1 = max(0,x1), max(0,y1)
            x2, y2 = min(W,x2), min(H,y2)
            if (x2-x1) < MIN_DETECTION_PX or (y2-y1) < MIN_DETECTION_PX:
                continue
            mask_xy   = results.masks.xy[idx]   if results.masks and idx < len(results.masks.xy)   else None
            full_mask = None
            if results.masks and idx < len(results.masks.data):
                m         = results.masks.data[idx].cpu().numpy()
                full_mask = np.array(
                    Image.fromarray((m*255).astype(np.uint8)).resize((W,H), Image.NEAREST)
                ) > 127
            crop, pmask = mask_to_crop(image, mask_xy, full_mask)
            regions.append({
                "df2_class": yolo.names[int(box.cls[0])],
                "bbox":      [x1,y1,x2,y2],
                "crop":      crop,
                "mask":      pmask,
            })

    if not regions:
        regions = [{"df2_class": "unknown", "bbox": [0,0,W,H], "crop": image, "mask": None}]

    # Remove overlapping duplicate detections before classifying
    regions = deduplicate_regions(regions)

    type_labels   = [t[0] for t in TYPE_MAP]
    gender_labels = list(GENDER_MAP.keys()) + ["unisex"]
    fit_labels    = [f[0] for f in FIT_MAP]

    items = []
    for i, r in enumerate(regions):
        crop   = r["crop"]
        colour = dominant_colour(crop, r["mask"])

        ctype,  type_conf   = clip_classify(crop, type_labels)
        gender, gender_conf = clip_classify(crop, gender_labels)
        fit,    fit_conf    = clip_classify(crop, fit_labels)

        # Log detected attributes to terminal, not shown on frontend
        print(f"  Item #{i+1}: class={r['df2_class']} | type={ctype} | colour={colour} | gender={gender} | fit={fit}")

        items.append({
            "item_id":        i + 1,
            "detected_class": r["df2_class"],
            "bbox":           r["bbox"],
            "type":           ctype,
            "colour":         colour,
            "gender":         gender,
            "fit":            fit,
            "_crop":          crop,
        })

    return items


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def index():
    return FileResponse(str(static_dir / "index.html"))


@app.get("/health")
def health():
    return {"status": "ok", "catalog_size": collection.count()}


@app.post("/match")
async def match(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image.")
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file.")

    image = fix_orientation(Image.open(io.BytesIO(data)).convert("RGB"))
    image = resize_input(image)
    items = detect_and_classify(image)

    output = []
    for item in items:
        attrs = {
            "type":   item["type"],
            "colour": item["colour"],
            "gender": item["gender"],
            "fit":    item["fit"],
            "style":  "casual",
        }
        matches = chroma_search(
            attrs      = attrs,
            crop       = item["_crop"],
            collection = collection,
            store      = store,
            processor  = processor,
            model      = clip,
            device     = device,
        )

        # Convert local image path to a served URL
        for m in matches:
            lp = m.get("local_image", "")
            if lp:
                fname = Path(lp).name
                m["served_image"] = f"/product_images/{fname}"
            else:
                m["served_image"] = m.get("image_url", "")

        output.append({
            "item_id":        item["item_id"],
            "detected_class": item["detected_class"],
            "detected_attrs": attrs,
            "matches":        matches,
            "found":          len(matches) > 0,
        })

    return JSONResponse({
        "item_count": len(output),
        "items":      output,
    })
