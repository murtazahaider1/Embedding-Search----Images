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

from extract_metadata import COLOUR_MAP   # used by nearest_colour fallback
from search import load_db, search as chroma_search
from product_store import ProductStore

# ── Config ────────────────────────────────────────────────────────────────────

IMAGE_DIR        = "zarr_data/image_resources"
DETECT_CONF      = 0.30
MIN_DETECTION_PX = 30
MIN_CROP_AREA_RATIO = 0.04   # crop must be at least 4% of image area to be a real garment
MAX_HEAD_ZONE_RATIO = 0.25   # detections entirely within top 25% of image are likely head/hair crops

# ── Gender classifier ────────────────────────────────────────────────────────
# Two polar-opposite prompts designed for Fashion-CLIP's training distribution.
# Using comma-separated fashion keywords rather than natural language gives
# more reliable discrimination — especially for Eastern/Pakistani garments.
# Confidence threshold: if below GENDER_CONF_MIN, we default to "unisex"
# so the search filter stays open rather than filtering the wrong gender.
GENDER_CONF_MIN = 0.62   # below this → unisex (safe fallback)

GENDER_LABELS = [
    "women clothing ladies fashion kurti shalwar kameez dupatta suit dress",
    "men clothing menswear kurta shirt trousers shalwar sherwani",
]
GENDER_LABEL_MAP = {
    GENDER_LABELS[0]: "women",
    GENDER_LABELS[1]: "men",
}

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

MAX_INPUT_HEIGHT = 224   # input image is capped at this height before detection

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


# ── Clothing presence check ───────────────────────────────────────────────────

# Positive labels — things that count as "clothing present"
_CLOTHING_PRESENT = [
    "a person wearing clothes",
    "a clothing item or garment",
    "a shirt, top, or jacket",
    "trousers, jeans, or a skirt",
    "a dress or traditional outfit",
    "a shalwar kameez or kurta",
    "a saree or ethnic outfit",
    "shoes or footwear",
    "a bag or accessory",
]

# Negative labels — things that are clearly not clothing
_NOT_CLOTHING = [
    "a landscape, sky, or outdoor scene",
    "a building, street, or architecture",
    "food or a meal",
    "a vehicle or transportation",
    "an animal or pet",
    "text, a document, or a screen",
    "furniture or an interior room without people",
    "nature, plants, or trees",
    "a blank or solid colour background",
]

# Combined label set — CLIP picks the single best match across all of them
_ALL_SCENE_LABELS = _CLOTHING_PRESENT + _NOT_CLOTHING

# Minimum fraction of probability mass that must sit on clothing labels
# for the image to be considered a valid clothing input.
# Set conservatively — we want to block obvious non-clothing without
# impacting any real-world clothing photo.
CLOTHING_PRESENCE_THRESHOLD = 0.55


def contains_clothing(image: Image.Image) -> tuple[bool, float]:
    """
    Run a zero-shot CLIP scene classification on the FULL image to check
    whether clothing is present before running the detection pipeline.

    Returns (is_clothing, clothing_score) where clothing_score is the
    summed probability across all positive clothing labels.

    Uses the same CLIP model already loaded at startup — no extra inference cost
    beyond a single forward pass on the resized input image.
    """
    # Resize to a fixed size for fast inference — no need for full resolution here
    thumb = image.copy()
    thumb.thumbnail((336, 336), Image.LANCZOS)

    inputs = processor(
        images=thumb,
        text=_ALL_SCENE_LABELS,
        return_tensors="pt",
        padding=True,
    ).to(device)

    with torch.no_grad():
        out    = clip(**inputs)
    logits = out.logits_per_image if hasattr(out, "logits_per_image") else out[0]
    probs  = logits.softmax(dim=1)[0].cpu().numpy()

    # Sum probability over all positive (clothing) labels
    n_positive      = len(_CLOTHING_PRESENT)
    clothing_score  = float(probs[:n_positive].sum())
    is_clothing     = clothing_score >= CLOTHING_PRESENCE_THRESHOLD

    print(f"  [clothing check] score={clothing_score:.3f} "
          f"({'PASS' if is_clothing else 'FAIL'}) — "
          f"top label: '{_ALL_SCENE_LABELS[int(probs.argmax())]}'")

    return is_clothing, round(clothing_score, 3)


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


def is_valid_garment_region(bbox: list, image_w: int, image_h: int) -> bool:
    """
    Filter out detections that are almost certainly not garments:
    1. Too small relative to image — likely noise or face/hand crop
    2. Entirely in the top quarter of the image — likely hair or head region
       misidentified as a scarf/hijab by YOLO
    """
    x1, y1, x2, y2 = bbox
    crop_w = x2 - x1
    crop_h = y2 - y1
    crop_area  = crop_w * crop_h
    image_area = image_w * image_h

    # Must cover at least MIN_CROP_AREA_RATIO of the total image
    if crop_area / image_area < MIN_CROP_AREA_RATIO:
        return False

    # If the entire bbox sits within the top MAX_HEAD_ZONE_RATIO of the image,
    # it is almost certainly a head/hair crop, not a garment
    if y2 < image_h * MAX_HEAD_ZONE_RATIO:
        return False

    return True


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
            bbox = [x1, y1, x2, y2]
            if not is_valid_garment_region(bbox, W, H):
                print(f"  [filter] Skipped small/head-zone region {bbox}")
                continue
            regions.append({
                "df2_class": yolo.names[int(box.cls[0])],
                "bbox":      bbox,
                "crop":      crop,
                "mask":      pmask,
            })

    if not regions:
        regions = [{"df2_class": "unknown", "bbox": [0,0,W,H], "crop": image, "mask": None}]

    # Remove overlapping duplicate detections before classifying
    regions = deduplicate_regions(regions)

    items = []
    for i, r in enumerate(regions):
        crop   = r["crop"]

        # Dominant colour via K-means pixel clustering
        colour = dominant_colour(crop, r["mask"])

        # Gender — binary CLIP classification with confidence fallback.
        # We intentionally skip type/fit CLIP classification here:
        # Fashion-CLIP misclassifies Eastern garments too often for those
        # labels to be useful in the query vector. The image embedding
        # itself carries the visual type signal — we just need gender
        # to enforce the search filter correctly.
        gender_raw, gender_conf = clip_classify(crop, GENDER_LABELS)
        if gender_conf >= GENDER_CONF_MIN:
            gender = GENDER_LABEL_MAP[gender_raw]
        else:
            gender = "unisex"   # low confidence → don't filter by gender

        df2 = r["df2_class"]
        print(
            f"  Item #{i+1}: yolo={df2} | colour={colour} | "
            f"gender={gender} (conf={gender_conf:.2f}{'*' if gender == 'unisex' else ''})"
        )

        items.append({
            "item_id":        i + 1,
            "detected_class": df2,
            "bbox":           r["bbox"],
            "type":           "unknown",   # not used — image vec drives search
            "colour":         colour,
            "gender":         gender,
            "fit":            "regular",   # not used in new filter
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

    # ── Clothing presence gate ─────────────────────────────────────────────
    is_clothing, clothing_score = contains_clothing(image)
    if not is_clothing:
        return JSONResponse({
            "item_count":      0,
            "items":           [],
            "clothing_found":  False,
            "clothing_score":  clothing_score,
            "message":         "No clothing detected in the uploaded image.",
        })

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
        "item_count":     len(output),
        "items":          output,
        "clothing_found": True,
        "clothing_score": clothing_score,
    })

