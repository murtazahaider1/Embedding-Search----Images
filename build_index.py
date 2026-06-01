"""


Rewritten for the live Shopify Facebook/Google feed CSV.


Usage:
    python build_index.py                   # full build from live URL
    python build_index.py --refresh         # incremental update from live URL
    python build_index.py --csv feed.csv    # use a local CSV file
    python build_index.py --csv feed.csv --refresh
"""

import argparse
import io
import shutil
import sys
import time
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import httpx
from PIL import Image

import chromadb
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification

from extract_metadata import extract_metadata
from product_store import ProductStore

# ── Config ────────────────────────────────────────────────────────────────────

FEED_URL        = "https://shopify-feedofy-prod.s3.us-east-2.amazonaws.com/d0vg0m-dv.myshopify.com-facebook.csv"
CHROMA_DIR      = "zarr_chroma_db"
COLLECTION_NAME = "zarr_catalog"
BATCH_SIZE      = 32
EMBED_IMAGE_MAX = 224   # CLIP native resolution
IMG_TIMEOUT     = 10    # seconds per image fetch
IMG_RETRIES     = 2     # retry failed image fetches once

# ── Load CLIP ─────────────────────────────────────────────────────────────────

def load_clip():
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained("patrickjohncyh/fashion-clip")
    model     = AutoModelForZeroShotImageClassification.from_pretrained(
        "patrickjohncyh/fashion-clip"
    ).to(device)
    model.eval()
    print(f"Fashion-CLIP loaded on {device}")
    return processor, model, device


# ── Tensor helper ─────────────────────────────────────────────────────────────

def _to_tensor(output) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state[:, 0]
    raise ValueError(f"Cannot extract tensor from {type(output)}")


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_texts(texts: list[str], processor, model, device) -> np.ndarray:
    all_vecs = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch  = texts[i:i + BATCH_SIZE]
        inputs = processor(
            text=batch, return_tensors="pt",
            padding=True, truncation=True,
        ).to(device)
        with torch.no_grad():
            out = model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
        feats = _to_tensor(out)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        all_vecs.append(feats.cpu().numpy())
    return np.vstack(all_vecs).astype("float32")


def _resize(img: Image.Image, max_px: int = EMBED_IMAGE_MAX) -> Image.Image:
    w, h    = img.size
    longest = max(w, h)
    if longest <= max_px:
        return img
    scale = max_px / longest
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def embed_image_url(url: str, processor, model, device) -> np.ndarray | None:
    """
    Fetch image URL into memory, embed with CLIP, discard bytes.
    Never writes to disk.
    """
    for attempt in range(IMG_RETRIES + 1):
        try:
            resp = httpx.get(url, timeout=IMG_TIMEOUT, follow_redirects=True)
            resp.raise_for_status()
            img    = _resize(Image.open(io.BytesIO(resp.content)).convert("RGB"))
            inputs = processor(images=img, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.get_image_features(pixel_values=inputs["pixel_values"])
            feats = _to_tensor(out)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats.cpu().numpy()[0].astype("float32")
        except Exception as e:
            if attempt < IMG_RETRIES:
                time.sleep(1)
            else:
                print(f"  Image fetch failed ({url[:60]}…): {e}")
    return None


# ── Synthetic description ─────────────────────────────────────────────────────

_GENDER_PHRASE = {
    "men":    "men's",
    "women":  "women's",
    "unisex": "unisex",
}

_FIT_PHRASE = {
    "slim":    "slim fit",
    "regular": "regular fit",
    "loose":   "loose fit",
    "baggy":   "baggy fit",
}

_STYLE_PHRASE = {
    "ethnic":      "ethnic pakistani fashion",
    "bridal":      "bridal formal ethnic wear",
    "modest wear": "modest wear islamic fashion",
    "formal":      "formal office wear",
    "casual":      "casual everyday wear",
    "streetwear":  "streetwear urban fashion",
    "athletic":    "athletic sportswear",
    "embroidered": "embroidered ethnic fashion",
    "printed":     "printed pattern fashion",
    "plain":       "solid plain fashion",
    "checkered":   "checkered pattern fashion",
    "striped":     "striped pattern fashion",
    "denim":       "denim fashion",
}

_TYPE_PHRASE = {
    "shalwar kameez":  "shalwar kameez pakistani suit",
    "kurta trouser":   "kurta with trouser pakistani",
    "kurta":           "kurta pakistani top",
    "kurti":           "kurti pakistani women's top",
    "kurta set":       "kurta set matching outfit",
    "dupatta set":     "shalwar kameez with dupatta pakistani suit",
    "sherwani":        "sherwani formal pakistani men's wear",
    "waistcoat":       "waistcoat vest jacket",
    "saree":           "saree sari south asian drape",
    "lehenga":         "lehenga skirt south asian bridal",
    "sharara":         "sharara wide leg south asian",
    "kaftan":          "kaftan loose dress",
    "abaya":           "abaya modest full length dress",
    "hijab":           "hijab headscarf modest wear",
    "scarf":           "scarf stole wrap accessory",
    "co-ord suit":     "co-ord matching set two piece outfit",
    "inner set":       "loungewear pyjama inner set",
    "unstitched":      "unstitched fabric suit",
    "t-shirt":         "t-shirt casual tee",
    "polo shirt":      "polo shirt",
    "shirt":           "shirt button-up",
    "hoodie":          "hoodie sweatshirt",
    "sweatshirt":      "sweatshirt crewneck",
    "sweater":         "sweater knitwear",
    "cardigan":        "cardigan knitwear",
    "top":             "top blouse",
    "jacket":          "jacket outerwear",
    "coat":            "coat overcoat",
    "blazer":          "blazer formal jacket",
    "puffer jacket":   "puffer jacket quilted",
    "denim jacket":    "denim jacket",
    "windbreaker":     "windbreaker light jacket",
    "jeans":           "jeans denim trousers",
    "trousers":        "trousers pants",
    "chinos":          "chinos casual trousers",
    "shorts":          "shorts",
    "leggings":        "leggings tights",
    "skirt":           "skirt",
    "joggers":         "joggers sweatpants",
    "dress":           "dress",
    "maxi":            "maxi dress floor length",
    "jumpsuit":        "jumpsuit one piece",
    "culotte":         "culotte wide leg crop",
    "romper":          "romper playsuit",
    "tracksuit":       "tracksuit set",
    "activewear":      "activewear sportswear gym wear",
    "sneakers":        "sneakers trainers",
    "formal shoes":    "formal shoes dress shoes",
    "sandals":         "sandals chappals",
    "boots":           "boots ankle boots",
    "sports shoes":    "sports shoes athletic shoes",
    "bag":             "bag handbag",
    "cap":             "cap hat",
    "belt":            "belt accessory",
    "jewellery":       "jewellery accessories",
    "sunglasses":      "sunglasses eyewear",
    "watch":           "watch timepiece",
    "socks":           "socks",
    "fragrance":       "fragrance perfume scent",
    "beauty":          "beauty skincare cosmetics",
    "grooming":        "grooming men's care",
}


def build_rich_description(item: dict, meta: dict) -> str:
    gender  = _GENDER_PHRASE.get(meta["gender"], meta["gender"])
    fit     = _FIT_PHRASE.get(meta["fit"], meta["fit"])
    colour  = meta["colour"] if meta["colour"] != "unknown" else ""
    type_p  = _TYPE_PHRASE.get(meta["type"], meta["type"])
    style_p = _STYLE_PHRASE.get(meta["style"], meta["style"])
    title   = str(item.get("title", "") or "")
    category = str(item.get("category", "") or "")
    sub_cat  = str(item.get("sub_cat", "") or "")
    desc     = str(item.get("description", "") or "")

    # Mine fabric and occasion words from description
    mined = []
    fabric_hits = re.findall(
        r"\b(cotton|silk|lawn|chiffon|velvet|linen|georgette|"
        r"organza|khaddar|karandi|cambric|satin|net|viscose|"
        r"jacquard|crepe|denim|wool|fleece|suede)\b",
        desc, re.IGNORECASE
    )
    mined.extend(list(dict.fromkeys(f.lower() for f in fabric_hits)))

    occasion_hits = re.findall(
        r"\b(casual|formal|wedding|bridal|party|eid|festive|"
        r"office|daily|ethnic|traditional|embroidered|printed|plain)\b",
        desc, re.IGNORECASE
    )
    mined.extend(list(dict.fromkeys(o.lower() for o in occasion_hits)))

    parts = [
        f"{gender} {fit}",
        colour,
        type_p,
        style_p,
        title,
        category,
        sub_cat,
        " ".join(mined),
    ]

    desc_out = " ".join(p for p in parts if p).strip()
    desc_out = re.sub(r"\s+", " ", desc_out)
    return desc_out


# ── CSV loading ───────────────────────────────────────────────────────────────

def load_feed(csv_path: str | None) -> pd.DataFrame:
    if csv_path:
        print(f"Loading CSV from local file: {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        print(f"Fetching live feed from:\n  {FEED_URL}")
        resp = httpx.get(FEED_URL, timeout=60, follow_redirects=True)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        print(f"Feed fetched — {len(df)} rows")

    # Rename columns to internal names
    df = df.rename(columns={
        "ID":             "product_id",
        "title":          "title",
        "description":    "description",
        "link":           "link",
        "image_link":     "image_link",
        "price":          "price",
        "sale_price":     "sale_price",
        "gender":         "gender_raw",      # "["Women"]" format — parsed later
        "color":          "color",
        "custom_label_1": "gender_label",    # "Women" | "Men" | "Kids"
        "custom_label_2": "category",        # "Eastern" | "Western" | etc.
        "custom_label_3": "sub_cat",
        "custom_label_4": "sub_cat2",
    })

    # Use custom_label_1 as gender (cleaner than the JSON ["Women"] field)
    if "gender_label" in df.columns:
        df["gender"] = df["gender_label"]
    elif "gender_raw" in df.columns:
        df["gender"] = df["gender_raw"]
    else:
        df["gender"] = "Unisex"

    df["product_id"] = df["product_id"].astype(str)

    # Parse display price — prefer sale_price if lower.
    # Store as clean "PKR X,XXX" — never the raw "PKR 5490.00" string
    # which causes parseFloat() to return NaN on the frontend.
    def _display_price(row):
        def _parse(v):
            try:
                return float(re.sub(r"[^\d.]", "", str(v)))
            except Exception:
                return None

        def _fmt(v):
            n = _parse(v)
            if n is None:
                return ""
            return f"PKR {int(n):,}"

        p  = _parse(row.get("price"))
        sp = _parse(row.get("sale_price"))
        if sp and p and sp < p:
            return _fmt(row.get("sale_price"))
        return _fmt(row.get("price"))

    df["display_price"] = df.apply(_display_price, axis=1)

    print(f"Loaded {len(df)} products | {df['category'].nunique()} categories")
    return df


def row_to_item(row) -> dict:
    """Convert a DataFrame row to the dict format extract_metadata expects."""
    return {
        "title":       str(row.get("title", "") or ""),
        "description": str(row.get("description", "") or ""),
        "link":        str(row.get("link", "") or ""),
        "image_link":  str(row.get("image_link", "") or ""),
        "gender":      str(row.get("gender", "") or ""),
        "category":    str(row.get("category", "") or ""),
        "sub_cat":     str(row.get("sub_cat", "") or ""),
        "sub_cat2":    str(row.get("sub_cat2", "") or ""),
        "color":       str(row.get("color", "") or ""),
        "price":       str(row.get("display_price", "") or ""),
    }


# ── Build ─────────────────────────────────────────────────────────────────────

def build(refresh: bool = False, csv_path: str | None = None):
    df = load_feed(csv_path)

    processor, model, device = load_clip()

    # ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        collection = client.get_collection(COLLECTION_NAME)
        print(f"ChromaDB collection exists: {collection.count()} vectors")
    except Exception:
        collection = client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        print("ChromaDB collection created")

    existing_ids = set()
    if collection.count() > 0:
        existing_ids = set(collection.get(include=[])["ids"])
    print(f"Already embedded: {len(existing_ids)}")

    store = ProductStore()

    # ── Refresh: remove delisted products ─────────────────────────────────────
    if refresh and existing_ids:
        feed_ids    = set(df["product_id"].astype(str))
        delisted    = existing_ids - feed_ids
        if delisted:
            print(f"Removing {len(delisted)} delisted products …")
            collection.delete(ids=list(delisted))
            for pid in delisted:
                store.delete(pid)
            existing_ids -= delisted
            print(f"  Removed {len(delisted)} products from ChromaDB + SQLite")

    # ── Refresh: detect changed title/price to force re-embed ─────────────────
    force_reembed: set[str] = set()
    if refresh:
        for _, row in df.iterrows():
            pid = str(row["product_id"])
            if pid not in existing_ids:
                continue
            sql = store.get(pid)
            if sql is None:
                continue
            if (sql.get("title") != str(row.get("title", "")) or
                    sql.get("price") != str(row.get("display_price", ""))):
                force_reembed.add(pid)
        if force_reembed:
            print(f"Re-embedding {len(force_reembed)} changed products …")
            collection.delete(ids=list(force_reembed))
            existing_ids -= force_reembed

    # ── Main loop ─────────────────────────────────────────────────────────────
    new_embedded = 0
    text_only    = 0
    skipped      = 0

    total = len(df)
    for i, (_, row) in enumerate(df.iterrows()):
        pid  = str(row["product_id"])
        item = row_to_item(row)
        meta = extract_metadata(item)

        image_url = str(row.get("image_link", "") or "").strip()

        # SQLite upsert — always runs so price/title changes are captured
        store.upsert(
            product_id  = pid,
            title       = item["title"],
            price       = item["price"],
            url         = item["link"],
            image_url   = image_url,
            local_image = "",   # not used — frontend uses image_url directly
        )

        # Skip embedding if already in ChromaDB
        if pid in existing_ids:
            skipped += 1
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{total} — skipped {skipped}, "
                      f"embedded {new_embedded}, text-only {text_only}")
            continue

        # Build synthetic description
        desc_text = build_rich_description(item, meta)

        # Text embedding
        txt_vec = embed_texts([desc_text], processor, model, device)[0]

        # Image embedding (in-memory, not saved to disk)
        vec = txt_vec  # fallback
        if image_url and image_url != "nan":
            img_vec = embed_image_url(image_url, processor, model, device)
            if img_vec is not None:
                # 40% text, 60% image (same weights as v2)
                combined = 0.4 * txt_vec + 0.6 * img_vec
                vec      = combined / np.linalg.norm(combined)
            else:
                text_only += 1
        else:
            text_only += 1

        chroma_meta = {
            "product_id": pid,
            "gender":     meta["gender"],
            "type":       meta["type"],
            "colour":     meta["colour"],
            "fit":        meta["fit"],
            "style":      meta["style"],
        }

        collection.add(
            ids        = [pid],
            embeddings = [vec.tolist()],
            metadatas  = [chroma_meta],
            documents  = [desc_text],
        )
        new_embedded += 1

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{total} — embedded: {new_embedded}, "
                  f"text-only: {text_only}, skipped: {skipped}")

    print(f"\nDone.")
    print(f"  SQLite rows upserted:              {total}")
    print(f"  New embeddings added to ChromaDB:  {new_embedded}")
    print(f"    of which text-only (img failed): {text_only}")
    print(f"  Skipped (already embedded):        {skipped}")
    print(f"  Total vectors in ChromaDB:         {collection.count()}")
    print(f"  Total products in SQLite:          {store.count()}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Zarr ChromaDB index from feed CSV")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Incremental update: remove delisted products, re-embed changed ones",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Wipe ChromaDB and SQLite completely before building. "
             "Use this when switching data sources or after a major feed change.",
    )
    parser.add_argument(
        "--csv",
        metavar="PATH",
        default=None,
        help="Path to a local CSV file. If omitted, fetches the live feed URL.",
    )
    args = parser.parse_args()

    if args.clean:
        print("--clean: wiping ChromaDB and SQLite …")
        if Path(CHROMA_DIR).exists():
            shutil.rmtree(CHROMA_DIR)
            print(f"  Deleted {CHROMA_DIR}/")
        db_path = "zarr_products.db"
        if Path(db_path).exists():
            Path(db_path).unlink()
            print(f"  Deleted {db_path}")
        print("  Clean done — starting fresh build\n")

    build(refresh=args.refresh, csv_path=args.csv)