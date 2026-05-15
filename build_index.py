"""
build_index.py

Reads zarr_products.json and builds two stores:

1. zarr_chroma_db/  — ChromaDB with embeddings + semantic metadata only
                      (gender, type, colour, fit, style, product_id)
                      Never needs rebuilding for title/price changes.

2. zarr_products.db — SQLite with mutable display fields
                      (title, price, url, image paths)
                      Update anytime without touching embeddings.

Run once to build from scratch, or run again to add/remove products.
Existing products in ChromaDB are skipped (embeddings reused).
"""

import json
import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path

import chromadb
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification

from extract_metadata import extract_metadata
from product_store import ProductStore

# ── Config ────────────────────────────────────────────────────────────────────

PRODUCTS_JSON   = "zarr_products.json"
IMAGE_DIR       = "zarr_data/image_resources"
CHROMA_DIR      = "zarr_chroma_db"
COLLECTION_NAME = "zarr_catalog"
BATCH_SIZE      = 32

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


# ── Embedding helpers ─────────────────────────────────────────────────────────

def _to_tensor(output) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state[:, 0]
    raise ValueError(f"Cannot extract tensor from {type(output)}")


def embed_texts(texts, processor, model, device):
    all_vecs = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch  = texts[i:i + BATCH_SIZE]
        inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            out = model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
        feats = _to_tensor(out)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        all_vecs.append(feats.cpu().numpy())
    return np.vstack(all_vecs).astype("float32")


EMBED_IMAGE_MAX_PX = 224   # product images resized to this before embedding

def embed_image_file(path, processor, model, device):
    try:
        img = Image.open(path).convert("RGB")
        # Resize so the longest side is at most EMBED_IMAGE_MAX_PX
        w, h   = img.size
        longest = max(w, h)
        if longest > EMBED_IMAGE_MAX_PX:
            scale = EMBED_IMAGE_MAX_PX / longest
            img   = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.get_image_features(pixel_values=inputs["pixel_values"])
        feats = _to_tensor(out)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy()[0].astype("float32")
    except Exception as e:
        print(f"  Image embed failed ({path}): {e}")
        return None


def find_local_image(image_urls, image_dir):
    img_dir = Path(image_dir)
    for img in image_urls:
        fname     = img.get("filename", "")
        candidate = img_dir / fname
        if candidate.exists():
            return str(candidate)
        stem = Path(fname).stem
        for f in img_dir.iterdir():
            if f.stem == stem:
                return str(f)
    return None


# ── Build ─────────────────────────────────────────────────────────────────────

def build():
    products = json.load(open(PRODUCTS_JSON))
    print(f"Loaded {len(products)} products from JSON")

    processor, model, device = load_clip()

    # ── ChromaDB: get or create collection ────────────────────────────────────
    client     = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        collection = client.get_collection(COLLECTION_NAME)
        print(f"ChromaDB collection exists with {collection.count()} vectors")
    except Exception:
        collection = client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        print("ChromaDB collection created")

    # IDs already embedded — skip these
    existing_ids = set()
    if collection.count() > 0:
        existing_ids = set(collection.get(include=[])["ids"])
    print(f"Already embedded: {len(existing_ids)} products")

    # ── SQLite: open store ────────────────────────────────────────────────────
    store = ProductStore()

    new_embedded = 0
    updated_meta = 0

    for i, product in enumerate(products):
        pid = str(i)

        meta      = extract_metadata(product)
        local_img = find_local_image(product.get("image_urls", []), IMAGE_DIR)

        first_img_url = ""
        if product.get("image_urls"):
            first_img_url = product["image_urls"][0].get("url", "")

        local_img_rel = ""
        if local_img:
            try:
                local_img_rel = str(Path(local_img).relative_to(Path(".")))
            except ValueError:
                local_img_rel = local_img

        # Always upsert mutable fields into SQLite — no embedding needed
        store.upsert(
            product_id  = pid,
            title       = product.get("title", ""),
            price       = str(product.get("price", "")),
            url         = product.get("url", ""),
            image_url   = first_img_url,
            local_image = local_img_rel,
        )
        updated_meta += 1

        # Only embed if not already in ChromaDB
        if pid in existing_ids:
            continue

        desc_text = (
            f"{meta['gender']} {meta['fit']} {meta['colour']} "
            f"{meta['type']} {meta['style']}"
        )

        txt_vec = embed_texts([desc_text], processor, model, device)[0]

        if local_img:
            img_vec = embed_image_file(local_img, processor, model, device)
            if img_vec is not None:
                vec = (txt_vec + img_vec) / 2
                vec = vec / np.linalg.norm(vec)
            else:
                vec = txt_vec
        else:
            vec = txt_vec

        # ChromaDB stores only semantic metadata + product_id link to SQLite
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

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(products)}...")

    print(f"\nDone.")
    print(f"  SQLite rows upserted (title/price): {updated_meta}")
    print(f"  New embeddings added to ChromaDB:   {new_embedded}")
    print(f"  Total vectors in ChromaDB:          {collection.count()}")
    print(f"  Total products in SQLite:           {store.count()}")


if __name__ == "__main__":
    build()
