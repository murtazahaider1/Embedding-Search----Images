"""
build_index.py

Reads zarr_products.json, extracts metadata, embeds each product using
Fashion-CLIP, and stores everything in a local ChromaDB collection.

ChromaDB was chosen over FAISS because:
- Stores embeddings + metadata + documents together (no separate JSON file)
- Supports metadata filtering (filter by gender, type, etc. before vector search)
- Persists to disk automatically, no manual save/load
- Simple Python API, no index management needed

Run once to build the database:
    python3 build_index.py

Re-run any time to rebuild from scratch.
"""

import json
import os
import re
import torch
import numpy as np
from PIL import Image
from io import BytesIO
from pathlib import Path

import chromadb
from chromadb.config import Settings
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification

from extract_metadata import extract_metadata

# ── Config ────────────────────────────────────────────────────────────────────

PRODUCTS_JSON   = "zarr_products.json"
IMAGE_DIR       = "zarr_data/image_resources"   # local folder with downloaded images
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
    """Extract embedding tensor regardless of whether model returns a raw tensor
    or a ModelOutput dataclass (varies by transformers version)."""
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state[:, 0]
    raise ValueError(f"Cannot extract tensor from {type(output)}")


def embed_texts(texts: list[str], processor, model, device) -> np.ndarray:
    all_vecs = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch  = texts[i:i + BATCH_SIZE]
        inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            out   = model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
        feats = _to_tensor(out)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        all_vecs.append(feats.cpu().numpy())
    return np.vstack(all_vecs).astype("float32")


def embed_image_file(path: str, processor, model, device) -> np.ndarray | None:
    try:
        img    = Image.open(path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            out   = model.get_image_features(pixel_values=inputs["pixel_values"])
        feats = _to_tensor(out)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy()[0].astype("float32")
    except Exception as e:
        print(f"  Image embed failed ({path}): {e}")
        return None


def find_local_image(image_urls: list[dict], image_dir: str) -> str | None:
    """Find the first matching local file for a product's image list."""
    img_dir = Path(image_dir)
    for img in image_urls:
        fname = img.get("filename", "")
        # Direct match
        candidate = img_dir / fname
        if candidate.exists():
            return str(candidate)
        # Stem match (ignore extension differences)
        stem = Path(fname).stem
        for f in img_dir.iterdir():
            if f.stem == stem:
                return str(f)
    return None


# ── Build ─────────────────────────────────────────────────────────────────────

def build():
    products = json.load(open(PRODUCTS_JSON))
    print(f"Loaded {len(products)} products")

    processor, model, device = load_clip()

    client     = chromadb.PersistentClient(path=CHROMA_DIR)
    # Delete existing collection if rebuilding
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids         = []
    embeddings  = []
    metadatas   = []
    documents   = []

    skipped = 0
    for i, product in enumerate(products):
        meta = extract_metadata(product)

        # Build rich text description for text embedding
        desc_text = (
            f"{meta['gender']} {meta['fit']} {meta['colour']} {meta['type']} {meta['style']} — "
            f"{product['title']}"
        )

        # Try to embed the local product image; fall back to text embedding
        local_img = find_local_image(product.get("image_urls", []), IMAGE_DIR)
        if local_img:
            img_vec = embed_image_file(local_img, processor, model, device)
        else:
            img_vec = None

        # Text embedding
        txt_vec = embed_texts([desc_text], processor, model, device)[0]

        # Final vector: average image + text if image is available
        if img_vec is not None:
            vec = (txt_vec + img_vec) / 2
            vec = vec / np.linalg.norm(vec)
        else:
            vec = txt_vec
            skipped += 1

        # First image URL for frontend display
        first_img_url = ""
        if product.get("image_urls"):
            first_img_url = product["image_urls"][0]["url"]

        # First local image path (relative) for serving
        local_img_rel = ""
        if local_img:
            local_img_rel = str(Path(local_img).relative_to(".")) if Path(local_img).is_absolute() else local_img

        chroma_meta = {
            "title":         product["title"],
            "url":           product.get("url", ""),
            "price":         str(product.get("price", "")),
            "image_url":     first_img_url,
            "local_image":   local_img_rel,
            "gender":        meta["gender"],
            "type":          meta["type"],
            "colour":        meta["colour"],
            "fit":           meta["fit"],
            "style":         meta["style"],
        }

        ids.append(str(i))
        embeddings.append(vec.tolist())
        metadatas.append(chroma_meta)
        documents.append(desc_text)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(products)}...")

    # Insert in batches
    batch = 100
    for i in range(0, len(ids), batch):
        collection.add(
            ids=ids[i:i+batch],
            embeddings=embeddings[i:i+batch],
            metadatas=metadatas[i:i+batch],
            documents=documents[i:i+batch],
        )

    print(f"\nDone. {collection.count()} vectors in ChromaDB.")
    print(f"Products without local images (text-only embed): {skipped}")
    print(f"Database saved to: {CHROMA_DIR}/")


if __name__ == "__main__":
    build()
