"""
search.py

Given detected clothing attributes + an optional image crop,
queries ChromaDB and returns ranked matching Zarr products.
"""

import numpy as np
import torch
from PIL import Image

import chromadb

# ── Config ────────────────────────────────────────────────────────────────────

CHROMA_DIR        = "zarr_chroma_db"
COLLECTION_NAME   = "zarr_catalog"
SIMILARITY_CUTOFF = 0.20   # cosine similarity — below this means no real match
TOP_K             = 5      # results to return per detected item
PREFETCH_K        = 30     # candidates to retrieve before metadata filtering


# ── Load DB ───────────────────────────────────────────────────────────────────

def load_db():
    client     = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(COLLECTION_NAME)
    print(f"ChromaDB loaded: {collection.count()} products")
    return collection


# ── Query embedding ───────────────────────────────────────────────────────────

def _to_tensor(output) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state[:, 0]
    raise ValueError(f"Cannot extract tensor from {type(output)}")


def _embed_text(text: str, processor, model, device) -> np.ndarray:
    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        out   = model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
    feats = _to_tensor(out)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy()[0].astype("float32")


def _embed_image(crop: Image.Image, processor, model, device) -> np.ndarray:
    inputs = processor(images=crop, return_tensors="pt").to(device)
    with torch.no_grad():
        out   = model.get_image_features(pixel_values=inputs["pixel_values"])
    feats = _to_tensor(out)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy()[0].astype("float32")


def build_query_vector(attrs: dict, crop: Image.Image | None, processor, model, device) -> list:
    desc = (
        f"{attrs.get('gender', '')} {attrs.get('fit', '')} "
        f"{attrs.get('colour', '')} {attrs.get('type', '')} {attrs.get('style', 'casual')}"
    ).strip()

    t_vec = _embed_text(desc, processor, model, device)

    if crop is not None:
        i_vec    = _embed_image(crop, processor, model, device)
        combined = (t_vec + i_vec) / 2
        combined = combined / np.linalg.norm(combined)
        return combined.tolist()

    return t_vec.tolist()


# ── Metadata hard-filter ──────────────────────────────────────────────────────

# Broad category groups — items within a group are considered same category
CATEGORY_GROUPS = [
    {"kurta", "shalwar kameez", "kurta trouser", "kurta set", "sherwani", "waistcoat", "kurti"},
    {"t-shirt", "polo shirt", "shirt", "top", "blouse", "hoodie", "sweatshirt", "sweater", "cardigan"},
    {"jeans", "trousers", "chinos", "shorts", "leggings", "skirt"},
    {"jacket", "coat", "blazer"},
    {"dress", "co-ord suit"},
    {"tracksuit", "activewear"},
    {"unstitched"},
]

def same_category(a: str, b: str) -> bool:
    if a == "unknown" or b == "unknown":
        return True
    if a == b:
        return True
    for group in CATEGORY_GROUPS:
        if a in group and b in group:
            return True
    return False


def passes_filter(meta: dict, attrs: dict) -> bool:
    q_gender = attrs.get("gender", "unisex")
    q_type   = attrs.get("type", "unknown")

    # Gender: must match unless one side is unisex
    if q_gender != "unisex" and meta.get("gender") != "unisex" and meta.get("gender") != q_gender:
        return False

    # Type: must be in same broad category
    if not same_category(q_type, meta.get("type", "unknown")):
        return False

    return True


# ── Search ────────────────────────────────────────────────────────────────────

def search(
    attrs: dict,
    crop: Image.Image | None,
    collection,
    processor,
    model,
    device,
) -> list[dict]:
    query_vec = build_query_vector(attrs, crop, processor, model, device)

    results = collection.query(
        query_embeddings=[query_vec],
        n_results=PREFETCH_K,
        include=["metadatas", "distances"],
    )

    metadatas = results["metadatas"][0]
    distances = results["distances"][0]   # ChromaDB cosine = 1 - similarity

    matches = []
    seen    = set()

    for meta, dist in zip(metadatas, distances):
        similarity = 1.0 - dist          # convert distance → similarity
        if similarity < SIMILARITY_CUTOFF:
            continue
        uid = meta.get("url", meta.get("title"))
        if uid in seen:
            continue
        if not passes_filter(meta, attrs):
            continue
        seen.add(uid)
        matches.append({
            "title":       meta.get("title", ""),
            "url":         meta.get("url", ""),
            "price":       meta.get("price", ""),
            "image_url":   meta.get("image_url", ""),
            "local_image": meta.get("local_image", ""),
            "type":        meta.get("type", ""),
            "colour":      meta.get("colour", ""),
            "gender":      meta.get("gender", ""),
            "fit":         meta.get("fit", ""),
            "style":       meta.get("style", ""),
            "similarity":  round(similarity, 4),
        })
        if len(matches) >= TOP_K:
            break

    return matches
