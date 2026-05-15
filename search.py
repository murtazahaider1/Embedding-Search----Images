"""
search.py

Queries ChromaDB for semantic matches, then joins with SQLite (ProductStore)
to get current title, price, and image paths.

ChromaDB  → semantic similarity (type, colour, gender, fit, style, image embedding)
SQLite    → display fields        (title, price, url, local_image, image_url)
"""

import numpy as np
import torch
from PIL import Image

import chromadb
from product_store import ProductStore

# ── Config ────────────────────────────────────────────────────────────────────

CHROMA_DIR        = "zarr_chroma_db"
COLLECTION_NAME   = "zarr_catalog"
SIMILARITY_CUTOFF = 0.20
TOP_K             = 5
PREFETCH_K        = 30

# ── Load ──────────────────────────────────────────────────────────────────────

def load_db():
    client     = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(COLLECTION_NAME)
    store      = ProductStore()
    print(f"ChromaDB: {collection.count()} vectors | SQLite: {store.count()} products")
    return collection, store


# ── Embedding ─────────────────────────────────────────────────────────────────

def _to_tensor(output) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state[:, 0]
    raise ValueError(f"Cannot extract tensor from {type(output)}")


def _embed_text(text, processor, model, device):
    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        out = model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
    feats = _to_tensor(out)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy()[0].astype("float32")


def _embed_image(crop, processor, model, device):
    inputs = processor(images=crop, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.get_image_features(pixel_values=inputs["pixel_values"])
    feats = _to_tensor(out)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy()[0].astype("float32")


def build_query_vector(attrs, crop, processor, model, device):
    desc  = f"{attrs.get('gender','')} {attrs.get('fit','')} {attrs.get('colour','')} {attrs.get('type','')} {attrs.get('style','casual')}".strip()
    t_vec = _embed_text(desc, processor, model, device)
    if crop is not None:
        i_vec    = _embed_image(crop, processor, model, device)
        combined = (t_vec + i_vec) / 2
        combined = combined / np.linalg.norm(combined)
        return combined.tolist()
    return t_vec.tolist()


# ── Metadata filter ───────────────────────────────────────────────────────────

CATEGORY_GROUPS = [
    {"kurta", "shalwar kameez", "kurta trouser", "kurta set", "sherwani", "waistcoat", "kurti"},
    {"t-shirt", "polo shirt", "shirt", "top", "blouse", "hoodie", "sweatshirt", "sweater", "cardigan"},
    {"jeans", "trousers", "chinos", "shorts", "leggings", "skirt"},
    {"jacket", "coat", "blazer"},
    {"dress", "co-ord suit"},
    {"tracksuit", "activewear"},
    {"unstitched"},
]

def same_category(a, b):
    if a == "unknown" or b == "unknown":
        return True
    if a == b:
        return True
    for group in CATEGORY_GROUPS:
        if a in group and b in group:
            return True
    return False


def passes_filter(chroma_meta, attrs):
    q_gender = attrs.get("gender", "unisex")
    q_type   = attrs.get("type", "unknown")
    if q_gender != "unisex" and chroma_meta.get("gender") != "unisex" and chroma_meta.get("gender") != q_gender:
        return False
    if not same_category(q_type, chroma_meta.get("type", "unknown")):
        return False
    return True


# ── Search ────────────────────────────────────────────────────────────────────

def search(attrs, crop, collection, store, processor, model, device):
    query_vec = build_query_vector(attrs, crop, processor, model, device)

    results   = collection.query(
        query_embeddings = [query_vec],
        n_results        = PREFETCH_K,
        include          = ["metadatas", "distances"],
    )

    chroma_metas = results["metadatas"][0]
    distances    = results["distances"][0]

    # Collect product_ids that pass semantic filter
    candidates = []
    seen_pids  = set()
    for meta, dist in zip(chroma_metas, distances):
        similarity = 1.0 - dist
        if similarity < SIMILARITY_CUTOFF:
            continue
        pid = meta.get("product_id", "")
        if pid in seen_pids:
            continue
        if not passes_filter(meta, attrs):
            continue
        seen_pids.add(pid)
        candidates.append((pid, similarity, meta))
        if len(candidates) >= TOP_K:
            break

    if not candidates:
        return []

    # Batch fetch mutable fields from SQLite
    pids      = [c[0] for c in candidates]
    sql_rows  = store.get_many(pids)

    matches = []
    for pid, similarity, chroma_meta in candidates:
        sql = sql_rows.get(pid, {})
        matches.append({
            "title":       sql.get("title", ""),
            "url":         sql.get("url", ""),
            "price":       sql.get("price", ""),
            "image_url":   sql.get("image_url", ""),
            "local_image": sql.get("local_image", ""),
            "similarity":  round(similarity, 4),
        })

    return matches
