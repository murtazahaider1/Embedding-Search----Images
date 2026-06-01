"""
search.py  v4  —  image-first

Architecture change: query vector is now 85% image / 15% text.
The text component is a minimal gender+category hint only — we no longer
rely on CLIP zero-shot type classification (which misclassifies Eastern wear).

Filter is simplified to three hard rules:
  1. Gender — enforced only when confidence is high (passed in via attrs)
  2. Super-category — blocks accessories from clothing queries and vice versa
  3. Non-clothing — blocks fragrances/beauty/grooming from clothing queries

Type-level filtering is removed. Visual similarity does precision work.
PREFETCH_K raised to 200 so the filter has enough candidates.
"""

import numpy as np
import torch
from PIL import Image

import chromadb
from product_store import ProductStore

# ── Config ────────────────────────────────────────────────────────────────────

CHROMA_DIR        = "zarr_chroma_db"
COLLECTION_NAME   = "zarr_catalog"
SIMILARITY_CUTOFF = 0.18   # lowered — image similarity is the precision gate now
TOP_K             = 5
PREFETCH_K        = 200    # fetch many; super-category filter then similarity rank


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
    inputs = processor(
        text=[text], return_tensors="pt", padding=True, truncation=True
    ).to(device)
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
    """
    85% image embedding from the detected crop + 15% text hint.

    The text hint is intentionally minimal — just gender and a broad
    category word. We do NOT encode the CLIP-classified type (e.g. "dress",
    "top") because for Eastern wear those classifications are unreliable and
    pull the query vector in the wrong direction.

    If no crop is available (fallback to full image), we still use 85/15
    because the full image is more informative than any text label.
    """
    # Minimal text hint: just gender + broad style signal
    gender_word = attrs.get("gender", "")
    colour_word = attrs.get("colour", "")
    # Only include colour if it's not unknown — unknown colour adds noise
    colour_part = colour_word if colour_word and colour_word != "unknown" else ""
    hint = f"{gender_word} {colour_part} clothing fashion".strip()

    t_vec = _embed_text(hint, processor, model, device)

    if crop is not None:
        i_vec    = _embed_image(crop, processor, model, device)
        # 85% image, 15% text
        combined = 0.85 * i_vec + 0.15 * t_vec
        combined = combined / np.linalg.norm(combined)
        return combined.tolist()

    # No crop — text-only fallback (rare, only if YOLO gives nothing)
    return t_vec.tolist()


# ── Super-category groups ─────────────────────────────────────────────────────
# These are BROAD groups used only to prevent category-crossing results.
# We no longer do fine-grained type matching — visual similarity handles that.

# Non-clothing product types that should never appear in clothing searches
NON_CLOTHING_TYPES = {
    "fragrance", "beauty", "grooming",
}

# Accessory types — blocked from clothing queries and vice versa
ACCESSORY_TYPES = {
    "scarf", "hijab", "bag", "cap", "belt", "jewellery",
    "sunglasses", "watch", "socks", "gloves",
}

# All clothing types — used to block accessories from clothing queries
CLOTHING_TYPES = {
    "t-shirt", "polo shirt", "shirt", "hoodie", "sweatshirt", "sweater",
    "cardigan", "top", "blouse", "jacket", "coat", "blazer", "puffer jacket",
    "denim jacket", "windbreaker", "jeans", "trousers", "chinos", "shorts",
    "leggings", "skirt", "dress", "maxi", "jumpsuit", "tracksuit", "activewear",
    "joggers", "kurta", "shalwar kameez", "kurta trouser", "sherwani",
    "kurti", "kurta set", "dupatta set", "co-ord suit", "saree", "lehenga",
    "sharara", "kaftan", "abaya", "inner set", "culotte", "romper",
    "waistcoat", "kimono", "shrug", "unstitched", "unknown",
}

# Footwear — only matches footwear queries
FOOTWEAR_TYPES = {
    "sneakers", "formal shoes", "sandals", "boots", "sports shoes",
}


def _super_category(type_str: str) -> str:
    """Return the broad super-category for a product type."""
    t = (type_str or "").lower().strip()
    if t in NON_CLOTHING_TYPES:
        return "non_clothing"
    if t in ACCESSORY_TYPES:
        return "accessory"
    if t in FOOTWEAR_TYPES:
        return "footwear"
    if t in CLOTHING_TYPES or not t:
        return "clothing"
    return "clothing"   # unknown types default to clothing


def passes_filter(chroma_meta: dict, attrs: dict) -> bool:
    """
    Simplified three-rule filter. Type-level matching removed.

    Rule 1 — Gender: only enforced when query gender is specific (not unisex)
             AND catalog gender is specific AND they differ. Both must be
             definite for this to block — gives maximum recall on Eastern wear
             where gender signals are weaker.

    Rule 2 — Super-category: clothing queries never return accessories,
             footwear, or non-clothing (fragrances/beauty/grooming).
             Footwear queries only return footwear.
             Accessory queries only return accessories.

    Rule 3 — Non-clothing products never returned for any clothing query.
    """
    q_gender = attrs.get("gender", "unisex")
    q_type   = attrs.get("type", "unknown")
    cat_type = chroma_meta.get("type", "unknown")
    cat_gender = chroma_meta.get("gender", "unisex")

    # Rule 1 — Gender (strict only when both sides are definite)
    if (q_gender not in ("unisex", "")
            and cat_gender not in ("unisex", "")
            and q_gender != cat_gender):
        return False

    # Rule 2 & 3 — Super-category cross-blocking
    q_super   = _super_category(q_type)
    cat_super = _super_category(cat_type)

    # Non-clothing never returned for clothing/accessory/footwear queries
    if cat_super == "non_clothing" and q_super != "non_clothing":
        return False

    # Footwear only matches footwear
    if q_super == "footwear" and cat_super != "footwear":
        return False
    if cat_super == "footwear" and q_super != "footwear":
        return False

    # Accessories don't mix with clothing (but accessories can match accessories)
    if q_super == "clothing" and cat_super == "accessory":
        return False
    if q_super == "accessory" and cat_super == "clothing":
        return False

    return True


# ── Search ────────────────────────────────────────────────────────────────────

def search(attrs, crop, collection, store, processor, model, device):
    query_vec = build_query_vector(attrs, crop, processor, model, device)

    results = collection.query(
        query_embeddings = [query_vec],
        n_results        = PREFETCH_K,
        include          = ["metadatas", "distances"],
    )

    chroma_metas = results["metadatas"][0]
    distances    = results["distances"][0]

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

    pids     = [c[0] for c in candidates]
    sql_rows = store.get_many(pids)

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
