"""
extract_metadata.py

Parses raw Zarr product JSON and extracts structured metadata fields:
type, colour, gender, fit, style.

All logic is keyword-based against the product title + description.
"""

import re

# ── Keyword maps ──────────────────────────────────────────────────────────────

GENDER_MAP = {
    "men":   ["men's", "mens", " men ", "gents", "male", "boys", "boy's"],
    "women": ["women's", "womens", " women ", "ladies", "female", "girls", "girl's", "ladies'"],
}

# Ordered longest-first so "shalwar kameez" matches before "kameez"
TYPE_MAP = [
    ("shalwar kameez",  ["shalwar kameez", "shalwar-kameez"]),
    ("kurta trouser",   ["kurta trouser", "kurta pant"]),
    ("sherwani",        ["sherwani"]),
    ("waistcoat",       ["waistcoat", "vest"]),
    ("kurti",           ["kurti"]),
    ("kurta set",       ["kurta set", "co-ord kurta"]),
    ("kurta",           ["kurta"]),
    ("unstitched",      ["unstitched"]),
    ("co-ord suit",     ["co-ord", "coord suit", "co ord"]),
    ("tracksuit",       ["tracksuit", "track suit"]),
    ("hoodie",          ["hoodie"]),
    ("sweatshirt",      ["sweatshirt"]),
    ("sweater",         ["sweater", "knitwear", "knit"]),
    ("cardigan",        ["cardigan"]),
    ("polo shirt",      ["polo shirt", "polo"]),
    ("t-shirt",         ["t-shirt", "tshirt", "t shirt", "tee"]),
    ("shirt",           ["shirt"]),
    ("top",             ["top", "blouse", "crop top"]),
    ("jacket",          ["jacket"]),
    ("coat",            ["coat", "overcoat"]),
    ("blazer",          ["blazer"]),
    ("jeans",           ["jeans", "denim"]),
    ("chinos",          ["chinos", "chino"]),
    ("trousers",        ["trouser", "trousers", "pant", "pants", "slacks"]),
    ("shorts",          ["shorts", "short"]),
    ("leggings",        ["legging", "tights"]),
    ("skirt",           ["skirt"]),
    ("dress",           ["dress", "frock", "maxi", "jumpsuit"]),
    ("activewear",      ["activewear", "sportswear", "gym wear"]),
]

COLOUR_MAP = [
    ("navy blue",    ["navy blue", "navy"]),
    ("royal blue",   ["royal blue"]),
    ("sky blue",     ["sky blue"]),
    ("light blue",   ["light blue", "baby blue"]),
    ("dark blue",    ["dark blue"]),
    ("blue",         ["blue"]),
    ("black",        ["black"]),
    ("white",        ["white"]),
    ("light grey",   ["light grey", "light gray"]),
    ("dark grey",    ["dark grey", "dark gray", "charcoal"]),
    ("grey",         ["grey", "gray"]),
    ("red",          ["red"]),
    ("burgundy",     ["burgundy", "maroon", "wine"]),
    ("coral",        ["coral"]),
    ("salmon",       ["salmon"]),
    ("hot pink",     ["hot pink", "fuchsia", "magenta"]),
    ("light pink",   ["light pink", "baby pink", "blush"]),
    ("pink",         ["pink"]),
    ("orange",       ["orange"]),
    ("yellow",       ["yellow", "mustard"]),
    ("forest green", ["forest green", "dark green", "bottle green"]),
    ("olive green",  ["olive green", "olive", "khaki green"]),
    ("mint green",   ["mint green", "mint"]),
    ("green",        ["green"]),
    ("brown",        ["brown", "chocolate"]),
    ("tan",          ["tan"]),
    ("camel",        ["camel"]),
    ("beige",        ["beige", "sand"]),
    ("cream",        ["cream", "off white", "off-white", "ivory"]),
    ("purple",       ["purple", "violet"]),
    ("lavender",     ["lavender", "lilac"]),
    ("multicolor",   ["multi", "multicolor", "multicolour", "printed", "check", "stripe", "plaid"]),
]

FIT_MAP = [
    ("baggy",   ["baggy", "balloon", "parachute"]),
    ("loose",   ["loose", "oversized", "relaxed", "wide leg", "wide-leg"]),
    ("slim",    ["slim fit", "skinny", "fitted", "tapered"]),
    ("regular", ["regular fit", "regular"]),
]

STYLE_MAP = [
    ("formal",      ["formal", "office wear", "business"]),
    ("ethnic",      ["ethnic", "traditional", "eastern"]),
    ("streetwear",  ["streetwear", "street wear", "urban"]),
    ("athletic",    ["athletic", "sport", "gym", "active"]),
    ("casual",      ["casual"]),
    ("embroidered", ["embroidered", "embroidery"]),
    ("printed",     ["printed", "print"]),
    ("plain",       ["plain", "solid"]),
    ("checkered",   ["check", "checked", "checkered", "plaid", "tartan"]),
    ("striped",     ["stripe", "striped"]),
]


def _find(text: str, mapping: list) -> str:
    """Return first matching label from a keyword map, or empty string."""
    t = f" {text.lower()} "
    for label, keywords in mapping:
        if any(f" {kw} " in t or t.startswith(f"{kw} ") or kw in t for kw in keywords):
            return label
    return ""


def extract_metadata(item: dict) -> dict:
    text = f"{item.get('title', '')} {item.get('description', '')}"

    gender = "unisex"
    for g, kws in GENDER_MAP.items():
        if any(kw in text.lower() for kw in kws):
            gender = g
            break

    clothing_type = _find(text, TYPE_MAP) or "unknown"
    colour        = _find(text, COLOUR_MAP) or "unknown"
    fit           = _find(text, FIT_MAP) or "regular"
    style         = _find(text, STYLE_MAP) or "casual"

    # Extract explicit colour from "Color: X" pattern in description
    colour_match = re.search(r"[Cc]olor[:\s]+([A-Za-z\s]+)", text)
    if colour_match and colour == "unknown":
        raw = colour_match.group(1).strip().lower().split("\n")[0].split(".")[0]
        for label, kws in COLOUR_MAP:
            if any(kw in raw for kw in kws):
                colour = label
                break

    # Extract fit from "Fit Type - X" pattern
    fit_match = re.search(r"[Ff]it [Tt]ype[\s\-:]+([A-Za-z\s]+)", text)
    if fit_match:
        raw = fit_match.group(1).strip().lower().split("\n")[0]
        for label, kws in FIT_MAP:
            if any(kw in raw for kw in kws):
                fit = label
                break

    return {
        "gender": gender,
        "type":   clothing_type,
        "colour": colour,
        "fit":    fit,
        "style":  style,
    }
