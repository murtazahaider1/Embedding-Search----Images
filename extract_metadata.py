"""
extract_metadata.py  v4

Rewritten for the Shopify Facebook/Google feed CSV format.

Input dict keys (from the CSV row):
    title, description, link, image_link,
    gender      (custom_label_1): "Women" | "Men" | "Kids" | "Unisex"
    category    (custom_label_2): "Eastern" | "Western" | "Ready To Wear" |
                                  "Unstitched" | "Accessories" | "Footwear" |
                                  "Boys" | "Girls" | "Modest Wear" |
                                  "Fragrances" | "Beauty" | "Grooming" |
                                  "Loungewear"
    sub_cat     (custom_label_3): "Shalwar Kameez" | "Co-ord Sets" | etc.
    sub_cat2    (custom_label_4): finer grain
    color:      "Black, " | "Hotpink / Blue, " | nan
    price, sale_price, size, sku, availability, brand, vendor

Compared to v3:
- Gender is read directly from custom_label_1 — no keyword scanning needed
- Category / sub_cat give strong prior for type → less fallback needed
- Colour is read from the `color` column first, then keyword scan as backup
- Kids gender is normalised to "unisex" so the search filter doesn't block them
- Non-clothing categories (Fragrances, Beauty, Grooming) get type = category name
  so they can still be indexed but never clash with clothing searches
"""

import re

# ── Type map ──────────────────────────────────────────────────────────────────
# Order = specificity descending. First match wins on title scan.

TYPE_MAP = [
    # Eastern full-length garments
    ("saree",           ["saree", "sari", " sari "]),
    ("lehenga",         ["lehenga", "lehnga", "lehenga choli", "bridal lehenga"]),
    ("sharara",         ["sharara", "gharara"]),
    ("shalwar kameez",  ["shalwar kameez", "shalwar-kameez", "kameez shalwar",
                         "kameez-shalwar", "salwar kameez", "salwar-kameez",
                         "lucknow suit", "lucknow style"]),
    ("kurta trouser",   ["kurta trouser", "kurta pant", "kurta bottom"]),
    ("sherwani",        ["sherwani"]),
    ("waistcoat",       ["waistcoat", "vasket", "nehru jacket"]),
    ("kaftan",          ["kaftan", "kaftaan", "caftan"]),
    ("abaya",           ["abaya", "abayas", "open abaya"]),
    ("hijab",           ["hijab", "niqab", "khimar"]),
    ("scarf",           ["scarf", "scarves", "stole", "odhni",
                         "shawl", "muffler", "dupatta"]),
    ("kurti",           ["kurti"]),
    ("kurta set",       ["kurta set", "co-ord kurta"]),
    ("kurta",           ["kurta", "kameez"]),
    # Suits / sets (Eastern women)
    ("dupatta set",     ["3-piece", "3 piece", "3pc", "2pc", "2-piece",
                         "with dupatta", "shirt dupatta", "shirt trouser dupatta",
                         "embroidered suit", "embroidered pret",
                         "lawn suit", "stitched suit", "unstitched suit",
                         "pret suit", "luxury pret"]),
    ("co-ord suit",     ["co-ord", "coord suit", "co ord", "coordinate set",
                         "matching set", "ensemble"]),
    ("unstitched",      ["unstitched", "stitched lawn"]),
    ("inner set",       ["inner set", "loungewear set", "pyjama set",
                         "night suit", "sleep set"]),
    ("romper",          ["romper", "bodysuit set", "romper set"]),
    ("culotte",         ["culotte", "culottes", "cullote"]),
    ("kimono",          ["kimono", "shrug kimono"]),
    ("shrug",           ["shrug"]),
    # Outerwear
    ("windbreaker",     ["windbreaker", "wind breaker"]),
    ("puffer jacket",   ["puffer jacket", "quilted jacket", "puffer"]),
    ("denim jacket",    ["denim jacket"]),
    ("jacket",          ["jacket", "shacket", "blouson"]),
    ("coat",            ["coat", "overcoat", "trench"]),
    ("blazer",          ["blazer"]),
    # Tops
    ("hoodie",          ["hoodie", "hoody"]),
    ("sweatshirt",      ["sweatshirt", "high neck", "mockneck", "mock neck",
                         "turtleneck", "zipper upper", "zipper hoody",
                         "half zip", "quarter zip", "zip through", "upper"]),
    ("cardigan",        ["cardigan", "zipcardi", "zip cardi"]),
    ("sweater",         ["sweater", "knitwear", "knit", "pullover",
                         "jumper", "woolens", "fleece top"]),
    ("polo shirt",      ["polo shirt", "polo"]),
    ("t-shirt",         ["t-shirt", "tshirt", "t shirt", "tee",
                         "graphic tee", "round neck tee"]),
    ("shirt",           ["shirt", "casual shirt", "dress shirt",
                         "formal shirt", "oxford", "flannel shirt"]),
    ("top",             ["top", "blouse", "crop top", "tank top", "cami"]),
    # Bottoms
    ("tracksuit",       ["tracksuit", "track suit", "jogger set"]),
    ("joggers",         ["jogger", "joggers", "sweat pants", "sweatpants"]),
    ("jeans",           ["jeans", "denim trouser", "denim pant"]),
    ("chinos",          ["chinos", "chino"]),
    ("trousers",        ["trouser", "trousers", "pant", "pants", "slacks",
                         "bottoms", "bottom wear"]),
    ("shorts",          ["shorts", "short pant"]),
    ("leggings",        ["legging", "tights", "stocking"]),
    ("skirt",           ["skirt"]),
    # Full length
    ("dress",           ["dress", "frock", "shirt dress", "gown"]),
    ("maxi",            ["maxi dress", "maxi", "floor length"]),
    ("jumpsuit",        ["jumpsuit", "playsuit", "dungaree"]),
    # Active
    ("activewear",      ["activewear", "sportswear", "gym wear",
                         "athletic wear", "workout"]),
    # Footwear
    ("sneakers",        ["sneaker", "sneakers", "trainer", "runners",
                         "canvas shoe", "plimsoll"]),
    ("formal shoes",    ["formal shoe", "oxford shoe", "derby", "brogues",
                         "loafer", "monk strap"]),
    ("sandals",         ["sandal", "sandals", "slipper", "chappal",
                         "chappals", "slides", "flip flop", "kolhapuri"]),
    ("boots",           ["boot", "boots", "ankle boot", "chelsea boot"]),
    ("sports shoes",    ["sports shoe", "running shoe", "athletic shoe"]),
    # Accessories
    ("cap",             ["cap", "hat", "beanie", "bucket hat", "snapback"]),
    ("belt",            ["belt", "belts"]),
    ("bag",             ["bag", "tote", "backpack", "clutch", "satchel",
                         "handbag", "wallet", "purse", "pouch"]),
    ("jewellery",       ["necklace", "ring", "earring", "bracelet",
                         "bangle", "jewellery", "jewelry", "cufflink"]),
    ("sunglasses",      ["sunglasses", "eyewear"]),
    ("watch",           ["watch", "timepiece"]),
    ("socks",           ["socks", "sock"]),
]

# sub_cat (custom_label_3) → clothing type.
# Used when title scan doesn't resolve type.
SUB_CAT_TYPE_MAP = {
    "shalwar kameez":        "shalwar kameez",
    "co-ord sets":           "co-ord suit",
    "co-ord suit":           "co-ord suit",
    "abayas":                "abaya",
    "dupattas & shawls":     "scarf",
    "hoodies & sweatshirts": "hoodie",
    "t-shirts & polos":      "t-shirt",
    "t-shirts & polos":      "t-shirt",
    "jackets & coats":       "jacket",
    "shirts":                "shirt",
    "bottoms":               "trousers",
    "unstitched":            "unstitched",
    "3 piece":               "dupatta set",
    "2 piece":               "dupatta set",
    "luxury pret":           "dupatta set",
    "daily pret":            "dupatta set",
    "festive":               "dupatta set",
    "eastern":               "shalwar kameez",
    "sandals":               "sandals",
    "sneakers":              "sneakers",
    "boots":                 "boots",
    "formal shoes":          "formal shoes",
    "bags":                  "bag",
    "caps & hats":           "cap",
    "scarves":               "scarf",
    "kurti":                 "kurti",
    "kurta set":             "kurta set",
    "western":               "shirt",          # generic fallback
    "skincare":              "beauty",
    "haircare":              "beauty",
    "hair care":             "beauty",
}

# category (custom_label_2) → type fallback when everything else fails
CATEGORY_TYPE_FALLBACK = {
    "Eastern":       "shalwar kameez",
    "Ready To Wear": "dupatta set",
    "Ready To wear": "dupatta set",
    "Modest Wear":   "abaya",
    "Unstitched":    "unstitched",
    "Loungewear":    "inner set",
    "Footwear":      "sandals",
    "Accessories":   "scarf",
    "Fragrances":    "fragrance",
    "Beauty":        "beauty",
    "Grooming":      "grooming",
    "Boys":          "shalwar kameez",
    "Girls":         "dupatta set",
}

CATEGORY_STYLE_FALLBACK = {
    "Eastern":       "ethnic",
    "Ready To Wear": "ethnic",
    "Ready To wear": "ethnic",
    "Modest Wear":   "modest wear",
    "Unstitched":    "ethnic",
    "Loungewear":    "casual",
    "Western":       "casual",
    "Footwear":      "casual",
    "Accessories":   "casual",
    "Boys":          "ethnic",
    "Girls":         "ethnic",
}

# ── Colour map ────────────────────────────────────────────────────────────────

COLOUR_MAP = [
    # Blues
    ("navy blue",      ["navy blue", "navy"]),
    ("royal blue",     ["royal blue"]),
    ("cobalt blue",    ["cobalt blue", "cobalt"]),
    ("electric blue",  ["electric blue"]),
    ("sky blue",       ["sky blue", "powder blue", "ice blue",
                        "baby blue", "pale blue"]),
    ("light blue",     ["light blue"]),
    ("dark blue",      ["dark blue", "deep blue"]),
    ("teal",           ["teal", "teal blue", "turquoise", "peacock blue"]),
    ("indigo",         ["indigo"]),
    ("blue",           ["blue", "denim blue"]),
    # Blacks / whites / greys
    ("black",          ["black"]),
    ("white",          ["white", "off-white", "off white", "pearl white",
                        "opal white", "snow white", "pastel white"]),
    ("light grey",     ["light grey", "light gray", "ash grey", "ash gray",
                        "silver grey", "silver gray", "mist grey",
                        "slate grey", "slate gray", "slate"]),
    ("dark grey",      ["dark grey", "dark gray", "charcoal grey",
                        "charcoal gray", "charcoal"]),
    ("grey",           ["grey", "gray", "smoke", "stone grey"]),
    # Reds / pinks
    ("red",            ["red", "brick red", "tomato red", "ruby red",
                        "cherry red", "fire red"]),
    ("burgundy",       ["burgundy", "maroon", "wine", "oxblood",
                        "crimson", "deep red"]),
    ("rust",           ["rust", "burnt orange", "terracotta",
                        "brick orange"]),
    ("coral",          ["coral"]),
    ("hot pink",       ["hot pink", "fuchsia", "magenta", "electric pink",
                        "neon pink", "deep pink", "hotpink"]),
    ("salmon",         ["salmon", "dusty rose", "rose wood"]),
    ("light pink",     ["light pink", "baby pink", "pastel pink", "blush",
                        "rose pink", "powder pink", "blush pink"]),
    ("pink",           ["pink", "rose"]),
    # Oranges / yellows
    ("peach",          ["peach", "apricot", "blush peach"]),
    ("orange",         ["orange", "pumpkin", "amber", "tangerine"]),
    ("gold",           ["gold", "golden", "metallic gold", "mustard gold"]),
    ("yellow",         ["yellow", "mustard", "lemon yellow", "butter yellow",
                        "marigold", "saffron", "chartreuse"]),
    # Greens
    ("forest green",   ["forest green", "dark green", "bottle green",
                        "hunter green", "deep green"]),
    ("olive green",    ["olive green", "olive", "khaki green", "army green",
                        "military green", "moss green"]),
    ("sage green",     ["sage green", "sage"]),
    ("mint green",     ["mint green", "mint"]),
    ("emerald green",  ["emerald green", "emerald"]),
    ("green",          ["green", "lime green", "grass green", "pistachio"]),
    # Browns / neutrals
    ("brown",          ["brown", "chocolate", "cocoa brown", "coffee brown",
                        "dark brown", "mahogany"]),
    ("copper",         ["copper", "bronze"]),
    ("tan",            ["tan", "caramel", "sand brown", "saddle"]),
    ("camel",          ["camel", "camel brown"]),
    ("beige",          ["beige", "sand", "nude", "latte", "mushroom",
                        "wheat", "khaki"]),
    ("cream",          ["cream", "ecru", "ivory", "vanilla",
                        "off white", "off-white"]),
    # Purples
    ("purple",         ["purple", "violet", "plum", "grape", "deep purple"]),
    ("lavender",       ["lavender", "lilac", "mauve", "wisteria",
                        "periwinkle"]),
    # Metallics
    ("silver",         ["silver", "metallic silver", "chrome"]),
    # Multi
    ("multicolor",     ["multi", "multicolor", "multicolour", "multi-color",
                        "tie dye", "tie-dye", "ombre", "gradient",
                        "rainbow", "abstract", "patchwork", "printed"]),
]

FABRIC_WORDS = {
    "lawn", "khaddar", "karandi", "cambric", "chiffon", "georgette",
    "organza", "silk", "velvet", "linen", "cotton", "denim", "jacquard",
    "muzlin", "muslin", "dobby", "crepe", "satin", "net", "wool",
    "fleece", "suede", "leather", "nylon", "polyester", "viscose",
}

# ── Fit map ───────────────────────────────────────────────────────────────────

FIT_MAP = [
    ("baggy",   ["baggy", "balloon", "parachute", "extra loose", "oversize"]),
    ("loose",   ["loose", "oversized", "over-sized", "relaxed",
                 "wide leg", "wide-leg", "boxy", "flowy"]),
    ("slim",    ["slim fit", "slim-fit", "skinny", "fitted", "tapered",
                 "body fit", "figure hugging"]),
    ("regular", ["regular fit", "regular-fit", "regular", "straight fit",
                 "classic fit", "comfort fit"]),
]

# ── Style map ─────────────────────────────────────────────────────────────────

STYLE_MAP = [
    ("formal",       ["formal", "office wear", "business", "corporate"]),
    ("bridal",       ["bridal", "wedding", "barat", "walima", "mehndi",
                      "nikah wear"]),
    ("ethnic",       ["ethnic", "traditional", "eastern", "desi",
                      "festive", "eid wear", "party wear", "cultural",
                      "embellished"]),
    ("modest wear",  ["modest", "abaya", "hijab", "niqab", "islamic wear"]),
    ("streetwear",   ["streetwear", "street wear", "urban", "hype"]),
    ("athletic",     ["athletic", "sport", "gym", "active", "fitness",
                      "workout", "performance"]),
    ("casual",       ["casual", "everyday", "daily wear", "weekend"]),
    ("embroidered",  ["embroidered", "embroidery", "hand embroidered",
                      "zari", "gota", "thread work"]),
    ("printed",      ["printed", "print", "floral print", "abstract print",
                      "digital print"]),
    ("plain",        ["plain", "solid", "solid colour", "self"]),
    ("checkered",    ["check", "checked", "checkered", "plaid", "tartan"]),
    ("striped",      ["stripe", "striped", "stripes", "pinstripe"]),
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def _strip_fabric(text: str) -> str:
    words = text.split()
    return " ".join(w for w in words if w.lower() not in FABRIC_WORDS)


def _find(text: str, mapping: list) -> str:
    t = f" {text.lower()} "
    for label, keywords in mapping:
        for kw in keywords:
            if re.search(rf"(?<![a-z]){re.escape(kw)}(?![a-z])", t):
                return label
    return ""


def _parse_csv_colour(raw: str) -> str:
    """
    Parse the CSV `color` field: "Hotpink / Blue, " → "hot pink"
    Takes only the first colour token before '/' or ',' and keyword-matches it.
    """
    if not raw or str(raw).lower() == "nan":
        return ""
    first = re.split(r"[/,]", raw)[0].strip()
    result = _find(_strip_fabric(first), COLOUR_MAP)
    return result


def _url_to_handle(url: str) -> str:
    # Strip query string (?utm_source=...) then extract handle
    url = url.split("?")[0]
    if "/products/" in url:
        return url.split("/products/")[-1].rstrip("/")
    return url


# ── Public API ────────────────────────────────────────────────────────────────

def extract_metadata(item: dict) -> dict:
    """
    item is a dict built from one CSV row with these keys:
        title, description, link, image_link,
        gender (custom_label_1), category (custom_label_2),
        sub_cat (custom_label_3), sub_cat2 (custom_label_4),
        color, price, sale_price
    """
    title    = str(item.get("title", "") or "")
    desc     = str(item.get("description", "") or "")
    category = str(item.get("category", "") or "")          # custom_label_2
    sub_cat  = str(item.get("sub_cat", "") or "").lower()   # custom_label_3
    color_raw = str(item.get("color", "") or "")
    handle   = _url_to_handle(str(item.get("link", "") or ""))
    text     = f"{title} {desc}"

    # ── Gender ────────────────────────────────────────────────────────────────
    # CSV custom_label_1 is authoritative: "Women" | "Men" | "Kids" | "Unisex"
    raw_gender = str(item.get("gender", "") or "").strip().lower()
    raw_gender = raw_gender.strip('[""]')   # handle '["Women"]' format
    if "women" in raw_gender or "girl" in raw_gender:
        gender = "women"
    elif "men" in raw_gender:
        gender = "men"
    elif "kid" in raw_gender or "boy" in raw_gender:
        gender = "unisex"   # Kids indexed as unisex so they surface for both
    else:
        gender = "unisex"

    # ── Type ──────────────────────────────────────────────────────────────────
    # Priority: title/desc keyword → sub_cat label → category fallback
    clothing_type = _find(text, TYPE_MAP)

    if not clothing_type:
        clothing_type = SUB_CAT_TYPE_MAP.get(sub_cat.strip(), "")

    if not clothing_type and category in CATEGORY_TYPE_FALLBACK:
        clothing_type = CATEGORY_TYPE_FALLBACK[category]

    if not clothing_type:
        clothing_type = "unknown"

    # ── Colour ────────────────────────────────────────────────────────────────
    # Priority: CSV color column → keyword scan on title/desc → handle scan
    colour = _parse_csv_colour(color_raw)
    if not colour:
        colour = _find(_strip_fabric(text), COLOUR_MAP)
    if not colour:
        colour = _find(text, COLOUR_MAP)
    if not colour:
        colour = _find(handle.replace("-", " "), COLOUR_MAP)
    if not colour:
        colour = "unknown"

    # ── Fit ───────────────────────────────────────────────────────────────────
    fit = _find(text, FIT_MAP)
    if not fit:
        fit = "regular"

    # ── Style ─────────────────────────────────────────────────────────────────
    style = _find(text, STYLE_MAP)
    if not style and category in CATEGORY_STYLE_FALLBACK:
        style = CATEGORY_STYLE_FALLBACK[category]
    if not style:
        style = "casual"

    return {
        "gender": gender,
        "type":   clothing_type,
        "colour": colour,
        "fit":    fit,
        "style":  style,
    }
