"""Vocabulary that maps everyday words to the dataset's real values.

This is NOT hard-coding user filters: nothing here is applied unless the
user's own words trigger it. It is a translation table from natural
language ("shampoo") to the dataset's category names
("Hair > Shampoo & Conditioner > Shampoo"). Every value below was checked
against products_clean.csv (see tests/test_query_parsing.py).
"""

from __future__ import annotations

from sephra.query.schema import CategoryFilter as C

# ---------------------------------------------------------------------------
# Product types -> category paths.
# Order matters: more specific phrases first ("dry shampoo" before
# "shampoo", "eye cream" before "cream"). A word already matched by an
# earlier pattern is not matched again.
# ---------------------------------------------------------------------------
PRODUCT_TYPE_PATTERNS: list[tuple[str, list[C]]] = [
    (r"dry shampoos?", [C(primary="Hair", tertiary="Dry Shampoo")]),
    (r"leave[- ]in(?: conditioners?)?", [C(primary="Hair", tertiary="Leave-In Conditioner")]),
    (r"shampoos?", [C(primary="Hair", tertiary="Shampoo")]),
    (r"conditioners?", [C(primary="Hair", tertiary="Conditioner")]),
    (r"hair masks?", [C(primary="Hair", tertiary="Hair Masks")]),
    (r"hair oils?", [C(primary="Hair", tertiary="Hair Oil")]),
    (r"hair ?sprays?", [C(primary="Hair", tertiary="Hair Spray")]),
    (r"scalp (?:treatments?|serums?|scrubs?)", [C(primary="Hair", tertiary="Scalp Treatments")]),
    (r"hair ?dryers?|blow ?dryers?", [C(primary="Hair", tertiary="Hair Dryers")]),
    (r"flat ?irons?|straighteners?", [C(primary="Hair", tertiary="Hair Straighteners & Flat Irons")]),
    (r"curling irons?", [C(primary="Hair", tertiary="Curling Irons")]),
    (r"eye creams?|eye treatments?|under[- ]eye creams?", [C(primary="Skincare", tertiary="Eye Creams & Treatments")]),
    (r"eye masks?|eye patch(?:es)?", [C(primary="Skincare", tertiary="Eye Masks")]),
    (r"lip masks?|lip balms?|lip treatments?",
     [C(primary="Skincare", secondary="Lip Balms & Treatments"), C(primary="Makeup", tertiary="Lip Balm & Treatment")]),
    (r"face oils?|facial oils?", [C(primary="Skincare", tertiary="Face Oils")]),
    (r"night creams?", [C(primary="Skincare", tertiary="Night Creams")]),
    (r"body mists?|hair mists?", [C(primary="Fragrance", tertiary="Body Mist & Hair Mist")]),
    (r"(?:face |facial )?mists?|essences?", [C(primary="Skincare", tertiary="Mists & Essences")]),
    (r"tinted moisturi[sz]ers?", [C(primary="Makeup", tertiary="Tinted Moisturizer")]),
    (r"bb creams?|cc creams?",
     [C(primary="Makeup", tertiary="BB & CC Cream"), C(primary="Skincare", tertiary="BB & CC Creams")]),
    (r"body (?:lotions?|oils?|creams?|butters?|moisturi[sz]ers?)",
     [C(primary="Bath & Body", tertiary="Body Lotions & Body Oils")]),
    (r"hand creams?|foot creams?", [C(primary="Bath & Body", tertiary="Hand Cream & Foot Cream")]),
    (r"body washe?s?|shower gels?", [C(primary="Bath & Body", tertiary="Body Wash & Shower Gel")]),
    (r"body scrubs?", [C(primary="Bath & Body", tertiary="Scrub & Exfoliants")]),
    (r"deodorants?|antiperspirants?", [C(primary="Bath & Body", tertiary="Deodorant & Antiperspirant")]),
    (r"moisturi[sz]ers?|moisturi[sz]ing creams?|face creams?|day creams?",
     [C(primary="Skincare", tertiary="Moisturizers"), C(primary="Skincare", tertiary="Night Creams")]),
    (r"(?:face |facial )?serums?", [C(primary="Skincare", tertiary="Face Serums")]),
    (r"(?:facial )?peels?", [C(primary="Skincare", tertiary="Facial Peels")]),
    (r"acne treatments?|spot treatments?|blemish treatments?",
     [C(primary="Skincare", tertiary="Blemish & Acne Treatments")]),
    (r"(?:face |facial )?cleansers?|face ?washe?s?|cleansing (?:balms?|oils?|gels?)",
     [C(primary="Skincare", tertiary="Face Wash & Cleansers"), C(primary="Men", tertiary="Face Wash")]),
    (r"toners?", [C(primary="Skincare", tertiary="Toners")]),
    (r"exfoliators?|exfoliants?", [C(primary="Skincare", tertiary="Exfoliators")]),
    (r"makeup removers?", [C(primary="Skincare", tertiary="Makeup Removers"), C(primary="Makeup", tertiary="Makeup Removers")]),
    (r"sheet masks?", [C(primary="Skincare", tertiary="Sheet Masks")]),
    (r"(?:face |facial )?masks?", [C(primary="Skincare", secondary="Masks")]),
    (r"sunscreens?|sun ?screens?|sunblocks?|sun protection",
     [C(primary="Skincare", secondary="Sunscreen"), C(primary="Men", tertiary="Sunscreen")]),
    (r"self[- ]tann(?:ers?|ing)", [C(secondary="Self Tanners")]),
    (r"liquid lipsticks?", [C(primary="Makeup", tertiary="Liquid Lipstick")]),
    (r"lipsticks?", [C(primary="Makeup", tertiary="Lipstick"), C(primary="Makeup", tertiary="Liquid Lipstick")]),
    (r"lip gloss(?:es)?", [C(primary="Makeup", tertiary="Lip Gloss")]),
    (r"lip liners?", [C(primary="Makeup", tertiary="Lip Liner")]),
    (r"lip stains?", [C(primary="Makeup", tertiary="Lip Stain")]),
    (r"lip plumpers?", [C(primary="Makeup", tertiary="Lip Plumper")]),
    (r"mascaras?", [C(primary="Makeup", tertiary="Mascara")]),
    (r"eye ?liners?", [C(primary="Makeup", tertiary="Eyeliner")]),
    (r"(?:eyeshadow|eye shadow|eye) palettes?", [C(primary="Makeup", tertiary="Eye Palettes")]),
    (r"eye ?shadows?", [C(primary="Makeup", tertiary="Eyeshadow"), C(primary="Makeup", tertiary="Eye Palettes")]),
    (r"eye primers?", [C(primary="Makeup", tertiary="Eye Primer")]),
    (r"(?:eye)?brow (?:gels?|pencils?|pomades?|products?)", [C(primary="Makeup", tertiary="Eyebrow")]),
    (r"false (?:eye)?lashes", [C(primary="Makeup", tertiary="False Eyelashes")]),
    (r"foundations?", [C(primary="Makeup", tertiary="Foundation")]),
    (r"concealers?", [C(primary="Makeup", tertiary="Concealer"), C(primary="Makeup", tertiary="Under-Eye Concealer")]),
    (r"hair primers?", [C(primary="Hair", tertiary="Hair Primers")]),
    (r"(?:face )?primers?", [C(primary="Makeup", tertiary="Face Primer")]),
    (r"setting (?:sprays?|powders?)", [C(primary="Makeup", tertiary="Setting Spray & Powder")]),
    (r"highlighters?", [C(primary="Makeup", tertiary="Highlighter")]),
    (r"blush(?:es)?", [C(primary="Makeup", tertiary="Blush")]),
    (r"bronzers?", [C(primary="Makeup", tertiary="Bronzer")]),
    (r"contour", [C(primary="Makeup", tertiary="Contour")]),
    (r"makeup brush(?:es)?|brush sets?", [C(primary="Makeup", secondary="Brushes & Applicators")]),
    (r"(?:makeup )?sponges?", [C(primary="Makeup", tertiary="Sponges & Applicators")]),
    (r"nail polish(?:es)?|nails?", [C(primary="Makeup", secondary="Nail")]),
    (r"candles?", [C(primary="Fragrance", tertiary="Candles")]),
    (r"diffusers?", [C(primary="Fragrance", tertiary="Diffusers")]),
    (r"colognes?", [C(primary="Fragrance", tertiary="Cologne")]),
    (r"perfumes?|eau de parfum|eau de toilette",
     [C(primary="Fragrance", tertiary="Perfume"), C(primary="Fragrance", tertiary="Rollerballs & Travel Size"),
      C(primary="Fragrance", tertiary="Cologne")]),
    (r"(?:beauty |hair )?supplements?",
     [C(primary="Skincare", tertiary="Beauty Supplements"), C(primary="Hair", tertiary="Hair Supplements")]),
    (r"gift sets?|value sets?", [C(secondary="Value & Gift Sets")]),
]

# Broad categories: only used when no specific product type was found.
BROAD_CATEGORY_PATTERNS: list[tuple[str, list[C]]] = [
    (r"skin ?care", [C(primary="Skincare")]),
    (r"make ?up", [C(primary="Makeup")]),
    (r"hair ?care|hair products?", [C(primary="Hair")]),
    (r"bath (?:and|&) body|body care|body products?", [C(primary="Bath & Body")]),
    (r"fragrances?|scents?", [C(primary="Fragrance")]),
]

# ---------------------------------------------------------------------------
# Required attributes. Each maps a phrase to a dataset highlight tag.
# A product satisfies the attribute if it carries the tag OR its name
# contains the phrase (e.g. "Hydrating Oil-Free Gel Moisturizer" has no
# "Oil Free" tag but clearly claims it in its name).
# ---------------------------------------------------------------------------
ATTRIBUTE_PATTERNS: dict[str, str] = {
    "Fragrance Free": r"fragrance[- ]free|unscented|no fragrance|without fragrance|scent[- ]free",
    "Vegan": r"vegan",
    "Cruelty-Free": r"cruelty[- ]free",
    "Without Parabens": r"paraben[- ]free|without parabens|no parabens",
    "Without Sulfates SLS & SLES": r"sulfate[- ]free|without sulfates|no sulfates",
    "Without Silicones": r"silicone[- ]free|without silicones|no silicones",
    "Oil Free": r"oil[- ]free",
    "Alcohol Free": r"alcohol[- ]free",
    "Gluten Free": r"gluten[- ]free",
    "Clean at Sephora": r"clean beauty|clean at sephora",
}

# ---------------------------------------------------------------------------
# Ingredients: canonical name -> spellings found in INCI ingredient lists.
# (e.g. hyaluronic acid usually appears as "Sodium Hyaluronate").
# A product matches if its ingredient list, its highlight tags or its name
# contains any spelling.
# ---------------------------------------------------------------------------
INGREDIENT_SYNONYMS: dict[str, list[str]] = {
    "hyaluronic acid": ["hyaluronic", "hyaluronate"],
    "niacinamide": ["niacinamide"],
    "retinol": ["retinol"],
    "retinoids": ["retinol", "retinal", "retinyl", "retinoate"],
    "vitamin c": ["ascorbic acid", "ascorbyl", "ascorbate", "vitamin c"],
    "vitamin e": ["tocopherol", "vitamin e"],
    "salicylic acid": ["salicylic acid"],
    "glycolic acid": ["glycolic acid"],
    "lactic acid": ["lactic acid"],
    "ceramides": ["ceramide"],
    "squalane": ["squalane"],
    "peptides": ["peptide"],
    "bakuchiol": ["bakuchiol"],
    "caffeine": ["caffeine"],
    "centella asiatica": ["centella", "cica"],
    "zinc oxide": ["zinc oxide"],
    "shea butter": ["shea", "butyrospermum"],
    "argan oil": ["argan"],
    "aloe": ["aloe"],
}

# Phrases in the query -> canonical ingredient name.
INGREDIENT_QUERY_PATTERNS: dict[str, str] = {
    r"hyaluronic(?: acid)?|\bha\b": "hyaluronic acid",
    r"niacinamide": "niacinamide",
    r"retinoids?": "retinoids",
    r"retinol": "retinol",
    r"vitamin c|vit c|ascorbic acid": "vitamin c",
    r"vitamin e": "vitamin e",
    r"salicylic(?: acid)?|\bbha\b": "salicylic acid",
    r"glycolic(?: acid)?": "glycolic acid",
    r"lactic acid": "lactic acid",
    r"ceramides?": "ceramides",
    r"squalane": "squalane",
    r"peptides?": "peptides",
    r"bakuchiol": "bakuchiol",
    r"caffeine": "caffeine",
    r"centella|cica": "centella asiatica",
    r"zinc oxide|mineral sunscreen": "zinc oxide",
    r"shea butter": "shea butter",
    r"argan oil": "argan oil",
    r"aloe(?: vera)?": "aloe",
}

# ---------------------------------------------------------------------------
# Skin type suitability tags (describe the PRODUCT, from highlights).
# ---------------------------------------------------------------------------
SKIN_TYPE_TAGS: dict[str, set[str]] = {
    "dry": {"Best for Dry Skin", "Best for Dry, Combo, Normal Skin", "Good for: Dryness"},
    "oily": {"Best for Oily Skin", "Best for Oily, Combo, Normal Skin", "Oil Free"},
    "combination": {"Best for Combination Skin", "Best for Dry, Combo, Normal Skin",
                    "Best for Oily, Combo, Normal Skin"},
    "normal": {"Best for Normal Skin", "Best for Dry, Combo, Normal Skin",
               "Best for Oily, Combo, Normal Skin"},
    "sensitive": {"Fragrance Free"},  # no explicit tag exists; weak proxy, documented
}

# Brands whose names are ordinary English words. They are only treated as a
# brand filter when the query makes that clear ("from Fresh", "Fresh brand").
AMBIGUOUS_BRANDS: set[str] = {
    "fresh", "slip", "merit", "prima", "virtue", "verb", "dame", "habit", "wishful",
    "duo", "origins", "lawless", "heretic", "google", "commodity", "philosophy",
    "topicals", "hourglass", "the maker", "the outset", "peace out", "the ordinary",
    "naturally serious",
}
