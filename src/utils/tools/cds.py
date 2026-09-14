"""Category-aware detector for cognitive-distortion (CDS) n-grams.

This is the validated detector: word-boundary, case-insensitive, one compiled
regex per CDS category. `network_evolution` uses it to recompute CDS from raw
post text, and `checks/check_cds_tracks_phq9.py` uses it on the training corpus.
It imports only the standard library, so it is cheap to import anywhere.
"""
import csv
import json
import re


def load_ngrams_by_category(filepath: str, skip_header=True) -> dict:
    """Load the CDS n-gram TSV, grouped by category.

    Same TSV format as `metrics.load_ngrams_tsv` (category, base marker,
    optional JSON list of variants), but keyed by category so each hit can be
    attributed to a distortion type.

    Args:
        filepath (str): path to `data/distorted_language_ngrams.tsv`.
        skip_header (bool): skip the first row.

    Returns:
        dict[str, set[str]]: category -> lowercased n-grams.
    """
    by_cat: dict[str, set] = {}
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        if skip_header:
            next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            category = row[0].strip() or "uncategorized"
            bucket = by_cat.setdefault(category, set())
            base = row[1].strip().lower()
            if base:
                bucket.add(base)
            if len(row) > 2 and row[2].strip():
                variants_str = row[2].strip()
                try:
                    variants = json.loads(variants_str)
                    if isinstance(variants, list):
                        for v in variants:
                            clean = v.strip().lower()
                            if clean:
                                bucket.add(clean)
                except json.JSONDecodeError:
                    clean = variants_str.lower()
                    if clean:
                        bucket.add(clean)
    return by_cat


def compile_category_patterns(by_cat: dict) -> dict:
    """Compile one word-boundary regex per category.

    Matches all n-grams of a category in one pass, which is much faster than
    one regex per n-gram over tens of thousands of posts.

    Args:
        by_cat (dict): output of `load_ngrams_by_category`.

    Returns:
        dict[str, re.Pattern]: category -> compiled case-insensitive regex.
    """
    patterns = {}
    for cat, ngrams in by_cat.items():
        if not ngrams:
            continue
        alt = "|".join(re.escape(ng) for ng in sorted(ngrams, key=len, reverse=True))
        patterns[cat] = re.compile(r"\b(?:" + alt + r")\b", re.IGNORECASE)
    return patterns
