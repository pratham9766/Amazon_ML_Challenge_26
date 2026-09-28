"""Offline string diagnostics, with an explicit equivalent-formula fallback."""
import numpy as np
from .normalization import norm, suffix, numbers, postal
try:
    from rapidfuzz import fuzz
    from rapidfuzz.distance import Levenshtein
    BACKEND = 'RapidFuzz'
except ImportError:
    fuzz = None
    BACKEND = 'Python exact normalized-Indel/token formulas; RapidFuzz package unavailable offline'

def ratio(a, b):
    if fuzz:
        return fuzz.ratio(a, b) / 100
    if not a and not b:
        return 1.0
    masks = {}
    for i, c in enumerate(a):
        masks[c] = masks.get(c, 0) | (1 << i)
    state = 0
    for c in b:
        x = state | masks.get(c, 0)
        state = x & ~(x - ((state << 1) | 1))
    return 2 * state.bit_count() / (len(a) + len(b))

def lev(a, b):
    if fuzz:
        return Levenshtein.normalized_similarity(a, b)
    if not a and not b:
        return 1.0
    # Myers bit-vector edit distance, unit insertion/deletion/substitution costs.
    if not a or not b:
        return 0.0
    masks = {}
    for i, c in enumerate(a):
        masks[c] = masks.get(c, 0) | (1 << i)
    vp, vn, distance = ~0, 0, len(a)
    top = 1 << (len(a) - 1)
    for c in b:
        eq = masks.get(c, 0)
        x = eq | vn
        d0 = (((eq & vp) + vp) ^ vp) | eq | vn
        hp, hn = vn | ~(d0 | vp), vp & d0
        distance += bool(hp & top) - bool(hn & top)
        hp = (hp << 1) | 1
        hn <<= 1
        vp, vn = hn | ~(x | hp), d0 & hp
    return 1 - distance / max(len(a), len(b))

def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else np.nan

def token_set(a, b):
    if fuzz:
        return fuzz.token_set_ratio(a, b) / 100
    a, b = set(a.split()), set(b.split())
    if not a or not b:
        return 0.0
    common = ' '.join(sorted(a & b))
    aa = ' '.join(filter(None, [common, ' '.join(sorted(a - b))]))
    bb = ' '.join(filter(None, [common, ' '.join(sorted(b - a))]))
    return max(ratio(aa, bb), ratio(common, aa), ratio(common, bb)) if common else ratio(aa, bb)

def features(a, b):
    out = {}
    for field, col in [('name', 'name'), ('address', 'address')]:
        x, y = a[col], b[col]
        nx, ny = norm(x), norm(y)
        raw_valid = bool(x.strip() and y.strip())
        valid = bool(nx and ny)
        out[field + '_raw_equal'] = float(raw_valid and x == y)
        out[field + '_normalized_equal'] = float(valid and nx == ny)
        out[field + '_jaccard'] = jaccard(nx.split(), ny.split()) if valid else np.nan
        out[field + '_ratio'] = ratio(nx, ny) if valid else np.nan
        out[field + '_token_sort_ratio'] = ratio(' '.join(sorted(nx.split())), ' '.join(sorted(ny.split()))) if valid else np.nan
        out[field + '_token_set_ratio'] = token_set(nx, ny) if valid else np.nan
        out[field + '_levenshtein'] = lev(nx, ny) if valid else np.nan
        if field == 'name':
            out.update(name_lowercase_equal=float(raw_valid and x.lower() == y.lower()),
                       name_punctuation_equal=float(valid and norm(x) == norm(y)),
                       name_whitespace_equal=float(raw_valid and ' '.join(x.split()) == ' '.join(y.split())),
                       name_suffix_equal=float(bool(suffix(x)) and suffix(x) == suffix(y)))
        else:
            xx, yy = numbers(x), numbers(y)
            px, py = set(postal(x)), set(postal(y))
            out['address_numeric_overlap'] = jaccard(xx, yy) if xx and yy else np.nan
            out['address_first_number_equal'] = float(xx[0] == yy[0]) if xx and yy else np.nan
            out['address_postal_equal'] = float(bool(px & py)) if px and py else np.nan
    return out

PERCENTILES = [1, 5, 10, 25, 50, 75, 90, 95, 99]

def distribution(values):
    a = np.asarray(values, dtype=float)
    valid = a[np.isfinite(a)]
    return {'n': len(a), 'valid_n': len(valid), 'missing_n': len(a)-len(valid),
            'mean': float(valid.mean()) if len(valid) else None,
            **{f'p{p}': float(np.percentile(valid, p)) if len(valid) else None for p in PERCENTILES}}
