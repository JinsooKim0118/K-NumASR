import re
import unicodedata
from collections import defaultdict

from .itn import korean_itn

_SEP_BETWEEN_DIGITS = re.compile(r"(?<=\d)[\s\-.,:/]+(?=\d)")
_PUNCT = re.compile(r"[^\w\s]")


def normalize_numbers(text):
    text = unicodedata.normalize("NFKC", text)
    prev = None
    while prev != text:
        prev = text
        text = _SEP_BETWEEN_DIGITS.sub("", text)
    return text


def normalize_text(text):
    text = normalize_numbers(text).lower()
    text = _PUNCT.sub(" ", text)
    return re.sub(r"\s+", "", text)


def extract_numbers(text):
    return re.findall(r"\d+", normalize_numbers(text))


try:
    from rapidfuzz.distance import Levenshtein as _lev
except ImportError:
    _lev = None


def edit_distance(a, b):
    if _lev is not None:
        return _lev.distance(a, b)
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def score_utterance(ref, hyp):
    r, h = normalize_text(ref), normalize_text(hyp)
    rn, hn = extract_numbers(ref), extract_numbers(hyp)
    hn_itn = extract_numbers(korean_itn(hyp))
    rd, hd = "".join(rn), "".join(hn)
    return {
        "char_err": edit_distance(r, h),
        "char_len": len(r),
        "digit_err": edit_distance(rd, hd),
        "digit_len": len(rd),
        "nem": int(rn == hn),
        "nem_norm": int(rn == hn_itn),
    }


def aggregate(rows):
    if not rows:
        return {}
    ce = sum(r["char_err"] for r in rows)
    cl = sum(r["char_len"] for r in rows)
    de = sum(r["digit_err"] for r in rows)
    dl = sum(r["digit_len"] for r in rows)
    n = len(rows)
    return {
        "n": n,
        "NEM": 100.0 * sum(r["nem"] for r in rows) / n,
        "NEM_norm": 100.0 * sum(r["nem_norm"] for r in rows) / n,
        "DER": 100.0 * de / max(dl, 1),
        "CER": 100.0 * ce / max(cl, 1),
    }


def evaluate(refs, hyps, groups=None):
    scored = [score_utterance(r, h) for r, h in zip(refs, hyps)]
    out = {"overall": aggregate(scored)}
    if groups is not None:
        by = defaultdict(list)
        for g, s in zip(groups, scored):
            by[g].append(s)
        out["by_group"] = {g: aggregate(v) for g, v in sorted(by.items())}
    return out, scored
