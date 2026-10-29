import re
import unicodedata

SINO_DIGITS = {"영": 0, "공": 0, "일": 1, "이": 2, "삼": 3, "사": 4, "오": 5, "육": 6, "륙": 6, "칠": 7, "팔": 8, "구": 9}
SMALL = {"십": 10, "백": 100, "천": 1000}
BIG = {"만": 10**4, "억": 10**8, "조": 10**12}

NATIVE_TENS = {"열": 10, "스물": 20, "스무": 20, "서른": 30, "마흔": 40, "쉰": 50, "예순": 60, "일흔": 70, "여든": 80, "아흔": 90}
NATIVE_ONES = {"하나": 1, "한": 1, "둘": 2, "두": 2, "셋": 3, "세": 3, "석": 3, "넷": 4, "네": 4,
               "다섯": 5, "여섯": 6, "일곱": 7, "여덟": 8, "아홉": 9}

SINO_COUNTERS = ["퍼센트", "프로", "킬로그램", "킬로미터", "킬로", "센티미터", "센티", "미터", "그램", "달러", "유로",
                 "원", "년", "월", "일", "분", "초", "층", "호", "동", "회", "위", "점", "도", "인분", "주년", "세기", "명", "개"]
NATIVE_COUNTERS = ["번째", "시간", "마리", "사람", "살", "명", "개", "시", "번", "권", "잔", "곳", "달", "켤레"]
PARTICLE_NEXT = "이가은는을를에의와과도만로으부까쯤씩째반입예였요짜정께며면인야라랑엔"
_BOUNDARY = rf"(?=$|[^가-힣]|[{PARTICLE_NEXT}])"

_SINO_CHARS = "".join(SINO_DIGITS) + "".join(SMALL) + "".join(BIG)
_NUM_TOKEN = rf"(?:\d[\d,]*(?![\d,])|[{_SINO_CHARS}])"
_SINO_SPAN = rf"{_NUM_TOKEN}(?:\s?{_NUM_TOKEN})*"
_SINO_COUNTER_RE = "|".join(sorted(SINO_COUNTERS, key=len, reverse=True))
_NATIVE_COUNTER_RE = "|".join(sorted(NATIVE_COUNTERS, key=len, reverse=True))

SINO_WITH_COUNTER = re.compile(rf"(?<![가-힣])({_SINO_SPAN})(\s?)({_SINO_COUNTER_RE}){_BOUNDARY}")
DIGIT_SEQUENCE = re.compile(rf"(?<![가-힣])([{''.join(SINO_DIGITS)}]+(?:[\s\-][{''.join(SINO_DIGITS)}]+)*){_BOUNDARY}")
MIXED_UNITS = re.compile(rf"(?<![가-힣\d])(\d[\d,]*(?![\d,])\s?[{''.join(SMALL)}{''.join(BIG)}](?:\s?(?:\d[\d,]*(?![\d,])|[{''.join(SMALL)}{''.join(BIG)}]))*)")

_tens = "|".join(sorted(NATIVE_TENS, key=len, reverse=True))
_ones = "|".join(sorted(NATIVE_ONES, key=len, reverse=True))
NATIVE_WITH_COUNTER = re.compile(rf"(?<![가-힣])((?:{_tens})(?:{_ones})?|(?:{_ones}))(\s?)({_NATIVE_COUNTER_RE}){_BOUNDARY}")

SPACED_BLOCKLIST = {("이", "분"), ("이", "번"), ("그", "분")}
NOSPACE_SINGLE_OK = {"월", "년"}
NOSPACE_BLOCKLIST = {("이", "월")}


def _tokens(span):
    out = []
    for tok in re.findall(rf"\d[\d,]*|[{_SINO_CHARS}]", span):
        if tok[0].isdigit():
            digits = tok.replace(",", "")
            if not digits or len(digits) > 18:
                return []
            out.append(("num", int(digits)))
        elif tok in SINO_DIGITS:
            out.append(("num", SINO_DIGITS[tok]))
        elif tok in SMALL:
            out.append(("small", SMALL[tok]))
        else:
            out.append(("big", BIG[tok]))
    return out


def parse_sino(span):
    toks = _tokens(span)
    if not toks:
        return None
    if all(t == "num" for t, _ in toks) and len(toks) > 1:
        return None
    total, section, cur = 0, 0, None
    for kind, val in toks:
        if kind == "num":
            if cur is not None:
                return None
            cur = val
        elif kind == "small":
            section += (1 if cur is None else cur) * val
            cur = None
        else:
            section += 0 if cur is None else cur
            total += (section or 1) * val
            section, cur = 0, None
    return total + section + (cur or 0)


def parse_native(span):
    for t, tv in sorted(NATIVE_TENS.items(), key=lambda x: -len(x[0])):
        if span.startswith(t):
            rest = span[len(t):]
            if not rest:
                return tv
            return tv + NATIVE_ONES[rest] if rest in NATIVE_ONES else None
    return NATIVE_ONES.get(span)


def _sino_sub(m):
    span, sp, counter = m.group(1), m.group(2), m.group(3)
    if sp and (span, counter) in SPACED_BLOCKLIST:
        return m.group(0)
    if not sp and len(span) == 1 and not span.isdigit():
        if counter not in NOSPACE_SINGLE_OK or (span, counter) in NOSPACE_BLOCKLIST:
            return m.group(0)
    if re.fullmatch(rf"[{''.join(SINO_DIGITS)}](?:\s?[{''.join(SINO_DIGITS)}])+", span) and counter not in ("원",):
        digits = "".join(str(SINO_DIGITS[c]) for c in span if c in SINO_DIGITS)
        return digits + counter
    value = parse_sino(span)
    if value is None:
        return m.group(0)
    return f"{value}{counter}"


def _digit_seq_sub(m):
    span = m.group(1)
    n = sum(1 for c in span if c in SINO_DIGITS)
    if n < 3:
        return span
    return "".join(str(SINO_DIGITS[c]) if c in SINO_DIGITS else ("-" if c == "-" else " ") for c in span)


def _mixed_sub(m):
    value = parse_sino(m.group(1))
    return m.group(0) if value is None else str(value)


def _native_sub(m):
    value = parse_native(m.group(1))
    if value is None:
        return m.group(0)
    return f"{value}{m.group(3)}"


def korean_itn(text):
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"(?<![가-힣])유월", "6월", text)
    text = re.sub(r"(?<![가-힣])시월", "10월", text)
    text = DIGIT_SEQUENCE.sub(_digit_seq_sub, text)
    text = NATIVE_WITH_COUNTER.sub(_native_sub, text)
    text = SINO_WITH_COUNTER.sub(_sino_sub, text)
    text = MIXED_UNITS.sub(_mixed_sub, text)
    return text
