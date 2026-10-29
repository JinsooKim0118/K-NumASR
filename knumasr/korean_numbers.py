import re

DIGIT_READ = ["공", "일", "이", "삼", "사", "오", "육", "칠", "팔", "구"]
SINO_DIGIT = ["", "일", "이", "삼", "사", "오", "육", "칠", "팔", "구"]
SMALL_UNITS = [(1000, "천"), (100, "백"), (10, "십")]
BIG_UNITS = [(10**12, "조"), (10**8, "억"), (10**4, "만")]

NATIVE_ONES_ATTR = ["", "한", "두", "세", "네", "다섯", "여섯", "일곱", "여덟", "아홉"]
NATIVE_TENS = ["", "열", "스물", "서른", "마흔", "쉰", "예순", "일흔", "여든", "아흔"]


def _sino_under_10000(n):
    out = []
    for value, unit in SMALL_UNITS:
        d = n // value
        n %= value
        if d:
            out.append(("" if d == 1 else SINO_DIGIT[d]) + unit)
    if n:
        out.append(SINO_DIGIT[n])
    return "".join(out)


def sino_read(n, spaced=True):
    if n == 0:
        return "영"
    parts = []
    for value, unit in BIG_UNITS:
        g = n // value
        n %= value
        if g:
            if g == 1 and unit == "만":
                parts.append("만")
            else:
                parts.append(_sino_under_10000(g) + unit)
    if n:
        parts.append(_sino_under_10000(n))
    return (" " if spaced else "").join(parts)


def digit_read(s):
    return "".join(DIGIT_READ[int(c)] for c in s if c.isdigit())


def native_read(n):
    if not 1 <= n <= 99:
        raise ValueError(n)
    tens, ones = divmod(n, 10)
    if tens == 2 and ones == 0:
        return "스무"
    return NATIVE_TENS[tens] + NATIVE_ONES_ATTR[ones]


def month_read(m):
    if m == 6:
        return "유월"
    if m == 10:
        return "시월"
    return sino_read(m) + " 월"


def group_digit_read(s):
    return " ".join(digit_read(g) for g in re.split(r"[-\s]", s) if g)
