import argparse
import csv
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from knumasr.korean_numbers import digit_read, month_read, native_read, sino_read

BANKS = ["국민은행", "신한은행", "우리은행", "하나은행", "농협", "기업은행", "카카오뱅크", "토스뱅크"]
NAMES = ["민수", "지영", "현우", "서연", "도윤", "하은", "준호", "수빈"]
ITEMS = ["사과", "커피", "볼펜", "공책", "우산", "빵", "의자", "상자"]


def rand_digits(rng, k):
    return "".join(str(rng.randint(0, 9)) for _ in range(k))


def make_phone(rng):
    kind = rng.choice(["mobile", "mobile", "mobile", "seoul", "region", "rep"])
    if kind == "mobile":
        groups = ["010", rand_digits(rng, 4), rand_digits(rng, 4)]
    elif kind == "seoul":
        groups = ["02", rand_digits(rng, rng.choice([3, 4])), rand_digits(rng, 4)]
    elif kind == "region":
        groups = [rng.choice(["031", "032", "051", "053", "062", "064"]), rand_digits(rng, 3), rand_digits(rng, 4)]
    else:
        groups = [rng.choice(["1588", "1577", "1544", "1661"]), rand_digits(rng, 4)]
    written = "-".join(groups)
    spoken = " ".join(digit_read(g) for g in groups)
    templates = [
        ("제 전화번호는 {w}입니다.", "제 전화번호는 {s}입니다."),
        ("{w}로 연락 주세요.", "{s}로 연락 주세요."),
        ("고객센터 번호는 {w}입니다.", "고객센터 번호는 {s}입니다."),
        ("{n} 씨 번호가 {w} 맞나요?", "{n} 씨 번호가 {s} 맞나요?"),
        ("문의는 {w}로 해 주시기 바랍니다.", "문의는 {s}로 해 주시기 바랍니다."),
    ]
    tw, ts = rng.choice(templates)
    n = rng.choice(NAMES)
    return tw.format(w=written, n=n), ts.format(s=spoken, n=n)


def make_account(rng):
    layout = rng.choice([[3, 2, 6], [4, 3, 6], [3, 6, 2, 3], [6, 2, 6], [3, 4, 4, 2]])
    groups = [rand_digits(rng, k) for k in layout]
    written = "-".join(groups)
    spoken = " ".join(digit_read(g) for g in groups)
    bank = rng.choice(BANKS)
    templates = [
        ("계좌번호는 {b} {w}입니다.", "계좌번호는 {b} {s}입니다."),
        ("{b} {w}로 입금해 주세요.", "{b} {s}로 입금해 주세요."),
        ("환불은 {w} 계좌로 보내 드리겠습니다.", "환불은 {s} 계좌로 보내 드리겠습니다."),
        ("받는 분 계좌는 {b} {w}, 예금주는 {n}입니다.", "받는 분 계좌는 {b} {s}, 예금주는 {n}입니다."),
    ]
    tw, ts = rng.choice(templates)
    n = rng.choice(NAMES)
    return tw.format(w=written, b=bank, n=n), ts.format(s=spoken, b=bank, n=n)


def make_amount(rng):
    scale = rng.choice(["small", "small", "mid", "mid", "large", "huge"])
    if scale == "small":
        v = rng.randint(10, 990) * 100
    elif scale == "mid":
        v = rng.randint(100, 9990) * 1000
    elif scale == "large":
        v = rng.randint(1000, 99000) * 1000
    else:
        v = rng.randint(11, 990) * 10**7
    written = f"{v:,}원"
    spoken = sino_read(v) + " 원"
    templates = [
        ("총 금액은 {w}입니다.", "총 금액은 {s}입니다."),
        ("{w}을 송금했어요.", "{s}을 송금했어요."),
        ("이번 달 관리비는 {w}이 나왔어요.", "이번 달 관리비는 {s}이 나왔어요."),
        ("결제하실 금액은 {w}입니다.", "결제하실 금액은 {s}입니다."),
        ("월급에서 {w}이 빠져나갔다.", "월급에서 {s}이 빠져나갔다."),
    ]
    tw, ts = rng.choice(templates)
    return tw.format(w=written), ts.format(s=spoken)


def make_datetime(rng):
    if rng.random() < 0.5:
        y, m, d = rng.randint(1990, 2030), rng.randint(1, 12), rng.randint(1, 28)
        written = f"{y}년 {m}월 {d}일"
        spoken = f"{sino_read(y)} 년 {month_read(m)} {sino_read(d)} 일"
        templates = [
            ("{w}에 만나요.", "{s}에 만나요."),
            ("계약 만료일은 {w}입니다.", "계약 만료일은 {s}입니다."),
            ("저는 {w}에 태어났어요.", "저는 {s}에 태어났어요."),
        ]
    else:
        ampm = rng.choice(["오전", "오후"])
        h, mi = rng.randint(1, 12), rng.choice([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 7, 13, 48])
        written = f"{ampm} {h}시 {mi}분"
        spoken = f"{ampm} {native_read(h)} 시 {sino_read(mi)} 분"
        templates = [
            ("회의는 {w}에 시작합니다.", "회의는 {s}에 시작합니다."),
            ("기차는 {w}에 출발해요.", "기차는 {s}에 출발해요."),
            ("예약 시간은 {w}입니다.", "예약 시간은 {s}입니다."),
        ]
    tw, ts = rng.choice(templates)
    return tw.format(w=written), ts.format(s=spoken)


def make_native(rng):
    kind = rng.choice(["age", "people", "items", "times"])
    if kind == "age":
        n = rng.randint(5, 89)
        pairs = [("제 동생은 {w}살이에요.", "제 동생은 {s} 살이에요."), ("할머니는 올해 {w}살이 되셨어요.", "할머니는 올해 {s} 살이 되셨어요.")]
    elif kind == "people":
        n = rng.randint(2, 49)
        pairs = [("학생 {w}명이 참석했습니다.", "학생 {s} 명이 참석했습니다."), ("손님 {w}명 예약할게요.", "손님 {s} 명 예약할게요.")]
    elif kind == "items":
        n = rng.randint(2, 30)
        item = rng.choice(ITEMS)
        pairs = [(f"{item} {{w}}개 주세요.", f"{item} {{s}} 개 주세요."), (f"{item} {{w}}개를 샀어요.", f"{item} {{s}} 개를 샀어요.")]
    else:
        n = rng.randint(2, 20)
        pairs = [("벌써 {w}번이나 말했잖아.", "벌써 {s} 번이나 말했잖아."), ("하루에 {w}번 약을 먹어요.", "하루에 {s} 번 약을 먹어요.")]
    tw, ts = rng.choice(pairs)
    return tw.format(w=n), ts.format(s=native_read(n))


MAKERS = {"phone": make_phone, "account": make_account, "amount": make_amount, "datetime": make_datetime, "native": make_native}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic/metadata.csv")
    ap.add_argument("--per_category", type=int, default=60)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    rows, seen = [], set()
    for cat, fn in MAKERS.items():
        k = 0
        while k < args.per_category:
            w, s = fn(rng)
            if w in seen:
                continue
            seen.add(w)
            k += 1
            rows.append({"id": f"{cat}_{k:03d}", "category": cat, "text": w, "spoken": s})
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["id", "category", "text", "spoken"])
        wr.writeheader()
        wr.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
