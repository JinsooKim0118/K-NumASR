# K-NumASR: Korean Numeric Speech Recognition Benchmark

전화번호, 계좌번호, 금액, 날짜·시간, 고유어 수량 표현처럼 **숫자가 포함된 한국어 발화**를 음성인식 모델이 얼마나 정확하게 받아쓰는지 평가하는 벤치마크입니다.
일반적인 CER은 숫자 한 자리 오류를 거의 감점하지 않지만, 실제 서비스(콜센터, 송금, 예약)에서는 한 자리만 틀려도 실패입니다. K-NumASR은 숫자 단위의 엄격한 지표로 이 차이를 측정합니다.

Speech Recognition Challenge 2026 (Ajou University) · Team 김진수

## 구성

```text
K-NumASR/
├── knumasr/
│   ├── korean_numbers.py    한국어 숫자 읽기 규칙 (한자어, 고유어, 자릿수 읽기, 유월/시월)
│   ├── itn.py               한국어 숫자 ITN (한글 숫자 표현 → 아라비아 숫자)
│   └── metrics.py           CER, NEM, DER, NEM-norm 계산
├── scripts/
│   ├── build_synthetic_text.py   유형별 템플릿 문장 생성 (쓰기형 정답 + 발음형 TTS 입력)
│   ├── synthesize_tts.py         TTS로 합성 음성 생성 (edge-tts, 실패 시 facebook/mms-tts-kor)
│   ├── build_fleurs_subset.py    google/fleurs ko_kr에서 숫자 포함 발화 추출
│   ├── run_asr.py                Baseline / 제안 방법 추론
│   └── evaluate.py               지표 계산 및 리더보드 생성
├── notebooks/K_NumASR_Colab.ipynb   전체 파이프라인을 한 번에 실행하는 Colab 노트북
├── data/
│   ├── synthetic/   K-Num 합성 세트 (metadata.csv + audio/*.flac)
│   └── fleurs_num/  FLEURS-ko 숫자 부분집합 (metadata.csv + audio/*.flac)
├── predictions/     시스템별 예측 결과 (jsonl)
└── results/         leaderboard.csv / leaderboard.md / leaderboard.json / details/
```

## Evaluation Dataset

| Subset | 출처 | 규모 | 설명 |
|---|---|---|---|
| K-Num Synthetic | 자체 구축 (템플릿 + TTS) | 5개 유형 × 60 = 300 발화 | phone, account, amount, datetime, native |
| FLEURS-Num | [google/fleurs](https://huggingface.co/datasets/google/fleurs) ko_kr dev+test | 136 발화 (dev 50 + test 86) | 실제 사람 낭독 음성, 정답에 아라비아 숫자 포함 |

- 합성 세트의 정답(`text`)은 쓰기형(예: `010-1234-5678`, `35,000원`, `23살`)이고, TTS 입력(`spoken`)은 발음형(예: `공일공 일이삼사 오육칠팔`, `삼만 오천 원`, `스물세 살`)입니다. 발음을 코드로 통제하기 때문에 TTS 엔진이 숫자를 임의로 읽는 문제를 피합니다.
- 유형별 특징
  - phone: 휴대전화, 지역번호, 대표번호. 자릿수 읽기(공/일/이…)
  - account: 은행별 계좌 형식(3-2-6, 4-3-6 등). 긴 자릿수 열
  - amount: 백 원 ~ 억 단위 금액. 만/억 단위 한자어 수사
  - datetime: 연월일(유월, 시월 등 불규칙 읽기 포함), 시각(시는 고유어, 분은 한자어)
  - native: 나이, 인원, 개수, 횟수. 고유어 수사(스물세 살, 열다섯 개)
- 합성 음성은 edge-tts 한국어 음성 3종(ko-KR-SunHiNeural, ko-KR-InJoonNeural, ko-KR-HyunsuMultilingualNeural)과 세 가지 발화 속도(+0%, +15%, -10%)를 순환해 만들었습니다.
- 전체 평가 세트는 436 발화이며 모든 음성은 16 kHz mono FLAC입니다.
- 생성은 `seed=2026`으로 고정되어 같은 문장이 재현됩니다.

## Metrics

| Metric | 설명 |
|---|---|
| **NEM** (Numeric Exact Match, 주 지표) ↑ | 발화 안의 모든 숫자열이 정답과 완전히 일치한 발화의 비율 |
| DER (Digit Error Rate) ↓ | 숫자만 이어 붙인 문자열의 편집 거리 / 정답 숫자 수 |
| CER (Character Error Rate) ↓ | 문장부호와 공백을 제거한 문자 오류율 |
| NEM-norm ↑ | 정답과 예측 모두에 K-ITN을 적용해 표기를 통일한 뒤의 NEM. 발음은 맞게 들었지만 숫자로 표기하지 못한 경우를 구분하는 진단 지표 |

숫자 사이의 구분자(`-`, `,`, `.`, `:`, 공백)는 무시합니다. 따라서 `010-1234-5678`과 `010 1234 5678`은 같고, `3만 5천원`은 NEM에서는 오답, NEM-norm에서는 정답입니다.

## Baseline & Method

- Baseline: Whisper (tiny, base, small, medium, large-v3-turbo, large-v3), wav2vec2-xlsr-ko (CTC)
- 제안 방법
  1. **K-ITN**: 한국어 숫자 표현(한자어, 고유어, 자릿수 읽기, 혼합 표기 `3만 5천`)을 아라비아 숫자로 바꾸는 규칙 기반 후처리
  2. **NumPrompt**: Whisper의 `prompt_ids`에 숫자 표기 예시 문장을 넣어 쓰기형 출력을 유도
  3. 두 방법의 결합

## Results

Colab GPU에서 greedy decoding(fp16)으로 측정했습니다. 단위는 %이고, 전체 NEM 순으로 정렬했습니다.

| rank | system | NEM | NEM_syn | NEM_fleurs | DER | CER | NEM_norm | NEM_phone | NEM_account | NEM_amount | NEM_datetime | NEM_native |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | whisper-large-v3-turbo+NumPrompt+K-ITN | 84.9 | 88.3 | 77.2 | 4.8 | 4.9 | 86.0 | 68.3 | 76.7 | 98.3 | 98.3 | 100.0 |
| 2 | whisper-large-v3+K-ITN | 83.0 | 86.0 | 76.5 | 5.0 | 4.9 | 84.2 | 78.3 | 60.0 | 96.7 | 96.7 | 98.3 |
| 3 | whisper-large-v3-turbo+K-ITN | 82.8 | 87.0 | 73.5 | 5.4 | 5.1 | 83.9 | 65.0 | 75.0 | 98.3 | 98.3 | 98.3 |
| 4 | whisper-medium+K-ITN | 82.3 | 79.7 | 88.2 | 4.9 | 5.3 | 83.5 | 56.7 | 50.0 | 95.0 | 98.3 | 98.3 |
| 5 | whisper-medium | 72.9 | 65.3 | 89.7 | 10.0 | 6.7 | 83.5 | 56.7 | 50.0 | 35.0 | 98.3 | 86.7 |
| 6 | whisper-large-v3-turbo+NumPrompt | 71.6 | 73.3 | 67.6 | 12.2 | 6.9 | 86.0 | 68.3 | 76.7 | 25.0 | 98.3 | 98.3 |
| 7 | whisper-small+NumPrompt+K-ITN | 68.8 | 64.0 | 79.4 | 12.2 | 9.5 | 69.5 | 33.3 | 16.7 | 83.3 | 90.0 | 96.7 |
| 8 | whisper-small+K-ITN | 68.6 | 63.0 | 80.9 | 10.0 | 8.1 | 69.5 | 31.7 | 26.7 | 86.7 | 81.7 | 88.3 |
| 9 | whisper-large-v3-turbo | 67.7 | 71.0 | 60.3 | 13.9 | 7.4 | 83.9 | 65.0 | 75.0 | 26.7 | 98.3 | 90.0 |
| 10 | whisper-large-v3 | 64.7 | 66.7 | 60.3 | 14.3 | 7.4 | 84.2 | 78.3 | 60.0 | 28.3 | 96.7 | 70.0 |
| 11 | whisper-small+NumPrompt | 61.0 | 52.7 | 79.4 | 14.7 | 10.5 | 69.5 | 35.0 | 16.7 | 28.3 | 90.0 | 93.3 |
| 12 | whisper-small | 59.6 | 50.0 | 80.9 | 13.4 | 9.3 | 69.5 | 36.7 | 26.7 | 25.0 | 81.7 | 80.0 |
| 13 | whisper-base+K-ITN | 47.7 | 38.7 | 67.6 | 40.4 | 21.1 | 48.2 | 5.0 | 0.0 | 75.0 | 56.7 | 56.7 |
| 14 | whisper-base | 38.3 | 25.3 | 66.9 | 44.2 | 22.4 | 48.2 | 6.7 | 0.0 | 25.0 | 55.0 | 40.0 |
| 15 | wav2vec2-xlsr-ko+K-ITN | 36.5 | 35.7 | 38.2 | 45.3 | 25.8 | 36.7 | 0.0 | 1.7 | 86.7 | 68.3 | 21.7 |
| 16 | whisper-tiny+K-ITN | 28.7 | 22.7 | 41.9 | 56.0 | 29.1 | 28.9 | 3.3 | 0.0 | 51.7 | 26.7 | 31.7 |
| 17 | whisper-tiny | 20.2 | 10.7 | 41.2 | 65.2 | 31.6 | 28.9 | 0.0 | 0.0 | 8.3 | 26.7 | 18.3 |
| 18 | wav2vec2-xlsr-ko | 0.0 | 0.0 | 0.0 | 100.0 | 39.2 | 36.7 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

자세한 발화별 결과는 `results/details/`에 있습니다.

## 실행 방법

### Colab (권장)

`notebooks/K_NumASR_Colab.ipynb`를 Colab에서 열고 GPU 런타임으로 `모두 실행`합니다.

### 로컬

```bash
pip install -r requirements.txt
python scripts/build_synthetic_text.py
python scripts/synthesize_tts.py --backend edge
python scripts/build_fleurs_subset.py
python scripts/run_asr.py --model openai/whisper-small --system whisper-small
python scripts/run_asr.py --model openai/whisper-small --system whisper-small+NumPrompt --prompt
python scripts/evaluate.py
```

`evaluate.py`는 `predictions/*.jsonl`을 모두 읽고, 각 시스템에 K-ITN을 적용한 `+K-ITN` 변형까지 함께 채점합니다.

### 제출 형식

새 시스템은 아래 형식의 jsonl을 `predictions/`에 넣으면 리더보드에 추가됩니다.

```json
{"id": "phone_001", "category": "phone", "ref": "…", "hyp": "…"}
```

## License

- FLEURS: CC-BY-4.0 (Google)
- K-Num Synthetic: 연구 및 교육 목적의 평가용으로만 사용합니다. 합성 음성은 edge-tts(Microsoft) 또는 facebook/mms-tts-kor(CC-BY-NC-4.0)로 생성되었습니다.
- Code: MIT
