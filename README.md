# Customer Feedback Analyzer고객 리뷰 분석기  

Qwen2.5-7B-Instruct를 QLoRA 방식으로 파인튜닝한 모델이다.
영어 상품 리뷰를 입력으로 받아 해당 리뷰를 설명하는 하나의 JSON 객체만 출력한다.

**포맷에 대한 설명 없이 한 줄짜리 지시문만 제공해도,**
모델은 감성(sentiment), 1점에서 5점 사이의 점수(score), 언급된 주제(topics), 그리고 짧은 요약(summary)을 출력한다.

**또한 JSON 객체 외의 추가 텍스트는 출력하지 않는다.**

**이 프로젝트의 목적은 짧은 프롬프트만으로도 신뢰할 수 있는 구조화된 출력을 얻는 것이었다.**

**일반적인 대화형 모델은 리뷰 내용을 자연어로 다시 설명하는 경우가 많지만,**
대시보드나 데이터베이스에서는 프로그램이 데이터를 해석할 수 있도록 항상 동일한 필드가 필요하다.

**따라서 출력 스키마는 고정되어 있다.**


| Field | Meaning | Allowed values |
| --- | --- | --- |
| `sentiment` | overall sentiment | one of `positive`, `negative`, `neutral`, `mixed` |
| `score` | rating | integer from 1 to 5 |
| `topics` | aspects the review touches on | list of lowercase aspect words |
| `summary` | one-line summary | a single short English sentence |

출력이 엄격한 JSON 형식이기 때문에 평가는 매우 간단하다. 출력 결과를 파싱한 후 각 필드를 확인하면 정확한 JSON 유효성(valid-JSON) 비율을 계산할 수 있다.검증용으로 분리한 리뷰 데이터에서 기본 모델은 유효한 JSON을 한 번도 생성하지 못해 유효 JSON 비율이 0%였으며, 모든 응답을 일반 문장 형태로 출력했다. 반면 파인튜닝 이후에는 유효 JSON 비율이 100%를 기록했다.

전체 보고서는 `report/Customer-Feedback-Analyzer-Finetuning-Report（en），report/Customer-Feedback-Analyzer-Finetuning-Report（ko）`, 정리되어 있으며, 학습 코드는`notebook/1.ipynb`에 포함되어 있다.

## What fine-tuning changed 파인튜닝을 통해 달라진 점

동일한 짧은 프롬프트를 사용하고 출력 형식에 대한 별도의 지시를 제공하지 않았을 때, 검증용 리뷰 데이터에서 유효한 JSON 생성 비율(valid-JSON rate)은 0%에서 100%로 향상되었다.

기본 모델은 대화형 모델의 특성에 따라 리뷰 내용을 하나의 문단으로 설명하는 방식으로 응답하는 경향이 있다.

반면 파인튜닝 이후에는 고정된 출력 구조를 생성하는 습관이 모델의 가중치에 학습되어 내재화되었다. 그 결과 모델은 JSON이라는 형식을 전혀 언급하지 않은 한 줄짜리 프롬프트만으로도 파싱 가능한 JSON을 안정적으로 생성할 수 있게 되었다.

또한 출력되는 네 개의 필드는 모두 완전하고 규칙에 맞는 형태로 제공되며, 추가적인 후처리 없이 바로 후속 프로그램이나 코드에서 활용할 수 있다.

## Tech stack 기술 스택

| Piece | Choice |
| --- | --- |
| Base model | Qwen2.5-7B-Instruct, loaded in 4-bit |
| Framework | Unsloth |
| Method | QLoRA: 4-bit frozen base plus LoRA adapters |
| Trainer | TRL `SFTTrainer` with `train_on_responses_only` |
| Chat template | qwen-2.5 (ChatML), matching the base model |
| Data | 280 synthetic review/JSON pairs, generated in code |
| Export | LoRA adapter and GGUF (`q4_k_m`) for Ollama or llama.cpp |

## Base model

강의에서는 8B급 지시형(Instruct) 모델 또는 0.5B~3B 규모의 소형 모델 사용을 권장하였다. 우리는 여러 후보 모델을 비교한 후 Qwen2.5-7B-Instruct를 기본 모델로 선택하였다.

| Model | Size | Note | Verdict |
| --- | --- | --- | --- |
| Qwen2.5-7B-Instruct | 7B | good English and Chinese; no thinking mode, so cleaner JSON; fits 16GB after 4-bit; mature tooling | chosen |
| Qwen2.5-3B-Instruct | 3B | lighter and faster, quality slightly lower | fallback |
| Qwen3-4B | 4B | newer, but its thinking mode can leak reasoning text into the JSON | not used |
| Llama-3.1-8B-Instruct | 8B | strong in English, weaker in Chinese, no edge here | not used |

선정의 주요 이유는 다음과 같다. 먼저 이 모델은 별도의 사고(Thinking) 모드를 사용하지 않아 엄격한 JSON 출력에 불필요한 추론 텍스트가 섞여 들어갈 가능성이 낮다. 또한 양자화(Quantization) 적용 후 16GB 메모리 환경에서도 충분한 여유를 확보할 수 있다. 마지막으로, 문제가 발생했을 때 참고할 수 있는 커뮤니티의 파인튜닝 자료와 사례가 가장 풍부하다는 장점이 있었다.

## Repository layout 저장소 구조

```
.
├── README.md
├── notebook/
│   └── 1.ipynb          full pipeline, with outputs and the figure code
├── data/
│   └── dataset.jsonl           280 role-tagged training examples
├── img/                        figures written by the notebook
└── report/
    └── Customer-Feedback-Analyzer-Finetuning-Report（en）
    └──Customer-Feedback-Analyzer-Finetuning-Report（ko）
```

## Setup 설정 환경

모든 실험은 Windows 환경에서 단일 16GB GPU인 RTX 5060 Ti(Blackwell 아키텍처) 를 사용하여 수행하였다.

정상적으로 동작한 주요 라이브러리 및 소프트웨어 버전은 다음과 같다.

| Component | Version |
| --- | --- |
| Python | 3.11 |
| PyTorch | 2.12.0 (a build compiled for CUDA 12.x) |
| Triton | 3.7.0 |
| Unsloth | 2026.6.7 |
| Transformers | 5.5.0 |

실제로 상당한 시간을 소모했던 두 가지 문제는 다시 한번 강조할 가치가 있다.

첫째, 사용한 그래픽카드는 Blackwell 아키텍처 기반이었기 때문에 기본으로 설치된 PyTorch는 이전 세대 GPU만 지원하도록 컴파일되어 있어 GPU를 인식하지 못했다. 따라서 최신 CUDA를 지원하는 PyTorch 버전을 별도로 설치해야 했다.

둘째, 설치 순서도 매우 중요했다. 먼저 **Unsloth**를 설치한 뒤 `pip install torch`를 실행하면, 두 번째 명령이 기존 CUDA 지원 버전을 CPU 전용 버전으로 덮어쓰게 된다. 이 경우 GPU를 더 이상 사용할 수 없게 된다.

따라서 우리는 각 설치 단계를 마칠 때마다 `torch.cuda.is_available()`의 반환값이 `True`인지 확인한 후 다음 단계로 진행하였다. 이를 통해 GPU가 정상적으로 인식되고 있는지 지속적으로 검증할 수 있었다.


```bash
pip install unsloth
# if the GPU is not detected, install a PyTorch matching your CUDA first, then install unsloth last
```

또한 로컬 환경 설정 과정을 생략하고 싶다면, 이 노트북은 Google Colab의 무료 T4 GPU 환경**에서도 실행할 수 있다. 따라서 별도의 로컬 설치 없이도 학습 및 실험을 진행할 수 있다.


## Running the training 학습 실행

`notebook/finetune.ipynb` 파일을 열고 모든 셀을 위에서 아래 순서대로 실행하면 된다.
실행 순서는 다음과 같다. 먼저 4비트로 양자화된 기본 모델을 불러오고, LoRA 어댑터를 연결한다. 이후 qwen-2.5 채팅 템플릿을 사용하여 데이터를 로드하고, 기본 모델의 응답을 기록한다. 그다음 학습을 진행하고 손실(loss) 그래프를 생성한 뒤, 파인튜닝된 모델의 응답을 기록한다. 마지막으로 두 모델의 결과를 비교하고 학습된 모델을 내보낸다(export).

## Inference 추론

기본 모델 위에 LoRA 어댑터를 로드하는 방법은 다음과 같다.

```python
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained("lora_model", load_in_4bit=True)
FastLanguageModel.for_inference(model)

messages = [
    {"role": "system", "content": "You are a customer review analysis assistant."},
    {"role": "user", "content": "Battery dies in three hours and gets hot while charging."},
]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                       return_tensors="pt").to("cuda")
print(tokenizer.decode(model.generate(inputs, max_new_tokens=160)[0], skip_special_tokens=True))
```

Ollama를 이용한 GGUF 모델 실행：

```bash
ollama create feedback-analyzer -f Modelfile
ollama run feedback-analyzer "The hotel location was great but the walls were paper thin."
```

## 데이터셋

사용 가능한 레이블링 데이터셋이 없었기 때문에 코드로 280개의 학습 예제를 생성하였다. 각 예제는 하나의 리뷰와 이에 대응하는 JSON 정답으로 구성된다.

데이터 생성기는 이어폰, 노트북, 러닝화, 호텔, 앱, 커피머신 등 총 12개의 상품 카테고리를 포함하고 있으며, 각 카테고리별 평가 항목과 각 항목에 대한 긍정 및 부정 표현을 보유하고 있다.

예제를 생성할 때는 먼저 감성을 선택한 뒤 몇 개의 평가 항목을 고르고, 이를 바탕으로 자연스러운 리뷰 문장을 생성한다. 이후 생성 과정에서 선택된 정보를 이용하여 JSON 레이블을 자동으로 생성한다.

혼합 감성 리뷰의 경우 먼저 긍정적인 문장을 작성한 후 "However"로 시작하는 부정적인 내용을 이어 붙여 인위적으로 조합된 문장처럼 보이지 않도록 구성하였다.

우리는 데이터의 단순한 수량보다 카테고리, 감성, 표현 방식의 다양성을 더 중요하게 고려하였으며, 학습 전에 두 가지 검증 절차를 수행하였다.

첫째, 모든 리뷰 문장이 서로 다르도록 하여 특정 표현이 반복적으로 등장하면서 과도한 영향력을 갖지 않도록 하였다.

둘째, 모든 JSON이 정상적으로 파싱되는지 확인하였으며, 네 개의 필드가 모두 존재하고 유효한 감성 값, 정수형 점수, 그리고 비어 있지 않은 토픽 목록을 포함하도록 검증하였다.

280개의 모든 데이터가 검증을 통과하였으며, 이를 통해 오류가 있거나 품질이 낮은 데이터가 학습 과정에 포함되는 것을 방지하였다.

참고로 감성 분포는 긍정 94개, 부정 87개, 혼합 66개, 중립 33개로 구성되어 있다. 리뷰 길이는 평균 약 21단어이며 최소 9단어에서 최대 37단어 사이의 길이를 가진다.

## 하이퍼파라미터

| Setting | Value | Reason |
| --- | --- | --- |
| LoRA rank `r` | 16 | safe default for a narrow task, enough capacity without bloat |
| `lora_alpha` | 16 | equal to the rank, a stable and common starting point |
| `lora_dropout` | 0 | Unsloth's recommended setting |
| `target_modules` | q, k, v, o, gate, up, down (7) | all attention and feed-forward projections |
| `learning_rate` | 2e-4 | a common rate for LoRA fine-tuning |
| `num_train_epochs` | 3 | enough to learn the format without over-memorizing |
| batch x grad-accum | 2 x 4 = 8 | fits 16GB while keeping training stable |
| `max_seq_length` | 2048 | well above review-plus-answer length, so no truncation |
| `optimizer` | adamw_8bit | 8-bit optimizer to save memory |
| scheduler | linear, 5 warmup steps | steady decay with a brief warmup |
| seed | 3407 | fixed for reproducibility |

손실 함수는 어시스턴트의 응답 부분에 대해서만 계산된다. 따라서 시스템 프롬프트와 입력 리뷰는 손실 계산에 포함되지 않으며, 모델은 매 학습 단계에서 입력을 그대로 반복하는 대신 JSON 출력 형식을 생성하는 데 집중하게 된다.

학습 과정에서는 LoRA 어댑터만 업데이트되며, 약 4천만 개의 파라미터가 학습된다. 이는 기본 모델 전체 파라미터의 약 0.5% 수준에 해당하며, 나머지 파라미터는 모두 고정된 상태로 유지된다.

평가에 사용된 프롬프트는 의도적으로 매우 짧게 구성되었으며 출력 형식에 대한 지시를 포함하지 않는다. 이것이 기본 모델과 파인튜닝 모델을 구분하는 핵심 요소이다. 기본 모델은 해당 프롬프트만으로 JSON을 출력할 이유가 없지만, 파인튜닝된 모델은 별도의 형식 지시가 없어도 안정적으로 JSON을 생성할 수 있다.

## Results 결과

학습 데이터에 포함되지 않은 리뷰를 대상으로 테스트를 수행하였다.

| Test set | Base | Fine-tuned |
| --- | --- | --- |
| Domain reviews (8) | 0% | 100% |
| General questions (4) | 0% | 25% |

도메인 리뷰에서 유효한 JSON 생성률이 0%에서 100%로 향상된 것은 이번 프로젝트에서 목표로 했던 핵심 성과이다.

일반 질문은 모델이 파인튜닝 과정에서 기존 능력을 잃었는지 확인하기 위한 망각 테스트이다. 이 경우 JSON 출력은 목표가 아니므로 기본 모델의 0%는 오히려 올바른 동작이라고 볼 수 있다.

반면 파인튜닝 모델의 25%는 형식 전이(format bleeding) 현상을 보여준다. 이는 리뷰 분석과 무관한 질문에도 JSON 형식을 강제로 적용하는 경우가 있음을 의미한다. 이러한 현상은 특정 출력 형식에 집중하여 파인튜닝할 때 발생할 수 있는 알려진 부작용으로, 숨기기보다는 명확히 언급하고 평가하는 것이 중요하다.

동일한 리뷰를 두 모델에 입력했을 때의 결과는 다음과 같다.


```
REVIEW       Battery dies in about three hours and it gets uncomfortably hot while charging.

BASE         It sounds like you're experiencing some significant issues with your device's
             battery life and heat generation during charging. Here's a breakdown...   (prose)

FINE-TUNED   {"sentiment": "negative", "score": 2, "topics": ["battery life",
             "heat during charging"], "summary": "Customer is dissatisfied, mainly due to
             battery life and heat."}                                                   (clean JSON)
```

## Figures 그림

노트북은 생성된 그래프를 img/ 폴더에 저장한다.

They are not embedded here; the files are:

| File | Shows |
| --- | --- |
| `img/01_dataset_composition.png` | sentiment counts and share |
| `img/01_sentiment_distribution.png` | distribution of sentiment labels |
| `img/02_review_length.png` | distribution of review lengths |
| `img/03_top_topics.png` | most frequent aspects |
| `img/04_training_loss.png` | training loss across steps |
| `img/05_json_rate_comparison.png` | valid-JSON rate, base vs fine-tuned |


손실값은 총 105단계의 학습 동안 약 3.35에서 0.01까지 감소하였다. 이는 약 3분 정도의 학습 시간에 해당한다.

이처럼 작은 데이터셋에서 최종 손실값이 매우 낮게 나타난 것은 모델이 출력 형식을 거의 암기했음을 의미한다. 이러한 결과는 본 프로젝트와 같은 구조화 출력 작업에서는 예상 가능한 현상이다. 만약 모델의 응답이 지나치게 경직되거나 반복적으로 느껴진다면, 학습 에포크 수를 줄이거나 더 다양한 데이터를 추가하는 것이 도움이 될 수 있다.

## Limitations 한계점

가장 큰 한계는 앞서 언급한 형식 전이(format bleeding) 현상이다. 또한 제한된 어휘 체계로 인해 정확도가 일부 감소하는 문제도 존재한다. 모든 리뷰가 미리 정의된 평가 항목 집합에 매핑되기 때문에 일반적이지 않은 사례는 정보가 단순화되는 경향이 있다. 예를 들어 물이 새는 믹서기에 대한 리뷰는 실제 불만 사항을 반영하지 못한 채 neutral로 분류되었고, 토픽 역시 ease of use로 처리되어 핵심 문제를 놓쳤다. 생성된 요약문도 다소 템플릿 기반으로 작성된 것처럼 보인다. 또한 데이터셋 규모가 작기 때문에 거의 0에 가까운 손실값은 일반화 성능이 입증되었다기보다 데이터 암기 효과를 반영한 결과라고 볼 수 있다. 실제 일반화 능력을 확인하기 위해서는 더 크고 현실적인 데이터셋이 필요하다.

이를 개선하기 위해 몇 가지 방법을 고려할 수 있다. 학습 데이터에 일반적인 질문-답변 예제를 일부 추가하여 모델이 언제 JSON을 사용하지 말아야 하는지 학습하도록 할 수 있으며, 고정된 평가 항목 체계를 완화하거나 리뷰 내용에서 직접 토픽을 추출하도록 개선할 수도 있다. 또한 별도의 검증 데이터셋을 구성하고 Early Stopping을 적용하는 방법도 도움이 된다. 마지막으로 동일한 데이터를 더 작은 모델에 학습시켜 기준선(Baseline) 성능을 비교하는 것도 의미 있는 실험이 될 수 있다.

## 모델 내보내기 및 로컬 사용

학습이 완료된 후 LoRA 어댑터는 `lora_model` 디렉터리에 저장된다. 저장된 어댑터는 크기가 작고 기본 모델 위에 다시 로드할 수 있으며, 파인튜닝된 동작을 그대로 복원할 수 있다. 이 과정은 안정적으로 수행되었다. 또한 Ollama 또는 llama.cpp에서 사용하기 위해 `model_gguf` 디렉터리에 GGUF(`q4_k_m`) 형식의 모델도 생성하였다. 이 과정에서는 어댑터를 기본 모델과 병합해야 하므로 전체 정밀도의 기본 모델 가중치를 다운로드한 뒤 로컬 환경에서 변환 작업을 수행한다. 따라서 안정적인 인터넷 연결과 충분한 저장 공간이 필요하다. 내보내기 과정에는 예외 처리 기능이 포함되어 있어 변환이 실패하더라도 이미 저장된 LoRA 어댑터는 영향을 받지 않으며, 필요할 경우 나중에 다시 시도할 수 있다. GGUF 파일이 생성된 이후에는 간단한 Modelfile을 작성한 뒤 `ollama create`와 `ollama run` 명령을 실행하는 것만으로 모델을 로컬 환경에 로드하여 사용할 수 있으며, 로컬에서 직접 모델과 대화형으로 상호작용할 수 있다.

# Customer Feedback Analyzer

A QLoRA fine-tune of Qwen2.5-7B-Instruct that reads an English product review and returns a single JSON object describing it. Given only a one-line instruction and nothing about formatting, the model outputs the sentiment, a 1 to 5 score, the topics mentioned, and a short summary, with no surrounding text.

The point of the project was to get dependable structured output from a short prompt. A general chat model will happily restate a review in prose, but a dashboard or database needs the same fields every time so a program can parse them. The output schema is fixed:

| Field | Meaning | Allowed values |
| --- | --- | --- |
| `sentiment` | overall sentiment | one of `positive`, `negative`, `neutral`, `mixed` |
| `score` | rating | integer from 1 to 5 |
| `topics` | aspects the review touches on | list of lowercase aspect words |
| `summary` | one-line summary | a single short English sentence |

Because the output is strict JSON, scoring is straightforward: parse it and check the fields, which gives a clean valid-JSON rate. On held-out reviews the base model produced valid JSON 0% of the time and answered every one in prose; after fine-tuning it was 100%.

The full write-up is in `report/Customer-Feedback-Analyzer-Finetuning-Report.docx`, and the training code is in `notebook/finetune.ipynb`.

## What fine-tuning changed

Under the same short prompt with no format instructions, the valid-JSON rate on held-out reviews went from 0% to 100%. The base model falls back on its chat habits and explains the review in a paragraph. After training, the habit of emitting the fixed structure sits in the weights, so the model produces parseable JSON from a one-line prompt that never mentions JSON. The four fields come back complete and legal, ready to feed downstream code.

## Tech stack

| Piece | Choice |
| --- | --- |
| Base model | Qwen2.5-7B-Instruct, loaded in 4-bit |
| Framework | Unsloth |
| Method | QLoRA: 4-bit frozen base plus LoRA adapters |
| Trainer | TRL `SFTTrainer` with `train_on_responses_only` |
| Chat template | qwen-2.5 (ChatML), matching the base model |
| Data | 280 synthetic review/JSON pairs, generated in code |
| Export | LoRA adapter and GGUF (`q4_k_m`) for Ollama or llama.cpp |

## Base model

The course suggested either an 8B-class instruct model or a smaller 0.5 to 3B one. We compared a few candidates and went with Qwen2.5-7B-Instruct.

| Model | Size | Note | Verdict |
| --- | --- | --- | --- |
| Qwen2.5-7B-Instruct | 7B | good English and Chinese; no thinking mode, so cleaner JSON; fits 16GB after 4-bit; mature tooling | chosen |
| Qwen2.5-3B-Instruct | 3B | lighter and faster, quality slightly lower | fallback |
| Qwen3-4B | 4B | newer, but its thinking mode can leak reasoning text into the JSON | not used |
| Llama-3.1-8B-Instruct | 8B | strong in English, weaker in Chinese, no edge here | not used |

The deciding factors were the lack of a thinking mode (no extra reasoning text bleeding into strict JSON), comfortable headroom in 16GB once quantized, and the largest pool of community fine-tuning material when something breaks.

## Repository layout

```
.
├── README.md
├── notebook/
│   └── 1.ipynb          full pipeline, with outputs and the figure code
├── data/
│   └── dataset.jsonl           280 role-tagged training examples
├── img/                        figures written by the notebook
└── report/
    └── Customer-Feedback-Analyzer-Finetuning-Report（en）
    └──Customer-Feedback-Analyzer-Finetuning-Report（ko）
```

```

## Setup

Everything ran on a single 16GB GPU (an RTX 5060 Ti, Blackwell architecture) on Windows. The versions that worked:

| Component | Version |
| --- | --- |
| Python | 3.11 |
| PyTorch | 2.12.0 (a build compiled for CUDA 12.x) |
| Triton | 3.7.0 |
| Unsloth | 2026.6.7 |
| Transformers | 5.5.0 |

Two things cost real time and are worth repeating. The card is Blackwell, so the default PyTorch was compiled only for older GPUs and could not see it; a build targeting a newer CUDA had to go in instead. Install order also matters. If you install Unsloth first and then run `pip install torch`, the second command overwrites the CUDA build with a CPU-only one and the GPU disappears. After each install step we confirmed that `torch.cuda.is_available()` returned `True` before continuing.

```bash
pip install unsloth
# if the GPU is not detected, install a PyTorch matching your CUDA first, then install unsloth last
```

The notebook also runs on a free Colab T4 if you would rather skip the local setup.

## Running the training 

Open `notebook/finetune.ipynb` and run the cells top to bottom. The order is: load the 4-bit base, attach the LoRA adapter, load the data with the qwen-2.5 chat template, record the base model's answers, train, plot the loss, record the fine-tuned answers, compare the two, and export.

## Inference

Loading the adapter on top of the base model:

```python
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained("lora_model", load_in_4bit=True)
FastLanguageModel.for_inference(model)

messages = [
    {"role": "system", "content": "You are a customer review analysis assistant."},
    {"role": "user", "content": "Battery dies in three hours and gets hot while charging."},
]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                       return_tensors="pt").to("cuda")
print(tokenizer.decode(model.generate(inputs, max_new_tokens=160)[0], skip_special_tokens=True))
```

Running the exported GGUF with Ollama:

```bash
ollama create feedback-analyzer -f Modelfile
ollama run feedback-analyzer "The hotel location was great but the walls were paper thin."
```

## Dataset

There was no ready-made labeled corpus, so we generated 280 examples in code, each pairing a review with its target JSON. The generator holds twelve product categories (earbuds, laptops, running shoes, hotels, apps, coffee machines, and so on), a set of aspects per category, and positive and negative phrasings for every aspect. To build an example it picks a sentiment, selects a few aspects, writes a review that reads naturally, and derives the JSON label from the choices it just made. Mixed reviews are written as one positive sentence followed by a "However" turn so they do not look stitched together.

We cared more about variety across categories, sentiments, and phrasings than about raw count, and ran two checks before training. First, every review text is distinct, so repeated phrasings do not quietly gain weight. Second, every JSON parses, with all four fields present, a legal sentiment, an integer score, and a non-empty topics list. All 280 passed, which kept dirty data out of training.

For reference, the sentiment split is 94 positive, 87 negative, 66 mixed, and 33 neutral, and reviews run about 21 words on average (9 to 37).

## Hyperparameters

| Setting | Value | Reason |
| --- | --- | --- |
| LoRA rank `r` | 16 | safe default for a narrow task, enough capacity without bloat |
| `lora_alpha` | 16 | equal to the rank, a stable and common starting point |
| `lora_dropout` | 0 | Unsloth's recommended setting |
| `target_modules` | q, k, v, o, gate, up, down (7) | all attention and feed-forward projections |
| `learning_rate` | 2e-4 | a common rate for LoRA fine-tuning |
| `num_train_epochs` | 3 | enough to learn the format without over-memorizing |
| batch x grad-accum | 2 x 4 = 8 | fits 16GB while keeping training stable |
| `max_seq_length` | 2048 | well above review-plus-answer length, so no truncation |
| `optimizer` | adamw_8bit | 8-bit optimizer to save memory |
| scheduler | linear, 5 warmup steps | steady decay with a brief warmup |
| seed | 3407 | fixed for reproducibility |

Loss is computed only on the assistant's reply, so the system prompt and the review do not count toward it and the model spends every step learning to produce the JSON rather than echoing the input. Only the adapters train, about 40 million parameters, roughly half a percent of the base, while the rest stays frozen. The evaluation prompt is kept short and says nothing about format on purpose; that is what separates the two models, since the base has no reason to output JSON from it and the fine-tune does it anyway.

## Results

Tested on reviews that were not in the training data:

| Test set | Base | Fine-tuned |
| --- | --- | --- |
| Domain reviews (8) | 0% | 100% |
| General questions (4) | 0% | 25% |

The jump from 0 to 100 on domain reviews is the result we were after. The general questions are a forgetting check, and there JSON is not the goal, so the base model's 0 is the correct behavior. The fine-tuned model's 25 is format bleeding: it sometimes forces JSON onto unrelated questions, a known side effect worth flagging rather than hiding.

The same review through both models:

```
REVIEW       Battery dies in about three hours and it gets uncomfortably hot while charging.

BASE         It sounds like you're experiencing some significant issues with your device's
             battery life and heat generation during charging. Here's a breakdown...   (prose)

FINE-TUNED   {"sentiment": "negative", "score": 2, "topics": ["battery life",
             "heat during charging"], "summary": "Customer is dissatisfied, mainly due to
             battery life and heat."}                                                   (clean JSON)
```

## Figures

The notebook writes its plots to the `img/` folder. They are not embedded here; the files are:

| File | Shows |
| --- | --- |
| `img/01_dataset_composition.png` | sentiment counts and share |
| `img/01_sentiment_distribution.png` | distribution of sentiment labels |
| `img/02_review_length.png` | distribution of review lengths |
| `img/03_top_topics.png` | most frequent aspects |
| `img/04_training_loss.png` | training loss across steps |
| `img/05_json_rate_comparison.png` | valid-JSON rate, base vs fine-tuned |

The loss curve falls from about 3.35 to 0.01 over 105 steps, which is roughly three minutes of training. A final loss that low on a small set means the model has largely memorized the format, which is expected for this kind of task; if outputs start to feel rigid, drop an epoch or add more varied data.

## Limitations

The format bleeding above is the main one. The controlled vocabulary also costs some accuracy: because every review is mapped onto a fixed set of aspect words, unusual cases get flattened. The leaking blender, for example, came back as `neutral` with the topic `ease of use` and missed the actual complaint. The summaries read a little templated. And the dataset is small, so the near-zero loss reflects memorization more than proven generalization, which would need a larger and more realistic set to confirm.

A few things would help: mixing a small number of ordinary question-answer pairs into the training set so the model learns when not to use JSON; loosening the aspect vocabulary or pulling topics straight from the review; adding a held-out validation set with early stopping; and training the same data on a smaller model as a baseline.

## Export and local use

After training we save the LoRA adapter to `lora_model`. It is small, reloads on top of the base, and restores the fine-tuned behavior; this step ran reliably. We also export a GGUF (`q4_k_m`) to `model_gguf` for Ollama or llama.cpp. That step merges the adapter into the base, so it downloads the full-precision weights and converts locally, which needs a stable connection and some disk space. The export is wrapped so that if it fails the saved adapter is untouched and you can retry it later. Once the GGUF exists, a short Modelfile plus `ollama create` and `ollama run` is enough to load the model and chat with it locally.
