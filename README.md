# FINE_TUNE_LLM

## 1. 허깅페이스 소개
모델  
데이터셋  
파이프라인  
도큐먼트 : [https://huggingface.co/docs]  
리더보드 : [https://huggingface.co/spaces/upstage/open-ko-llm-leaderboard]  

## 2. 랭귀지 모델 소개
CausalLM  
SequenceClassification  
TokenClassification  
Summarize  
QuestionAnswering  
Ner  

## 3. 트랜스포머 모델 구성
tokenizer  
```
input = "나는 학교에 간다"
1 = "나는" "학교에" "간다"
2 = "나" "는" "학교" "에" "간다"
3 = [SEP] "나" "는" "학교" "에" "간다" [CLS]
output = 1, 333, 234, 33, 44, 55, 0
```
position_encoding  
![ex_screenshot](./img/position_encoding.png)  

    시퀀스 위치 ids 임베딩을 input_ids 임베딩과 합침  

인코더/디코더  
![ex_screenshot](./img/transformers.png)  


어텐션 메커니즘 바디 (qkv) + bertviz [참고영상](https://youtu.be/MJYBdTCwxDY?si=Rkhm3G1Ff9ZzjX68)  
분류 헤드  
### **파인튜닝**  
> pretrained model의 weights를 목적에 맞는 결과를 출력하도록 튜닝하는 것  
0에서 부터 학습하는 경우 데이터양이 많이 필요하고 학습 시간, gpu리소스도 많이 소요됨  
![ex_screenshot](./img/body_head.png)  
- 임베딩 튜닝  
- 분류기 튜닝  
- 임베딩+분류기 튜닝  
- 프롬프트 엔지니어링  
    - 퓨샷/원샷 러닝  
    - chain of thought  
    
- 프롬프트 튜닝  
    - p-tune/ prefix  
    
- adapter 튜닝  
    - LoRA/ IA3  

## 4. 학습 코드 구성
tensor 변환  
```python
import torch
torch.Tensor([1,5,768])
```
device 설정  
```python
#숫자는 전부 device 설정을 해야함
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
```
model 불러오기  
```python
from transformers import AutoModel, GPTNeoXModel, AutoModelForCausalLM, AutoModelForSequenceClassification
model = AutoModel.pretrained(model_name)
model = GPTNeoXModel.pretrained(model_name)
model = AutoModelForCausalLM.pretrained(model_name)
```
tokenizer 불러오기  

    input_ids : 단어사전 매칭 ids  
    attention_mask : padding 여부  
    token_type_ids : 문장 구분  
    label : 정답  
    
```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer(model_name)
```
datasets  
```python
from datasets import Dataset
dataset = Dataset.from_pandas(df)
```

|sequence_num|batch2_num|epoch1_num|num1|num2|num3|num4|num5|
|---|---|---|---|---|---|---|---|
|1|1|1|나|는|학교|에|간다|
|2|1|1|학교|종|이|땡|padding|
|3|2|1|어서|모이자|padding|||
|4|2|1|선생님|이|우리를|||
|5|3|1|…|…|…|…|…|
|6|3|1|…|…|…|…|…|
|7|4|1|…|…|…|…|…|
|8|4|1|…|…|…|…|…|
|9|5|1|…|…|…|…|…|
|10|5|1|…|…|…|…|…|

data_loader  
```python
from torch.utils.data import DataLoader
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
```
data_collator  
```python
from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```
***
### **활성함수**  

> sigmoid  
$$S(x) = \frac {1}{1+e^{-x}}$$

![ex_screenshot](./img/sigmoid.png)  

> softmax  
$$\sigma = \frac {e^{z_{i}}} {\displaystyle\sum_{j=1}^{k} e^{z_{j}}}$$

### **손실함수**  

> mse  
$$\mathrm{MSE} = \frac{1}{n} \sum_{i=1}^{n}(Y_{i}-\hat{Y}_{i})^2$$

> rmse  
$$\mathrm{RMSD} = \sqrt{\frac{\displaystyle\sum_{i=1}^{N}\left(x_{i}-\hat{x}_{i}\right)^{2}}{N}}$$

> cross_entropy  
$$-Y*log(y)-(1-Y)*log(1-y)$$

![ex_screenshot](./img/cross.png)  
***

optimizer  
> 손실함수(loss)가 최저가 되는 부분에 도달하도록 W에 손실함수의 순간변화량(미분, 기울기) 만큼 조금씩 뺀다  
$$W-기울기 = W-d(손실함수)$$  

![ex_screenshot](./img/loss.png)

학습  
```python
from transformers import Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    optimizers=(optimizer, lr_scheduler),
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
```
학습 모니터링
```python
from transformers import TrainingArguments
training_args = TrainingArguments(
    output_dir="./bert_test",
    learning_rate= 1e-5, 
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64, 
    num_train_epochs=5,
    optim="adamw_hf",
    weight_decay= 0.5 ,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="wandb",
)
```


모델 저장 
```python
model.save_pretrained(model_output_dir)
```
hub 업로드  
```python
model.push_to_hub("[내계정]/[레포지토리명]", create_pr=1,use_auth_token=True)
```

## 5. 기본 모델과 데이터셋
klue-로버타 : [https://huggingface.co/klue/roberta-base]  
klue-데이터셋 : [https://huggingface.co/datasets/klue]  

## 6.로버타 전체 코드(keyword+sentiment)
키워드 분류 시행착오  

- CausalLM 모델 사용  
Causal 모델은 학습데이터에 target 없이 입력 시퀀스만 입력됨, 그래서 시퀀스에 일부분을 [MASK] 처리 후 해당 부분 예측을 학습  
> [MASK]는 콜레이터로 일괄 생성
```python
from transformers import DataCollatorForLanguageModeling
data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
```
> 학습데이터

|sequence|target|학습데이터|
|---|---|---|
|옷은 예쁘지만 배송은 늦어요|디자인1, 배송-1|###입력: 옷은 예쁘지만 배송은 늦어오 \n ###명령어: 리뷰를 분석해줘 \n###출력: 디자인1, 배송-1|
|색이 정말 이뻐요|색상2|###입력: 색이 정말 이뻐요 \n ###명령어: 리뷰를 분석해줘 \n###출력: 색상2|
|품질이 좀 떨어지지만 가격이 싸니까|품질-1, 가격1|###입력: 품질이 좀 떨어지지만 가격이 싸니까 \n ###명령어: 리뷰를 분석해줘 \n###출력: 품질-1, 가격1|

> **추론**

```python
inputs = tokenizer(
    f'###입력 : \n{리뷰} ###명령어: 리뷰를 분석해줘 \n###출력 :',
    return_tensors="pt",)
```

- SequenceClassification 모델 사용  
시퀀스 분류 시 target(라벨)이 필요하고 동시에 여러 라벨이 존재하는 멀티라벨 분류와 하나의 라벨로 구성된 싱글라벨 분류로 나뉜다  
***
> 멀티라벨

|sequence|배송|가격|디자인|색상|품질|
|---|---|---|---|---|---|
|옷은 예쁘지만 배송은 늦어요|1|0|1|0|0|
|색이 정말 이뻐요|0|0|0|1|0|
|품질이 좀 떨어지지만 가격이 싸니까|1|0|0|0|1|

```python
def __init__(self):
    super(BERTClass, self).__init__()
    self.bert_model = RobertaModel.from_pretrained(model_name_or_path, return_dict=True)
    self.dropout = torch.nn.Dropout(0.2)
    self.linear = torch.nn.Linear(768, 14)
```
> **멀티라벨 학습**  
모델에서 시퀀스 각 키워드의 로짓을 출력하고 각 로짓을 시그모이드 함수에 넣어 최종 예측 확률을 출력한다

|class|sequence|배송|가격|디자인|색상|품질|
|---|---|---|---|---|---|---|
|정답|옷은 예쁘지만 배송은 늦어요|1|0|1|0|0|
|예측|옷은 예쁘지만 배송은 늦어요|0.9|0.03|0.8|0.1|0.01|
***
> 싱글라벨  

|학습데이터|라벨|
|---|---|
|###입력: 옷은 예쁘지만 배송은 늦어오 \n###키워드: 디자인|긍정|
|###입력: 옷은 예쁘지만 배송은 늦어오 \n###키워드: 배송|부정|

> **싱글라벨 학습**  
모델에서 시퀀스의 로짓을 소프트맥스 함수에 넣어 최종 예측 확률을 출력한다

|학습데이터|정답|예측|
|---|---|---|
|###입력: 옷은 예쁘지만 배송은 늦어오 \n###키워드: 디자인|0|0.0|
|###입력: 옷은 예쁘지만 배송은 늦어오 \n###키워드: 디자인|0|0.05|
|###입력: 옷은 예쁘지만 배송은 늦어오 \n###키워드: 디자인|0|0.1|
|###입력: 옷은 예쁘지만 배송은 늦어오 \n###키워드: 디자인|1|0.6|
|###입력: 옷은 예쁘지만 배송은 늦어오 \n###키워드: 디자인|0|0.25|
***

모델 구성

- 키워드 분류
- 감성 분석
- 하이라이트 추출  

|키워드|결과|예측확률|제외확률|
|---|---|---|---|
|디자인|[MASK] 예쁘지만 배송은 늦어오|0.9|0.8|
|디자인|옷은 [MASK] 배송은 늦어오|0.9|0.05|
|디자인|옷은 예쁘지만 [MASK] 늦어오|0.9|0.8|
|디자인|옷은 예쁘지만 배송은 [MASK]|0.9|0.7|



과적합 해결과제  

## 7. 모델 경량화
quantization  
    부동소수점  
distillation_knowledge, ONNX, 가지치기  
streamlit : [https://docs.streamlit.io/]  
```bash
streamlit run [파일명.py]
```

## 8. 도커, 클라우드 배포
[체크포인트 다운](https://drive.google.com/file/d/1-5zsnJVR_kF0MoQeredj7DTSwSF-1Ikh/view?usp=drive_link)  
  
도커파일 이미지 빌드  
```bash
docker build --tag [이미지명]:[버전]
```
인스턴스 생성 및 도커 구동
```bash
docker run -it --rm [이미지명]:[버전] /bin/bash
```

## *파인튠참고 구름,코알파카 / 기타 W&B
고려대-kullm : [https://github.com/nlpai-lab/KULLM]  
koalpaca : [https://github.com/Beomi/KoAlpaca]  
weights & biases : [https://wandb.ai/home]  
