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
position_encoding  
인코더/디코더  
어텐션 메커니즘 바디 (qkv) + bertviz [참고영상](https://youtu.be/MJYBdTCwxDY?si=Rkhm3G1Ff9ZzjX68)  
분류 헤드  
파인튜닝(freezing, 분류기)  
제로샷 러닝  

## 4. 학습 코드 구성
tensor 변환  
device 설정  
model 불러오기  
tokenizer (input_ids, attention_mask, token_type_ids, label)   
dataset 전처리  
batch/collator  
손실함수  
optimizer  
학습  
weight 저장  
hub 업로드  

## 5. 기본 모델과 데이터셋
klue-로버타 : [https://huggingface.co/klue/roberta-base]  
klue-데이터셋 : [https://huggingface.co/datasets/klue]  

## 6.로버타 전체 코드(keyword+sentiment)
시행착오  
데이터셋 구축 : 멀티 라벨, 싱글 라벨  
과적합  

## 7. 모델 경량화 배포
부동소수점  
streamlit : [https://docs.streamlit.io/]  
```
streamlit run [파일명.py]
```

## 8. 도커, 클라우드
[체크포인트 다운](https://drive.google.com/file/d/1-5zsnJVR_kF0MoQeredj7DTSwSF-1Ikh/view?usp=drive_link)  
  
도커파일 이미지 빌드  
```
docker build --tag [이미지명]:[버전]
```
인스턴스 생성 및 도커 구동
```
docker run -it --rm [이미지명]:[버전] /bin/bash
```

## *파인튠참고 구름,코알파카 / 기타 W&B
고려대-kullm : [https://github.com/nlpai-lab/KULLM]  
koalpaca : [https://github.com/Beomi/KoAlpaca]  
weights & biases : [https://wandb.ai/home]  
