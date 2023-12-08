from huggingface_hub import notebook_login
from transformers import AutoModelForSequenceClassification,  AutoTokenizer, AutoModel, RobertaModel
import torch
import torch.nn.functional as F
import numpy as np
import streamlit as st
from torch.quantization import quantize_dynamic
import os

HUGGINGFACEHUB_API_TOKEN = "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" 
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

# 디바이스 설정
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# 라벨 준비
id2label_sent = {0: "매우부정", 1: "부정", 2: "중립", 3: "긍정", 4: "매우긍정"}
label2id_sent = {"매우부정": 0, "부정": 1, "중립": 2, "긍정": 3, "매우긍정": 4}
target_list = ['가격', '기능성', '길이', '디자인', '라인(핏)', '마감처리', '배송', '사이즈', '색상', '소재', '스타일', '신축성', '착용감', '품질']

## 모델 준비
# 키워드 모델
model_name = 'klue/roberta-small'
class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.bert_model = RobertaModel.from_pretrained(model_name, return_dict=True)
        self.dropout = torch.nn.Dropout(0.2)
        self.linear = torch.nn.Linear(768, 14) 

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.bert_model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        output_dropout = self.dropout(output.pooler_output)
        output = self.linear(output_dropout)
        return output

    def push(self,path):
        self.bert_model.push_to_hub(path,  create_pr=1, use_auth_token = True)

model = BERTClass()
model_dict = torch.load('C:/Users/jeong/langchain/best_model_weights.pt', map_location=torch.device('cpu'))
model.load_state_dict(model_dict)
model_q = quantize_dynamic(model, {torch.nn.Linear}, dtype = torch.qint8)

# 긍부정 모델
sent_name = "dongseon/cllama_sentiment"
model_sent = AutoModelForSequenceClassification.from_pretrained(
    sent_name
    , num_labels=5, id2label=id2label_sent, label2id=label2id_sent
)
model_sent_q = quantize_dynamic(model_sent, {torch.nn.Linear}, dtype = torch.qint8)

# 토크나이저 준비
model_name_or_path = "klue/roberta-small"
if any(k in model_name for k in ("gpt", "opt", "bloom","poly")):
    padding_side = "left"
else:
    padding_side = "right"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side=padding_side)
if getattr(tokenizer, "pad_token_id") is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id 

# 모델 평가 모드
model_q.eval()
model_sent_q.eval()

## 앱 평션
def results(txt):
    # 키워드/확률 추출
    inputs = tokenizer(txt, return_tensors="pt")
    probabilities = []

    with torch.no_grad():
        inputs = {key: value.to(device) for key, value in inputs.items()}
        ids = inputs['input_ids'].to(device, dtype = torch.long)
        mask = inputs['attention_mask'].to(device, dtype = torch.long)
        token_type_ids = inputs['token_type_ids'].to(device, dtype = torch.long)
        outputs = model_q(ids, mask, token_type_ids)
        probabilities.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    # 시그모이드 0.5 이상 클래스만 연산
    for idx, probability in enumerate(probabilities[0]):
        if probability> 0.5 :   # 노출조건
            lbl = target_list[idx]
            token_contributions = []

            # 감성분석
            inputs_sent = tokenizer(f'### review: {txt} ### keyword: {lbl} ', return_tensors="pt")
            with torch.no_grad():
                inputs_sent = {key: value.to(model_sent_q.device) for key, value in inputs_sent.items()}
                # 모델 호출
                logits_sent = model_sent_q(**inputs_sent).logits
                predicted_id = logits_sent.argmax().item() 
            sentiment = model_sent_q.config.id2label[predicted_id]

            # 모델에 입력 토큰 단위로 대체하면서 확률 계산
            for token_index in range(len(txt.split())):
                replaced_txt = txt.split()
                replaced_txt[token_index] = "[MASK]"
                replaced_txt = " ".join(replaced_txt)

                # 각 토큰을 마스크로 변환하여 변환 전후 확률 계산으로 기여도 확인
                inputs = tokenizer(replaced_txt, return_tensors="pt")
                with torch.no_grad():
                    inputs = {key: value.to(device) for key, value in inputs.items()}
                    ids = inputs['input_ids'].to(device, dtype = torch.long)
                    mask = inputs['attention_mask'].to(device, dtype = torch.long)
                    token_type_ids = inputs['token_type_ids'].to(device, dtype = torch.long)
                    outputs = model_q(ids, mask, token_type_ids)
                    probabilities2 = torch.sigmoid(outputs).cpu().detach().numpy().tolist()

                token_quality_probability = probabilities2[0][idx]
                token_contribution = probability - token_quality_probability
                token_contributions.append(token_contribution)

            # 가장 큰 기여도를 가진 토큰 찾기
            max_contribution_index = np.argmax(token_contributions)
            max_txt = txt.split()[max_contribution_index]
            def find_second_largest_index(nums):
                if len(nums) < 2:
                    return None  # 리스트에 두 개 이상의 원소가 없는 경우 None 반환

                largest_index = 0
                second_largest_index = -1

                for i in range(1, len(nums)):
                    if nums[i] > nums[largest_index]:
                        second_largest_index = largest_index
                        largest_index = i
                    elif nums[i] != nums[largest_index]:
                        if second_largest_index == -1 or nums[i] > nums[second_largest_index]:
                            second_largest_index = i

                return second_largest_index

            # 두 번째로 큰 숫자의 인덱스 찾기
            second_largest_index = find_second_largest_index(token_contributions)
            second_txt = txt.split()[second_largest_index]

            f"Class: {lbl}, Probability: {probability:.4f}, Highlight: {max_txt,second_txt} ,Sentiment: {sentiment} "





## 웹ui
st.title('키워드 분류 테스트')
content = st.text_input('리뷰를 입력해주세요.')

text = []
if st.button('분석'):
    st.write(results(content))

