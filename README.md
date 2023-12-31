# Forecasting_product_sales_LGAimers
> 온라인 쇼핑몰의 일별 제품별 판매 데이터를 바탕으로 향후 21일간의 제품별 판매량을 예측하는 AI 모델을 개발한다
> [https://dacon.io/competitions/official/236129/overview/description](https://dacon.io/competitions/official/236129/overview/description)
- 대회기간 : 23.08.01 ~ 23.08.28

## Metric
- 평가 산식 : Pseudo SFA(PSFA)
  - i : 대분류 내에서 제품 index  
  - M : 대분류 index  
  - $y_i^{day}$ : i번째 제품의 day일의 판매량  
  - $p_i^{day}$ : i번째 제품의 day일의 예측량

  $PSFA_m=1 - {1 \over n} {\sum}\_{day=1}^N {\sum}\_{i=1}^N ({{|y_i^{day} - p_i^{day}|} \over {max (y_i^{day}, p_i^{day})}} \times {y_i^{day} \over {{\sum}\_{i=1}^N y_i^{day}}})$

  $PSFA = {1 \over M} {\sum}\_{m=1}^M PSFA_m$

  

## Data
15890개의 제품별 데이터(시계열 데이터)
- train.csv
- sales.csv
- sample_submission.csv

15890개의 제품별 설명 데이터(자연어 데이터)
- product_info.csv

3170개의 브랜드별 데이터(시계열 데이터)
- brand_keyword_cnt.csv

## Run
```
# train
python train.py

# inference
python inference.py
```

## LB
- Public LB : 0.54446 (66th / 757)
- Private LB : 0.52261 (110th / 747)
