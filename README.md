# Math Word Problem Solving

LLM PEFT(Parameter-Efficient Fine-Tuning)

## More infomation

[MWPS | Notion](https://yeonghun.notion.site/Problem-Solving-with-LLM-PEFT-20f18a1c065a45cdb810bd6c01528865?pvs=4)

## Introduction

This project supports these pipeline for llm

- Train
  - Train
- Inference
  - Huggingface_transformer_inf
  - PEFT_transformer_inf
  - Batch inf
- Evaluation
  - llm evaluation
- Data
  - Data Loader
  - Data Processing
  - Data Analysis
- Model
  - Model loader
- Test
  - prompt test

## Getting started

1. Install requirements & libararies (with git)  
   :
2. Load Model & Data on local drive  
   : Look "data/data_process_prm800k.py" and "models/model_loader.py"
3. Train  
   : Look "train_mwps.py"
4. Inference  
   : Look "inference\*\*\*\*.py
5. Evaluation  
   : Look "evaluate.py" (working version will be updated in later)

# Management

## Branch

master #안정적인 가장 최신 버전  
└ develop #개발용 Endpoint  
　 └ {personal branch} #개별 개발 브랜치  
└ relese #배포 버전

## Commit Rule(Convention)

| Type명 | 설명                                                           |
| ------ | -------------------------------------------------------------- |
| feat   | 새로운 기능                                                    |
| chore  | 간단한 수정 사항                                               |
| bugfix | 버그 수정, 어떤 버그를 수정했는지 Commit Message에 상세히 기록 |
| doc    | 문서 업데이트                                                  |
| refac  | 신규 기능 추가 없는 코드 리팩토링 시 사용                      |
| test   | 테스트 코드 업데이트 시 사용                                   |

## Naming Convention

- 디렉토리명 : CamelCase (AbcDef)
- 파일명 : CamelCase (AbcDef)
- 함수명 : lowerCamelCase (abcDef)
- 외 변수 등 : 연구원 재량
