# Steps to run 
## install Python, Java, FastApi

## Step - 1 download Standford Core NLP 
Link : https://stanfordnlp.github.io/CoreNLP/

## Step - 2 Run Standford Core NLP 

Open Terminal to / at path PARA_QG\stanford-corenlp-4.5.7

Paste this bash Command :
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer \
  -port 9000 -timeout 15000

## step-3 start FastAPI Server 

open terminal to / at path Question-Generation\paraqg_system

Paste this bash Command : uvicorn main:app --reload

open this link in browser :  http://127.0.0.1:8000/

