
# Steps to Run the Question Generation System

## Prerequisites
1. Install **Python**.
2. Install **Java**.

## Step 1: Download Stanford CoreNLP
- Download Stanford CoreNLP from the following link: [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/).
- Extract the downloaded file into a folder named **Question-Generation**.

## Step 2: Run Stanford CoreNLP
1. Open a terminal and navigate to the **Question-Generation/stanford-corenlp-4.5.7** directory.
2. Paste the following command to start the Stanford CoreNLP server:
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer \
  -port 9000 -timeout 15000

## Step 3: Download glove.6B.300d.txt 
- Download glove.6B.300d.txt from the following link: [glove.6B.300d.txt](https://www.kaggle.com/datasets/thanakomsn/glove6b300dtxt).
- Extract the downloaded file into a folder named **Question-Generation\code\data**.
