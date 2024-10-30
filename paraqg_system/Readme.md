# Steps to Run the Question Generation System

## Prerequisites
1. Install **Python**.
2. Install **Java**.

## python Packages to Install
1. Install **fastapi**.
2. Install **uvicorn**.
3. Install **pydantic**.
4. Install **jinja2**.
5. Install **python-multipart**.
6. Install **nltk**

## Step 1: Download Stanford CoreNLP
- Download Stanford CoreNLP from the following link: [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/).
- Extract the downloaded file into a folder named **Question-Generation**.

## Step 2: Run Stanford CoreNLP
1. Open a terminal and navigate to the **Question-Generation/stanford-corenlp-4.5.7** directory.
2. Paste the following command to start the Stanford CoreNLP server:
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer \
  -port 9000 -timeout 15000

## step-3 Start the FastAPI Server 
1. Open another terminal and navigate to the Question-Generation/paraqg_system directory.
2. Paste the following command to start the FastAPI server:
uvicorn main:app --reload
3. Open your web browser and visit the following link: http://127.0.0.1:8000/.
