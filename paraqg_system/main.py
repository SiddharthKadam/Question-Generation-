from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from classes.part_of_speech import POS
from classes.named_entity_recognition import NER
from classes.bio_notation import BIO
# from classes.QuestionGenrationModel import GenerateQuestion
from classes.QuestionGenerator import QuestionGenerator
import re
import string 

app = FastAPI()
ner = NER()
pos = POS()
bio = BIO()
# QG = GenerateQuestion()
QG = QuestionGenerator()
def default_variable():
    return  {
        # Buttons
        "new_content" : True , 
        "review_content" : False,
        "select_answers" : False,
        "question_answers" : False,


        "noun_phrase" : False,
        "named_entity" : False,
        "custom_answers" : False,
        
        # text
        "original_text" : "",
        "selected_texts":[]

    }

variables = default_variable()
# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    variables["request"]=request
    return templates.TemplateResponse("index.html", variables)

@app.post("/selected_text")
async def toggle_selected_text(selected: str = Form(...)):
    print(selected)
    if selected in variables["selected_texts"]:
        variables["selected_texts"].remove(selected)
        action = "removed"
    else:
        variables["selected_texts"].append(selected)
        action = "added"

    return JSONResponse({"message": f"{selected} {action}", "selected_texts": variables["selected_texts"]})


@app.post("/update_pivot_ans")
async def update_pivot_ans(request: Request, button_name: str = Form(...)):
    print("This actually worked")
    print(button_name)
@app.post("/set_button")
async def set_button(request: Request, button_name: str = Form(...),original_text: str = Form(...)):
    print(button_name)
    print(variables["selected_texts"])
    if button_name != "update":
        variables["original_text"]=original_text
    if button_name == "new_Content":
        on_new_content()
    elif button_name == "Review_Content":
        on_review_content()
    elif button_name == "Select_Answers":
        on_select_answers()
    elif button_name == "Question_Answers":
        if len(variables["selected_texts"]) > 0:
            on_question_answers()

    elif button_name == "Noun_Phrase":
        on_noun_pharse()
    elif button_name == "Named_Entity":
        on_named_entity()
    elif button_name == "Custom_Answers":
        on_custom_answers()
    
    
    variables["request"]=request
    return templates.TemplateResponse("index.html", variables)



# Local Methods
def on_new_content():
    global variables
    variables = default_variable()

def on_review_content():
    if len(variables["original_text"]) != 0:
        review_text()
        variables["new_content"] = False  
        variables["review_content"] = True
        variables["select_answers"] = False
        variables["question_answers"] = False

        variables["noun_phrase"] = False
        variables["named_entity"] = False
        variables["custom_answers"] = False

def on_select_answers():
    print(variables["original_text"])
    if len(variables["original_text"]) != 0:

        variables["new_content"] = False  
        variables["review_content"] = False
        variables["select_answers"] = True
        variables["question_answers"] = False

        variables["noun_phrase"] = True
        variables["named_entity"] = False
        variables["custom_answers"] = False


        # Extract Pivotal answers

        # Extract NER
        _,ner_res_sentence_arr = ner.get_ner_context_pivot_ans(variables["original_text"])
        ner_pivotal_ans = []
        for i in ner_res_sentence_arr:
            # print(i["answer"])
            ner_pivotal_ans.append(i["answer"])

        # Extract POS
        _,pos_res_sentence_arr = pos.get_pos_context_pivot_ans(variables["modified_text"])
        pos_pivotal_ans = []
        for i in pos_res_sentence_arr:
            # print(i["answer"])
            pos_pivotal_ans.append(i["answer"])

        print(ner_pivotal_ans)
        print(pos_pivotal_ans)
        
        variables["ner_pivotal_ans"] = ner_pivotal_ans
        variables["pos_pivotal_ans"] = pos_pivotal_ans



def on_question_answers():
    if len(variables["original_text"]) != 0:

        variables["new_content"] = False  
        variables["review_content"] = False
        variables["select_answers"] = False
        variables["question_answers"] = True

        variables["noun_phrase"] = False
        variables["named_entity"] = False
        variables["custom_answers"] = False

    print("orignal_text:",variables["original_text"])

    if variables["original_text"] in "The First epoch is Always Hard , We are the Chaotic Noob , project Git Link : https://github.com/SiddharthKadam/Question-Generation-":
        variables   ["original_text"]= "who are you?"
    else:

        print("Ans_List:",variables["selected_texts"])
        text = ""
        genrated_questions = []
        for answer in variables["selected_texts"]:
            bio_sentence = bio.convert_BIO(context=variables["original_text"].lower(),answers=answer.lower())
            # context_seq_test = QG.tokenizer_text.texts_to_sequences([bio_sentence])
            # from tensorflow.keras.preprocessing.sequence import pad_sequences
            # context_seq_test = pad_sequences(context_seq_test, maxlen=QG.max_len_context, padding='post')
            print(answer)
            gen_que = QG.generate_question(context=bio_sentence)
            print(gen_que)
            genrated_questions.append(gen_que)

        
        for question in genrated_questions:
            text += " " + question +"\n"
            print(len(question.split()))
        variables["original_text"]= text
        # text = ""
        # for answer in variables["selected_texts"]:
        #     text+= " "+ bio.convert_BIO(context=variables["original_text"].lower(),answers=answer) +"\n"
        # variables["original_text"]= text


def on_noun_pharse():

    variables["noun_phrase"] = True
    variables["named_entity"] = False
    variables["custom_answers"] = False

def on_named_entity():
    variables["noun_phrase"] = False
    variables["named_entity"] = True
    variables["custom_answers"] = False

def on_custom_answers():
    print("intmethod")
    variables["noun_phrase"] = False
    variables["named_entity"] = False
    variables["custom_answers"] = True



# method to review Text
def review_text():
    text = re.sub(r',', r' ,', variables["original_text"])
    variables["original_text"] = text
    # Regex pattern for URLs
    url_pattern = r'https?://[^\s]+'
    # Find all URLs
    urls = re.findall(url_pattern, text)
    # Check for non-ASCII characters
    non_ascii = [char for char in text if ord(char) > 127]
    
    variables["urls"] = urls
    variables["non_ascii"]=non_ascii

    modified_text = ""
    for _ in text.split():
        if _ not in urls and _ not  in non_ascii:
            modified_text += " " + _

    punctuation_pattern = f"([{re.escape(string.punctuation)}])"

    modified_text = re.sub(punctuation_pattern, r' \1', modified_text)
    modified_text = modified_text.lower()
    variables["modified_text"] = modified_text
    
    print(urls,non_ascii)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
