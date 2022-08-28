import transformers
from transformers import pipeline

def load_models_trans():
#    # English to Chinese
    EN_ZH_MODEL = "Helsinki-NLP/opus-mt-en-zh"
    en_zh_tokenizer = transformers.AutoTokenizer.from_pretrained(EN_ZH_MODEL)
    en_zh_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(EN_ZH_MODEL)
    en_zh_translator = transformers.pipeline('text2text-generation', model=en_zh_model, tokenizer=en_zh_tokenizer, device=0)
    return en_zh_translator

en_zh_translator = load_models_trans()    

def load_models_qa():
    MODEL_NAME = "deepset/roberta-base-squad2"
    qa_tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    qa_model = transformers.AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME, return_dict=True)
    qa_pipeline = transformers.pipeline('question-answering', model=qa_model, tokenizer=qa_tokenizer)
    return qa_pipeline

qa_pipeline = load_models_qa()

def load_models_summary():
    MODEL_NAME = "mrm8488/t5-base-finetuned-summarize-news"
    sm_tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    sm_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    summarizer = transformers.pipeline("summarization", model=sm_model, tokenizer=sm_tokenizer)
    return summarizer

summarizer = load_models_summary()


