import transformers
import streamlit as st
import time

@st.cache(allow_output_mutation=True)
def load_models_trans():
    # English to Chinese
    EN_ZH_MODEL = "Helsinki-NLP/opus-mt-en-zh"
    en_zh_tokenizer = transformers.AutoTokenizer.from_pretrained(EN_ZH_MODEL)
    en_zh_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(EN_ZH_MODEL)
    en_zh_translator = transformers.pipeline("text2text-generation", model=en_zh_model, tokenizer=en_zh_tokenizer, device=0)
    return en_zh_translator

en_zh_translator = load_models_trans()

@st.cache(allow_output_mutation=True)
def load_models_qa():
    MODEL_NAME= "deepset/roberta-base-squad2"
    qa_tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    qa_model = transformers.AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME, return_dict=True)
    qa_pipeline = transformers.pipeline('question-answering', model=qa_model, tokenizer=qa_tokenizer)
    return qa_pipeline

qa_pipeline = load_models_qa()

@st.cache(allow_output_mutation=True)
def load_models_summary():
    MODEL_NAME_SUM = "mrm8488/t5-base-finetuned-summarize-news"
    sm_tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME_SUM)
    sm_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME_SUM)
    summarizer = transformers.pipeline("summarization", model=sm_model, tokenizer=sm_tokenizer)
    return summarizer

summarizer = load_models_summary()

st.title("Q and A App with Summary")

st.subheader("Input Context")
input_context = st.text_area("",value="Nvidia Corporation is an American multinational technology company incorporated in Delaware and based in Santa Clara, California.It is a software and fabless company which designs graphics processing units (GPUs), application programming interface (APIs) for Data Science and High-performance computing as well as system on a chip units (SoCs) for the mobile computing and automotive market. NVIDIA is a global leader in Artificial Intelligence hardware & software from edge to cloud computing and expanded its presence in the gaming industry with its handheld game consoles Shield Portable, Shield Tablet, and Shield Android TV and its cloud gaming service GeForce Now. Its professional line of GPUs are used in workstations for applications in such fields as architecture, engineering and construction, media and entertainment, automotive, scientific research, and manufacturing design.", height=150)

translate_input = en_zh_translator(input_context)[0]['generated_text']
translate_ans = ""
if len(input_context) > 0 :
    start_time = time.time()
    st.subheader("Abstractive Summary of Context")
    out_summary = summarizer(input_context, max_length=30)[0]['summary_text']
    st.markdown(out_summary)
    translate_sum = en_zh_translator(out_summary)[0]['generated_text']
    end_time = time.time()
    time_taken = str(round(end_time-start_time,2))
    st.markdown("Time taken: "+str(time_taken)+"s")
    st.subheader("Input Question")
    input_question = st.text_area("",value="What is Nvidia?", height=1)
    translate_q = en_zh_translator(input_question)[0]['generated_text']
    if len(input_question) > 0 :
        start_time = time.time()
        qa_input = {'question':input_question, 'context': input_context}
        qa_answer = qa_pipeline(qa_input)
        st.subheader("Answer to Question")
        st.markdown(qa_answer['answer'])
        translate_ans = en_zh_translator(qa_answer['answer'])[0]['generated_text']
        end_time = time.time()
        time_taken = str(round(end_time-start_time,2))
        st.markdown("Time taken: "+str(time_taken)+"s")
    else:
        st.markdown("Input the Question")
else:
    st.markdown("")

st.sidebar.header("Chinese Translation")
st.sidebar.subheader(en_zh_translator("Translation to chinese")[0]['generated_text'])
st.sidebar.markdown(en_zh_translator("**Input Context:** ")[0]['generated_text'])
st.sidebar.markdown(translate_input)
st.sidebar.markdown(en_zh_translator("**Summary:** ")[0]['generated_text'])
st.sidebar.markdown(translate_sum)
st.sidebar.markdown(en_zh_translator("**Input Question:** ")[0]['generated_text'])
st.sidebar.markdown(translate_q)
st.sidebar.markdown(en_zh_translator("**Answer to Question:** ")[0]['generated_text'])
st.sidebar.markdown(translate_ans)