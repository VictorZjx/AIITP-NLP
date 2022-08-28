import pickle
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import tensorflow as tf
import models
import transformers

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

DEFAULT_TEXT_INPUT = """This movie is really bad. Do not watch it!"""
CLASSES = ["negative", "positive"]
CLASSES_SCORE = ["1 Star", "2 Star", "3 Star", "4 Star", "5 Star"]

st.title("Sentiment Analysis")
st.markdown("**Stanford IMDB Large Movie Review Dataset**")
st.markdown("**Douban Movie Short Comments Dataset**")
st.caption("Note: The chinese roberta model should be typed in chinese. The prediction is based on 1 star to 5 star")

text_input = st.text_area(label="Input Text",
                                 value=DEFAULT_TEXT_INPUT,
                                 height=100)

st.sidebar.image("https://logos-download.com/wp-content/uploads/2016/10/Nvidia_logo.png", use_column_width=True)
st.sidebar.markdown("**Model selection**")
keras_models = ["MLP", "CNN", "LSTM"]
huggingface_models = ["distilroberta-base", "hfl/chinese-roberta-wwm-ext"]
model_name = st.sidebar.selectbox("Select model to use", keras_models+huggingface_models)

# note that when using keras models, the tokenizer is loaded from the saved pickle file
# and using transformers from huggingface, there is an autotokenizer.

@st.cache(allow_output_mutation=True)
def load_tokenizer(model_name, tokenizer_path="./data/tokenizer.pickle"):
    if model_name in keras_models:
        with open(tokenizer_path, "rb") as f:
            tokenizer = pickle.load(f)
    elif model_name in huggingface_models:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    return tokenizer

# this allows streamlit to save the cache of the loaded models, so 2nd run will not re-run model

@st.cache(allow_output_mutation=True)
def load_model_from_name(model_name="MLP"):
    model_choices = {
        "MLP": "./data/best_mlp.h5",
        "CNN": "./data/best_cnn.h5",
        "LSTM": "./data/best_lstm.h5",
        "distilroberta-base": "./data/best_transformer.h5",
        "hfl/chinese-roberta-wwm-ext": "./data/best_chinese_transformer.h5"
    }
    model_path = model_choices[model_name]
    if model_name in keras_models:
        model = tf.keras.models.load_model(model_path, compile=True)
    elif model_name in huggingface_models:
        if model_name == "distilroberta-base":
            model = transformers.TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
            model.load_weights(model_path)
        elif model_name == "hfl/chinese-roberta-wwm-ext":
            model = transformers.TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)
            model.load_weights(model_path)         
    return model

with st.spinner("Loading model"):
    tokenizer = load_tokenizer(model_name=model_name)
    model = load_model_from_name(model_name=model_name)
    tf.keras.utils.plot_model(model, to_file="./data/model.png", show_shapes=True, show_layer_names=True)
    image = Image.open("./data/model.png")
    trainable_count = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_count = np.sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
    num_params = trainable_count + non_trainable_count
    num_params = str(int(num_params/1e6))+"M"
    caption = num_params + " param " + model_name
    st.sidebar.image(image, caption=caption, use_column_width=True)

# for keras models, texts_to_sequences is used here
# for transformer models, encode is used instead

def predict(model_name, model, input_text, max_seq_len=128):
    if model_name in keras_models:
        input_text = input_text.strip().lower()
        input_seq = tokenizer.texts_to_sequences([input_text])
        input_seq = tf.keras.preprocessing.sequence.pad_sequences(input_seq,
                                                         padding="post",
                                                         maxlen=max_seq_len)
        preds = model.predict(input_seq)[0]
    elif model_name in huggingface_models:
        input_seq = [tokenizer.encode(input_text.strip())]
        input_seq = tf.keras.preprocessing.sequence.pad_sequences(input_seq,
                                                         padding="post",
                                                         maxlen=max_seq_len)
        preds = model.predict(input_seq)[0][0]
    return preds

with st.spinner("Running inference"):
    start_time = time.time()
    logits = predict(model_name, model, text_input)
    end_time = time.time()

prediction = np.argmax(logits)
if model_name == "hfl/chinese-roberta-wwm-ext":
    prediction = CLASSES_SCORE[prediction]
else:
    prediction = CLASSES[prediction]
st.markdown("### Prediction: "+prediction)
time_taken = str(round(end_time-start_time, 2))
st.markdown("Inference time: "+time_taken+" seconds")

with st.expander("See logits"):
    st.markdown(logits)
    fig, ax = plt.subplots()
    if model_name == "hfl/chinese-roberta-wwm-ext":
        sns.barplot(x=CLASSES_SCORE, y=logits)
    else:
        sns.barplot(x=CLASSES, y=logits)
    st.pyplot(fig)

with st.expander("Dataset citation"):
    st.markdown("""[**Stanford IMDB Large Movie Review Dataset**](https://ai.stanford.edu/~amaas/data/sentiment/)
    Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011). [*Learning Word Vectors for Sentiment Analysis*](https://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf). The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011).""")

    st.markdown('''[**Douban Movie Short Comments Dataset**](https://www.kaggle.com/datasets/utmhikari/doubanmovieshortcomments)
    This dataset is taken from kaggle to train chinese characters and classifying from 1 to 5 stars.''')
