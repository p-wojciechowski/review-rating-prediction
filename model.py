import tensorflow as tf 
from tensorflow import keras
from pickle import load
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

MAXLEN_SEQ=300

model = tf.keras.models.load_model('./lstm_model')
tokenizer = load(open('./lstm_model/tokenizer.pkl', 'rb'))
stopwords = load(open('./lstm_model/stopwords.pkl', 'rb'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r"n't", ' not', text)
    text = re.sub(r"\W|[\r\n]+", ' ', text) # remove not-words and newlines
    text = re.sub(r"\s{2,}", ' ', text) # remove '.', ',' and doubled whitespaces
    tokenized = [word for word in text.split() if word not in stopwords]
    text = ' '.join(tokenized)
    x_vector = tokenizer.texts_to_sequences([text])
    x_vector = pad_sequences(x_vector, padding='post', maxlen=MAXLEN_SEQ)
    return x_vector

def predict_rating(text):
    x_vector = preprocess(text)
    return model.predict(x_vector)[0,0]

def soft_preprocess(text):
    text = text.lower()
    text = re.sub(r"n't", ' not', text)
    text = re.sub(r"\W|[\r\n]+", ' ', text) # remove not-words and newlines
    text = re.sub(r"\s{2,}", ' ', text) # remove '.', ',' and doubled whitespaces
    tokenized = [word for word in text.split() if word not in stopwords]
    return tokenized