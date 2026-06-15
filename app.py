import streamlit as st
import pickle
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load tokenizer
with open(
    "models/tokenizer.pkl",
    "rb"
) as f:

    tokenizer = pickle.load(f)

total_words = len(tokenizer.word_index) + 1

# Must match training
max_sequence_len = 15

# Rebuild model
model = Sequential([
    Embedding(
        input_dim=total_words,
        output_dim=100,
        input_shape=(max_sequence_len-1,)
    ),

    SimpleRNN(150),

    Dense(
        total_words,
        activation='softmax'
    )
])

# Load weights
model.load_weights(
    "models/next_word.weights.h5"
)

st.title(
    "Next Word Prediction using RNN"
)

seed_text = st.text_input(
    "Enter text"
)

if st.button(
    "Predict"
):

    token_list = tokenizer.texts_to_sequences(
        [seed_text]
    )[0]

    token_list = pad_sequences(
        [token_list],
        maxlen=max_sequence_len-1,
        padding='pre'
    )

    predicted = np.argmax(
        model.predict(token_list, verbose=0),
        axis=-1
    )[0]

    next_word = ""

    for word, index in tokenizer.word_index.items():

        if index == predicted:

            next_word = word

            break

    st.success(
        seed_text + " " + next_word
    )