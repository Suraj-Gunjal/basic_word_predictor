import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the tokenizer and model
model = load_model('next_word_lstm.h5')

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
    
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    
    # Trim to max length - 1
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]
    
    # Pad the sequence
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    
    # Predict next word
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]
    
    # Map index to word
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

# Streamlit app
st.title("Next Word Prediction with LSTM")
st.write("Enter a sentence to predict the next word:")
input_text = st.text_input("Input Text")
if input_text:
    max_sequence_len = 20  # Adjust based on your model's training
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    
    if next_word:
        st.write(f"The predicted next word is: **{next_word}**")
    else:
        st.write("Could not predict the next word.")
        
# Add a footer
st.markdown("---")