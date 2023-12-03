import streamlit as st
import joblib
import re
import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Embedding, LSTM, Dense, Activation, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l1_l2

nltk.download('punkt')

# Define stemmer and stop_words
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def create_model(lstm_units=64, dense_units=256, l1_reg=0.01, l2_reg=0.01, learning_rate=0.001):
    inputs = Input(name='inputs', shape=[max_words])
    layer = Embedding(2000, 100, input_length=max_words)(inputs)
    layer = LSTM(lstm_units)(layer)
    layer = Dense(dense_units, name='FC1', kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg))(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1, name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs, outputs=layer)
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(learning_rate=learning_rate),
                  metrics=['accuracy'])
    return model

# Function to preprocess the input string
def preprocess_string(text):
    text = text.lower()
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\bRT\b|\brt\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\d', '', text)
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stop_words]
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    return stemmed_tokens

# Load the trained model and tokenizer
loaded_model = joblib.load('lstm_model.joblib')  # Update with your model file path
tokenizer = joblib.load('tokenizer_instance.joblib')  # Update with your tokenizer file path

# Set page configuration to widen the app
st.set_page_config(layout="wide")

# Display styled disclaimer
st.markdown("""
    <style>
        .disclaimer {
            font-size: 18px;
            font-weight: bold;
            color: #FF6347;  /* Tomato color for warning */
            margin-bottom: 20px;
        }

        .header {
            font-size: 50px;
            font-weight: bold;
            margin-top: 20px;
            margin-bottom: 20px;
        }

        .input-field {
            margin-bottom: 20px;
        }

        .prediction {
            font-size: 20px;
            margin-top: 20px;
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Display disclaimer
st.markdown("""
    <div class="disclaimer">Disclaimer: Hate speech classification is subjective. The model provides predictions based on the training data, and interpretations may vary. 
    Use the results with caution and consider the context before drawing conclusions.</div>
""", unsafe_allow_html=True)

# Adjust the app width using CSS
st.image('https://i0.wp.com/vitalflux.com/wp-content/uploads/2022/03/hate-speech-detection-using-machine-learning.png?resize=640%2C325&ssl=1', use_column_width=True)
st.markdown(
    """
    <style>
    .main {
        max-width: 1000px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Display header
st.markdown("""
    <div class="header">HATE SPEECH DETECTOR SYSTEM</div>
""", unsafe_allow_html=True)

# Input field for the user to enter their tweet
user_input = st.text_input("Enter your tweet here:")

# Display the input
if st.button('Check for Hate Speech'):
    # Preprocess the input
    processed_input = preprocess_string(user_input)
    
    # Convert tokens to sequences using the tokenizer
    input_sequence = tokenizer.texts_to_sequences([processed_input])

    # Pad sequences to match the input shape expected by the LSTM model
    max_words = 1000  # Assuming max_words during training was 500
    input_sequence_padded = pad_sequences(input_sequence, maxlen=max_words, padding='post')

    # Predict using the trained LSTM model
    prediction = loaded_model.predict(input_sequence_padded)
# ...



    # Display prediction and debug information
    st.write("Raw prediction:", prediction)

    # Check if any value in the prediction array is greater than 0.5
    if np.any(prediction > 0.5):
        st.warning("Hate speech detected! The model suggests that this might be hate speech.")
    else:
        st.info("No hate speech detected. The model suggests that this seems to be a regular comment.")
