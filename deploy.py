import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

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

        .main {
            max-width: 1000px;
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

# Load the pre-trained model and tokenizer
checkpoint = "distilbert-base-cased"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Display header
st.title("Hate Speech Detector")

# Input field for the user to enter text
user_input = st.text_area("Enter your text here:")

# Predict button
if st.button('Predict'):
    # Check if the user has entered any text
    if user_input:
        # Tokenize the text and prepare it for the model
        inputs = tokenizer(user_input, return_tensors="pt")

        # Forward pass through the model
        outputs = model(**inputs)

        # Get the predicted class probabilities using softmax
        probs = torch.softmax(outputs.logits, dim=1).detach().numpy()[0]

        # Swap predicted class labels (0 for hate speech, 1 for non-hate speech)
        predicted_class = 1 - int(torch.argmax(outputs.logits))

        # Display prediction and probabilities
        st.write("Predicted Class:", "Hate Speech" if predicted_class == 0 else "Non-Hate Speech")
        st.write("Class Probabilities:", f"Hate Speech: {probs[0]:.4f}, Non-Hate Speech: {probs[1]:.4f}")
