import streamlit as st
import torch
from transformers import AutoTokenizer
from model import LSTMClassifier  # assuming your model class is saved in model.py

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # change if using another tokenizer

# Set constants
MAX_LEN = 128
VOCAB_SIZE = len(tokenizer.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the trained model
model = LSTMClassifier(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM).to(DEVICE)
model.load_state_dict(torch.load("ai_human_classifier.pth", map_location=DEVICE))
model.eval()

# Prediction function
def predict(text):
    inputs = tokenizer.encode_plus(
        text,
        None,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_token_type_ids=True
    )

    input_ids = torch.tensor(inputs['input_ids'], dtype=torch.long).unsqueeze(0).to(DEVICE)
    length = torch.tensor([len([token for token in inputs['input_ids'] if token != tokenizer.pad_token_id])])

    with torch.no_grad():
        output = model(input_ids, length)
        prediction = torch.round(output).item()
        prob = output.item()

    label = "AI-generated" if prediction == 1 else "Human-written"
    return label, prob

# Streamlit UI
st.title("AI vs Human Text Classifier")

input_text = st.text_area("Enter text to classify:", height=200)

if st.button("Predict"):
    if input_text.strip():
        label, prob = predict(input_text)
        st.write(f"**Prediction:** {label}")
        st.write(f"**Confidence:** {prob:.2f}")
    else:
        st.warning("Please enter some text.")
