import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import os
import glob

st.title("📰 News Article Classifier")

# Check for available checkpoints
checkpoint_dirs = glob.glob("./results/checkpoint-*")
if checkpoint_dirs:
    # Use the latest checkpoint
    latest_checkpoint = max(checkpoint_dirs, key=os.path.getctime)
    st.info(f"Using trained model: {latest_checkpoint}")
    model_path = latest_checkpoint
else:
    st.warning("⚠️ No trained model found! Please run `python train_model.py` first to train the model.")
    st.info("For now, using a basic BERT model (not fine-tuned for news classification)")
    model_path = "bert-base-uncased"

# Sample articles for quick testing
sample_articles = {
    "Business": "Apple Inc. reported record quarterly earnings today, beating analyst expectations with revenue of $95 billion.",
    "Sports": "Manchester United defeats Barcelona 3-1 in Champions League semifinal match.",
    "World": "UN Security Council votes on new sanctions against North Korea over missile tests.",
    "Sci/Tech": "NASA's James Webb telescope discovers potentially habitable exoplanet 100 light-years away."
}

st.subheader("Quick Test Examples:")
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("📈 Business"):
        st.session_state.sample_text = sample_articles["Business"]
with col2:
    if st.button("⚽ Sports"):
        st.session_state.sample_text = sample_articles["Sports"]
with col3:
    if st.button("🌍 World"):
        st.session_state.sample_text = sample_articles["World"]
with col4:
    if st.button("🔬 Sci/Tech"):
        st.session_state.sample_text = sample_articles["Sci/Tech"]

# Get text from session state if sample was clicked
default_text = st.session_state.get('sample_text', '')

text = st.text_area("Enter news article text here", 
                   value=default_text,
                   placeholder="Example: Apple Inc. announced record quarterly earnings today...")

if st.button("Classify"):
    if text.strip():
        try:
            with st.spinner("Loading model and classifying..."):
                tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
                
                if model_path == "bert-base-uncased":
                    # Use base model with 4 labels for news classification
                    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4)
                    st.warning("Using untrained model - results may not be accurate!")
                else:
                    # Use fine-tuned model
                    model = BertForSequenceClassification.from_pretrained(model_path)

                inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                
                probs = torch.nn.functional.softmax(outputs.logits, dim=1)
                pred = torch.argmax(probs).item()
                confidence = probs[0][pred].item()

                labels = ["World", "Sports", "Business", "Sci/Tech"]
                
                st.success(f"**Predicted Category:** {labels[pred]}")
                st.info(f"**Confidence:** {confidence:.2%}")
                
                # Show all probabilities
                st.subheader("All Category Probabilities:")
                for i, (label, prob) in enumerate(zip(labels, probs[0])):
                    st.write(f"{label}: {prob.item():.2%}")
                    
        except Exception as e:
            st.error(f"Error during classification: {str(e)}")
    else:
        st.warning("Please enter some text to classify!")
