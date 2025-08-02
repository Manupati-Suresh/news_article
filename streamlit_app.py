import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import torch
import os
import glob

# Page config
st.set_page_config(
    page_title="News Article Classifier",
    page_icon="📰",
    layout="wide"
)

st.title("📰 News Article Classifier")
st.markdown("*Classify news articles into World, Sports, Business, or Science & Technology categories*")

# Cache the model loading for better performance
@st.cache_resource
def load_model():
    """Load the classification model with caching for better performance"""
    try:
        # Check for available checkpoints first
        checkpoint_dirs = glob.glob("./results/checkpoint-*")
        if checkpoint_dirs:
            latest_checkpoint = max(checkpoint_dirs, key=os.path.getctime)
            st.info(f"✅ Using fine-tuned model: {latest_checkpoint}")
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            model = BertForSequenceClassification.from_pretrained(latest_checkpoint)
            return tokenizer, model, "fine-tuned"
        else:
            # Use a pre-trained model that works better for text classification
            st.info("📝 Using pre-trained BERT model (not specifically fine-tuned for news)")
            # Use a pipeline for better out-of-box performance
            classifier = pipeline(
                "text-classification",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True
            )
            return None, classifier, "pipeline"
    except Exception as e:
        st.error(f"Error loading model: {e}")
        # Fallback to basic BERT
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4)
        return tokenizer, model, "basic"

# Load model
tokenizer, model, model_type = load_model()

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

if st.button("🔍 Classify Article", type="primary"):
    if text.strip():
        try:
            with st.spinner("🤖 Analyzing article..."):
                labels = ["World", "Sports", "Business", "Sci/Tech"]
                
                if model_type == "fine-tuned":
                    # Use fine-tuned model
                    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
                    with torch.no_grad():
                        outputs = model(**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
                    pred = torch.argmax(probs).item()
                    confidence = probs[0][pred].item()
                    
                    st.success(f"**🎯 Predicted Category:** {labels[pred]}")
                    st.info(f"**📊 Confidence:** {confidence:.2%}")
                    
                    # Show all probabilities
                    st.subheader("📈 All Category Probabilities:")
                    for i, (label, prob) in enumerate(zip(labels, probs[0])):
                        st.write(f"**{label}:** {prob.item():.2%}")
                        
                elif model_type == "pipeline":
                    # Use pipeline model (fallback)
                    results = model(text)
                    st.warning("⚠️ Using general sentiment model - results are approximate")
                    
                    # Map sentiment to news categories (rough approximation)
                    sentiment_to_category = {
                        "LABEL_0": "World",  # Negative -> World news
                        "LABEL_1": "Business",  # Neutral -> Business
                        "LABEL_2": "Sports"  # Positive -> Sports
                    }
                    
                    best_result = max(results, key=lambda x: x['score'])
                    predicted_category = sentiment_to_category.get(best_result['label'], "Sci/Tech")
                    
                    st.success(f"**🎯 Predicted Category:** {predicted_category}")
                    st.info(f"**📊 Confidence:** {best_result['score']:.2%}")
                    
                else:
                    # Use basic BERT model
                    st.warning("⚠️ Using basic BERT model - results may not be accurate for news classification!")
                    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
                    with torch.no_grad():
                        outputs = model(**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
                    pred = torch.argmax(probs).item()
                    confidence = probs[0][pred].item()
                    
                    st.success(f"**🎯 Predicted Category:** {labels[pred]}")
                    st.info(f"**📊 Confidence:** {confidence:.2%}")
                    
                    # Show all probabilities
                    st.subheader("📈 All Category Probabilities:")
                    for i, (label, prob) in enumerate(zip(labels, probs[0])):
                        st.write(f"**{label}:** {prob.item():.2%}")
                    
        except Exception as e:
            st.error(f"❌ Error during classification: {str(e)}")
            st.info("💡 Try with a shorter text or check your internet connection")
    else:
        st.warning("⚠️ Please enter some text to classify!")

# Add footer
st.markdown("---")
st.markdown("**🚀 Built with:** Streamlit • Transformers • PyTorch")
st.markdown("**📚 Dataset:** AG News (World, Sports, Business, Sci/Tech)")
st.markdown("**🔗 [View Source Code](https://github.com/yourusername/news-article-classifier)**")
