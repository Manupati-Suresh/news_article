import streamlit as st
import torch
import numpy as np

# Try to import transformers with error handling
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    st.error(f"Error importing transformers: {e}")
    TRANSFORMERS_AVAILABLE = False

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
def load_classifier():
    """Load a simple text classification pipeline"""
    if not TRANSFORMERS_AVAILABLE:
        return None
        
    try:
        # Use a lightweight model that works well for text classification
        classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )
        return classifier
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Check if transformers is available
if not TRANSFORMERS_AVAILABLE:
    st.error("❌ Transformers library is not properly installed. Please check the deployment logs.")
    st.info("💡 This might be a temporary issue. Try refreshing the page in a few minutes.")
    st.stop()

# Load classifier
classifier = load_classifier()

if classifier is None:
    st.error("❌ Failed to load the classification model. Please try refreshing the page.")
    st.info("💡 The model might be downloading. This can take a few minutes on first load.")
    st.stop()

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
                # Define the news categories
                candidate_labels = ["World News", "Sports", "Business", "Science and Technology"]
                
                # Classify the text
                result = classifier(text, candidate_labels)
                
                # Get the results
                predicted_label = result['labels'][0]
                confidence = result['scores'][0]
                
                # Map to simpler labels
                label_mapping = {
                    "World News": "World",
                    "Sports": "Sports", 
                    "Business": "Business",
                    "Science and Technology": "Sci/Tech"
                }
                
                predicted_category = label_mapping.get(predicted_label, predicted_label)
                
                st.success(f"**🎯 Predicted Category:** {predicted_category}")
                st.info(f"**📊 Confidence:** {confidence:.2%}")
                
                # Show all probabilities
                st.subheader("📈 All Category Probabilities:")
                for label, score in zip(result['labels'], result['scores']):
                    mapped_label = label_mapping.get(label, label)
                    st.write(f"**{mapped_label}:** {score:.2%}")
                    
        except Exception as e:
            st.error(f"❌ Error during classification: {str(e)}")
            st.info("💡 Try with a shorter text or check your internet connection")
            st.info("🔧 If this persists, the model might still be loading. Please wait a moment and try again.")
    else:
        st.warning("⚠️ Please enter some text to classify!")

# Add footer
st.markdown("---")
st.markdown("**🚀 Built with:** Streamlit • Transformers • PyTorch")
st.markdown("**📚 Dataset:** AG News (World, Sports, Business, Sci/Tech)")
st.markdown("**🔗 [View Source Code](https://github.com/yourusername/news-article-classifier)**")
