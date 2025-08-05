import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import time
import re
from collections import Counter
import base64

# Try to import transformers with error handling
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    st.error(f"Error importing transformers: {e}")
    TRANSFORMERS_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="AI News Article Classifier Pro",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .category-card {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    .category-card:hover {
        border-color: #4ECDC4;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .confidence-bar {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        height: 20px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .footer {
        text-align: center;
        padding: 2rem;
        background: #f8f9fa;
        border-radius: 10px;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🤖 AI News Classifier Pro</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced AI-powered news article classification with analytics and insights</p>', unsafe_allow_html=True)

# Initialize session state
if 'classification_history' not in st.session_state:
    st.session_state.classification_history = []
if 'total_classifications' not in st.session_state:
    st.session_state.total_classifications = 0
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

# Cache the model loading for better performance
@st.cache_resource
def load_classifier():
    """Load multiple classification models for better accuracy"""
    if not TRANSFORMERS_AVAILABLE:
        return None, None
        
    try:
        # Primary classifier for news categorization
        news_classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )
        
        # Secondary classifier for sentiment analysis
        sentiment_classifier = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest"
        )
        
        return news_classifier, sentiment_classifier
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

# Utility functions
def extract_keywords(text, top_k=5):
    """Extract key words from text"""
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    word_freq = Counter(words)
    return [word for word, _ in word_freq.most_common(top_k)]

def calculate_readability_score(text):
    """Simple readability score based on sentence and word length"""
    sentences = text.split('.')
    words = text.split()
    if len(sentences) == 0 or len(words) == 0:
        return 0
    avg_sentence_length = len(words) / len(sentences)
    avg_word_length = sum(len(word) for word in words) / len(words)
    # Simple readability formula (lower is easier to read)
    score = (avg_sentence_length * 0.39) + (avg_word_length * 11.8) - 15.59
    return max(0, min(100, score))

def get_article_stats(text):
    """Get comprehensive article statistics"""
    words = text.split()
    sentences = text.split('.')
    paragraphs = text.split('\n\n')
    
    return {
        'word_count': len(words),
        'sentence_count': len([s for s in sentences if s.strip()]),
        'paragraph_count': len([p for p in paragraphs if p.strip()]),
        'avg_words_per_sentence': len(words) / max(len(sentences), 1),
        'readability_score': calculate_readability_score(text),
        'keywords': extract_keywords(text)
    }

# Check if transformers is available
if not TRANSFORMERS_AVAILABLE:
    st.error("❌ Transformers library is not properly installed. Please check the deployment logs.")
    st.info("💡 This might be a temporary issue. Try refreshing the page in a few minutes.")
    st.stop()

# Load classifiers with progress bar
if not st.session_state.model_loaded:
    with st.spinner("🚀 Loading AI models... This may take a few minutes on first load."):
        progress_bar = st.progress(0)
        progress_bar.progress(25)
        
        news_classifier, sentiment_classifier = load_classifier()
        progress_bar.progress(75)
        
        if news_classifier is None:
            st.error("❌ Failed to load the classification models. Please try refreshing the page.")
            st.info("💡 The models might be downloading. This can take a few minutes on first load.")
            st.stop()
        
        progress_bar.progress(100)
        st.session_state.model_loaded = True
        st.success("✅ AI models loaded successfully!")
        time.sleep(1)
        st.rerun()
else:
    news_classifier, sentiment_classifier = load_classifier()

# Sidebar for settings and analytics
with st.sidebar:
    st.header("🎛️ Control Panel")
    
    # Model settings
    st.subheader("⚙️ Model Settings")
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    show_all_scores = st.checkbox("Show All Category Scores", True)
    enable_sentiment = st.checkbox("Enable Sentiment Analysis", True)
    enable_analytics = st.checkbox("Enable Text Analytics", True)
    
    # Analytics dashboard
    st.subheader("📊 Analytics Dashboard")
    st.metric("Total Classifications", st.session_state.total_classifications)
    
    if st.session_state.classification_history:
        # Category distribution
        categories = [item['category'] for item in st.session_state.classification_history]
        category_counts = Counter(categories)
        
        fig = px.pie(
            values=list(category_counts.values()),
            names=list(category_counts.keys()),
            title="Category Distribution"
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Average confidence over time
        confidences = [item['confidence'] for item in st.session_state.classification_history[-10:]]
        fig2 = px.line(
            x=range(len(confidences)),
            y=confidences,
            title="Recent Confidence Scores",
            labels={'x': 'Classification #', 'y': 'Confidence'}
        )
        fig2.update_layout(height=250)
        st.plotly_chart(fig2, use_container_width=True)
    
    # Export data
    if st.session_state.classification_history:
        if st.button("📥 Export History"):
            df = pd.DataFrame(st.session_state.classification_history)
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="classification_history.csv">Download CSV</a>'
            st.markdown(href, unsafe_allow_html=True)
    
    # Clear history
    if st.button("🗑️ Clear History"):
        st.session_state.classification_history = []
        st.session_state.total_classifications = 0
        st.rerun()

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Enhanced sample articles
    sample_articles = {
        "Business": {
            "text": "Apple Inc. reported record quarterly earnings today, beating analyst expectations with revenue of $95 billion. The tech giant's iPhone sales surged 15% year-over-year, driven by strong demand for the latest iPhone 15 series. CEO Tim Cook attributed the success to innovative features and expanding market presence in emerging economies.",
            "icon": "📈"
        },
        "Sports": {
            "text": "Manchester United secured a thrilling 3-1 victory over Barcelona in the Champions League semifinal at Old Trafford. Marcus Rashford scored twice in the first half, while Bruno Fernandes sealed the win with a spectacular free-kick in the 78th minute. The Red Devils will face Real Madrid in the final next month.",
            "icon": "⚽"
        },
        "World": {
            "text": "The UN Security Council voted unanimously to impose new economic sanctions on North Korea following its latest ballistic missile tests. The resolution, backed by all 15 member states, targets key sectors of the North Korean economy and restricts diplomatic travel. Secretary-General António Guterres called for immediate de-escalation.",
            "icon": "🌍"
        },
        "Sci/Tech": {
            "text": "NASA's James Webb Space Telescope has discovered a potentially habitable exoplanet located 100 light-years from Earth. The planet, designated K2-18b, shows signs of water vapor and methane in its atmosphere. Scientists believe it could harbor conditions suitable for life, marking a significant breakthrough in the search for extraterrestrial life.",
            "icon": "🔬"
        }
    }
    
    st.subheader("🚀 Quick Test Examples")
    
    # Create cards for sample articles
    cols = st.columns(2)
    for i, (category, article) in enumerate(sample_articles.items()):
        with cols[i % 2]:
            with st.container():
                st.markdown(f"""
                <div class="category-card">
                    <h4>{article['icon']} {category}</h4>
                    <p style="font-size: 0.9em; color: #666;">{article['text'][:100]}...</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button(f"Use {category} Example", key=f"btn_{category}"):
                    st.session_state.sample_text = article['text']
                    st.rerun()
    
    # Text input area
    st.subheader("📝 Enter Your Article")
    default_text = st.session_state.get('sample_text', '')
    
    text = st.text_area(
        "Paste your news article here:",
        value=default_text,
        height=200,
        placeholder="Paste a news article here to classify it into World, Sports, Business, or Science & Technology categories...",
        help="Enter any news article text. The AI will analyze and classify it automatically."
    )
    
    # File upload option
    uploaded_file = st.file_uploader(
        "Or upload a text file:",
        type=['txt'],
        help="Upload a .txt file containing your news article"
    )
    
    if uploaded_file is not None:
        text = str(uploaded_file.read(), "utf-8")
        st.success(f"✅ File uploaded successfully! ({len(text.split())} words)")

with col2:
    # Real-time text statistics
    if text.strip():
        st.subheader("📊 Article Statistics")
        stats = get_article_stats(text)
        
        # Display metrics in a nice format
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Words", stats['word_count'])
            st.metric("Sentences", stats['sentence_count'])
        with col_b:
            st.metric("Paragraphs", stats['paragraph_count'])
            st.metric("Readability", f"{stats['readability_score']:.1f}")
        
        # Keywords
        if stats['keywords']:
            st.subheader("🔑 Key Terms")
            keywords_html = " ".join([f"<span style='background: #e1f5fe; padding: 2px 8px; border-radius: 12px; margin: 2px; display: inline-block;'>{kw}</span>" for kw in stats['keywords']])
            st.markdown(keywords_html, unsafe_allow_html=True)

# Classification button
st.markdown("---")
classify_col1, classify_col2, classify_col3 = st.columns([1, 2, 1])

with classify_col2:
    classify_button = st.button(
        "🤖 Analyze Article with AI",
        type="primary",
        use_container_width=True,
        help="Click to classify the article using advanced AI models"
    )

if classify_button:
    if text.strip():
        try:
            # Create progress indicators
            progress_container = st.container()
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: Text preprocessing
                status_text.text("🔍 Preprocessing text...")
                progress_bar.progress(20)
                time.sleep(0.5)
                
                # Step 2: News classification
                status_text.text("🤖 Classifying article...")
                progress_bar.progress(50)
                
                candidate_labels = ["World News", "Sports", "Business", "Science and Technology"]
                result = news_classifier(text, candidate_labels)
                
                # Step 3: Sentiment analysis (if enabled)
                sentiment_result = None
                if enable_sentiment and sentiment_classifier:
                    status_text.text("😊 Analyzing sentiment...")
                    progress_bar.progress(75)
                    sentiment_result = sentiment_classifier(text[:512])  # Limit text length
                
                # Step 4: Final processing
                status_text.text("📊 Generating insights...")
                progress_bar.progress(100)
                time.sleep(0.5)
                
                # Clear progress indicators
                progress_container.empty()
                
                # Process results
                predicted_label = result['labels'][0]
                confidence = result['scores'][0]
                
                label_mapping = {
                    "World News": "World",
                    "Sports": "Sports", 
                    "Business": "Business",
                    "Science and Technology": "Sci/Tech"
                }
                
                predicted_category = label_mapping.get(predicted_label, predicted_label)
                
                # Category icons and colors
                category_config = {
                    "World": {"icon": "🌍", "color": "#FF6B6B"},
                    "Sports": {"icon": "⚽", "color": "#4ECDC4"},
                    "Business": {"icon": "📈", "color": "#45B7D1"},
                    "Sci/Tech": {"icon": "🔬", "color": "#96CEB4"}
                }
                
                # Display results with enhanced UI
                st.markdown("## 🎯 Classification Results")
                
                # Main result card
                config = category_config.get(predicted_category, {"icon": "📰", "color": "#666"})
                
                if confidence >= confidence_threshold:
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, {config['color']}22, {config['color']}44);
                        border-left: 5px solid {config['color']};
                        padding: 20px;
                        border-radius: 10px;
                        margin: 20px 0;
                    ">
                        <h2 style="color: {config['color']}; margin: 0;">
                            {config['icon']} {predicted_category}
                        </h2>
                        <p style="font-size: 1.2em; margin: 10px 0 0 0;">
                            Confidence: <strong>{confidence:.1%}</strong>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning(f"⚠️ Low confidence classification: {predicted_category} ({confidence:.1%})")
                    st.info("💡 Consider reviewing the article or trying a different text.")
                
                # Detailed scores
                if show_all_scores:
                    st.markdown("### 📊 Detailed Category Scores")
                    
                    for i, (label, score) in enumerate(zip(result['labels'], result['scores'])):
                        mapped_label = label_mapping.get(label, label)
                        config = category_config.get(mapped_label, {"icon": "📰", "color": "#666"})
                        
                        # Create progress bar for each category
                        col1, col2, col3 = st.columns([1, 3, 1])
                        with col1:
                            st.write(f"{config['icon']} **{mapped_label}**")
                        with col2:
                            st.progress(score)
                        with col3:
                            st.write(f"**{score:.1%}**")
                
                # Sentiment analysis results
                if enable_sentiment and sentiment_result:
                    st.markdown("### 😊 Sentiment Analysis")
                    sentiment_label = sentiment_result[0]['label']
                    sentiment_score = sentiment_result[0]['score']
                    
                    sentiment_mapping = {
                        'LABEL_0': {'name': 'Negative', 'icon': '😞', 'color': '#FF6B6B'},
                        'LABEL_1': {'name': 'Neutral', 'icon': '😐', 'color': '#FFA726'},
                        'LABEL_2': {'name': 'Positive', 'icon': '😊', 'color': '#4ECDC4'}
                    }
                    
                    sentiment_info = sentiment_mapping.get(sentiment_label, {'name': 'Unknown', 'icon': '❓', 'color': '#666'})
                    
                    st.markdown(f"""
                    <div style="
                        background: {sentiment_info['color']}22;
                        border: 1px solid {sentiment_info['color']};
                        padding: 15px;
                        border-radius: 8px;
                        text-align: center;
                    ">
                        <h4>{sentiment_info['icon']} {sentiment_info['name']} Sentiment</h4>
                        <p>Confidence: {sentiment_score:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Text analytics
                if enable_analytics:
                    st.markdown("### 📈 Article Analytics")
                    stats = get_article_stats(text)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Reading Time", f"{stats['word_count'] // 200 + 1} min")
                    with col2:
                        st.metric("Complexity", "Easy" if stats['readability_score'] < 50 else "Medium" if stats['readability_score'] < 70 else "Hard")
                    with col3:
                        st.metric("Avg Words/Sentence", f"{stats['avg_words_per_sentence']:.1f}")
                    with col4:
                        st.metric("Key Terms", len(stats['keywords']))
                
                # Save to history
                classification_record = {
                    'timestamp': datetime.now().isoformat(),
                    'category': predicted_category,
                    'confidence': confidence,
                    'text_length': len(text),
                    'word_count': len(text.split()),
                    'sentiment': sentiment_info['name'] if enable_sentiment and sentiment_result else None
                }
                
                st.session_state.classification_history.append(classification_record)
                st.session_state.total_classifications += 1
                
                # Success message
                st.success("✅ Analysis complete! Results saved to history.")
                
        except Exception as e:
            st.error(f"❌ Error during classification: {str(e)}")
            st.info("💡 Try with a shorter text or check your internet connection")
            st.info("🔧 If this persists, the model might still be loading. Please wait a moment and try again.")
            
            # Debug information
            with st.expander("🔍 Debug Information"):
                st.write(f"Error type: {type(e).__name__}")
                st.write(f"Error message: {str(e)}")
                st.write(f"Text length: {len(text)} characters")
                st.write(f"Model loaded: {st.session_state.model_loaded}")
    else:
        st.warning("⚠️ Please enter some text to classify!")
        st.info("💡 You can use one of the sample articles above or paste your own news article.")

# Enhanced footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <h3>🚀 AI News Classifier Pro</h3>
    <p><strong>Built with:</strong> Streamlit • Transformers • PyTorch • Plotly</p>
    <p><strong>Models:</strong> BART-Large-MNLI • RoBERTa-Base-Sentiment</p>
    <p><strong>Features:</strong> Zero-shot Classification • Sentiment Analysis • Text Analytics • Export Data</p>
    <p><strong>Categories:</strong> World News 🌍 • Sports ⚽ • Business 📈 • Science & Technology 🔬</p>
    <br>
    <p style="font-size: 0.9em; color: #666;">
        Made with ❤️ for accurate news classification | 
        <a href="https://github.com/yourusername/news-article-classifier" target="_blank">View Source Code</a>
    </p>
</div>
""", unsafe_allow_html=True)

# Additional features section
with st.expander("ℹ️ About This Application"):
    st.markdown("""
    ### 🤖 How It Works
    
    This advanced AI news classifier uses state-of-the-art natural language processing models to:
    
    1. **Zero-Shot Classification**: Uses Facebook's BART-Large-MNLI model to classify articles without specific training
    2. **Sentiment Analysis**: Analyzes the emotional tone using RoBERTa-based sentiment models
    3. **Text Analytics**: Provides comprehensive statistics about readability, keywords, and structure
    4. **Real-time Processing**: Instant classification with progress tracking
    5. **History Tracking**: Saves all classifications with analytics and export capabilities
    
    ### 📊 Categories
    
    - **🌍 World News**: International affairs, politics, global events
    - **⚽ Sports**: Athletic competitions, games, sports personalities
    - **📈 Business**: Financial markets, corporate news, economic updates
    - **🔬 Science & Technology**: Scientific discoveries, tech innovations, research
    
    ### 🎯 Accuracy & Confidence
    
    - **High Confidence**: 70%+ - Very reliable classification
    - **Medium Confidence**: 50-70% - Good classification with some uncertainty
    - **Low Confidence**: <50% - May need manual review
    
    ### 🔧 Advanced Features
    
    - **Adjustable Confidence Threshold**: Set minimum confidence for classifications
    - **Sentiment Analysis**: Understand the emotional tone of articles
    - **Text Statistics**: Word count, readability, keyword extraction
    - **Export Functionality**: Download classification history as CSV
    - **Real-time Analytics**: Visual charts and metrics
    """)

# Performance metrics (if history exists)
if st.session_state.classification_history:
    with st.expander("📈 Performance Metrics"):
        df = pd.DataFrame(st.session_state.classification_history)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            avg_confidence = df['confidence'].mean()
            st.metric("Average Confidence", f"{avg_confidence:.1%}")
        
        with col2:
            avg_word_count = df['word_count'].mean()
            st.metric("Average Article Length", f"{avg_word_count:.0f} words")
        
        with col3:
            most_common_category = df['category'].mode().iloc[0] if not df.empty else "N/A"
            st.metric("Most Common Category", most_common_category)
        
        # Confidence distribution
        fig = px.histogram(df, x='confidence', nbins=20, title="Confidence Score Distribution")
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
