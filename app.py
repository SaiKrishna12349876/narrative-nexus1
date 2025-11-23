import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from textblob import TextBlob
from transformers import pipeline
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from wordcloud import WordCloud

# --- Page Config ---
st.set_page_config(page_title="NarrativeNexus Analyzer", layout="wide")

# --- Cache NLTK Downloads (Run once) ---
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('wordnet')

download_nltk_data()

# --- Cache the Summarization Model (Heavy resource) ---
@st.cache_resource
def load_summarizer():
    # Using a smaller model for faster cloud performance
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# --- Helper Functions (From your Notebook) ---

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    return " ".join(lemmatized_tokens)

def get_topics_lda(text, n_topics=5, n_words=7):
    documents = [text]
    vectorizer = TfidfVectorizer(max_df=1.0, min_df=1, stop_words='english')
    tfidf = vectorizer.fit_transform(documents)

    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(tfidf)
    feature_names = vectorizer.get_feature_names_out()

    topics_data = []
    for topic_idx, topic_words in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic_words.argsort()[:-n_words - 1:-1]]
        topics_data.append(f"**Topic {topic_idx + 1}:** {', '.join(top_words)}")
    
    return topics_data

def get_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.1: sentiment = 'Positive 😊'
    elif polarity < -0.1: sentiment = 'Negative 😠'
    else: sentiment = 'Neutral 😐'
    return polarity, sentiment

# --- Main App Layout ---

st.title("📄 NarrativeNexus: Text Analysis Pipeline")
st.markdown("Upload your text files to perform Preprocessing, LDA Topic Modeling, Sentiment Analysis, and Summarization.")

# 1. File Upload
uploaded_files = st.file_uploader("Upload .txt files", type="txt", accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.divider()
        st.subheader(f"📂 Analyzing: {uploaded_file.name}")
        
        # Read file
        stringio = uploaded_file.getvalue().decode("utf-8")
        raw_text = stringio
        
        # Display Raw Text (Expandable)
        with st.expander("View Raw Text"):
            st.write(raw_text)

        # Create Tabs for the features
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🧹 Preprocessing", 
            "🔍 Topic Modeling (LDA)", 
            "📊 Sentiment Analysis", 
            "📝 Summarization",
            "🖼️ Visualizations"
        ])

        # --- Tab 1: Preprocessing ---
        with tab1:
            st.markdown("### Data Preprocessing")
            processed_text = preprocess_text(raw_text)
            st.success("Text cleaned, tokenized, and lemmatized.")
            st.text_area("Processed Text Preview", processed_text, height=150)

        # --- Tab 2: Topic Modeling ---
        with tab2:
            st.markdown("### LDA Topic Modeling")
            if len(processed_text.split()) > 10:
                topics = get_topics_lda(processed_text)
                for t in topics:
                    st.markdown(t)
            else:
                st.warning("Text is too short for Topic Modeling.")

        # --- Tab 3: Sentiment Analysis ---
        with tab3:
            st.markdown("### Sentiment Analysis")
            polarity, sentiment_label = get_sentiment(raw_text)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Sentiment Label", sentiment_label)
            with col2:
                st.metric("Polarity Score", f"{polarity:.2f}")

            # Sentiment Gauge Chart
            fig = go.Figure(go.Bar(
                x=[polarity], 
                y=['Sentiment'], 
                orientation='h', 
                marker=dict(color='green' if polarity > 0 else 'red')
            ))
            fig.update_layout(xaxis=dict(range=[-1, 1]), title="Polarity Scale (-1 to +1)")
            st.plotly_chart(fig, use_container_width=True)

        # --- Tab 4: Summarization ---
        with tab4:
            st.markdown("### Abstractive Summarization")
            with st.spinner("Generating summary... (this may take a moment)"):
                try:
                    summarizer = load_summarizer()
                    # Truncate to 1024 chars for speed/limit safety
                    summary_result = summarizer(raw_text[:1024], max_length=150, min_length=30, do_sample=False)
                    st.info(summary_result[0]['summary_text'])
                except Exception as e:
                    st.error(f"Error generating summary: {e}")

        # --- Tab 5: Visualizations ---
        with tab5:
            st.markdown("### Word Cloud")
            try:
                wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(processed_text)
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig)
            except ValueError:
                st.warning("Not enough text to generate Word Cloud.")
else:
    st.info("Please upload a .txt file to begin.")
