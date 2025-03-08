import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import cohere
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
import re
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# 🔹 Initialize Cohere API & Sentiment Analyzer
co = cohere.Client("3OX8jQ6maGWiw4KCbiQn95SS0t8aaDrhzwy318Xk")
sia = SentimentIntensityAnalyzer()
lemmatizer = nltk.WordNetLemmatizer()
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# 🔹 Load Emotion & Sarcasm Detection Models
emotion_model_name = "j-hartmann/emotion-english-distilroberta-base"
emotion_tokenizer = AutoTokenizer.from_pretrained(emotion_model_name)
emotion_model = AutoModelForSequenceClassification.from_pretrained(emotion_model_name)

# sarcasm_model_name = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
# sarcasm_tokenizer = AutoTokenizer.from_pretrained(sarcasm_model_name)
# sarcasm_model = AutoModelForSequenceClassification.from_pretrained(sarcasm_model_name)

# 🔹 Custom Styling
st.set_page_config(page_title="Review Analysis", page_icon="📊", layout="wide")

# 🔹 Header Section
st.markdown(
    """
    <style>
        body {background-color: #0f172a; color: #ffffff; font-family: 'Poppins', sans-serif;}
        .title {text-align: center; font-size: 40px; font-weight: bold; margin-bottom: 20px;}
        .subtitle {text-align: center; font-size: 18px; color: #A3A3A3; margin-bottom: 40px;}
        .stButton>button {background-color: #2563EB; color: white; font-size: 16px; border-radius: 10px; padding: 10px 20px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<h1 class="title">Review Analysis Dashboard</h1>', unsafe_allow_html=True)
st.markdown(
    '<h3 class="subtitle">Analyze emotions and topics in customer feedback </h3>',
    unsafe_allow_html=True,
)

# 🔹 Input Section
st.markdown("###  Enter a Review Below:")
input_text = st.text_area("", height=150, placeholder="Type your review here...")

# 🔹 Button with Loading Animation
analyze_btn = st.button("Analyze Review", help="Click to analyze the review")

# Text Preprocessing function
def preprocess_text(text):
    if not text:
        return ""

    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    tokens = [word for word in tokens if len(word) > 2]
    return " ".join(tokens)



def classify_emotions(text):
    inputs = emotion_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = emotion_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
    emotion_labels = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
    return {emotion: round(score.item(), 2) for emotion, score in zip(emotion_labels, probs)}


# 🔹 Radar Chart for Emotions
def plot_emotion_radar(emotions):
    labels = list(emotions.keys())
    values = list(emotions.values())
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(r=values + [values[0]], theta=labels + [labels[0]], fill="toself", name="Emotions")
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=False, title="Emotion Radar Chart"
    )
    return fig

# Functions for Topic Finder
def clean_topics(topics):
    return [topic.strip().rstrip(".") for topic in topics if len(topic) > 1]

def extract_main_topics(text):
    response = co.generate(
        model="command",
        prompt=f"""Analyze the following customer review and extract **exactly 3** key topics.
        - Topics should summarize the **core themes** of the review.
        - Ensure they are **concise and relevant** (e.g., 'Service' instead of 'Service quality').
        - Avoid generic words like 'good' or 'bad.'
        - Output only the topics as a **comma-separated list**.

        Review: "{text}"

        Main Topics:
        """,
        max_tokens=15,
        temperature=0.2,
        stop_sequences=["\n"]
    )
    try:
        topics = response.generations[0].text.strip().split(",")
        return clean_topics(topics)
    except Exception:
        return ["Error extracting topics"]

def extract_subtopics(text, topic):
    response = co.generate(
        model="command",
        prompt=f"""
        For the topic '{topic}', extract **1 relevant subtopic** based on the review.
        - Each subtopic should be **one word or a very short phrase** (e.g., 'Speed' instead of 'Fast Delivery').
        - Avoid generic words like 'good' or 'bad.'
        - Output only the subtopics as a **comma-separated list**.

        Review: "{text}"

        Subtopics:
        """,
        max_tokens=10,
        temperature=0.2,
        stop_sequences=["\n"]
    )
    try:
        subtopics = response.generations[0].text.strip().split(",")
        return clean_topics(subtopics[:3])  # Max 3 subtopics
    except Exception:
        return ["Error extracting subtopics"]

def process_review(text):
    main_topics = extract_main_topics(text)
    subtopics = {topic: extract_subtopics(text, topic) for topic in main_topics}

    return {
        "review": text,
        "topics": {
            "main": main_topics,
            "subtopics": subtopics
        }
    }

# Functions for Adorescore
def analyze_sentiment(text):
    sentiment = sia.polarity_scores(text)
    return int(sentiment["compound"] * 100)

def clean_topicsm(topics_with_scores):
    cleaned_topics = {}
    
    for item in topics_with_scores:
        match = re.match(r"(.*)\((\d+)%\)", item.strip())  # Extract topic & score
        if match:
            topic, score = match.groups()
            cleaned_topics[topic.strip()] = int(score)
    
    return cleaned_topics

def extract_main_topicsm(text):
    response = co.generate(
        model="command",
        prompt=f"""
        Analyze the following customer review and extract **exactly 3** key topics along with relevance scores.
        - Topics should summarize the **core themes** of the review.
        - Assign a relevance percentage (0-100%) to each topic, based on its importance in the review.
        - Format the response as: Topic1 (X%), Topic2 (Y%), Topic3 (Z%)

        Review: "{text}"

        Main Topics:
        """,
        max_tokens=50
    )
    
    try:
        topics = response.generations[0].text.strip().split(",")  # Get raw topics
        return clean_topicsm(topics)  # Extract and clean topics with scores
    except Exception:
        return {"Error extracting topics": 100}  # Default if extraction fails

def compute_adorescore(text, topics):
    overall_sentiment = analyze_sentiment(text)
    topic_breakdown = {}
    weighted_sum = 0
    total_weight = sum(topics.values())  # Sum of topic relevance scores

    for topic, weight in topics.items():
        topic_sentiment = analyze_sentiment(topic)  # Get sentiment of topic itself
        weighted_score = (topic_sentiment + overall_sentiment) / 2  # Average topic + review sentiment
        weighted_score *= (weight / 100)  # Adjust by topic relevance

        topic_breakdown[topic] = round(weighted_score, 2)
        weighted_sum += weighted_score
    adorescore = round(weighted_sum / total_weight * 100) if total_weight > 0 else 0

    return {
        "adorescore": {
            "overall": adorescore,
            "breakdown": topic_breakdown
        }
    }


import plotly.express as px
import pandas as pd

def plot_emotion_radar_chart(topic_emotion_distributions):
    emotions = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
    data = []

    for topic, emotions_data in topic_emotion_distributions.items():
        emotion_values = [
            emotions_data.get(emotion, {"intensity": 0})["intensity"] if isinstance(emotions_data.get(emotion), dict) 
            else emotions_data.get(emotion, 0)  # Handle float case
            for emotion in emotions
        ]
        data.append(dict(Topic=topic, **dict(zip(emotions, emotion_values))))

    df = pd.DataFrame(data)

    fig = px.line_polar(
        df.melt(id_vars=["Topic"], var_name="Emotion", value_name="Intensity"),
        r="Intensity",
        theta="Emotion",
        color="Topic",
        line_close=True,
        title="Emotional Distribution Across Topics"
    )
    return fig


def analyze_full_emotions_per_topic(text, topics):
    topic_emotions = {}
    for topic in topics["main"]:
        topic_text = topic + " " + " ".join(topics["subtopics"].get(topic, []))
        detected_emotions = classify_emotions(topic_text)
        topic_emotions[topic] = detected_emotions
    return topic_emotions

# 🔹 Run Analysis
if analyze_btn and input_text:
    with st.spinner("Analyzing..."):
        emotions = classify_emotions(input_text)
        extracted_topics = process_review(input_text)  
        main_topics_with_scores = extract_main_topicsm(input_text)  
        adorescore_data = compute_adorescore(input_text, main_topics_with_scores)

    # 🔹 Results Section
    st.markdown("---")
    
    st.markdown("---")
    st.markdown("###  **Emotion Analysis**")
    st.plotly_chart(plot_emotion_radar(emotions))
    st.markdown("###  **Top Emotions Detected:**")
    for emotion, intensity in sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]:
        st.write(f"**{emotion.capitalize()}:** {intensity * 100:.0f}%")

    # 🏷 **Topic Extraction**
    st.markdown("---")
    st.markdown("### 🏷 **Extracted Topics**")
    st.write("**Main Topics:**", extracted_topics["topics"]["main"])
    st.write("**Subtopics:**", extracted_topics["topics"]["subtopics"])

     #  ** Emotional Distributions per Topic**
    result1 = process_review(input_text) 
    topic_emotion_distributions = analyze_full_emotions_per_topic(input_text, result1["topics"])
    st.subheader("🔹 Emotional Distributions per Topic")
    fig2 = plot_emotion_radar_chart(topic_emotion_distributions)
    st.plotly_chart(fig2)

    #  **Adorescore Analysis**
    st.markdown("---")
    st.markdown("###  **Adorescore Analysis**")
    st.write(f"**Overall Adorescore:** {adorescore_data['adorescore']['overall']} / 100")

    # 🔹 Progress Bars for Topic Breakdown
    st.markdown("####  **Adorescore Breakdown by Topic**")
    for topic, score in adorescore_data["adorescore"]["breakdown"].items():
        st.write(f"**{topic}:** {score}%")
        st.progress(score / 100)

    # 🔹 Sidebar Summary
    with st.sidebar:
        st.markdown("##  **Quick Summary**")
        st.metric(label="Adorescore", value=f"{adorescore_data['adorescore']['overall']} / 100")
        st.metric(label="Top Emotion", value=max(emotions, key=emotions.get).capitalize())
        st.markdown("### 🏷 **Key Topics:**")
        for topic in extracted_topics["topics"]["main"]:
            st.write(f"✅ {topic}")

st.markdown("---")
st.markdown("💡 **Tip:** Enter different types of reviews to see how emotions are detected!")
