import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import cohere
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# -------------------- Setup: Load NLTK Resources & Initialize API -------------------- #
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

co = cohere.Client("3OX8jQ6maGWiw4KCbiQn95SS0t8aaDrhzwy318Xk")

# -------------------- Load Saved Fine-Tuned Model & Tokenizer -------------------- #
model_path = "Tone-Track\Customer-Emotion-Analysis\finetuned"  
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
print("Loaded fine-tuned model and tokenizer.")

# -------------------- Helper Functions -------------------- #
def preprocess_text(text):
    """
    Basic text preprocessing: lowercases, removes URLs, non-alphabetic characters,
    tokenizes, removes stopwords, and then reconstructs the text.
    """
    if not text:
        return ""
    
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

def classify_emotions(text):
    """
    Classifies emotions using the fine-tuned model.
    Assumes the model outputs logits for emotion labels:
    ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Apply softmax to convert logits to probabilities
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
    emotion_labels = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
    return {label: round(score.item(), 2) for label, score in zip(emotion_labels, probs)}

def extract_main_topics(text):
    """
    Uses Cohere's API to extract exactly 3 key topics from the review.
    Topics are returned as a list.
    """
    response = co.generate(
        model="command",
        prompt=f"""Analyze the following customer review and extract exactly 3 key topics.
- Topics should summarize the core themes of the review.
- Output only the topics as a comma-separated list.
Review: "{text}"
Main Topics:""",
        max_tokens=15,
        temperature=0.2,
        stop_sequences=["\n"]
    )
    try:
        topics = response.generations[0].text.strip().split(",")
        return [topic.strip() for topic in topics if topic.strip()]
    except Exception:
        return ["Error extracting topics"]

def analyze_sentiment(text):
    """
    Uses NLTK's VADER to analyze sentiment.
    Returns a compound sentiment score as an integer percentage.
    """
    sentiment = sia.polarity_scores(text)
    return int(sentiment["compound"] * 100)

def compute_adorescore(text, topics):
    """
    Computes an adorescore as a simple function of the overall sentiment
    and the number of extracted topics.
    (This formula can be adjusted based on your methodology.)
    """
    sentiment_score = analyze_sentiment(text)
    # For demonstration, let's say each topic adds 5 points to the score.
    topic_bonus = len(topics) * 5
    adorescore = sentiment_score + topic_bonus
    return adorescore

# -------------------- Process a Sample Review -------------------- #
sample_review = ("The food was incredibly bland, and the service was disappointing. "
                 "I left the restaurant feeling let down and unimpressed.")
print("\nOriginal Review:")
print(sample_review)

# Preprocess the review text
processed_review = preprocess_text(sample_review)
print("\nProcessed Review:")
print(processed_review)

# Classify emotions using the fine-tuned model
emotions = classify_emotions(processed_review)
print("\nDetected Emotions:")
print(emotions)

# Extract main topics using Cohere API
topics = extract_main_topics(sample_review)
print("\nExtracted Topics:")
print(topics)

# Compute the Adorescore based on the review and topics
adorescore = compute_adorescore(sample_review, topics)
print("\nAdorescore:")
print(adorescore)
