import pymongo
import time
import random

# ---------------------- MongoDB Loader ---------------------- #
def load_data_from_mongodb(uri, db_name, collection_name):   
    """
    Connects to MongoDB and loads data from the specified collection.
    """
    client = pymongo.MongoClient(uri)
    db = client[db_name]
    collection = db[collection_name]
    data = list(collection.find())
    client.close()
    return data

# ---------------------- Simulated Modules ---------------------- #
def simulate_emotion_detection(text):
    emotions = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust']
    selected = random.sample(emotions, k=random.randint(1, 2))
    confidence = round(random.uniform(0.6, 0.95), 2)
    return [{"label": emo, "score": confidence} for emo in selected]

def simulate_topic_extraction(text):
    topics = ['product quality', 'customer service', 'pricing', 'delivery', 'usability']
    subtopics = ['packaging', 'speed', 'interface', 'support', 'value for money']
    return {
        "topics": random.sample(topics, k=1),
        "subtopics": random.sample(subtopics, k=2)
    }

def simulate_sentiment_score(text):
    return round(random.uniform(-1, 1), 2)

def simulate_adorescore(sentiment, num_emotions, topic_relevance=0.8):
    """
    Simulates Adorescore calculation.
    """
    intensity = abs(sentiment)
    adorescore = round((sentiment + intensity + topic_relevance + (0.1 * num_emotions)) / 3.1, 2)
    return adorescore

# ---------------------- Fine-tuning Simulation ---------------------- #
def simulate_fine_tuning_pipeline(data):
    print(f"Simulating fine-tuning with {len(data)} records...")
    time.sleep(1)
    
    for i, doc in enumerate(data[:5]):  # Simulate first 5 for demo
        text = doc.get("text", "No text found")
        print(f"\n--- Processing Document {i+1} ---")
        print("Text:", text)

        emotions = simulate_emotion_detection(text)
        topics = simulate_topic_extraction(text)
        sentiment = simulate_sentiment_score(text)
        adorescore = simulate_adorescore(sentiment, len(emotions))

        print("Emotions:", emotions)
        print("Topics:", topics)
        print("Sentiment Score:", sentiment)
        print("Adorescore:", adorescore)
    
    return "Simulated Fine-Tuned Model with App Pipeline Logic"

# ---------------------- Main ---------------------- #
if __name__ == "__main__":
    # Replace with your actual MongoDB Atlas URI
    mongo_uri = "mongodb+srv://smrithimln:mongosmr25@cluster0.4mspg.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    
    database_name = "customer_emotion_db"
    collection_name = "processed_reviews"
    
    print("Loading data from MongoDB Atlas...")
    review_data = load_data_from_mongodb(mongo_uri, database_name, collection_name)
    print(f"Loaded {len(review_data)} reviews from MongoDB.")
    
    simulated_model = simulate_fine_tuning_pipeline(review_data)
    print("\nFine-tuning simulation complete.")
    print("Simulated Model:", simulated_model)
