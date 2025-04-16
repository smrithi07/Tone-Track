import json
import threading
from kafka import KafkaConsumer
from pymongo import MongoClient

# Use your MongoDB Atlas connection string here.
atlas_uri = "mongodb+srv://smrithimln:mongosmr25@cluster0.4mspg.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
mongo_client = MongoClient(atlas_uri)
db = mongo_client["emotions"]

def consume_and_insert(topic, collection_name, bootstrap_servers='localhost:9092'):
    """
    Consume messages from the given Kafka topic and insert them into the corresponding MongoDB collection.
    """
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap_servers,
        auto_offset_reset='earliest',
        group_id=f'{topic}_group',
        value_deserializer=lambda m: json.loads(m.decode('utf-8'))
    )
    
    collection = db[collection_name]
    print(f"Started consumer for topic '{topic}'. Inserting into collection '{collection_name}'...")
    
    for message in consumer:
        try:
            document = message.value
            collection.insert_one(document)
            print(f"Inserted document into '{collection_name}' (review_id: {document.get('review_id', 'N/A')})")
        except Exception as e:
            print(f"Error inserting document from topic '{topic}': {e}")

# Define your topics and corresponding collection names
topics_collections = {
    "amazon_reviews_clean": "amazon_reviews",
    "imdb_reviews_clean": "imdb_reviews",
    "yelp_reviews_clean": "yelp_reviews"
}

threads = []
for topic, collection in topics_collections.items():
    thread = threading.Thread(target=consume_and_insert, args=(topic, collection))
    thread.start()
    threads.append(thread)

for thread in threads:
    thread.join()