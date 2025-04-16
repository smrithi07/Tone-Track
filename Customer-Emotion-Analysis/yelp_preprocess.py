import json
import re
from kafka import KafkaConsumer, KafkaProducer
import spacy

# Load spaCy model (with some components disabled for performance)
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "textcat"])

def clean_text(text):
    """
    Clean text using regex and convert to lowercase.
    """
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-z\s]', '', text)        # Remove non-letters
    text = " ".join(text.split())               # Remove extra spaces
    return text

def process_topic(raw_topic, processed_topic, batch_size=100, bootstrap_servers='localhost:9092'):
    consumer = KafkaConsumer(
        raw_topic,
        bootstrap_servers=bootstrap_servers,
        auto_offset_reset='earliest',
        group_id='preprocessing_group',
        value_deserializer=lambda m: json.loads(m.decode('utf-8'))
    )

    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

    batch_messages = []
    print(f"🔄 Listening to '{raw_topic}' and writing processed messages to '{processed_topic}'...")

    try:
        for message in consumer:
            batch_messages.append(message.value)

            if len(batch_messages) >= batch_size:
                texts = [clean_text(msg.get("text", "")) for msg in batch_messages]
                processed_texts = [
                    " ".join([token.lemma_ for token in doc if token.is_alpha and not token.is_stop])
                    for doc in nlp.pipe(texts, batch_size=batch_size)
                ]

                # Create enriched messages
                for idx, original in enumerate(batch_messages):
                    enriched = original.copy()
                    enriched["processed_text"] = processed_texts[idx]
                    producer.send(processed_topic, enriched)

                producer.flush()
                print(f" Processed and forwarded {len(batch_messages)} messages.")
                batch_messages = []

    except KeyboardInterrupt:
        print("❗ Interrupted by user.")

    finally:
        # Handle remaining messages
        if batch_messages:
            texts = [clean_text(msg.get("text", "")) for msg in batch_messages]
            processed_texts = [
                " ".join([token.lemma_ for token in doc if token.is_alpha and not token.is_stop])
                for doc in nlp.pipe(texts, batch_size=len(batch_messages))
            ]
            for idx, original in enumerate(batch_messages):
                enriched = original.copy()
                enriched["processed_text"] = processed_texts[idx]
                producer.send(processed_topic, enriched)

            producer.flush()
            print(f" Final batch of {len(batch_messages)} messages processed.")

        consumer.close()
        producer.close()
        print(" Kafka connections closed.")

# Run the processor
if __name__ == "__main__":
    process_topic(
        raw_topic="yelp_reviews",             # Your input Kafka topic
        processed_topic="yelp_reviews_clean", # Output Kafka topic
        batch_size=100
    )
