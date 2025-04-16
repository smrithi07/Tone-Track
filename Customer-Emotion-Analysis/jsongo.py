import json
from kafka import KafkaProducer

# Configure the Kafka Producer with a JSON serializer
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],  # Adjust if your broker is on a different host/port
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Path to your GoEmotions JSON file
json_file_path = 'goemotions_cleaned.json'

# Load the GoEmotions data from the JSON file
with open(json_file_path, 'r') as f:
    goemotion_data = json.load(f)

# Ingest each record into the Kafka topic "GoEmotion"
for record in goemotion_data:
    producer.send('GoEmotion', record)

# Ensure all messages are sent
producer.flush()
producer.close()

print("Success: GoEmotions data has been ingested into the Kafka topic 'GoEmotion'.")