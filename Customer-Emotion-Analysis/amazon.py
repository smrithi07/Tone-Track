import sqlite3
import json
from kafka import KafkaProducer

def ingest_table_data(db_path, table_name, kafka_topic, kafka_servers='localhost:9092'):
    """
    Ingests data from a specified SQLite table into a Kafka topic.
    
    Parameters:
        db_path (str): Path to the SQLite database.
        table_name (str): Name of the table to ingest data from.
        kafka_topic (str): Kafka topic to send data.
        kafka_servers (str or list): Kafka bootstrap servers. Default is 'localhost:9092'.
    """
    # Set up the Kafka Producer with JSON serializer
    producer = KafkaProducer(
        bootstrap_servers=kafka_servers,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Fetch all records from the table
    cursor.execute(f"SELECT * FROM {table_name}")
    rows = cursor.fetchall()
    
    # Retrieve column names for creating dictionary records
    columns = [desc[0] for desc in cursor.description]
    
    # Send each record to the Kafka topic
    for row in rows:
        record = dict(zip(columns, row))
        producer.send(kafka_topic, record)
    
    # Flush the producer to ensure all messages are sent
    producer.flush()
    
    # Close the database connection and Kafka producer
    conn.close()
    producer.close()
    
    print(f"Success: All data from table '{table_name}' have been ingested into Kafka topic '{kafka_topic}'.")

# Example usage:
# Assuming your SQLite database is at 'your_database.sqlite', you can ingest the 'amazon_table'
ingest_table_data("yelp_data.db", "amazon_reviews", "amazon_reviews")
