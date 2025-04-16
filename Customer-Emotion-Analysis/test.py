from pymongo import MongoClient

atlas_uri = "mongodb+srv://smrithimln:mongosmr25@cluster0.4mspg.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
try:
    client = MongoClient(atlas_uri)
    # List databases to verify connection
    print(client.list_database_names())
    print("Connection successful!")
except Exception as e:
    print("Connection error:", e)