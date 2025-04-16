import pymongo
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset

# -------------------- MongoDB Data Loader -------------------- #
def load_data_from_mongodb(uri, db_name, collection_name):
  
    client = pymongo.MongoClient(uri)
    db = client[db_name]
    collection = db[collection_name]
    data = list(collection.find())
    client.close()
    return data

# -------------------- Dataset Preparation -------------------- #
def prepare_dataset(data):
  
    texts = []
    labels = []
    for doc in data:
        text = doc.get("text", "")
        # If your data includes labels, use them; otherwise, assign a default.
        label = doc.get("label", 0)
        texts.append(text)
        labels.append(label)
    return {"text": texts, "label": labels}

def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True)

# -------------------- Main Fine-Tuning and Saving Pipeline -------------------- #
if __name__ == "__main__":
    # MongoDB Atlas connection string (replace <db_password> with your actual password)
    mongo_uri = "mongodb+srv://sar:<db_password>@cluster0.gq4fcrj.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    database_name = "customer_emotion_db"
    collection_name = "processed_reviews"
    
    print("Loading data from MongoDB Atlas...")
    data = load_data_from_mongodb(mongo_uri, database_name, collection_name)
    print(f"Loaded {len(data)} records from MongoDB.")
    
    # Prepare the dataset (expects each record to have a 'text' field)
    dataset_dict = prepare_dataset(data)
    raw_dataset = Dataset.from_dict(dataset_dict)
    
    # Choose a pre-trained model (e.g., DistilRoBERTa for sequence classification)
    model_name = "distilroberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_dataset = raw_dataset.map(lambda examples: tokenize_function(examples, tokenizer), batched=True)
    
    # Set up training arguments (adjust parameters as needed)
    training_args = TrainingArguments(
        output_dir="./fine_tuned",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=2,
        logging_dir="./logs",
    )
    
    # Initialize the model (adjust num_labels according to your task)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # Data collator for padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Set up the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Fine-tune the model (this step might take a while depending on your data size and hardware)
    print("Starting fine-tuning process...")
    trainer.train()
    print("Fine-tuning complete.")
    
    # Save the fine-tuned model and tokenizer to disk
    model.save_pretrained("./fine_tuned")
    tokenizer.save_pretrained("./fine_tuned")
    print("Fine-tuned model and tokenizer saved to './fine_tuned'.")
