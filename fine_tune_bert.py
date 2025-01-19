from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import sqlite3
import pandas as pd

# Connect to the SQL database and fetch data
def fetch_data_from_sql():
    conn = sqlite3.connect('emotions.db')  # Adjust the path to match your database location
    query = "SELECT * FROM connections"  # Fetch all data from the 'connections' table
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# Fetch the data
df = fetch_data_from_sql()

# For demonstration, we'll label all entries as relevant (1). Adjust this based on your actual classification criteria
df['label'] = 1

# Split the data into train and test sets
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['text'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42
)

# Tokenize and encode the data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=64)

train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
test_dataset = Dataset.from_dict({"text": test_texts, "label": test_labels})

tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_test = test_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Load the BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./bert_results',
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=360,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=lambda p: {"accuracy": (p.predictions.argmax(-1) == p.label_ids).astype(np.float32).mean().item()},
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_bert")
tokenizer.save_pretrained("./fine_tuned_bert")

# Example of how to use the fine-tuned model for inference
def predict_relevance(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = logits.argmax(dim=1).item()
    return "Relevant" if predicted_class_id == 1 else "Not Relevant"

# Test the model with some sample texts from your dataset
sample_texts = df['text'].sample(5).tolist()  # Sample 5 random texts from your dataset
for text in sample_texts:
    print(f"Text: {text}")
    print(f"Predicted Relevance: {predict_relevance(text)}\n")