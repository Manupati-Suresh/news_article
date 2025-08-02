from datasets import load_dataset, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch

print("Using cached Hugging Face authentication...")

# Try to load ag_news dataset
try:
    print("Loading ag_news dataset...")
    dataset = load_dataset("ag_news")
    print("Successfully loaded ag_news dataset!")
except Exception as e:
    print(f"Failed to load ag_news: {e}")
    print("Creating a sample dataset for testing...")
    
    # Create a sample dataset with the same structure as ag_news
    sample_texts = [
        "Apple Inc. reported strong quarterly earnings today, beating analyst expectations.",
        "The new smartphone features advanced AI capabilities and improved battery life.",
        "Scientists discover new species of marine life in the Pacific Ocean.",
        "Local football team wins championship after thrilling overtime victory.",
        "Stock markets rally as inflation concerns ease across major economies.",
        "Breakthrough in renewable energy technology promises cheaper solar panels.",
        "Olympic athletes prepare for upcoming games with intensive training programs.",
        "New restaurant opens downtown featuring fusion cuisine and local ingredients."
    ]
    
    sample_labels = [2, 3, 3, 1, 2, 3, 1, 0]  # 0: World, 1: Sports, 2: Business, 3: Sci/Tech
    
    # Create train and test splits
    train_size = 6
    train_data = Dataset.from_dict({
        'text': sample_texts[:train_size],
        'label': sample_labels[:train_size]
    })
    test_data = Dataset.from_dict({
        'text': sample_texts[train_size:],
        'label': sample_labels[train_size:]
    })
    
    dataset = {'train': train_data, 'test': test_data}
    print("Created sample dataset for testing!")
# Use a smaller subset for faster training/testing
train_data = dataset['train'].select(range(1000))  # Use only 1000 samples
test_data = dataset['test'].select(range(200))      # Use only 200 samples

print(f"Train dataset size: {len(train_data)}")
print(f"Test dataset size: {len(test_data)}")

# Tokenize
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
def tokenize_function(example):
    return tokenizer(example['text'], padding='max_length', truncation=True, max_length=128)
tokenized_train = train_data.map(tokenize_function, batched=True)
tokenized_test = test_data.map(tokenize_function, batched=True)
tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "label"])
tokenized_test.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# Load model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4)

# Training args - optimized for faster training
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    per_device_train_batch_size=8,  # Smaller batch size for faster training
    per_device_eval_batch_size=32,
    num_train_epochs=1,  # Reduced epochs for quick testing
    weight_decay=0.01,
    logging_dir='./logs',
    max_steps=100,  # Limit training steps for quick testing
    save_steps=50,
    eval_steps=50,
    logging_steps=10,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
)

# Train
trainer.train()
