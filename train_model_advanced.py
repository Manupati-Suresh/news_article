#!/usr/bin/env python3
"""
Advanced News Article Classifier Training Script
================================================

This script provides enhanced training capabilities with:
- Better error handling and logging
- Model evaluation metrics
- Visualization of training progress
- Model comparison and selection
- Automated hyperparameter tuning options
"""

from datasets import load_dataset, Dataset
from transformers import (
    BertTokenizer, BertForSequenceClassification, 
    Trainer, TrainingArguments, EarlyStoppingCallback,
    AutoTokenizer, AutoModelForSequenceClassification
)
import torch
import os
import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
import argparse
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NewsClassifierTrainer:
    """Advanced trainer for news article classification"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Create output directories
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(config.logging_dir).mkdir(parents=True, exist_ok=True)
        Path("./plots").mkdir(parents=True, exist_ok=True)
        
        # Label mapping
        self.label_names = ["World", "Sports", "Business", "Sci/Tech"]
        self.id2label = {i: label for i, label in enumerate(self.label_names)}
        self.label2id = {label: i for i, label in enumerate(self.label_names)}
    
    def load_dataset(self):
        """Load and prepare the dataset"""
        try:
            logger.info("📥 Loading AG News dataset...")
            dataset = load_dataset("ag_news")
            logger.info("✅ Successfully loaded AG News dataset!")
            
            # Print dataset info
            logger.info(f"Train samples: {len(dataset['train'])}")
            logger.info(f"Test samples: {len(dataset['test'])}")
            
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to load ag_news: {e}")
            logger.info("🔄 Creating enhanced sample dataset...")
            
            return self._create_sample_dataset()
    
    def _create_sample_dataset(self):
        """Create a comprehensive sample dataset"""
        sample_data = {
            'Business': [
                "Apple Inc. reported record quarterly earnings today, beating analyst expectations with revenue of $95 billion.",
                "Tesla stock surged 15% after announcing plans to expand manufacturing in Europe and Asia markets.",
                "Federal Reserve raises interest rates by 0.25% to combat rising inflation concerns across the economy.",
                "Amazon acquires healthcare startup for $2.3 billion to expand into medical services sector.",
                "Oil prices climb to $85 per barrel amid supply chain disruptions and geopolitical tensions.",
                "Microsoft announces new AI partnership with OpenAI worth $10 billion over multiple years.",
                "Cryptocurrency market rebounds as Bitcoin surges past $45,000 following regulatory clarity.",
                "Major banks report strong Q3 earnings despite concerns about economic slowdown and recession fears."
            ],
            'Sports': [
                "Manchester United secured a thrilling 3-1 victory over Barcelona in Champions League semifinal match.",
                "Serena Williams announces retirement from professional tennis after illustrious 23-year career.",
                "Lakers trade Russell Westbrook to Miami Heat in blockbuster NBA deal worth $47 million.",
                "Olympic swimming records broken at World Championships in Budapest, Hungary this summer.",
                "Premier League season kicks off with Manchester City favored to win title for fifth consecutive year.",
                "Tennis star Novak Djokovic wins Wimbledon championship for the seventh time in his career.",
                "NFL draft sees quarterback prospects dominate first round selections across multiple teams.",
                "World Cup preparations intensify as national teams finalize rosters for upcoming tournament."
            ],
            'World': [
                "UN Security Council votes unanimously on new sanctions against North Korea over recent missile tests.",
                "Earthquake measuring 7.2 magnitude strikes Japan, tsunami warning issued for Pacific coast regions.",
                "European Union announces ambitious climate change policies to reduce carbon emissions by 55% by 2030.",
                "Presidential elections in Brazil see record voter turnout amid economic and environmental concerns.",
                "NATO allies pledge additional military aid to Ukraine following latest round of diplomatic talks.",
                "Climate summit in Dubai reaches historic agreement on fossil fuel transition timeline.",
                "International trade tensions escalate as new tariffs imposed on technology imports.",
                "Humanitarian crisis deepens in conflict zones as aid organizations struggle with funding shortfalls."
            ],
            'Sci/Tech': [
                "NASA's James Webb telescope discovers potentially habitable exoplanet 100 light-years from Earth.",
                "Scientists develop breakthrough gene therapy treatment for rare genetic disorders affecting children worldwide.",
                "Google announces major advancement in quantum computing with 1000-qubit processor breakthrough achievement.",
                "Climate researchers warn Arctic ice melting 30% faster than predicted by current climate models.",
                "SpaceX successfully launches Mars mission with advanced rover equipped with AI technology systems.",
                "Artificial intelligence breakthrough enables real-time language translation with 99% accuracy rates.",
                "Medical researchers develop new cancer treatment using CRISPR gene editing technology.",
                "Renewable energy milestone reached as solar power costs drop below fossil fuel alternatives."
            ]
        }
        
        # Create balanced dataset
        texts, labels = [], []
        label_map = {'World': 0, 'Sports': 1, 'Business': 2, 'Sci/Tech': 3}
        
        for category, articles in sample_data.items():
            for article in articles:
                texts.append(article)
                labels.append(label_map[category])
        
        # Shuffle data
        combined = list(zip(texts, labels))
        np.random.shuffle(combined)
        texts, labels = zip(*combined)
        
        # Create train/test split (80/20)
        train_size = int(0.8 * len(texts))
        
        train_data = Dataset.from_dict({
            'text': texts[:train_size],
            'label': labels[:train_size]
        })
        test_data = Dataset.from_dict({
            'text': texts[train_size:],
            'label': labels[train_size:]
        })
        
        dataset = {'train': train_data, 'test': test_data}
        logger.info("✅ Created enhanced sample dataset!")
        return dataset
    
    def prepare_data(self, dataset):
        """Prepare and tokenize the dataset"""
        # Use subset if specified
        if self.config.train_samples and len(dataset['train']) > self.config.train_samples:
            train_data = dataset['train'].select(range(self.config.train_samples))
            logger.info(f"📊 Using subset of {self.config.train_samples} training samples")
        else:
            train_data = dataset['train']
            logger.info(f"📊 Using full training dataset: {len(train_data)} samples")

        if self.config.test_samples and len(dataset['test']) > self.config.test_samples:
            test_data = dataset['test'].select(range(self.config.test_samples))
            logger.info(f"📊 Using subset of {self.config.test_samples} test samples")
        else:
            test_data = dataset['test']
            logger.info(f"📊 Using full test dataset: {len(test_data)} samples")
        
        # Load tokenizer
        logger.info(f"🔤 Loading tokenizer: {self.config.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        # Tokenization function
        def tokenize_function(examples):
            return tokenizer(
                examples['text'], 
                padding='max_length', 
                truncation=True, 
                max_length=self.config.max_length
            )
        
        # Tokenize datasets
        logger.info("🔤 Tokenizing datasets...")
        tokenized_train = train_data.map(tokenize_function, batched=True)
        tokenized_test = test_data.map(tokenize_function, batched=True)
        
        # Set format for PyTorch
        tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        tokenized_test.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        
        return tokenized_train, tokenized_test, tokenizer
    
    def load_model(self):
        """Load the pre-trained model"""
        logger.info(f"🤖 Loading model: {self.config.model_name}")
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=self.config.num_labels,
            id2label=self.id2label,
            label2id=self.label2id
        )
        return model
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def train(self):
        """Main training function"""
        logger.info("🚀 Starting Advanced News Classifier Training")
        logger.info("=" * 60)
        
        # Load dataset
        dataset = self.load_dataset()
        
        # Prepare data
        train_dataset, test_dataset, tokenizer = self.prepare_data(dataset)
        
        # Load model
        model = self.load_model()
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.train_batch_size,
            per_device_eval_batch_size=self.config.eval_batch_size,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            logging_dir=self.config.logging_dir,
            logging_steps=self.config.logging_steps,
            evaluation_strategy="steps",
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            report_to=None,  # Disable wandb/tensorboard
            learning_rate=self.config.learning_rate,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train the model
        logger.info("🏋️ Starting training...")
        train_result = trainer.train()
        
        # Save the model and tokenizer
        logger.info("💾 Saving model and tokenizer...")
        trainer.save_model()
        tokenizer.save_pretrained(self.config.output_dir)
        
        # Evaluate the model
        logger.info("📊 Evaluating model...")
        eval_result = trainer.evaluate()
        
        # Generate predictions for detailed analysis
        predictions = trainer.predict(test_dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
        y_true = predictions.label_ids
        
        # Generate detailed report
        self.generate_report(y_true, y_pred, eval_result, train_result)
        
        logger.info("✅ Training completed successfully!")
        return trainer, eval_result
    
    def generate_report(self, y_true, y_pred, eval_result, train_result):
        """Generate comprehensive training report"""
        logger.info("📋 Generating training report...")
        
        # Classification report
        report = classification_report(y_true, y_pred, target_names=self.label_names, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create visualizations
        self.plot_confusion_matrix(cm)
        self.plot_metrics(report)
        
        # Save detailed report
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'model_name': self.config.model_name,
            'training_config': vars(self.config),
            'evaluation_metrics': eval_result,
            'training_metrics': train_result.metrics,
            'classification_report': report,
            'confusion_matrix': cm.tolist()
        }
        
        with open(f"{self.config.output_dir}/training_report.json", 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        # Print summary
        logger.info("📈 Training Summary:")
        logger.info(f"  Accuracy: {eval_result['eval_accuracy']:.4f}")
        logger.info(f"  F1 Score: {eval_result['eval_f1']:.4f}")
        logger.info(f"  Precision: {eval_result['eval_precision']:.4f}")
        logger.info(f"  Recall: {eval_result['eval_recall']:.4f}")
    
    def plot_confusion_matrix(self, cm):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.label_names, yticklabels=self.label_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('./plots/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("📊 Confusion matrix saved to ./plots/confusion_matrix.png")
    
    def plot_metrics(self, report):
        """Plot performance metrics"""
        # Extract metrics for each class
        classes = self.label_names
        precision = [report[cls]['precision'] for cls in classes]
        recall = [report[cls]['recall'] for cls in classes]
        f1_score = [report[cls]['f1-score'] for cls in classes]
        
        # Create bar plot
        x = np.arange(len(classes))
        width = 0.25
        
        plt.figure(figsize=(12, 6))
        plt.bar(x - width, precision, width, label='Precision', alpha=0.8)
        plt.bar(x, recall, width, label='Recall', alpha=0.8)
        plt.bar(x + width, f1_score, width, label='F1-Score', alpha=0.8)
        
        plt.xlabel('Categories')
        plt.ylabel('Score')
        plt.title('Performance Metrics by Category')
        plt.xticks(x, classes)
        plt.legend()
        plt.ylim(0, 1)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('./plots/metrics_by_category.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("📊 Metrics plot saved to ./plots/metrics_by_category.png")


class Config:
    """Training configuration"""
    def __init__(self):
        self.model_name = "bert-base-uncased"
        self.num_labels = 4
        self.max_length = 128
        self.train_batch_size = 16
        self.eval_batch_size = 32
        self.learning_rate = 2e-5
        self.num_epochs = 3
        self.warmup_steps = 500
        self.weight_decay = 0.01
        self.output_dir = "./results"
        self.logging_dir = "./logs"
        self.save_steps = 500
        self.eval_steps = 500
        self.logging_steps = 100
        
        # Dataset configuration
        self.use_full_dataset = False
        self.train_samples = 5000 if not self.use_full_dataset else None
        self.test_samples = 1000 if not self.use_full_dataset else None


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train News Article Classifier')
    parser.add_argument('--full-dataset', action='store_true', 
                       help='Use full dataset instead of subset')
    parser.add_argument('--epochs', type=int, default=3, 
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, 
                       help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=2e-5, 
                       help='Learning rate')
    
    args = parser.parse_args()
    
    # Create configuration
    config = Config()
    config.use_full_dataset = args.full_dataset
    config.num_epochs = args.epochs
    config.train_batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    
    if config.use_full_dataset:
        config.train_samples = None
        config.test_samples = None
    
    # Initialize trainer
    trainer = NewsClassifierTrainer(config)
    
    # Start training
    model, results = trainer.train()
    
    print("\n🎉 Training completed successfully!")
    print(f"📊 Final Results:")
    print(f"   Accuracy: {results['eval_accuracy']:.4f}")
    print(f"   F1 Score: {results['eval_f1']:.4f}")
    print(f"   Model saved to: {config.output_dir}")


if __name__ == "__main__":
    main()