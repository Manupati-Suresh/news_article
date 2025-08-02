# 📰 News Article Classifier

A BERT-based news article classifier that categorizes articles into four categories: **World**, **Sports**, **Business**, and **Science & Technology**.

## 🚀 Features

- **BERT-based Classification**: Uses pre-trained BERT model fine-tuned on AG News dataset
- **Interactive Web Interface**: Streamlit app with sample articles for quick testing
- **Real-time Predictions**: Get instant classification results with confidence scores
- **Easy Setup**: Simple installation and usage

## 📋 Categories

- **World** 🌍: International news, politics, global events
- **Sports** ⚽: Sports news, games, athletes, competitions  
- **Business** 📈: Financial news, company updates, market trends
- **Sci/Tech** 🔬: Technology, science discoveries, innovations

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/news-article-classifier.git
   cd news-article-classifier
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Hugging Face authentication**:
   ```bash
   huggingface-cli login
   ```
   Enter your Hugging Face token when prompted.

## 🎯 Usage

### Training the Model

```bash
python train_model.py
```

This will:
- Download the AG News dataset
- Fine-tune BERT on news classification
- Save the trained model to `./results/`

### Running the Web Interface

```bash
streamlit run streamlit_app.py
```

The app will open in your browser with:
- Text input for custom articles
- Quick test buttons for each category
- Real-time classification results
- Confidence scores for all categories

## 📊 Model Performance

The model is trained on the AG News dataset with:
- **Training samples**: 120,000 articles
- **Test samples**: 7,600 articles
- **Architecture**: BERT-base-uncased with classification head
- **Categories**: 4 (World, Sports, Business, Sci/Tech)

## 🔧 Configuration

You can modify training parameters in `train_model.py`:
- `num_train_epochs`: Number of training epochs
- `per_device_train_batch_size`: Batch size for training
- `max_steps`: Maximum training steps
- Dataset size for faster experimentation

## 📁 Project Structure

```
news-article-classifier/
├── train_model.py          # Model training script
├── streamlit_app.py        # Web interface
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
├── .gitignore            # Git ignore rules
└── results/              # Trained model outputs (created after training)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co/) for the transformers library and AG News dataset
- [Streamlit](https://streamlit.io/) for the web interface framework
- [PyTorch](https://pytorch.org/) for the deep learning framework