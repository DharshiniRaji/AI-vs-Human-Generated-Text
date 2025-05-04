# 🤖 AI vs Human Text Classification

This project focuses on building a machine learning model to distinguish between **AI-generated** and **human-written** text using **BERT** and **PyTorch**. The application includes a Streamlit interface for real-time text classification.

---

## 📁 Dataset
- Source: `/kaggle/input/ai-vs-human-text/AI_Human.csv`
- Contains labeled text:
  - `generated = 0` → Human-written
  - `generated = 1` → AI-generated

To balance the dataset:
- 50,000 samples are taken from each class.

---

## 🧪 Key Features

### ✅ Preprocessing
- Cleans and balances the dataset.
- Uses `BertTokenizer` (`bert-base-uncased`) for tokenization.
- Analyzes token length distribution for optimal input sizing.

### 📊 Visualization
- Histogram showing distribution of token lengths in the dataset.

### 🧠 Model Architecture
- LSTM-based classifier with embedding layer
- Dropout regularization (p=0.5)
- Sigmoid activation for binary classification
- Trained on BERT tokenized inputs

### 🖥️ Web Interface
- Interactive Streamlit application
- Real-time text classification
- Confidence score display

---

## 🛠 Libraries Used
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `torch`, `torch.nn`, `transformers`
- `sklearn` for evaluation metrics
- `nltk` and `string` for text analysis
- `tqdm` for progress tracking
- `streamlit` for the web interface

---

## 🚀 Getting Started

1. Clone the repository:
```bash
git clone https://github.com/DharshiniRaji/AI-vs-Human-Generated-Text.git
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit application:
```bash
streamlit run app.py
```

4. Enter text in the provided text area and click "Predict" to classify it as AI-generated or human-written.

---

## 📌 Notes
- The model (`ai_human_classifier.pth`) and tokenizer are loaded locally.
- Maximum sequence length is set to 128 tokens.
- The application runs on CPU if CUDA is not available.

---

## 📈 Future Improvements
- Implement more advanced transformer models
- Add explainability features to highlight influential text segments
- Improve model accuracy with larger datasets
- Support for multiple languages
- Batch processing capability for large text corpora

---

## ✍️ Author
This project was built as an educational tool to explore how well machine learning can detect AI-generated text from human-written content.