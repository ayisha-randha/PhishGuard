# PhishGuard - AI-Powered Phishing Detection System

PhishGuard is a comprehensive Flask web application that uses CatBoost machine learning models to detect phishing attempts in:
- URLs
- Email messages
- SMS messages
- QR codes

## Abstract

Phishing remains one of the most common and effective attack vectors in cybersecurity, and it no longer arrives through a single channel — attackers now spoof URLs, craft deceptive emails, send smishing texts, and even embed malicious links inside QR codes. Most detection tools address only one of these vectors in isolation, leaving users to juggle separate checkers or, more often, no checker at all. **PhishGuard** addresses this by unifying phishing detection across all four channels in a single Flask web application backed by two purpose-built **CatBoost gradient-boosting classifiers**. The first model evaluates URLs using 24 engineered features spanning lexical structure, character composition, domain characteristics, and entropy. The second model evaluates free text (covering both emails and SMS messages) using 26+ handcrafted linguistic/structural features combined with a 500-feature TF-IDF representation, allowing it to capture both surface-level red flags (urgency language, suspicious links, excessive capitalization) and deeper semantic patterns. QR codes are handled by decoding the embedded payload and routing any URL found inside through the same URL model. Every prediction returns a confidence score and risk level in real time, and the project ships with a full suite of training visualizations (confusion matrices, ROC curves, feature importance, and class distribution) so the models' behavior is auditable rather than opaque. The result is a single, practical reference implementation of end-to-end, multi-channel phishing detection — useful both as a deployable tool and as a teaching example of applied ML for security.

## Features

- **URL Phishing Detection**: Analyzes URLs using 24+ features to identify phishing attempts
- **Email Phishing Detection**: Examines email content using linguistic features and TF-IDF analysis
- **SMS Phishing Detection**: Detects smishing attempts in SMS messages
- **QR Code Scanning**: Decodes QR codes and checks if they contain phishing URLs
- **Real-time Analysis**: Instant predictions with confidence scores and risk levels
- **Visualization Graphs**: Comprehensive training graphs for model performance analysis

## Project Structure

This is the structure of the repository as packaged (some folders are created automatically at runtime and are not part of the repo itself):

```
PhishGuard-main/
├── models/                     # ✅ Pre-trained models (already included in this repo)
│   ├── url_phishing_model.cbm
│   ├── url_feature_extractor.pkl
│   ├── text_phishing_model.cbm
│   ├── text_feature_extractor.pkl
│   └── tfidf_vectorizer.pkl
├── graphs/                     # ✅ Training visualization graphs (already included)
│   ├── url_feature_importance.png / url_confusion_matrix.png / url_roc_curve.png / url_metrics.png / url_class_distribution.png
│   └── text_feature_importance.png / text_confusion_matrix.png / text_roc_curve.png / text_metrics.png / text_class_distribution.png
├── templates/                  # HTML templates
│   ├── base.html
│   ├── index.html
│   ├── url_checker.html
│   ├── email_checker.html
│   ├── sms_checker.html
│   ├── qr_checker.html
│   ├── graphs.html
│   ├── about.html
│   ├── 404.html
│   └── 500.html
├── train/                      # Model training scripts (only needed if you want to retrain)
│   ├── train_url_model.py
│   └── train_text_model.py
├── features.py                  # URLFeatureExtractor & TextFeatureExtractor classes
├── predictor.py                 # Unified PhishingPredictor (loads models, runs inference)
├── app.py                       # Flask application & routes
└── requirements.txt              # Python dependencies

# Not included in this repo — required only if you want to retrain the models:
# dataset/url_dataset.csv, dataset/email_messages_clean.csv
# Created automatically at runtime: uploads/ (temporary QR image storage)
```

> **Note:** Unlike what earlier versions of this README implied, the trained models (`.cbm`/`.pkl` files) and training graphs (`.png`) are **already included** in this repository — you do **not** need to run the training scripts before using the app. Training is only necessary if you want to retrain on your own/updated dataset. The training scripts also now live under `train/`, not the project root.

## Installation

### 1. Clone or Download the Project

```bash
cd PhishGuard-main
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv .venv
```

### 3. Activate Virtual Environment

**Windows:**
```bash
.venv\Scripts\activate
```

**Linux/Mac:**
```bash
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

## Running the App (using the included pre-trained models)

The `models/` folder already contains trained `.cbm`/`.pkl` files, so you can skip straight to [Running the Application](#running-the-application) below. The training section that follows is only needed if you want to retrain the models yourself (e.g. on an updated dataset).

## (Optional) Retraining the Models

If you want to retrain from scratch, you'll need your own copies of `dataset/url_dataset.csv` and `dataset/email_messages_clean.csv` (not included in this repo — see [Dataset Information](#dataset-information)), then run the scripts from the `train/` folder:

### Train URL Phishing Detection Model

```bash
python train/train_url_model.py
```

This will:
- Load and process the URL dataset (450K+ URLs)
- Extract 24+ features from each URL
- Train a CatBoost classifier
- Generate visualization graphs in `graphs/` folder:
  - `url_feature_importance.png`
  - `url_confusion_matrix.png`
  - `url_roc_curve.png`
  - `url_metrics.png`
  - `url_class_distribution.png`
- Save the trained model to `models/url_phishing_model.cbm`
- Save feature extractor to `models/url_feature_extractor.pkl`

**Expected Output:**
- Training time: 5-15 minutes (depending on your CPU)
- Model accuracy: ~95%+
- AUC-ROC score: ~0.98+

### Train Email/SMS Phishing Detection Model

```bash
python train/train_text_model.py
```

This will:
- Load and process the email dataset (170K+ messages)
- Sample 100,000 messages for training (configurable)
- Extract 26+ handcrafted features
- Create TF-IDF features (500 features)
- Train a CatBoost classifier
- Generate visualization graphs in `graphs/` folder:
  - `text_feature_importance.png`
  - `text_confusion_matrix.png`
  - `text_roc_curve.png`
  - `text_metrics.png`
  - `text_class_distribution.png`
- Save models to:
  - `models/text_phishing_model.cbm`
  - `models/text_feature_extractor.pkl`
  - `models/tfidf_vectorizer.pkl`

**Expected Output:**
- Training time: 10-20 minutes
- Model accuracy: ~95%+
- AUC-ROC score: ~0.97+

## Running the Application

The pre-trained models are already included, so you can start the app directly (no training required unless you've deliberately retrained the models above):

```bash
python app.py
```

The application will be available at: **http://127.0.0.1:5000**

## Usage

### 1. URL Checker
- Navigate to "URL Checker"
- Enter a URL (e.g., `https://example.com`)
- Click "Check URL"
- View results with confidence scores and risk levels

### 2. Email Checker
- Navigate to "Email Checker"
- Paste email content into the text area
- Click "Analyze Email"
- View phishing detection results

### 3. SMS Checker
- Navigate to "SMS Checker"
- Paste SMS message content
- Click "Analyze SMS"
- View results with phishing probability

### 4. QR Code Checker
- Navigate to "QR Code Checker"
- Upload a QR code image (PNG, JPG, etc.)
- Click "Scan QR Code"
- The system will decode the QR code and check if it contains a phishing URL

## Model Features

### URL Detection Features (24 features)
- URL length, domain length, path length
- Special character counts (dots, hyphens, slashes, etc.)
- Protocol security (HTTPS)
- Domain characteristics (subdomains, IP addresses)
- Suspicious keyword presence
- Character ratios (digits, letters)
- URL entropy
- Query parameters

### Email/SMS Detection Features (26+ handcrafted + 500 TF-IDF features)
- Text length and word count
- Special character patterns
- Capitalization analysis
- Suspicious keyword detection
- Urgency word indicators
- Link and email address presence
- Money-related symbols
- Grammatical features
- TF-IDF vectorization for semantic analysis

## API Endpoints

The application provides REST API endpoints:

- `POST /api/check-url` - Check URL for phishing
- `POST /api/check-email` - Check email for phishing
- `POST /api/check-sms` - Check SMS for phishing
- `POST /api/check-qr` - Check QR code for phishing

### Example API Usage

```python
import requests

# Check URL
response = requests.post('http://127.0.0.1:5000/api/check-url',
    json={'url': 'https://example.com'})
print(response.json())

# Check Email
response = requests.post('http://127.0.0.1:5000/api/check-email',
    json={'email_text': 'Your email content here...'})
print(response.json())
```

## Performance Metrics

The trained models generate comprehensive visualizations:

1. **Feature Importance**: Shows which features contribute most to predictions
2. **Confusion Matrix**: Displays true/false positives and negatives
3. **ROC Curve**: Illustrates model performance across different thresholds
4. **Performance Metrics**: Accuracy, Precision, Recall, F1-Score, AUC
5. **Class Distribution**: Shows balance between legitimate and phishing samples

All graphs are saved in the `graphs/` folder after training.

## Troubleshooting

### Models Not Found Error
This repo ships with pre-trained models, so this shouldn't normally happen. If you see "URL model not found" or "Text model not found":
1. Check that `models/` still contains all five files: `url_phishing_model.cbm`, `url_feature_extractor.pkl`, `text_phishing_model.cbm`, `text_feature_extractor.pkl`, `tfidf_vectorizer.pkl`
2. If any are missing (e.g. stripped out by Git LFS or `.gitignore`), retrain them using the scripts in `train/` (see [Optional: Retraining the Models](#optional-retraining-the-models))

### Memory Issues During Training
If you encounter memory errors:
1. Edit `train/train_text_model.py`
2. Reduce sample size: Change `n=100000` to a smaller number (e.g., `n=50000`)

### QR Code Detection Issues
Ensure you have the required dependencies:
```bash
pip install opencv-python pyzbar pillow
```

## Dataset Information

### URL Dataset
- **File**: `dataset/url_dataset.csv`
- **Rows**: ~450,000
- **Columns**: `url`, `type` (legitimate/phishing)

### Email/SMS Dataset
- **File**: `dataset/email_messages_clean.csv`
- **Rows**: ~172,000
- **Columns**: `Text`, `Type` (legitimate/phishing)

## Technology Stack

- **Backend**: Flask (Python web framework)
- **Machine Learning**: CatBoost (Gradient Boosting)
- **Feature Engineering**: scikit-learn, pandas, numpy
- **Text Processing**: TF-IDF Vectorization
- **QR Code**: OpenCV, pyzbar
- **Visualization**: matplotlib, seaborn

## Security Notes

- Uploaded QR code images are automatically deleted after processing
- No user data is stored or logged
- All analysis is performed locally on the server
- Results are provided in real-time

## Future Enhancements

- [ ] Add support for more file formats
- [ ] Implement batch processing
- [ ] Add API rate limiting
- [ ] Create user dashboard for history tracking
- [ ] Add multilingual support
- [ ] Implement real-time URL reputation checking

## License

This project is for educational and security research purposes.

## Support

For issues or questions, please ensure:
1. You've installed all dependencies from `requirements.txt`
2. The `models/` folder still contains all five pre-trained model/extractor files
3. If retraining, your dataset files are in the correct `dataset/` location

## Contributing

Contributions are welcome! Please ensure:
- Code follows PEP 8 style guidelines
- Models are tested before committing
- Documentation is updated for new features

---

**Built with CatBoost and Flask** | **Stay Safe from Phishing!** 🛡️
