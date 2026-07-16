import pandas as pd
import numpy as np
import re
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os


class TextFeatureExtractor:
    """Extract features from text messages (Email/SMS) for phishing detection"""

    def __init__(self):
        self.suspicious_keywords = [
            'urgent', 'verify', 'account', 'suspended', 'click', 'confirm',
            'password', 'update', 'security', 'alert', 'winner', 'prize',
            'congratulations', 'free', 'claim', 'limited', 'expire', 'act now',
            'bank', 'credit card', 'social security', 'tax', 'refund', 'pay',
            'transfer', 'transaction', 'locked', 'unusual activity'
        ]

        self.urgency_words = [
            'urgent', 'immediate', 'now', 'asap', 'quickly', 'hurry',
            'expire', 'expires', 'limited time', 'act now', 'don\'t wait'
        ]

    def extract_features(self, text):
        """Extract comprehensive features from text"""
        features = {}

        if pd.isna(text) or text == '':
            text = ''

        text_lower = text.lower()

        # Basic text features
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        features['char_count'] = len(text)

        # Special characters
        features['num_exclamation'] = text.count('!')
        features['num_question'] = text.count('?')
        features['num_dollar'] = text.count('$')
        features['num_urls'] = len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text))
        features['num_emails'] = len(re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text))
        features['num_phone'] = len(re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text))

        # Capitalization
        features['num_capitals'] = sum(1 for c in text if c.isupper())
        features['capital_ratio'] = features['num_capitals'] / len(text) if len(text) > 0 else 0
        features['has_all_caps_words'] = 1 if any(word.isupper() and len(word) > 2 for word in text.split()) else 0

        # Suspicious keywords
        features['num_suspicious_keywords'] = sum(1 for word in self.suspicious_keywords if word in text_lower)
        features['num_urgency_words'] = sum(1 for word in self.urgency_words if word in text_lower)

        # Money related
        features['has_money_symbol'] = 1 if '$' in text or '£' in text or '€' in text else 0
        features['num_digits'] = sum(1 for c in text if c.isdigit())
        features['digit_ratio'] = features['num_digits'] / len(text) if len(text) > 0 else 0

        # Link patterns
        features['has_link'] = 1 if 'http' in text_lower or 'www.' in text_lower else 0
        features['has_ip_address'] = 1 if re.search(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', text) else 0

        # Grammatical features
        features['num_spaces'] = text.count(' ')
        features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0

        # Suspicious patterns
        features['has_click_here'] = 1 if 'click here' in text_lower or 'click now' in text_lower else 0
        features['has_verify'] = 1 if 'verify' in text_lower else 0
        features['has_confirm'] = 1 if 'confirm' in text_lower else 0

        # Sentence count
        features['num_sentences'] = len(re.findall(r'[.!?]+', text))

        # Special character ratio
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        features['special_char_ratio'] = special_chars / len(text) if len(text) > 0 else 0

        return features

    def get_feature_names(self):
        """Return list of all feature names"""
        return [
            'text_length', 'word_count', 'char_count', 'num_exclamation',
            'num_question', 'num_dollar', 'num_urls', 'num_emails', 'num_phone',
            'num_capitals', 'capital_ratio', 'has_all_caps_words',
            'num_suspicious_keywords', 'num_urgency_words', 'has_money_symbol',
            'num_digits', 'digit_ratio', 'has_link', 'has_ip_address',
            'num_spaces', 'avg_word_length', 'has_click_here', 'has_verify',
            'has_confirm', 'num_sentences', 'special_char_ratio'
        ]


def load_and_prepare_data(dataset_path):
    """Load and prepare text dataset"""
    print(f"Loading dataset from {dataset_path}...")

    # Try to read with different encodings, only keeping first 2 columns
    try:
        df = pd.read_csv(dataset_path, encoding='utf-8', usecols=[0, 1], on_bad_lines='skip')
    except:
        try:
            df = pd.read_csv(dataset_path, encoding='latin-1', usecols=[0, 1], on_bad_lines='skip')
        except:
            try:
                df = pd.read_csv(dataset_path, encoding='iso-8859-1', usecols=[0, 1], on_bad_lines='skip')
            except:
                # Last resort - read without column specification
                df = pd.read_csv(dataset_path, encoding='utf-8', on_bad_lines='skip')
                # Take only first 2 columns
                df = df.iloc[:, :2]

    print(f"Dataset shape: {df.shape}")

    # Clean column names
    df.columns = ['text', 'type']

    # Remove any rows with missing values
    df = df.dropna()

    # Remove rows where text or type is empty
    df = df[df['text'].astype(str).str.strip() != '']
    df = df[df['type'].astype(str).str.strip() != '']

    print(f"\nCleaned dataset shape: {df.shape}")
    print(f"\nClass distribution:")
    # Safely print value counts
    try:
        value_counts = df['type'].value_counts()
        for label, count in value_counts.items():
            print(f"  {label}: {count}")
    except:
        print("  (Unable to display due to encoding)")

    return df


def extract_features_from_dataset(df):
    """Extract features from all texts in dataset"""
    print("\nExtracting features from texts...")
    extractor = TextFeatureExtractor()

    features_list = []
    for idx, text in enumerate(df['text']):
        if idx % 10000 == 0:
            print(f"Processed {idx}/{len(df)} texts...")
        features = extractor.extract_features(str(text))
        features_list.append(features)

    features_df = pd.DataFrame(features_list)
    print(f"\nFeatures extracted. Shape: {features_df.shape}")

    return features_df, extractor


def create_tfidf_features(texts, max_features=1000, tfidf_vectorizer=None):
    """Create TF-IDF features from texts"""
    print("\nCreating TF-IDF features...")

    # Convert all texts to string to avoid issues with numeric values
    texts = [str(text) for text in texts]

    if tfidf_vectorizer is None:
        tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=2,
            max_df=0.95,
            ngram_range=(1, 2),
            stop_words='english'
        )
        tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
    else:
        tfidf_matrix = tfidf_vectorizer.transform(texts)

    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

    return tfidf_matrix, tfidf_vectorizer


def train_catboost_model(X_train, y_train, X_val, y_val):
    """Train CatBoost classifier"""
    print("\nTraining CatBoost model...")

    # Create CatBoost pools
    train_pool = Pool(X_train, y_train)
    val_pool = Pool(X_val, y_val)

    # Initialize model
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.1,
        depth=6,
        loss_function='Logloss',
        eval_metric='AUC',
        random_seed=42,
        verbose=100,
        early_stopping_rounds=50,
        task_type='CPU'
    )

    # Train model
    model.fit(
        train_pool,
        eval_set=val_pool,
        use_best_model=True,
        plot=False
    )

    print("\nModel training completed!")
    return model


def plot_training_graphs(model, X_train, y_train, X_test, y_test, feature_names):
    """Generate and save training visualization graphs"""
    print("\nGenerating visualization graphs...")

    os.makedirs('graphs', exist_ok=True)

    # 1. Feature Importance (top 20)
    plt.figure(figsize=(12, 10))
    feature_importance = model.get_feature_importance()
    indices = np.argsort(feature_importance)[::-1][:20]

    plt.barh(range(len(indices)), feature_importance[indices], color='steelblue')
    plt.yticks(range(len(indices)), [feature_names[i] if i < len(feature_names) else f'tfidf_{i-len(feature_names)}' for i in indices])
    plt.xlabel('Importance Score')
    plt.title('Top 20 Feature Importance - Email/SMS Phishing Detection')
    plt.tight_layout()
    plt.savefig('graphs/text_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: graphs/text_feature_importance.png")

    # 2. Confusion Matrix
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Legitimate', 'Phishing'],
                yticklabels=['Legitimate', 'Phishing'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix - Email/SMS Phishing Detection')
    plt.tight_layout()
    plt.savefig('graphs/text_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: graphs/text_confusion_matrix.png")

    # 3. ROC Curve
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Email/SMS Phishing Detection')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('graphs/text_roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: graphs/text_roc_curve.png")

    # 4. Training Metrics
    train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
    test_auc = roc_auc_score(y_test, y_pred_proba)

    metrics = {
        'Train AUC': train_auc,
        'Test AUC': test_auc,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred)
    }

    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(metrics)), list(metrics.values()), color='teal', alpha=0.7)
    plt.xticks(range(len(metrics)), list(metrics.keys()), rotation=45, ha='right')
    plt.ylabel('Score')
    plt.title('Model Performance Metrics - Email/SMS Phishing Detection')
    plt.ylim([0, 1.05])

    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, metrics.values())):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.4f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig('graphs/text_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: graphs/text_metrics.png")

    # 5. Class Distribution
    plt.figure(figsize=(8, 6))
    unique, counts = np.unique(y_train, return_counts=True)
    colors = ['#2ecc71', '#e74c3c']
    plt.pie(counts, labels=['Legitimate', 'Phishing'], autopct='%1.1f%%',
            startangle=90, colors=colors, explode=(0.05, 0.05))
    plt.title('Training Data Class Distribution')
    plt.tight_layout()
    plt.savefig('graphs/text_class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: graphs/text_class_distribution.png")

    return metrics


def save_models_and_artifacts(model, extractor, tfidf_vectorizer):
    """Save trained model and artifacts"""
    print("\nSaving model and artifacts...")

    os.makedirs('models', exist_ok=True)

    # Save CatBoost model
    model.save_model('models/text_phishing_model.cbm')
    print("[OK] Saved: models/text_phishing_model.cbm")

    # Save feature extractor
    joblib.dump(extractor, 'models/text_feature_extractor.pkl')
    print("[OK] Saved: models/text_feature_extractor.pkl")

    # Save TF-IDF vectorizer
    joblib.dump(tfidf_vectorizer, 'models/tfidf_vectorizer.pkl')
    print("[OK] Saved: models/tfidf_vectorizer.pkl")


def main():
    """Main training pipeline"""
    print("=" * 60)
    print("Email/SMS Phishing Detection - CatBoost Training")
    print("=" * 60)

    # Load dataset
    df = load_and_prepare_data('dataset/email_messages_clean.csv')

    # Take a sample if dataset is very large (for faster training)
    if len(df) > 50000:
        print(f"\nDataset is large ({len(df)} rows). Sampling 50,000 rows for training...")
        df = df.sample(n=50000, random_state=42)
        print(f"Sampled dataset shape: {df.shape}")
    else:
        print(f"\nUsing full dataset: {len(df)} samples")

    # Extract handcrafted features
    features_df, extractor = extract_features_from_dataset(df)

    # Create TF-IDF features
    tfidf_matrix, tfidf_vectorizer = create_tfidf_features(df['text'].values, max_features=500)

    # Combine handcrafted and TF-IDF features
    print("\nCombining handcrafted and TF-IDF features...")
    X_handcrafted = features_df.values
    X_tfidf = tfidf_matrix.toarray()
    X = np.hstack([X_handcrafted, X_tfidf])

    print(f"Combined feature matrix shape: {X.shape}")

    # Prepare labels
    print("\nMapping labels...")

    # Convert type column to string and lowercase
    df['type'] = df['type'].astype(str).str.lower().str.strip()

    # Map labels to 0 (legitimate/safe) and 1 (phishing/spam)
    # Handle various label formats
    def map_label(label):
        label_lower = label.lower()
        # Check for safe/legitimate patterns
        if any(word in label_lower for word in ['safe', 'ham', 'legitimate', 'normal']):
            return 0
        # Check for phishing/spam patterns
        elif any(word in label_lower for word in ['phish', 'spam', 'unsafe', 'fraud', 'scam']):
            return 1
        else:
            # Default to looking at the label itself
            return 0 if 'safe' in label_lower else 1

    y = df['type'].apply(map_label).values

    print(f"\nFinal dataset size: {len(df)}")
    print(f"Label distribution:")
    print(f"  Legitimate/Safe: {np.sum(y == 0)}")
    print(f"  Phishing/Spam: {np.sum(y == 1)}")
    print(f"Labels shape: {y.shape}")

    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
    )

    print(f"\nTrain set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")

    # Train model
    model = train_catboost_model(X_train, y_train, X_val, y_val)

    # Evaluate
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                              target_names=['Legitimate', 'Phishing']))

    # Generate graphs
    feature_names = extractor.get_feature_names() + [f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])]
    metrics = plot_training_graphs(model, X_train, y_train, X_test, y_test, feature_names)

    # Save model and artifacts
    save_models_and_artifacts(model, extractor, tfidf_vectorizer)

    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
