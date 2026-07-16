import pandas as pd
import numpy as np
from urllib.parse import urlparse
import re
import tldextract
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

class URLFeatureExtractor:
    """Extract features from URLs for phishing detection"""

    def __init__(self):
        self.suspicious_words = [
            'account', 'update', 'secure', 'banking', 'confirm', 'login',
            'verify', 'password', 'suspended', 'locked', 'security'
        ]

    def extract_features(self, url):
        """Extract comprehensive features from URL"""
        features = {}

        try:
            # Parse URL
            parsed = urlparse(url)
            domain_info = tldextract.extract(url)

            # Basic URL features
            features['url_length'] = len(url)
            features['domain_length'] = len(parsed.netloc)
            features['path_length'] = len(parsed.path)

            # Count special characters
            features['num_dots'] = url.count('.')
            features['num_hyphens'] = url.count('-')
            features['num_underscores'] = url.count('_')
            features['num_slashes'] = url.count('/')
            features['num_questionmarks'] = url.count('?')
            features['num_equals'] = url.count('=')
            features['num_at'] = url.count('@')
            features['num_ampersand'] = url.count('&')
            features['num_percent'] = url.count('%')

            # Protocol features
            features['is_https'] = 1 if parsed.scheme == 'https' else 0
            features['has_port'] = 1 if parsed.port else 0

            # Domain features
            features['subdomain_count'] = len(domain_info.subdomain.split('.')) if domain_info.subdomain else 0
            features['is_ip'] = 1 if self._is_ip_address(parsed.netloc) else 0

            # Suspicious patterns
            features['has_suspicious_words'] = sum(1 for word in self.suspicious_words if word in url.lower())
            features['digit_ratio'] = sum(c.isdigit() for c in url) / len(url) if len(url) > 0 else 0
            features['letter_ratio'] = sum(c.isalpha() for c in url) / len(url) if len(url) > 0 else 0

            # URL entropy
            features['entropy'] = self._calculate_entropy(url)

            # Query parameters
            features['num_params'] = len(parsed.query.split('&')) if parsed.query else 0

            # Path depth
            features['path_depth'] = len([x for x in parsed.path.split('/') if x])

            # Domain age indicators (heuristics)
            features['domain_has_numbers'] = 1 if any(c.isdigit() for c in parsed.netloc) else 0
            features['domain_has_hyphens'] = 1 if '-' in parsed.netloc else 0

        except Exception as e:
            print(f"Error extracting features from {url}: {e}")
            # Return default features on error
            features = {key: 0 for key in self._get_feature_names()}

        return features

    def _is_ip_address(self, netloc):
        """Check if netloc is an IP address"""
        ip_pattern = r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$'
        return bool(re.match(ip_pattern, netloc.split(':')[0]))

    def _calculate_entropy(self, text):
        """Calculate Shannon entropy of text"""
        if not text:
            return 0
        entropy = 0
        for x in range(256):
            p_x = text.count(chr(x)) / len(text)
            if p_x > 0:
                entropy += - p_x * np.log2(p_x)
        return entropy

    def _get_feature_names(self):
        """Return list of all feature names"""
        return [
            'url_length', 'domain_length', 'path_length',
            'num_dots', 'num_hyphens', 'num_underscores', 'num_slashes',
            'num_questionmarks', 'num_equals', 'num_at', 'num_ampersand',
            'num_percent', 'is_https', 'has_port', 'subdomain_count',
            'is_ip', 'has_suspicious_words', 'digit_ratio', 'letter_ratio',
            'entropy', 'num_params', 'path_depth', 'domain_has_numbers',
            'domain_has_hyphens'
        ]


def load_and_prepare_data(dataset_path):
    """Load and prepare URL dataset"""
    print(f"Loading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path)

    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nClass distribution:")
    print(df['type'].value_counts())

    return df


def extract_features_from_dataset(df):
    """Extract features from all URLs in dataset"""
    print("\nExtracting features from URLs...")
    extractor = URLFeatureExtractor()

    features_list = []
    for idx, url in enumerate(df['url']):
        if idx % 10000 == 0:
            print(f"Processed {idx}/{len(df)} URLs...")
        features = extractor.extract_features(url)
        features_list.append(features)

    features_df = pd.DataFrame(features_list)
    print(f"\nFeatures extracted. Shape: {features_df.shape}")

    return features_df


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

    # 1. Feature Importance
    plt.figure(figsize=(12, 8))
    feature_importance = model.get_feature_importance()
    indices = np.argsort(feature_importance)[::-1][:15]

    plt.barh(range(len(indices)), feature_importance[indices], color='steelblue')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Importance Score')
    plt.title('Top 15 Feature Importance - URL Phishing Detection')
    plt.tight_layout()
    plt.savefig('graphs/url_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: graphs/url_feature_importance.png")

    # 2. Confusion Matrix
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Legitimate', 'Phishing'],
                yticklabels=['Legitimate', 'Phishing'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix - URL Phishing Detection')
    plt.tight_layout()
    plt.savefig('graphs/url_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: graphs/url_confusion_matrix.png")

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
    plt.title('ROC Curve - URL Phishing Detection')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('graphs/url_roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: graphs/url_roc_curve.png")

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
    plt.title('Model Performance Metrics - URL Phishing Detection')
    plt.ylim([0, 1.05])

    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, metrics.values())):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.4f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig('graphs/url_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: graphs/url_metrics.png")

    # 5. Class Distribution
    plt.figure(figsize=(8, 6))
    unique, counts = np.unique(y_train, return_counts=True)
    colors = ['#2ecc71', '#e74c3c']
    plt.pie(counts, labels=['Legitimate', 'Phishing'], autopct='%1.1f%%',
            startangle=90, colors=colors, explode=(0.05, 0.05))
    plt.title('Training Data Class Distribution')
    plt.tight_layout()
    plt.savefig('graphs/url_class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: graphs/url_class_distribution.png")

    return metrics


def save_model_and_extractor(model, extractor):
    """Save trained model and feature extractor"""
    print("\nSaving model and feature extractor...")

    os.makedirs('models', exist_ok=True)

    # Save CatBoost model
    model.save_model('models/url_phishing_model.cbm')
    print("[OK] Saved: models/url_phishing_model.cbm")

    # Save feature extractor
    joblib.dump(extractor, 'models/url_feature_extractor.pkl')
    print("[OK] Saved: models/url_feature_extractor.pkl")


def main():
    """Main training pipeline"""
    print("=" * 60)
    print("URL Phishing Detection - CatBoost Training")
    print("=" * 60)

    # Load dataset
    df = load_and_prepare_data('dataset/url_dataset.csv')

    # Extract features
    extractor = URLFeatureExtractor()
    features_df = extract_features_from_dataset(df)

    # Prepare labels (convert text labels to binary)
    label_map = {'legitimate': 0, 'phishing': 1}
    y = df['type'].map(label_map).values
    X = features_df.values

    print(f"\nFeature matrix shape: {X.shape}")
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
    metrics = plot_training_graphs(model, X_train, y_train, X_test, y_test,
                                   extractor._get_feature_names())

    # Save model
    save_model_and_extractor(model, extractor)

    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
