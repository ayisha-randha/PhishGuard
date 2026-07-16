import numpy as np
import joblib
import re
from catboost import CatBoostClassifier
import cv2
from pyzbar.pyzbar import decode
from PIL import Image
import os
from features import URLFeatureExtractor, TextFeatureExtractor


class PhishingPredictor:
    """Unified predictor for URL, Email, and SMS phishing detection"""

    def __init__(self):
        self.url_model = None
        self.url_extractor = None
        self.text_model = None
        self.text_extractor = None
        self.tfidf_vectorizer = None
        self.models_loaded = False

    def load_models(self):
        """Load all trained models and extractors"""
        try:
            # Load URL model and create extractor
            if os.path.exists('models/url_phishing_model.cbm'):
                self.url_model = CatBoostClassifier()
                self.url_model.load_model('models/url_phishing_model.cbm')
                self.url_extractor = URLFeatureExtractor()
                print("[OK] URL model loaded successfully")
            else:
                print("⚠ URL model not found. Please train the model first.")

            # Load text model and extractors
            if os.path.exists('models/text_phishing_model.cbm'):
                self.text_model = CatBoostClassifier()
                self.text_model.load_model('models/text_phishing_model.cbm')
                self.text_extractor = TextFeatureExtractor()
                self.tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
                print("[OK] Text model loaded successfully")
            else:
                print("⚠ Text model not found. Please train the model first.")

            self.models_loaded = True
            return True

        except Exception as e:
            print(f"Error loading models: {e}")
            return False

    def predict_url(self, url):
        """Predict if a URL is phishing or legitimate"""
        if not self.url_model or not self.url_extractor:
            return {
                'error': 'URL model not loaded',
                'prediction': None,
                'confidence': 0.0
            }

        try:
            # Store original URL for display
            original_url = url

            # Normalize URL - add https:// if no scheme present (default to secure)
            # This is important because the model is trained to expect HTTPS for legitimate sites
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url

            # Extract features
            features = self.url_extractor.extract_features(url)
            X = np.array(list(features.values())).reshape(1, -1)

            # Make prediction
            prediction = self.url_model.predict(X)[0]
            probabilities = self.url_model.predict_proba(X)[0]

            result = {
                'url': url,
                'prediction': 'Phishing' if prediction == 1 else 'Legitimate',
                'confidence': float(probabilities[prediction]) * 100,
                'phishing_probability': float(probabilities[1]) * 100,
                'legitimate_probability': float(probabilities[0]) * 100,
                'risk_level': self._get_risk_level(probabilities[1])
            }

            return result

        except Exception as e:
            return {
                'error': f'Error processing URL: {str(e)}',
                'prediction': None,
                'confidence': 0.0
            }

    def predict_text(self, text):
        """Predict if an email or SMS is phishing or legitimate"""
        if not self.text_model or not self.text_extractor or not self.tfidf_vectorizer:
            return {
                'error': 'Text model not loaded',
                'prediction': None,
                'confidence': 0.0
            }

        try:
            # Extract handcrafted features
            features = self.text_extractor.extract_features(text)
            X_handcrafted = np.array(list(features.values())).reshape(1, -1)

            # SHORT MESSAGE HEURISTIC FIX
            # The model was trained on emails (typically 100+ chars)
            # Short SMS messages (<30 chars) like "hi", "ok", "thanks" create sparse TF-IDF vectors
            # Apply special rules for short messages to prevent false positives
            text_length = len(text.strip())
            if text_length < 30:
                # Check for suspicious indicators
                has_suspicious = (
                    features['num_suspicious_keywords'] > 0 or
                    features['num_urgency_words'] > 1 or
                    features['has_link'] == 1 or
                    features['special_char_ratio'] > 0.3 or
                    features['capital_ratio'] > 0.5 or
                    features['has_money_symbol'] == 1
                )

                # If short message has no suspicious features, classify as legitimate
                if not has_suspicious:
                    return {
                        'text': text,
                        'prediction': 'Legitimate',
                        'confidence': 95.0,
                        'phishing_probability': 5.0,
                        'legitimate_probability': 95.0,
                        'risk_level': 'Safe',
                        'note': 'Short message with no suspicious indicators'
                    }

            # Extract TF-IDF features
            X_tfidf = self.tfidf_vectorizer.transform([text]).toarray()

            # Combine features
            X = np.hstack([X_handcrafted, X_tfidf])

            # Make prediction
            prediction = self.text_model.predict(X)[0]
            probabilities = self.text_model.predict_proba(X)[0]

            result = {
                'text': text[:200] + '...' if len(text) > 200 else text,
                'prediction': 'Phishing' if prediction == 1 else 'Legitimate',
                'confidence': float(probabilities[prediction]) * 100,
                'phishing_probability': float(probabilities[1]) * 100,
                'legitimate_probability': float(probabilities[0]) * 100,
                'risk_level': self._get_risk_level(probabilities[1])
            }

            return result

        except Exception as e:
            return {
                'error': f'Error processing text: {str(e)}',
                'prediction': None,
                'confidence': 0.0
            }

    def predict_qr_code(self, image_path):
        """Decode QR code and predict if the content is phishing (URL or text)"""
        try:
            # Read image
            img = cv2.imread(image_path)

            if img is None:
                # Try with PIL
                img = np.array(Image.open(image_path))

            # Decode QR code
            decoded_objects = decode(img)

            if not decoded_objects:
                return {
                    'error': 'No QR code found in image',
                    'prediction': None,
                    'confidence': 0.0
                }

            # Get the decoded data
            qr_data = decoded_objects[0].data.decode('utf-8').strip()

            # Check if it's a URL (more robust detection)
            is_url = qr_data.lower().startswith(('http://', 'https://', 'www.'))
            if not is_url:
                # Check for domain-like pattern (e.g., example.com)
                # No spaces, contains a dot, and has a TLD-like ending
                if '.' in qr_data and ' ' not in qr_data:
                    parts = qr_data.split('.')
                    if len(parts[-1]) >= 2 and parts[-1].isalpha():
                        is_url = True

            if is_url:
                # It's a URL - use URL phishing detection
                result = self.predict_url(qr_data)
                result['qr_content'] = qr_data
                result['qr_type'] = decoded_objects[0].type
                result['content_type'] = 'URL'
            else:
                # It's text content - use text phishing detection
                result = self.predict_text(qr_data)
                result['qr_content'] = qr_data
                result['qr_type'] = decoded_objects[0].type
                result['content_type'] = 'Text'

            return result

        except Exception as e:
            return {
                'error': f'Error processing QR code: {str(e)}',
                'prediction': None,
                'confidence': 0.0
            }

    def _get_risk_level(self, phishing_probability):
        """Determine risk level based on phishing probability"""
        if phishing_probability >= 0.8:
            return 'High Risk'
        elif phishing_probability >= 0.5:
            return 'Medium Risk'
        elif phishing_probability >= 0.3:
            return 'Low Risk'
        else:
            return 'Safe'


# Singleton instance
_predictor_instance = None


def get_predictor():
    """Get or create predictor instance"""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = PhishingPredictor()
        _predictor_instance.load_models()
    return _predictor_instance


if __name__ == "__main__":
    # Test the predictor
    predictor = PhishingPredictor()
    predictor.load_models()

    # Test URL prediction
    test_url = "https://www.google.com"
    print("\nTesting URL prediction:")
    print(f"URL: {test_url}")
    result = predictor.predict_url(test_url)
    print(f"Result: {result}")

    # Test text prediction
    test_text = "Congratulations! You've won $1000. Click here to claim your prize now!"
    print("\nTesting text prediction:")
    print(f"Text: {test_text}")
    result = predictor.predict_text(test_text)
    print(f"Result: {result}")
