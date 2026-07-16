"""Feature extraction classes for phishing detection"""
import numpy as np
import pandas as pd
from urllib.parse import urlparse
import re
import tldextract


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
