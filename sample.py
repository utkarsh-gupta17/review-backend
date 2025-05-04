from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import re
import nltk
import spacy
import pickle
import joblib
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as SIA

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Ensure that NLTK uses the local nltk_data folder
nltk.data.path.append(os.path.join(os.getcwd(), 'nltk_data'))

# Load pre-trained models and pipelines
try:
    with open('scaling_pipeline.pkl', 'rb') as f:
        scaling_pipeline = pickle.load(f)

    with open('vectorization_pipeline.pkl', 'rb') as f:
        vectorization_pipeline = pickle.load(f)

    review_classifier_model = joblib.load('Review_classifier_LG.pkl')
except Exception as e:
    app.logger.error(f"Error loading models: {e}")
    raise e

# Initialize components
analyzer = SIA()

# Try to load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    # If model not found, download it
    from spacy.cli import download
    download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

# Download NLTK resources if they are not present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

stop_words = set(nltk.corpus.stopwords.words('english'))

# Define contractions dictionary
contractions = {
    "can't": "cannot",
    "won't": "will not",
    "n't": " not",
    "'re": " are",
    "'s": " is",
    "'d": " would",
    "'ll": " will",
    "'t": " not",
    "'ve": " have",
    "'m": " am"
}

# Function to expand contractions
def expand_contractions(text, contractions_dict=contractions):
    contractions_pattern = re.compile('({})'.format('|'.join(contractions_dict.keys())), 
                                      flags=re.IGNORECASE | re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        expanded_contraction = contractions_dict.get(match.lower())
        return expanded_contraction
    
    expanded_text = contractions_pattern.sub(expand_match, text)
    return expanded_text

# Function to preprocess and lemmatize text
def preprocess_and_lemmatize(text):
    text = text.lower()
    text = expand_contractions(text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation and numbers
    text = text.strip()
    tokens = nltk.word_tokenize(text)
    tokens = [token for token in tokens if token not in stop_words]
    doc = nlp(' '.join(tokens))
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(tokens)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json()
        
        review_text = data['review_text']
        overall = int(data['overall'])
        helpful_ratio = float(data['helpful_ratio'])

        # Extract features from review text
        review_length = len(review_text)
        word_count = len(review_text.split())
        avg_word_length = np.mean([len(word) for word in review_text.split()]) if review_text else 0
        sentiment_score = analyzer.polarity_scores(review_text)['compound']

        # Classify sentiment score
        if sentiment_score >= 0.05:
            sentiment_label = 2
        elif sentiment_score <= -0.05:
            sentiment_label = 0
        else:
            sentiment_label = 1

        # Transform numerical features
        numerical_features = pd.DataFrame([[review_length, helpful_ratio, word_count, avg_word_length]], 
                                        columns=['reviewLength', 'helpfulRatio', 'wordCount', 'avgWordLength'])
        scaled_features = scaling_pipeline.transform(numerical_features)

        # Preprocess review text
        cleaned_review_text = preprocess_and_lemmatize(review_text)

        # Vectorize review text
        vectorized_text = vectorization_pipeline.transform([cleaned_review_text]).toarray()

        # Combine all features into a single dataframe
        final_features = np.hstack((scaled_features, np.array([[overall, sentiment_label]]), vectorized_text))

        # Create a DataFrame for the final features
        final_df = pd.DataFrame(final_features, columns=['reviewLength', 'helpfulRatio', 'wordCount', 'avgWordLength', 
                                                     'overall', 'encoded_sentimentLabel'] + [str(i) for i in range(2000)])

        # Predict using the classifier
        prediction_value = review_classifier_model.predict(final_df)[0]
        prediction = "FAKE" if prediction_value == 1 else "GENUINE"

        return jsonify({
            'prediction': prediction,
            'status': 'success'
        })

    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 400

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))