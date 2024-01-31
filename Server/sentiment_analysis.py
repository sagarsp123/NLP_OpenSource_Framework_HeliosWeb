import csv
from flask import Flask, request, jsonify
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import pos_tag, ne_chunk
from nltk.tokenize import word_tokenize
from nltk import download
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VSAnalyzer
import os

app = Flask(__name__)

# # Set the NLTK data path explicitly
# nltk_data_path = 'C:\\Users\\Admin/nltk_data'
# os.environ["NLTK_DATA"] = nltk_data_path


# Download NLTK data (uncomment the line below if not already downloaded)
# download('punkt')
# download('maxent_ne_chunker')
# download('words')
#download('averaged_perceptron_tagger')

class SentimentAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.vader_analyzer = VSAnalyzer()

    def nlp_nltk_sentiment(self, text):
        scores = self.sia.polarity_scores(text)
        # Exclude the compound score from the result
        return {key: value for key, value in scores.items() if key != 'compound'}

    # Performed Named Entity Recognition (NER) using NLTK and inbuilt methods:
    def nlp_ner(self, text):
        words = word_tokenize(text)
        pos_tags = pos_tag(words)
        named_entities = ne_chunk(pos_tags)
        #return named_entities
        # Perform sentiment analysis for each named entity
        entities_sentiment = {}
        for entity in named_entities:
            if isinstance(entity, tuple):
                # If the entity is a word tuple, perform sentiment analysis on the word
                word, pos = entity
                entity_text = f"{word}/{pos}"
                entity_sentiment = self.nlp_nltk_sentiment(word)
                entities_sentiment[entity_text] = entity_sentiment
        
        return entities_sentiment

    # Performed POS Tagging using NLTK and inbuilt methods:
    def nlp_pos_tagging(self, text):
        words = word_tokenize(text)
        pos_tags = pos_tag(words)
        return pos_tags

    def nlp_textblob_sentiment(self, text):
        blob = TextBlob(text)
        sentiment_scores = {
            'neg': blob.sentiment.polarity,
            'pos': blob.sentiment.polarity,
            'neu': 1.0 - abs(blob.sentiment.polarity)
        }
        return sentiment_scores

    def nlp_vader_sentiment(self, text):
        vader_scores = self.vader_analyzer.polarity_scores(text)
        return {'neg': vader_scores['neg'], 'pos': vader_scores['pos'], 'neu': vader_scores['neu']}


nlp_object = SentimentAnalyzer()

@app.route('/nlp_nltk_sentiment', methods=['POST'])
def analyze_sentiment_endpoint():
    try:
        if not request.is_json:
            return jsonify({"error": "No JSON data provided"}), 400

        json_data = request.get_json()

        results = {}
        for key, sentence in json_data.items():
            scores = nlp_object.nlp_nltk_sentiment(sentence)
            results[int(key)] = scores

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/nlp_ner', methods=['POST'])
def perform_ner_endpoint():
    try:
        if not request.is_json:
            return jsonify({"error": "No JSON data provided"}), 400

        json_data = request.get_json()

        results = {}
        for key, sentence in json_data.items():
            named_entities = nlp_object.nlp_ner(sentence)
            results[int(key)] = str(named_entities)

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/nlp_pos_tagging', methods=['POST'])
def perform_pos_tagging_endpoint():
    try:
        if not request.is_json:
            return jsonify({"error": "No JSON data provided"}), 400

        json_data = request.get_json()

        results = {}
        for key, sentence in json_data.items():
            pos_tags = nlp_object.nlp_pos_tagging(sentence)
            results[int(key)] = str(pos_tags)

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/nlp_textblob', methods=['POST'])
def textblob_sentiments():
    try:
        if not request.is_json:
            return jsonify({"error": "No JSON data provided"}), 400

        json_data = request.get_json()

        results = {}
        for key, sentence in json_data.items():
            sentiment_score = nlp_object.nlp_textblob_sentiment(sentence)
            results[int(key)] = sentiment_score

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/nlp_vader', methods=['POST'])
def vader_sentiments():
    try:
        if not request.is_json:
            return jsonify({"error": "No JSON data provided"}), 400

        json_data = request.get_json()

        results = {}
        for key, sentence in json_data.items():
            sentiment_scores = nlp_object.nlp_vader_sentiment(sentence)
            results[int(key)] = sentiment_scores

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
