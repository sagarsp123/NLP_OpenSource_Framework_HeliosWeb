import csv
from flask import Flask, request, jsonify
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import pos_tag, ne_chunk
from nltk.tokenize import word_tokenize
from nltk import download
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VSAnalyzer
import os
import torch
# from transformers import BertTokenizer, BertForSequenceClassification, DistilBertTokenizer, DistilBertForSequenceClassification, pipeline
# from transformers import XLNetTokenizer, XLNetForSequenceClassification

# # import tensorflow as tf
# # from tensorflow.keras.preprocessing.text import Tokenizer
# # from tensorflow.keras.preprocessing.sequence import pad_sequences

# import tensorflow as tf
# import tensorflow_hub as hub

from flask_cors import CORS  # Import CORS class


app = Flask(__name__)
CORS(app)  # Apply CORS to your Flask app

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
        # self.tokenizer_bert = DistilBertTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
        # self.model_bert = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
        # self.model_bert.eval()

        # # Load pre-trained model from TensorFlow Hub
        # self.model_lstm = hub.load("https://tfhub.dev/google/nnlm-en-dim128/2")

        # # Initialize XLNet tokenizer and model
        # self.tokenizer_xlnet = XLNetTokenizer.from_pretrained('xlnet-base-cased')
        # self.model_xlnet = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased')
        # self.model_xlnet.eval()


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

    # def bert_sentiment(self, text):
    #     inputs = self.tokenizer_bert(text, return_tensors='pt', padding=True, truncation=True)
    #     with torch.no_grad():
    #         outputs = self.model_bert(**inputs)
    #         logits = outputs.logits
    #         probabilities = torch.softmax(logits, dim=1).tolist()[0]
    #     return {'neg': probabilities[0], 'pos': probabilities[1]}
    #     # print("Probabilities:", probabilities)  # Add print statement for debugging
    #     # return probabilities

    # def lstm_sentiment(self, text):
    #     embeddings = self.model_lstm([text]).numpy()
    #     # Assuming a simple polarity classification based on embeddings
    #     polarity_score = embeddings[0][0]
    #     # Convert float32 to Python float for JSON serialization
    #     polarity_score = float(polarity_score)
    #     return {'neg': 1.0 - polarity_score, 'pos': polarity_score, 'neu': 0.0}  # Assuming a binary classification, 'neu' is set to 0


    # def cnn_sentiment(self, text):
    #     # Load pre-trained CNN model for sentiment analysis
    #     cnn_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    #     result = cnn_model(text)
    #     sentiment_scores = {'neg': result[0]['score'], 'pos': 1 - result[0]['score'], 'neu': 0}
    #     return sentiment_scores


    # def xlnet_sentiment(self, text):
    #     try:
    #         if not text.strip():  # Check if the input text is empty or whitespace
    #             return {'neg': 0.0, 'pos': 0.0, 'neu': 1.0}  # Return neutral sentiment scores

    #         # Tokenize input text
    #         inputs = self.tokenizer_xlnet(text, return_tensors="pt", padding=True, truncation=True, max_length=256)

    #         print(inputs)
    #         # Perform inference
    #         with torch.no_grad():
    #             outputs = self.model_xlnet(**inputs)
    #             logits = outputs.logits
    #             probabilities = torch.softmax(logits, dim=1).tolist()[0]
    #         # Return sentiment scores
    #         return {'neg': probabilities[0], 'pos': probabilities[1], 'neu': probabilities[2]}

    #     except Exception as e:
    #         return {'neg': 0.0, 'pos': 0.0, 'neu': 1.0}  # Return neutral sentiment scores in case of error


nlp_object = SentimentAnalyzer()

@app.route('/nlp_nltk_sentiment_bulk', methods=['POST'])
def analyze_sentiment_endpoint():
    try:
        if not request.is_json:
            return jsonify({"error": "No JSON data provided"}), 400

        json_data = request.get_json()

        results = {}
        for key, sentence in json_data.items():
            scores = nlp_object.nlp_nltk_sentiment(sentence)
            # Take the maximum score and determine its category
            max_score_key = max(scores, key=scores.get)
            if max_score_key == 'pos':
                max_score = scores['pos']
            elif max_score_key == 'neg':
                max_score = -scores['neg']
            else:
                max_score = 0  # Return 0 for 'neu'

            results[key] = max_score

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

# @app.route('/bert_sentiment', methods=['POST'])
# def bert_sentiment_endpoint():
#     try:
#         if not request.is_json:
#             return jsonify({"error": "No JSON data provided"}), 400

#         json_data = request.get_json()

#         results = {}
#         for key, sentence in json_data.items():
#             scores = nlp_object.bert_sentiment(sentence)
#             results[int(key)] = scores  # Convert key to integer

#         return jsonify(results)

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


# @app.route('/lstm_endpoint', methods=['POST'])
# def lstm_sentiment_endpoint():
#     try:
#         if not request.is_json:
#             return jsonify({"error": "No JSON data provided"}), 400

#         json_data = request.get_json()

#         results = {}
#         for key, sentence in json_data.items():
#             scores = nlp_object.lstm_sentiment(sentence)
#             results[int(key)] = scores

#         return jsonify(results)

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


# @app.route('/cnn', methods=['POST'])
# def cnn_sentiment_endpoint():
#     try:
#         if not request.is_json:
#             return jsonify({"error": "No JSON data provided"}), 400

#         json_data = request.get_json()

#         results = {}
#         for key, sentence in json_data.items():
#             scores = nlp_object.cnn_sentiment(sentence)
#             results[int(key)] = scores

#         return jsonify(results)

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


# @app.route('/xlnet_endpoint', methods=['POST'])
# def xlnet_endpoint():
#     try:
#         if not request.is_json:
#             return jsonify({"error": "No JSON data provided"}), 400

#         json_data = request.get_json()

#         results = {}
#         for key, sentence in json_data.items():
#             scores = nlp_object.xlnet_sentiment(sentence)
#             results[key] = scores

#         return jsonify(results)

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
