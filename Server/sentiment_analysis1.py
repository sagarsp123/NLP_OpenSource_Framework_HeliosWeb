import csv
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import pos_tag, ne_chunk
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VSAnalyzer
import torch

class SentimentAnalyzer1:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.vader_analyzer = VSAnalyzer()

    def nlp_nltk_sentiment(self, text):
        scores = self.sia.polarity_scores(text)
        return {key: value for key, value in scores.items() if key != 'compound'}

    def nlp_ner(self, text):
        words = word_tokenize(text)
        pos_tags = pos_tag(words)
        named_entities = ne_chunk(pos_tags)
        entities_sentiment = {}
        for entity in named_entities:
            if isinstance(entity, tuple):
                word, pos = entity
                entity_text = f"{word}/{pos}"
                entity_sentiment = self.nlp_nltk_sentiment(word)
                entities_sentiment[entity_text] = entity_sentiment
        return entities_sentiment

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
