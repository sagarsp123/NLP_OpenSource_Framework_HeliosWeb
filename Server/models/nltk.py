from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import pos_tag, ne_chunk
from nltk.tokenize import word_tokenize
from nltk import download



class NLTKSentimentAnalyzer(SentimentAnalyzer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Check if vander_lexicon is already downloaded, if not download it
        if(...):
            download('vader_lexicon')

        self.sia = SentimentIntensityAnalyzer()

    def analyze(self, input_text):
        scores = sia.polarity_scores(input_text)
        # Exclude the compound score from the result
        return {key: value for key, value in scores.items() if key != 'compound'}
