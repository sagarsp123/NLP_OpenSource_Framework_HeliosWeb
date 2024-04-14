from flask import Flask, request, jsonify
from flask_cors import CORS
from sentiment_analysis1 import SentimentAnalyzer1

app = Flask(__name__)
CORS(app)

nlp_object = SentimentAnalyzer1()

@app.route('/nlp_textblob', methods=['POST'])
def textblob_sentiments():
    try:
        if not request.is_json:
            return jsonify({"error": "No JSON data provided"}), 400

        json_data = request.get_json()

        results = {}
        for key, sentence in json_data.items():
            scores = nlp_object.nlp_textblob_sentiment(sentence)
            max_score_key = max(scores, key=scores.get)
            if max_score_key == 'pos':
                max_score = scores['pos']
            elif max_score_key == 'neg':
                max_score = -scores['neg']
            else:
                max_score = 0
            results[key] = max_score

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Add other routes as needed
@app.route('/sentiment_endpoint')
def sentiment_endpoint():
    return jsonify({"endpoint": "nlp_textblob"})

if __name__ == '__main__':
    app.run(debug=True)
