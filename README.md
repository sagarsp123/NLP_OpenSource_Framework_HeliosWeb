# NLP_OpenSource_Framework_HeliosWeb

## Project Use Case:
Network and embedding data can be associated with textual content. We need ways to explore not only visuals on this data but also on the associated content. In social media, I am exploring features such as sentiment analysis, narrative polarization, and topic summarization.

## Project Information
1. Server: Python server that processes text content from POST json data.
a) Calculate sentiment scores in JSON
b) Input: JSON with IDs as keys and pieces of texts (e.g., a set of sentences) as values
    Output: JSON with IDs and the different sentiment scores.
c) Fine-tuning model, used different Sentiment Analysis model and techniques like Named Entity Recognition (NER), POS Tagging, Sentiment packages(NLTK, Textblob, Vader), CNN architecture and LLM Modeling(LSTM, BERT, XLNET)

2. Frontend: Javascript/HTML/D3. JS frontend to test the python server calls and showcase sentiment scores based on the input in 3D visualized format using D3.
3. Integration of Frontend with Backend server and testing API calls for efficiency, time consumption and scalability.

## Tech Stack/ Methodologies:
1.Python

2.Data Extraction and Cleaning

3.NLP, LLM, CNN, Sentiment Analysis packages

4.HTML, CSS, Javascript, D3. JS
