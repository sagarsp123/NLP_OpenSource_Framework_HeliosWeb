const textInput = document.getElementById("textInput");

// Function to call the nlp_nltk_sentiment route
function analyzeSentiment() {
    const text = textInput.value;

    // Define the JSON data to send in the request body
    data = {
        text: "I'm sad with the service. It's very bad."
    };

    // Define the URL of your Flask server route
    url = 'https://zany-cod-7jxw4pvvp3wxjg-5000.app.github.dev/nlp_nltk_sentiment';

    // Make a POST request to the Flask server
    fetch(url, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data) // Convert data to JSON string
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json(); // Parse response body as JSON
    })
    .then(data => {
        // Handle the response data here
        console.log(data);
    })
    .catch(error => {
        // Handle errors here
        console.error('There was a problem with the fetch operation:', error);
    });
}

// Call the analyzeSentiment function when a button is clicked or a form is submitted
// For example, if you have a button with the id "analyzeButton":
const analyzeButton = document.getElementById("analyzeButton");
analyzeButton.addEventListener("click", analyzeSentiment);
