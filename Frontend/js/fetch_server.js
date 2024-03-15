// Function to call the nlp_nltk_sentiment route
function analyzeSentiment(text, scoreBox) {
    // Define the JSON data to send in the request body
    const data = {
        text: text
    };

    // Define the URL of your Flask server route
    const url = 'http://127.0.0.1:5000/nlp_nltk_sentiment';

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
        // Extract the score from the response
        const score = data.text || 0; // Assuming the keys are 'pos' and 'neg'
        // Update the UI score box with the extracted score
        scoreBox.textContent = score.toFixed(3); // Display score with 3 decimal places

    })
    .catch(error => {
        // Handle errors here
        console.error('There was a problem with the fetch operation:', error);
    });
}

const scoreBoxes = [];

// Add event listener to Submit button
const submitButton = document.getElementById("submit-btn");
submitButton.addEventListener("click", function() {
    var rows = document.querySelectorAll('.row');
    var texts = [];
    rows.forEach(function(row) {
        var input = row.querySelector('.readonly-text');
        var inputValue = input.value.trim();
        if (inputValue !== "") {
            texts.push(inputValue);
            // Store the corresponding score-box element
            var scoreBox = row.querySelector('.score-box');
            scoreBoxes.push(scoreBox);
        }
    });

    // Call the analyzeSentiment function for each text
    texts.forEach(function(text, index) {
        analyzeSentiment(text, scoreBoxes[index]);
    });

    console.log(texts);
});
