// Function to call the nlp_nltk_sentiment_bulk route
function analyzeSentimentBulk(texts, scoreBoxes, endpoint) {
    // Define the JSON data to send in the request body
    const data = texts.reduce((acc, text, index) => {
        acc[index] = text;
        return acc;
    }, {});

    const url = `http://127.0.0.1:5000/${endpoint}`;

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

        // Update the UI score boxes with the extracted scores
        Object.keys(data).forEach((key, index) => {
            const scoreBox = scoreBoxes[index];
            const score = data[key];
            scoreBox.textContent = score.toFixed(3); // Display score with 3 decimal places

            if (score < 0) {
                scoreBox.style.color = 'red';
            } else if (score > 0) {
                scoreBox.style.color = 'green';
            } else {
                scoreBox.style.color = 'grey';
            }
        });
    })
    .catch(error => {
        // Handle errors here
        console.error('There was a problem with the fetch operation:', error);
    });
}

// Add event listener to Submit button
const submitButton = document.getElementById("submit-btn");
submitButton.addEventListener("click", function() {
    const rows = document.querySelectorAll('.row');
    const texts = [];
    const scoreBoxes = [];

    rows.forEach(function(row) {
        const input = row.querySelector('.readonly-text');
        const inputValue = input.value.trim();
        if (inputValue !== "") {
            texts.push(inputValue);
            // Store the corresponding score-box element
            const scoreBox = row.querySelector('.score-box');
            scoreBoxes.push(scoreBox);
        }
    });

    // Get the endpoint from the Flask server
    fetch('http://127.0.0.1:5000/sentiment_endpoint')
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        // Call the analyzeSentimentBulk function with the determined endpoint
        analyzeSentimentBulk(texts, scoreBoxes, data.endpoint);
    })
    .catch(error => {
        console.error('There was a problem with the fetch operation:', error);
    });
    // Call the analyzeSentimentBulk function with all texts
    //analyzeSentimentBulk(texts, scoreBoxes, apiUrl);
});
