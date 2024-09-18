document.getElementById('query-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    const query = document.getElementById('query').value;
    const responseElement = document.getElementById('response');
    const loadingElement = document.getElementById('loading');

    // Clear previous response and show loading
    responseElement.textContent = '';
    loadingElement.classList.remove('hidden');

    try {
        console.log('Sending query:', query);  // Check if query is being sent
        const response = await fetch('/retrieve', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ query })
        });

        console.log('Response received:', response);  // Check the full response object

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const data = await response.json();
        console.log('Response data:', data);  // Check the received data

        // Update response element
        if (data.generated_text) {
            responseElement.textContent = data.generated_text;
        } else {
            responseElement.textContent = 'No response text generated.';
        }
    } catch (error) {
        console.error('Error:', error);
        responseElement.textContent = 'An error occurred. Please try again.';
    } finally {
        // Hide the loading animation
        loadingElement.classList.add('hidden');
    }
});
