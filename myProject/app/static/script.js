// Function to submit form data to the specified endpoint
async function submitForm(formId, modelTypeId, resultId, fraud = true) {
    const form = document.getElementById(formId);
    const modelType = document.getElementById(modelTypeId).value;
    const formData = new FormData(form);
    const data = Object.fromEntries(formData.entries());

    // Determine endpoint based on form and model type
    const endpoint = fraud
        ? (modelType === 'rf' ? '/predict_fraud_rf' : '/predict_fraud_dt')
        : (modelType === 'rf' ? '/predict_creditcard_rf' : '/predict_creditcard_dt');

    try {
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        });
        const result = await response.json();
        document.getElementById(resultId).textContent = `Prediction: ${result.prediction}`;
    } catch (error) {
        document.getElementById(resultId).textContent = `Error: ${error.message}`;
    }
}

// Event listener for fraud detection form
document.getElementById('fraudForm').addEventListener('submit', function (event) {
    event.preventDefault();
    submitForm('fraudForm', 'fraud_model_type', 'fraudResult', true);
});

// Event listener for credit card fraud detection form
document.getElementById('creditCardForm').addEventListener('submit', function (event) {
    event.preventDefault();
    submitForm('creditCardForm', 'credit_model_type', 'creditCardResult', false);
});
