document.getElementById("upload-form").addEventListener("submit", async function(e) {
    e.preventDefault();

    const formData = new FormData();
    formData.append("image", document.getElementById("image").files[0]);

    const response = await fetch("/predict", {
        method: "POST",
        body: formData,
    });
    
    const data = await response.json();
    document.getElementById("result").innerHTML = `
        <p>Prediction: ${data.prediction}</p>
        <p>Explanation: ${data.explanation}</p>
    `;
});
