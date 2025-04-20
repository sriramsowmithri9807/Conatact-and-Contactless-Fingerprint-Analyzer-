document.addEventListener('DOMContentLoaded', function() {
    const contactInput = document.getElementById('contact');
    const contactlessInput = document.getElementById('contactless');
    const contactPreview = document.getElementById('contactPreview');
    const contactlessPreview = document.getElementById('contactlessPreview');
    const comparisonForm = document.getElementById('comparisonForm');
    const resultDiv = document.getElementById('result');
    const probabilitySpan = document.getElementById('probability');
    const matchStatusSpan = document.getElementById('matchStatus');
    const submitButton = document.querySelector('button[type="submit"]');

    // Handle file upload previews
    function handleFilePreview(input, previewElement) {
        const file = input.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                previewElement.src = e.target.result;
                previewElement.style.display = 'block';
            };
            reader.readAsDataURL(file);
        }
    }

    contactInput.addEventListener('change', function() {
        handleFilePreview(this, contactPreview);
    });

    contactlessInput.addEventListener('change', function() {
        handleFilePreview(this, contactlessPreview);
    });

    // Handle form submission
    comparisonForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Show loading state
        submitButton.disabled = true;
        submitButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Comparing...';
        
        const formData = new FormData(this);
        
        try {
            const response = await fetch('/compare', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            
            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }
            
            // Update results
            resultDiv.style.display = 'block';
            probabilitySpan.textContent = `${(data.probability * 100).toFixed(2)}%`;
            matchStatusSpan.textContent = data.match ? 'Match' : 'No Match';
            
            // Update result alert styling
            const resultAlert = document.getElementById('resultAlert');
            resultAlert.className = `alert ${data.match ? 'alert-success' : 'alert-danger'}`;
            
        } catch (error) {
            console.error('Error:', error);
            resultDiv.style.display = 'block';
            const resultAlert = document.getElementById('resultAlert');
            resultAlert.className = 'alert alert-danger';
            resultAlert.innerHTML = `<p class="mb-0">Error: ${error.message}</p>`;
        } finally {
            // Reset button state
            submitButton.disabled = false;
            submitButton.textContent = 'Compare Fingerprints';
        }
    });

    // Initialize by hiding the previews
    contactPreview.style.display = 'none';
    contactlessPreview.style.display = 'none';
}); 