document.addEventListener('DOMContentLoaded', function() {
    const imageUpload = document.getElementById('imageUpload');
    const originalImage = document.getElementById('originalImage');
    const resultImage = document.getElementById('resultImage');
    const startWebcamBtn = document.getElementById('startWebcam');
    const captureImageBtn = document.getElementById('captureImage');
    const stopWebcamBtn = document.getElementById('stopWebcam');
    const webcamElement = document.getElementById('webcam');
    const canvasElement = document.getElementById('canvas');
    const webcamContainer = document.querySelector('.webcam-container');
    const faceInfoContainer = document.getElementById('faceInfo');
    
    let stream = null;
    
    // Handle image upload
    imageUpload.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(event) {
                originalImage.src = event.target.result;
                detectFaces(file);
            };
            reader.readAsDataURL(file);
        }
    });
    
    // Start webcam
    startWebcamBtn.addEventListener('click', async function() {
        try {
            stream = await navigator.mediaDevices.getUserMedia({ video: true });
            webcamElement.srcObject = stream;
            webcamContainer.style.display = 'block';
            startWebcamBtn.disabled = true;
            captureImageBtn.disabled = false;
            stopWebcamBtn.disabled = false;
        } catch (err) {
            console.error('Error accessing webcam:', err);
            alert('Could not access webcam. Please check permissions.');
        }
    });
    
    // Capture image from webcam
    captureImageBtn.addEventListener('click', function() {
        const context = canvasElement.getContext('2d');
        canvasElement.width = webcamElement.videoWidth;
        canvasElement.height = webcamElement.videoHeight;
        context.drawImage(webcamElement, 0, 0, canvasElement.width, canvasElement.height);
        
        canvasElement.toBlob(function(blob) {
            originalImage.src = canvasElement.toDataURL('image/jpeg');
            detectFaces(blob);
        }, 'image/jpeg');
    });
    
    // Stop webcam
    stopWebcamBtn.addEventListener('click', function() {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            webcamElement.srcObject = null;
            webcamContainer.style.display = 'none';
            startWebcamBtn.disabled = false;
            captureImageBtn.disabled = true;
            stopWebcamBtn.disabled = true;
        }
    });
    
    // Function to detect faces
    function detectFaces(imageData) {
        const formData = new FormData();
        formData.append('image', imageData);
        
        // Show loading state
        resultImage.src = '';
        resultImage.alt = 'Processing...';
        faceInfoContainer.innerHTML = '<p>Processing...</p>';
        
        fetch('/detect', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            if (data.error) {
                throw new Error(data.error);
            }
            resultImage.src = data.image;
            resultImage.alt = 'Detected faces';
            
            // Display face information
            displayFaceInfo(data.faces);
        })
        .catch(error => {
            console.error('Error detecting faces:', error);
            resultImage.alt = 'Error: ' + error.message;
            faceInfoContainer.innerHTML = '<p class="error">Error: ' + error.message + '</p>';
            alert('Error processing image: ' + error.message);
        });
    }
    
    // Function to display face information
    function displayFaceInfo(faces) {
        if (!faces || faces.length === 0) {
            faceInfoContainer.innerHTML = '<p>No faces detected</p>';
            return;
        }
        
        let html = '<h3>Detected Faces</h3>';
        html += '<p class="accuracy-note">Note: These predictions are estimates and may not be 100% accurate.</p>';
        html += '<div class="face-grid">';
        
        faces.forEach((face, index) => {
            html += `<div class="face-card">
                <h4>Face ${index + 1}</h4>
                <ul>`;
            
            if (face.emotion) {
                html += `<li><strong>Emotion:</strong> ${face.emotion}</li>`;
            }
            
            if (face.age) {
                html += `<li><strong>Age:</strong> ${face.age}</li>`;
            }
            
            if (face.gender) {
                html += `<li><strong>Gender:</strong> ${face.gender}</li>`;
            }
            
            html += `<li><strong>Confidence:</strong> ${(face.confidence * 100).toFixed(2)}%</li>
                </ul>
            </div>`;
        });
        
        html += '</div>';
        faceInfoContainer.innerHTML = html;
    }
    
    // For GitHub Pages demo (since we can't run the backend)
    // This is a simplified version that will work without the backend
    // In a real implementation, you would use the detectFaces function above
    if (window.location.hostname.includes('github.io')) {
        // Override the detectFaces function for GitHub Pages demo
        detectFaces = function(imageData) {
            // Just display the original image
            setTimeout(() => {
                resultImage.src = originalImage.src;
                resultImage.alt = 'Demo mode - backend not available';
                
                // Display a message
                faceInfoContainer.innerHTML = `
                    <p>This is a demo running on GitHub Pages without a backend.</p>
                    <p>In a real deployment, the image would be processed by the server to detect faces.</p>
                `;
            }, 1000);
        };
    }
});
