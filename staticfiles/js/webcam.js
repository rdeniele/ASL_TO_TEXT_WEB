// static/js/webcam.js
let video = document.getElementById('webcam');
let predictionElement = document.getElementById('prediction');
let confidenceElement = document.getElementById('confidence');

// Initialize webcam
async function setupWebcam() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: 640,
                height: 480
            }
        });
        video.srcObject = stream;
    } catch (error) {
        console.error('Error accessing webcam:', error);
    }
}

// Process frames and send to backend
async function processFrame() {
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    
    // Process frame every 100ms
    setInterval(() => {
        ctx.drawImage(video, 0, 0);
        const imageData = canvas.toDataURL('image/jpeg');
        
        // Send to backend
        fetch('/api/process-asl/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image: imageData
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                predictionElement.textContent = data.prediction;
                confidenceElement.textContent = 
                    `${(data.confidence * 100).toFixed(2)}%`;
            }
        })
        .catch(error => console.error('Error:', error));
    }, 100);
}

// Start everything
setupWebcam().then(() => {
    video.onloadedmetadata = () => {
        processFrame();
    };
});