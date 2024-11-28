// static/js/webcam.js
let currentDetectionType = 'cnn';

let video = document.getElementById('webcam');
let predictionElement = document.getElementById('prediction');
// let confidenceElement = document.getElementById('confidence');

// Feedback functionality
let captureBtn = document.getElementById('captureBtn');
let uploadInput = document.getElementById('uploadInput');
let previewArea = document.getElementById('previewArea');
let imagePreview = document.getElementById('imagePreview');
let videoPreview = document.getElementById('videoPreview');
let feedbackForm = document.getElementById('feedbackForm');
let submitFeedback = document.getElementById('submitFeedback');


const SUPPORTED_IMAGE_FORMATS = ['image/jpeg', 'image/png', 'image/gif', 'image/webp', 'image/bmp'];
const SUPPORTED_VIDEO_FORMATS = ['video/mp4', 'video/webm', 'video/ogg', 'video/quicktime'];

document.getElementById('cnnBtn').addEventListener('click', () => {
    currentDetectionType = 'cnn';
});

document.getElementById('rnnBtn').addEventListener('click', () => {
    currentDetectionType = 'rnn';
});

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

        video.style.transform = "scaleX(-1)";
        video.style.mozTransform = "scaleX(-1)";
        video.style.msTransform = "scaleX(-1)";
        video.style.OTransform = "scaleX(-1)";
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
    
    setInterval(() => {
        ctx.drawImage(video, 0, 0);
        const imageData = canvas.toDataURL('image/jpeg');
        
        const endpoint = currentDetectionType === 'cnn' 
            ? '/api/process-asl/'
            : '/api/process-asl-rnn/';
        
        fetch(endpoint, {
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

// Feedback HITL part
captureBtn.addEventListener('click', () => {
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    
    // Flip the image horizontally
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0);

    // Show preview
    previewArea.classList.remove('hidden');
    imagePreview.src = canvas.toDataURL('image/jpeg');
    imagePreview.style.display = 'block';
    videoPreview.style.display = 'none';
    feedbackForm.classList.remove('hidden');
});

async function preprocessImage(imageElement) {
    const canvas = document.createElement('canvas');
    canvas.width = imageElement.naturalWidth;
    canvas.height = imageElement.naturalHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(imageElement, 0, 0);
    
    // Send to backend for preprocessing
    const response = await fetch('/api/preprocess-media/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            image: canvas.toDataURL('image/jpeg'),
            type: 'image'
        })
    });
    
    const result = await response.json();
    if (result.success) {
        // Show the preprocessed image
        const preprocessedImage = new Image();
        preprocessedImage.src = 'data:image/jpeg;base64,' + result.preprocessed;
        return preprocessedImage;
    }
    throw new Error(result.error || 'Preprocessing failed');
}

uploadInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const fileType = file.type;

    // Validate file format
    if (!SUPPORTED_IMAGE_FORMATS.includes(fileType) && !SUPPORTED_VIDEO_FORMATS.includes(fileType)) {
        alert('Unsupported file format. Please upload an image (JPG, PNG, GIF, WebP, BMP) or video (MP4, WebM, OGG, MOV)');
        return;
    }

    previewArea.classList.remove('hidden');
    feedbackForm.classList.remove('hidden');

    try {
        if (SUPPORTED_IMAGE_FORMATS.includes(fileType)) {
            // For images
            const img = new Image();
            img.onload = async () => {
                const preprocessed = await preprocessImage(img);
                imagePreview.src = preprocessed.src;
                imagePreview.style.display = 'block';
                videoPreview.style.display = 'none';
            };
            img.src = URL.createObjectURL(file);
        } else {
            // For videos
            videoPreview.src = URL.createObjectURL(file);
            videoPreview.style.display = 'block';
            imagePreview.style.display = 'none';
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Error processing media. Please try again.');
    }
});

submitFeedback.addEventListener('click', async () => {
    const correctSign = document.getElementById('correctSign').value;
    const notes = document.getElementById('notes').value;
    
    if (!correctSign) {
        alert('Please enter the correct sign');
        return;
    }
    
    let mediaData;
    let mediaType;

    if (imagePreview.style.display !== 'none' && imagePreview.src) {
        // For images
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = imagePreview.naturalWidth;
        canvas.height = imagePreview.naturalHeight;
        ctx.drawImage(imagePreview, 0, 0);
        mediaData = canvas.toDataURL('image/jpeg', 0.8); // Add quality parameter
        mediaType = 'image';
    } else if (videoPreview.style.display !== 'none' && videoPreview.src) {
        // For videos
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = videoPreview.videoWidth;
        canvas.height = videoPreview.videoHeight;
        ctx.drawImage(videoPreview, 0, 0);
        mediaData = canvas.toDataURL('image/jpeg', 0.8);
        mediaType = 'video';
    } else {
        alert('Please capture or upload media first');
        return;
    }
    
    // Verify data URL format before sending
    if (!mediaData.startsWith('data:image/')) {
        alert('Error: Invalid image format');
        return;
    }

    // Show loading state
    submitFeedback.disabled = true;
    submitFeedback.textContent = 'Submitting...';

    try {
        const response = await fetch('/api/submit-feedback/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                media: mediaData,
                mediaType: mediaType,
                correctSign: correctSign,
                notes: notes
            })
        });

        const result = await response.json();
        if (result.success) {
            alert('Thank you for your feedback! This will help improve the ASL detection.');
            // Reset form
            previewArea.classList.add('hidden');
            feedbackForm.classList.add('hidden');
            document.getElementById('correctSign').value = '';
            document.getElementById('notes').value = '';
        } else {
            alert(`Error: ${result.error || 'Failed to submit feedback. Please try again.'}`);
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Network error. Please check your connection and try again.');
    } finally {
        // Reset button state
        submitFeedback.disabled = false;
        submitFeedback.textContent = 'Submit Feedback';
    }
});



// Start webcam
setupWebcam().then(() => {
    video.onloadedmetadata = () => {
        processFrame();
    };
});
