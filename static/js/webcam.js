// static/js/webcam.js
document.addEventListener('DOMContentLoaded', () => {
    let video = document.getElementById('webcam');
    const modelSwitchBtn = document.getElementById('modelSwitchBtn');
    let currentModel = 'cnn'; 
    let isProcessing = false;
    let videoStream = null;
    // let confidenceElement = document.getElementById('confidence');

    // Feedback functionality
    const captureBtn = document.getElementById('captureBtn');
    const uploadInput = document.getElementById('uploadInput');
    const previewArea = document.getElementById('previewArea');
    const imagePreview = document.getElementById('imagePreview');
    const videoPreview = document.getElementById('videoPreview');
    const feedbackForm = document.getElementById('feedbackForm');
    const submitFeedback = document.getElementById('submitFeedback');

    let frameCount = 0;
    let frameBuffer = [];
    const SEQUENCE_LENGTH = 30;

    const SUPPORTED_IMAGE_FORMATS = ['image/jpeg', 'image/png', 'image/gif', 'image/webp', 'image/bmp'];
    const SUPPORTED_VIDEO_FORMATS = ['video/mp4', 'video/webm', 'video/ogg', 'video/quicktime'];

    modelSwitchBtn.addEventListener('click', async () => {
        if (isProcessing) return;

        try {
            isProcessing = true;
            modelSwitchBtn.setAttribute('data-loading', 'true');
            modelSwitchBtn.querySelector('.default-text').classList.add('hidden');
            modelSwitchBtn.querySelector('.loading-text').classList.remove('hidden');

            // Stop current video stream
            if (videoStream) {
                videoStream.getTracks().forEach(track => track.stop());
            }

            // Clear video element
            if (video.srcObject) {
                const tracks = video.srcObject.getTracks();
                tracks.forEach(track => track.stop());
                video.srcObject = null;
            }

            // Switch model
            const newModel = currentModel === 'cnn' ? 'rnn' : 'cnn';

            // Send model switch request
            const response = await fetch('/api/switch-model/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': getCookie('csrftoken')
                },
                body: JSON.stringify({ model: newModel })
            });
            if (!response.ok) throw new Error('Model switch failed');

            // Start webcam
            setupWebcam().then(() => {
                video.onloadedmetadata = () => {
                    const frameProcessingLoop = async () => {
                        if (video.readyState === 4) { // Ensure video is ready
                            await processFrame();
                        }
                        requestAnimationFrame(frameProcessingLoop);
                    };
                    frameProcessingLoop();
                };
            });

            // Update UI and redirect
            currentModel = newModel;
            window.location.href = currentModel === 'rnn' ? '/rnn-detection/' : '/';

        } catch (error) {
            console.error('Model switch error:', error);
            alert('Failed to switch models. Please try again.');

        } finally {
            isProcessing = false;
            modelSwitchBtn.setAttribute('data-loading', 'false');
            modelSwitchBtn.querySelector('.loading-text').classList.add('hidden');
            modelSwitchBtn.querySelector('.default-text').classList.remove('hidden');
        }
    });

    function getCookie(name) {
        let cookieValue = null;
        if (document.cookie && document.cookie !== '') {
            const cookies = document.cookie.split(';');
            for (let i = 0; i < cookies.length; i++) {
                const cookie = cookies[i].trim();
                if (cookie.substring(0, name.length + 1) === (name + '=')) {
                    cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                    break;
                }
            }
        }
        return cookieValue;
    }

    // Initialize webcam
    async function setupWebcam() {
        try {
            videoStream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 }
                }
            });
            video.srcObject = videoStream;
            return new Promise((resolve) => {
                video.onloadedmetadata = () => {
                    resolve();
                };
            });
        } catch (error) {
            console.error('Webcam setup error:', error);
            throw error;
        }
    }

    // Process frames and send to backend
    async function processFrame() {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.drawImage(video, 0, 0);

        if (currentModel === 'rnn') {
            frameCount++;
            if (frameCount % 2 === 0) {
                frameBuffer.push(canvas.toDataURL('image/jpeg'));

                // Process when buffer is full
                if (frameBuffer.length >= SEQUENCE_LENGTH) {
                    try {
                        const response = await fetch('/process-frame/', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                                'X-CSRFToken': getCookie('csrftoken')
                            },
                            body: JSON.stringify({
                                frames: frameBuffer,
                                model: 'rnn'
                            })
                        });
                        frameBuffer = []; // Clear buffer after processing
                        return await response.json();
                    } catch (error) {
                        console.error('Frame processing error:', error);
                        return null;
                    }
                }
                return null;
            }
        } else {
            // CNN processing remains unchanged
            try {
                const response = await fetch('/process-frame/', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'X-CSRFToken': getCookie('csrftoken')
                    },
                    body: JSON.stringify({
                        frame: canvas.toDataURL('image/jpeg'),
                        model: 'cnn'
                    })
                });
                return await response.json();
            } catch (error) {
                console.error('Frame processing error:', error);
                return null;
            }
        }
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
            const frameProcessingLoop = async () => {
                if (video.readyState === 4) { // Ensure video is ready
                    await processFrame();
                }
                requestAnimationFrame(frameProcessingLoop);
            };
            frameProcessingLoop();
        };
    });
});
