# myapp/views.py
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.decorators import login_required
import json
import base64
import numpy as np
import cv2
import time
from .asl_processor import ASLProcessor
from .models import Feedback
from django.core.files.base import ContentFile


asl_processor = ASLProcessor()


def index(request):
    return render(request, 'myapp/index.html')

def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect('index')
    else:
        form = AuthenticationForm()
    return render(request, 'myapp/login.html', {'form': form})

def register_view(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('index')
    else:
        form = UserCreationForm()
    return render(request, 'myapp/register.html', {'form': form})

def logout_view(request):
    logout(request)
    return redirect('index')

@csrf_exempt
def process_asl(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            image_data = data.get('image')
            
            # Convert base64 to image
            image_bytes = base64.b64decode(image_data.split(',')[1])
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Process the frame
            result = asl_processor.process_frame(frame)
            
            return JsonResponse({
                'success': True,
                'prediction': result['prediction'],
                'confidence': result['confidence']
            })
            
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            })
    
    return JsonResponse({'success': False, 'error': 'Invalid request method'})

# FEEDBACK HITL PART

@csrf_exempt
def submit_feedback(request):
        try:
            data = json.loads(request.body)
            media_data = data.get('media')
            correct_sign = data.get('correctSign')
            notes = data.get('notes')
            
            # Validation
            if not media_data:
                return JsonResponse({'success': False, 'error': 'No media data provided'})
            if not correct_sign:
                return JsonResponse({'success': False, 'error': 'Please provide the correct sign'})
            
            # Validate media format
            if not media_data.startswith('data:'):
                return JsonResponse({'success': False, 'error': 'Invalid media format: Missing data URI scheme'})

            # Create new feedback instance
            feedback = Feedback(
                correct_sign=correct_sign,
                notes=notes
            )

            try:
                # Get content type and data
                header, data = media_data.split(',', 1)
                content_type = header.split(';')[0].split(':')[1]

                if content_type.startswith('image/'):
                    ext = '.jpg'
                else:
                    return JsonResponse({'success': False, 'error': 'Unsupported media type. Please use images only.'})
                
                # Save file to model
                feedback.media_file.save(
                    f'feedback_{int(time.time())}{ext}',
                    ContentFile(base64.b64decode(data)),
                    save=True
                )

                return JsonResponse({'success': True})
                
            except Exception as e:
                print(f"Error processing media: {str(e)}")
                return JsonResponse({
                    'success': False, 
                    'error': f'Error processing media file: {str(e)}'
                })
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            })
            
@csrf_exempt
def preprocess_media(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            image_data = data.get('image')
            media_type = data.get('type')
            
            # Convert base64 to image
            image_bytes = base64.b64decode(image_data.split(',')[1])
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Process using MediaPipe Hands
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = [(landmark.x, landmark.y, landmark.z) 
                               for landmark in hand_landmarks.landmark]
                    preprocessed_image = preprocess_landmarks(landmarks)
                    
                    # Convert preprocessed image back to base64
                    _, buffer = cv2.imencode('.jpg', preprocessed_image)
                    preprocessed_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    return JsonResponse({
                        'success': True,
                        'preprocessed': preprocessed_base64
                    })
            
            return JsonResponse({
                'success': False,
                'error': 'No hand landmarks detected'
            })
            
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            })
    
    return JsonResponse({'success': False, 'error': 'Invalid request method'})

@csrf_exempt
def process_asl_rnn(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            image_data = data.get('image')
            
            # Convert base64 to image
            image_bytes = base64.b64decode(image_data.split(',')[1])
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Create RNN processor instance
            rnn_processor = ASLProcessor(model_type='rnn')
            result = rnn_processor.process_frame(frame)
            
            return JsonResponse({
                'success': True,
                'prediction': result['prediction'],
                'confidence': result['confidence']
            })
            
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            })
    
    return JsonResponse({'success': False, 'error': 'Invalid request method'})