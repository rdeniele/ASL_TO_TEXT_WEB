from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('api/process-asl/', views.process_asl, name='process_asl'),
    path('api/submit-feedback/', views.submit_feedback, name='submit_feedback'),
    path('api/preprocess-media/', views.preprocess_media, name='preprocess_media'),
    path('rnn-detection/', views.rnn_detection, name='rnn_detection'),  # RNN detection
    path('api/switch-model/', views.switch_model, name='switch_model'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('register/', views.register_view, name='register'),
]
