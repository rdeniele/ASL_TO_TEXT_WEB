from django.db import models

# Create your models here.
class Feedback(models.Model):
    media_file = models.FileField(upload_to='feedback/')
    correct_sign = models.CharField(max_length=100)
    notes = models.TextField(blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Feedback for {self.correct_sign} at {self.timestamp}"