from django.contrib import admin
from django.utils.html import format_html
from .models import Feedback
from django.urls import reverse

@admin.register(Feedback)
class FeedbackAdmin(admin.ModelAdmin):
    list_display = ('correct_sign', 'notes', 'timestamp', 'media_preview', 'download_link')
    list_filter = ('timestamp', 'correct_sign')
    search_fields = ('correct_sign', 'notes')

    def media_preview(self, obj):
        if obj.media_file:
            if obj.media_file.url.lower().endswith(('.png', '.jpg', '.jpeg')):
                return format_html('<img src="{}" width="100" />', obj.media_file.url)
            elif obj.media_file.url.lower().endswith(('.mp4', '.mov')):
                return format_html('<video width="100" controls><source src="{}" type="video/mp4"></video>', obj.media_file.url)
        return "No media"
    
    def download_link(self, obj):
        if obj.media_file:
            url = reverse('download_file', args=[obj.pk])
            return format_html('<a href="{}">Download</a>', url)
        return "No file"

    media_preview.short_description = 'Preview'
    download_link.short_description = 'Download'