from rest_framework import serializers

class ASLPredictionSerializer(serializers.Serializer):
    label = serializers.CharField()
    confidence = serializers.FloatField()