from unittest import result
from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework import status
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models

# Create your views here.

@api_view(['GET', 'POST'])
def build_model(request):
    model = models.Sequential()
    model.add(layers.Dense(28, input_shape=(28, 1)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.build()
    result = model.to_json()
    return Response({result})
