import json
from unittest import result
from django.shortcuts import render
from django.http import JsonResponse
from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework import status
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models

# Create your views here.


def empty_model():
    model = models.Sequential()
    return model


def add_dense_layer(model, neurons_num, act):
    model.add(layers.Dense(neurons_num, activation=act))


@api_view(['GET', 'POST'])
def build_model(request):
    model = empty_model()
    model.add(layers.Dense(28, input_shape=(28, 1)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.build()
    result = model.to_json()
    result = json.loads(result)
    # print(result)
    return JsonResponse(result)
