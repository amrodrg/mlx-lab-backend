from calendar import day_abbr
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
import pandas as pd
from sklearn.model_selection import train_test_split

# Variables
INSURANCE_DATA_LINK = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"

# Create your views here.


def load_data(data_link):
    data = pd.read_csv(data_link)
    data_one_hot = pd.get_dummies(data=data)
    # print(data_one_hot.head())
    return data_one_hot


def split_x_y(data_link, labels_name):
    data = load_data(data_link)
    X = data.drop('charges', axis=1)
    y = data['charges']
    return X, y


def split_tein_test(X, y, test_size):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test


def empty_model():
    model = models.Sequential()
    return model


def add_dense_layer(model, neurons_num, act):
    model.add(layers.Dense(neurons_num, activation=act))


@api_view(['GET', 'POST'])
def build_model(request):
    X, y = split_x_y(INSURANCE_DATA_LINK, 'charges')
    X_train, X_test, y_train, y_test = split_tein_test(X, y, 0.2)
    print("--------------------------------------->")
    print(X_train.shape)
    print("--------------------------------------->")
    model = empty_model()
    model.add(layers.Dense(28))
    model.add(layers.Dense(100))
    model.add(layers.Dense(10))
    model.add(layers.Dense(10))
    model.add(layers.Dense(1))
    model.compile(loss=tf.keras.losses.mae,
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['mae']
                  )

    model.fit(X_train, y_train, epochs=5, verbose=0)
    result = model.to_json()
    result = json.loads(result)
    # print(X_train)
    return JsonResponse(result)
