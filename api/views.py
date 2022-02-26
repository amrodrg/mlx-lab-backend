import json
from unittest import result
from django.shortcuts import render
from django.http import JsonResponse
from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework import status
import tensorflow as tf
from django.http import HttpResponse, HttpResponseNotFound
from tensorflow.keras import layers
from tensorflow.keras import models

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline

import shap
import matplotlib.pyplot as plt
from IPython.core.display import display, HTML
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

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
    result = json.loads(result)
    # print(result)
    return JsonResponse(result)


@api_view(['GET'])
def explaine_model(request):

    X, y = load_diabetes(return_X_y=True)
    features = load_diabetes()['feature_names']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)

    model = make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(5,), activation='logistic',
                     max_iter=10000, learning_rate='invscaling', random_state=0)
    )

    model.fit(X_train, y_train)

    explainer = shap.KernelExplainer(model.predict, X_train)

    shap_values = explainer.shap_values(X_test, nsamples=100)

    print(shap_values)

    # shap_display_force = shap.force_plot(explainer.expected_value,
    #                                      shap_values[0, :], X_test[0, :], feature_names=features)

    shap_summary_plot = shap.summary_plot(shap_values, X_test, show=False)

    # shap_dependence_plot = shap.dependence_plot(0, shap_values, X_test)

    # shap_waterfall_plot = shap.waterfall_plot(shap_values, 10, show=True)

    shap.initjs()

    shap_html = f"<head>{shap.getjs()}</head><body>{shap_summary_plot}</body>"

    # plot_summary_plot = shap.summary_plot(shap_values, X_test, show=False)

    # savefig('test.svg', bbox_inches='tight') full size plot image
    # shap.save_html("index.htm", shap_summary_plot)

    return HttpResponse(shap_html)
