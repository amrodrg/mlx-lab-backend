from sklearn.datasets import load_diabetes
from IPython.core.display import display, HTML
import matplotlib.pyplot as plt
import shap
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from calendar import day_abbr
import json
from statistics import mode
from unittest import result
from django.shortcuts import render
from django.http import JsonResponse
from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework import status
import tensorflow as tf    
tf.compat.v1.disable_v2_behavior()
from django.http import HttpResponse, HttpResponseNotFound
from tensorflow.keras import layers
from tensorflow.keras import models
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


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
    X = data.drop(labels_name, axis=1)
    y = data[labels_name]
    return X, y


def split_train_test(X, y, test_size):
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

    if request.method == "POST":
        neuronsNumList = request.data['neuronsList']
        activation_functions_list = request.data['activationList']
        layersNum = request.data['layersNumber']
        model_name = request.data['modelName']
        data_link = request.data['dataLink']
        labels_name = request.data['labelsName']
        epochs_number = request.data['epochsNumber']
        testing_percentage = request.data['testingPercentage']/100
        loss_function = request.data['lossFunction']
        optimizer = request.data['optimizer']
        metrics = request.data['metrics']

        saving_formate = ".h5"
        saving_name = model_name + saving_formate
        saving_path = "saved_models/" + saving_name

        X, y = split_x_y(data_link, labels_name)
        X_train, X_test, y_train, y_test = split_train_test(
            X, y, testing_percentage)

        print("--------------------------------------->")
        print('Neurons Numbers: ', neuronsNumList)
        print('Activation Functions List: ', activation_functions_list)
        print('Layers Number: ', layersNum)
        print('Model Name: ', model_name)
        print('Saving Path: ', saving_path)
        print('Data Link: ', data_link)
        print('Labels name: ', labels_name)
        print('Epochs Nimber: ', epochs_number)
        print('Testing Percentage: ', testing_percentage)
        print('Loss Function: ', loss_function)
        print('Optimizer Function: ', optimizer)
        print('Evaluation Metrics: ', metrics)
        print('Username: ', request.user)
        print("--------------------------------------->")
        print('Data Shape: ', X_train.shape)
        print('****************************************>')

        model = empty_model()

        for l in range(layersNum):
            model.add(layers.Dense(neuronsNumList[l]))

        model.compile(loss=loss_function,
                      optimizer=optimizer,
                      metrics=[metrics]
                      )

        model.fit(X_train, y_train, epochs=epochs_number, verbose=0)

        model.save(saving_path)

        result = model.to_json()
        result = json.loads(result)
        return JsonResponse(result)

    else:
        X, y = split_x_y(INSURANCE_DATA_LINK, 'charges')
        X_train, X_test, y_train, y_test = split_train_test(X, y, 0.2)
        print("--------------------------------------->")
        print(X_train.shape)
        print("--------------------------------------->")
        model = empty_model()
        model.add(layers.Dense(28))
        model.add(layers.Dense(100))
        model.add(layers.Dense(10))
        model.add(layers.Dense(1))
        model.compile(loss=tf.keras.losses.mae,
                      optimizer=tf.keras.optimizers.Adam(),
                      metrics=['mae']
                      )

        model.fit(X_train, y_train, epochs=5, verbose=0)
        result = model.to_json()
        result = json.loads(result)
        # model.save(tensorflow_models/model_name+userID)
        return JsonResponse(result)

@api_view(['GET', 'POST'])
def evaluate_model(request):
    if request.method == "POST":
        model_name = request.data['modelName']
        data_link = request.data['dataLink']
        labels_name = request.data['labelsName']
        testing_percentage = request.data['testingPercentage']/100

        saving_formate = ".h5"
        saving_name = model_name + saving_formate
        saving_path = "saved_models/" + saving_name

        loaded_model = tf.keras.models.load_model(saving_path)

        X, y = split_x_y(data_link, labels_name)
        X_train, X_test, y_train, y_test = split_train_test(
            X, y, testing_percentage)

        evaluation = loaded_model.evaluate(X_test, y_test)
        medain_value = y_train.median()
        mean_value = y_train.mean()

        evaluation_dict = {
            'loss': float("{:.2f}".format(
                evaluation[0])),
            'accuracy': float("{:.2f}".format(
                evaluation[1])),
            'median': float("{:.2f}".format(
                medain_value)),
            'mean': float("{:.2f}".format(
                mean_value)),
        }

        print("==================> ", evaluation)
        return JsonResponse(evaluation_dict)
    return

def use_model(request):
    # model = import(/asdas/asda/modeical_ID)
    # model.predict(asdads/asd/asd.csv)
    return

##################################################### SHAP ######################################################

@api_view(['GET'])
def get_model_features(request):

    model_name = request.data['modelName']

    saving_formate = ".h5"
    saving_name = model_name + saving_formate
    saving_path = "saved_models/" + saving_name
    loaded_model = tf.keras.models.load_model(saving_path)

    






    return JsonResponse()

@api_view(['POST'])
def explain_model(request):
    model_name = request.data['modelName']
    # data_link = request.data['dataLink']
    # label_name = request.data['labelsName']
    background_percentage = request.data['backgroundValue']

    print(background_percentage)

    # saving_formate = ".h5"
    # saving_name = model_name + saving_formate
    # saving_path = "saved_models/" + saving_name
    # loaded_model = tf.keras.models.load_model(saving_path)
    
    # X, y = split_x_y(INSURANCE_DATA_LINK, 'charges')
    # X_train, X_test, y_train, y_test = split_train_test(X, y, 0.2)
    
    # print("--------------------------------------->")
    # print('Data Shape: ', X_train.shape)
    # print("--------------------------------------->")
    
    # explainer = shap.DeepExplainer(loaded_model, X_train)
    # # shap_values = explainer.shap_values(X_train)
    # shap_values = explainer.shap_values(X_test[:3].values)

    # print("--------------------------------------->")
    # print("Explainer: ", shap_values)
    # print("--------------------------------------->")
    
    return HttpResponse("test")