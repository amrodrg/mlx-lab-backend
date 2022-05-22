from cProfile import label
from doctest import Example
from urllib import response
from sklearn.datasets import load_diabetes
from IPython.core.display import display, HTML
import shap
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from calendar import day_abbr
import json
from statistics import mode
from unittest import result
from django.shortcuts import render
from django.http import JsonResponse
from rest_framework import status
from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework import status
import tensorflow as tf
from django.http import HttpResponse, HttpResponseNotFound
from tensorflow.keras import layers
from tensorflow.keras import models
from joblib import dump, load
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from json import JSONEncoder
import os
import glob
import datetime
import hashlib

# Variables
INSURANCE_DATA_LINK = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"


# Create your views here.


def load_data(data_link):
    data = pd.read_csv(data_link)
    data_one_hot = pd.get_dummies(data=data)
    # print(data_one_hot.head())
    return data_one_hot


def load_google_drive_data(data_link):
    path = 'https://drive.google.com/uc?export=download&id=' + \
        data_link.split('/')[-2]
    data = pd.read_csv(path)
    data_one_hot = pd.get_dummies(data=data)
    return data_one_hot


def split_x_y(data_link, labels_name):
    data = load_google_drive_data(data_link=data_link)  # load_data(data_link)
    X = data.drop(labels_name, axis=1)
    y = data[labels_name]
    return X, y


def split_train_test(X, y, test_size):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test


def normalize_data(fitting_data, x_data):
    scaler = MinMaxScaler()
    scaler.fit(fitting_data)
    normalized_data = scaler.transform(x_data)
    return normalized_data, scaler


def empty_model():
    model = models.Sequential()
    return model


def add_dense_layer(model, neurons_num, act):
    model.add(layers.Dense(neurons_num, activation=act))


def compile_model(model, loss_function, optimizer, learning_rate):
    if (optimizer == "adam"):
        model.compile(
            loss=loss_function,
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            metrics=['mae', 'accuracy']
        )
    elif (optimizer == "sgd"):
        model.compile(
            loss=loss_function,
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            metrics=['mae', 'accuracy']
        )
    else:
        model.compile(
            loss=loss_function,
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            metrics=['mae', 'accuracy']
        )
    return model


@api_view(['GET', 'POST'])
def check_data_link(request):
    if request.method == "POST":
        data_link = request.data['dataLink']
        try:
            path = 'https://drive.google.com/uc?export=download&id=' + \
                data_link.split('/')[-2]
            data = pd.read_csv(path)
        except:
            content = {'message': 'Data Link is Invalid!'}
            return Response(data=content, status=status.HTTP_503_SERVICE_UNAVAILABLE)

        content = {'message': 'Data Link is Valid!'}
        return Response(data=content, status=status.HTTP_200_OK)


@api_view(['GET', 'POST'])
def check_labels_column_name(request):
    if request.method == "POST":
        data_link = request.data['dataLink']
        labels_name = request.data['labelsName']
        try:
            path = 'https://drive.google.com/uc?export=download&id=' + \
                data_link.split('/')[-2]
            data = pd.read_csv(path)
            X = data.drop(labels_name, axis=1)
        except:
            content = {'message': 'Labels Column Name is Incorrect!'}
            return Response(data=content, status=status.HTTP_503_SERVICE_UNAVAILABLE)

        content = {'message': 'Labels Column Name is Correct!'}
        return Response(data=content, status=status.HTTP_200_OK)


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
        learning_rate = request.data['learningRate']
        do_normalize = request.data['doNormalize']

        host_ip_hash_string = hashlib.sha224(
            request.get_host().encode()).hexdigest()
        saving_formate = ".h5"
        saving_name = model_name + saving_formate
        saving_path = "saved_models/" + host_ip_hash_string + "/" + saving_name
        saving_folder = "saved_models/" + host_ip_hash_string + "/"

        try:
            X, y = split_x_y(data_link, labels_name)
            X_train, X_test, y_train, y_test = split_train_test(
                X, y, testing_percentage)
            X_train_normal = X_train
            if(do_normalize):
                X_train_normal, scaler = normalize_data(X_train, X_train)
                print("========== Data Normalized ================================>")
                scaler_filename = saving_folder + model_name + "_scaler"
                dump(scaler, scaler_filename)
        except:
            content = {'error_message': 'invalid data link!'}
            return Response(data=content, status=status.HTTP_503_SERVICE_UNAVAILABLE)

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
        print('Learning Rate: ', learning_rate)
        print('Username: ', request.user)
        print("--------------------------------------->")
        print('Data Shape: ', X_train.shape)
        print('Normalized Data Shape: ', X_train_normal.shape)
        print('****************************************>')

        model = empty_model()

        for l in range(layersNum):
            model.add(layers.Dense(
                neuronsNumList[l], activation_functions_list[l]))
        model.add(layers.Dense(1))

        print("========== Compiling =====================================>")
        model = compile_model(model=model, loss_function=loss_function,
                              optimizer=optimizer, learning_rate=learning_rate)

        print("========== Fitting Data ==================================>")
        try:
            if(do_normalize):
                model.fit(X_train_normal, y_train,
                          epochs=epochs_number, verbose=0)
            else:
                model.fit(X_train, y_train, epochs=epochs_number, verbose=0)
        except:
            content = {'error_message': 'data fitting failed!'}
            return Response(data=content, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        model.save(saving_path)

        original_data_shape = X_train.head()
        zeros_dataFrame = pd.DataFrame(
            0, index=np.arange(1), columns=list(original_data_shape.columns))
        zeros_dataFrame.to_csv(
            saving_folder + model_name + "_data_shabe.csv", index=False)

        result = model.to_json()
        result = json.loads(result)
        return JsonResponse(result)


@api_view(['GET', 'POST'])
def evaluate_model(request):
    if request.method == "POST":
        model_name = request.data['modelName']
        data_link = request.data['dataLink']
        labels_name = request.data['labelsName']
        testing_percentage = request.data['testingPercentage']/100
        do_normalize = request.data['doNormalize']

        host_ip_hash_string = hashlib.sha224(
            request.get_host().encode()).hexdigest()
        saving_formate = ".h5"
        saving_name = model_name + saving_formate
        saving_path = "saved_models/" + host_ip_hash_string + "/" + saving_name

        loaded_model = tf.keras.models.load_model(saving_path)

        X, y = split_x_y(data_link, labels_name)
        X_train, X_test, y_train, y_test = split_train_test(
            X, y, testing_percentage)
        X_test_normal = X_test
        if(do_normalize):
            print("========= Normalizing X_test =========> ")
            X_test_normal, scaler = normalize_data(X_train, X_test)

        evaluation = loaded_model.evaluate(X_test_normal, y_test)
        medain_value = y_train.median()
        mean_value = y_train.mean()

        evaluation_dict = {
            'mae': float("{:.2f}".format(
                evaluation[1])),
            'accuracy': float("{:.2f}".format(
                evaluation[2])),
            'median': float("{:.2f}".format(
                medain_value)),
            'mean': float("{:.2f}".format(
                mean_value)),
        }

        print("============ Evaluation =========> ", evaluation)
        return JsonResponse(evaluation_dict)
    return


@api_view(['GET', 'POST'])
def use_model(request):
    if request.method == "POST":
        model_name = request.data['modelName']
        prediction_data_link = request.data['predictionDataLink']
        do_normalize = request.data['doNormalize']

        host_ip_hash_string = hashlib.sha224(
            request.get_host().encode()).hexdigest()
        saving_formate = ".h5"
        saving_name = model_name + saving_formate
        saving_path = "saved_models/" + host_ip_hash_string + "/" + saving_name
        saving_folder = "saved_models/" + host_ip_hash_string + "/"

        loaded_model = tf.keras.models.load_model(saving_path)

        original_data_shape = pd.read_csv(
            saving_folder + model_name + "_data_shabe.csv")

        loaded_data = load_google_drive_data(prediction_data_link)

        filled_dataFrame = pd.DataFrame(
            0, index=np.arange(len(loaded_data)), columns=list(original_data_shape.columns))

        filled_dataFrame.update(loaded_data)

        if(do_normalize):
            scaler = load(saving_folder + model_name + '_scaler')
            filled_dataFrame = scaler.transform(filled_dataFrame)

        predictions = loaded_model.predict(filled_dataFrame)

        predictions = predictions.tolist()
        predictions_list = []
        for p in range(len(predictions)):
            predictions_list.append(
                {'idx': p, 'prediction': predictions[p][0]})

        return JsonResponse(predictions_list, safe=False)
    return
