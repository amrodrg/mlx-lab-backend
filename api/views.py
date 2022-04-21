from doctest import Example
from urllib import response
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
from django.http import HttpResponse, HttpResponseNotFound
from tensorflow.keras import layers
from tensorflow.keras import models
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from json import JSONEncoder
import numpy as np
import os
import datetime


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
    print(
        "========== Data Loaded From Drive ===================================================>")
    data_one_hot = pd.get_dummies(data=data)
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


def compile_model(model, loss_function, optimizer, metrics, learning_rate):
    if (optimizer == "adam"):
        model.compile(
            loss=loss_function,
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            metrics=[metrics]
        )
    elif (optimizer == "sgd"):
        model.compile(
            loss=loss_function,
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            metrics=[metrics]
        )
    else:
        model.compile(
            loss=loss_function,
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            metrics=[metrics]
        )
    return model


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
        metrics = 'accuracy'

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
        print('Learning Rate: ', learning_rate)
        print('Evaluation Metrics: ', metrics)
        print('Username: ', request.user)
        print("--------------------------------------->")
        print('Data Shape: ', X_train.shape)
        print('****************************************>')

        model = empty_model()

        for l in range(layersNum):
            model.add(layers.Dense(neuronsNumList[l]))
        model.add(layers.Dense(1))

        model = compile_model(model=model, loss_function=loss_function, optimizer=optimizer,
                              metrics=metrics, learning_rate=learning_rate)

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

@api_view(['GET', 'POST'])
def use_model(request):
    if request.method == "POST":
        print("========== Post Request ==========>")
        model_name = request.data['modelName']
        prediction_data_link = request.data['predictionDataLink']
        original_data_link = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
        labels_name = "charges"

        saving_formate = ".h5"
        saving_name = model_name + saving_formate
        saving_path = "saved_models/" + saving_name

        print(
            "========== Loading Model ==================================================>")
        loaded_model = tf.keras.models.load_model(saving_path)
        print(
            "========== Model Loaded ===================================================>")

        X, y = split_x_y(original_data_link, labels_name)
        X_train, X_test, y_train, y_test = split_train_test(
            X, y, 0.1)
        original_data_form = X_train.head()
        print(
            "========== Training Data imported ===================================================>")

        loaded_data = load_google_drive_data(prediction_data_link)
        print(
            "========== Prediction Data Loaded ========================================>", loaded_data)

        filled_dataFrame = pd.DataFrame(
            0, index=np.arange(len(loaded_data)), columns=list(original_data_form.columns))

        filled_dataFrame.update(loaded_data)

        predictions = loaded_model.predict(filled_dataFrame)
        print(filled_dataFrame)
        print(
            "========== Predictions ========================================>", predictions)

        predictions = predictions.tolist()

        predictions_list = []
        for p in range(len(predictions)):
            predictions_list.append(
                {'idx': p, 'prediction': predictions[p][0]})

        return JsonResponse(predictions_list, safe=False)
    return




#################################################################################################################
##################################################### SHAP ######################################################




def load_data_shap(data_link):
    data = pd.read_csv(data_link)
    return data

def split_x_y_shap(data_link, labels_name):
    data = load_data_shap(data_link)
    X = data.drop(labels_name, axis=1)
    y = data[labels_name]
    return X, y

@api_view(['POST'])
def get_model_explaination(request):
    shap_values_list = request.data['shapValues']
    label_name = request.data['labelName']
    model_name = request.data['modelName']
    data_link = request.data['dataLink']
    background_value = request.data['backgroundValue']
    plot = request.data['plot']
    example = request.data['example']

    saving_formate = ".h5"
    saving_name = model_name + saving_formate
    saving_path = "saved_models/" + saving_name
    last_modified = os.path.getmtime(saving_path)
    dt_m = datetime.datetime.fromtimestamp(last_modified)

    X, y = split_x_y_shap(data_link, label_name)

    decodedArrays = json.loads(shap_values_list)

    finalNumpyArray = np.asarray(decodedArrays["array"])

    # shap.force_plot(kernel_explainer.expected_value, shap_values, X_test, feature_names = X.columns)


    plotType = ""
    if plot == "1":
        plotType = "Force Plot"
    elif plot == "2":
        plotType = "Summery Plot"

    feature_string = ""
    for feature in X.columns:
        feature_string = feature_string + " " + feature

    resultDic = {
        "plot": plotType,
        "background_value": background_value,
        "modelName": model_name,
        "dataLink": data_link,
        "labelToPredict": label_name,
        "modelFeaturesString": feature_string,
        "lastModified": dt_m
    }

    return JsonResponse(resultDic)

@api_view(['POST'])
def get_model_information(request):
    model_name = request.data['modelName']
    data_link = request.data['dataLink']
    label_name = request.data['labelName']

    saving_formate = ".h5"
    saving_name = model_name + saving_formate
    saving_path = "saved_models/" + saving_name
    last_modified = os.path.getmtime(saving_path)
    dt_m = datetime.datetime.fromtimestamp(last_modified)

    X, y = split_x_y_shap(data_link, label_name)

    featureArray = []
    for feature in X.columns:
        item = {"name": feature}
        featureArray.append(item)

    feature_string = ""
    for feature in X.columns:
        feature_string = feature_string + " " + feature

    infoDic = {
        "modelName": model_name,
        "labelToPredict": label_name,
        "dataLink": data_link,
        "featureArray": featureArray,
        "modelFeaturesString": feature_string,
        "lastModified": dt_m
    }

    return JsonResponse(infoDic)


@api_view(['POST'])
def explain_model(request):
    model_name = request.data['modelName']
    background_value = request.data['backgroundValue']
    example = request.data['example']
    fBooleanArray = request.data['fBooleanArray']
    fExampleArray = request.data['fExampleArray']
    shap_plot = request.data['plot']
    data_link = request.data['dataLink']
    label_name = request.data['labelName']

    saving_formate = ".h5"
    saving_name = model_name + saving_formate
    saving_path = "saved_models/" + saving_name
    loaded_model = tf.keras.models.load_model(saving_path)

    background_value_int = int(background_value)

    X, y = split_x_y(data_link, label_name)
    X_train, X_test, y_train, y_test = split_train_test(X, y, background_value_int/100)

    kernel_explainer = shap.KernelExplainer(loaded_model, X_train)
    # explainer = shap.DeepExplainer(loaded_model, X_train)

    ###################### Feature importance






    








    ####################### For New Example
    cleanExampleDic = {}
    for key in fExampleArray.keys():
        itemValue = fExampleArray.get(key)
        if any(char.isdigit() for char in itemValue) :
            itemValue = int(itemValue)
        cleanExampleDic[key] = itemValue

    # wrap user example in a data frame
    exampleDataFrame = pd.DataFrame(cleanExampleDic, index=[0])
    data_one_hot = pd.get_dummies(data=exampleDataFrame)

    original_data_form = X_train.head()
    filled_dataFrame = pd.DataFrame(
        0, index=np.arange(len(data_one_hot)), columns=list(original_data_form.columns))
    filled_dataFrame.update(data_one_hot)
    shap_values_list = kernel_explainer.shap_values(filled_dataFrame.values)

    class NumpyArrayEncoder(JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return JSONEncoder.default(self, obj)

    numpyData = {"array": shap_values_list}
    encodedNumpyData = json.dumps(numpyData, cls=NumpyArrayEncoder)

    return JsonResponse(encodedNumpyData, safe=False)