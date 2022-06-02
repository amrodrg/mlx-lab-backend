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
from os.path import exists
from matplotlib.pyplot import figure
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
from sklearn.preprocessing import OneHotEncoder
from json import JSONEncoder
import os
import glob
import datetime
import hashlib
import matplotlib.pyplot as plt;
import matplotlib
matplotlib.use('Agg')

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
        metrics = 'accuracy'

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
        print('Evaluation Metrics: ', metrics)
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
        model = compile_model(model=model, loss_function=loss_function, optimizer=optimizer,
                              metrics=metrics, learning_rate=learning_rate)

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

        ###################### Save some stuff for the SHAP Info boxes ######################
        # This primitive way of saving data was used because of the deadline

        modelinfo_data_shap = [labels_name, data_link]
        
        evaluation = model.evaluate(X_test, y_test)
        medain_value = y_train.median()
        mean_value = y_train.mean()

        modelinfo_data_shap.append(evaluation[0])
        modelinfo_data_shap.append(medain_value)
        modelinfo_data_shap.append(mean_value)

        modelinfo_dataFrame = pd.DataFrame(0,
                                           index=np.arange(1), columns=list(modelinfo_data_shap))
        modelinfo_dataFrame.to_csv(
            saving_folder + model_name + "_modelinfo.csv", index=False)

        #######################################################################################
        result = model.to_json()
        result = json.loads(result)
        return JsonResponse(result)

    else:
        X, y = split_x_y(INSURANCE_DATA_LINK, 'charges')
        X_train, X_test, y_train, y_test = split_train_test(X, y, 0.2)
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
            'loss': float("{:.2f}".format(
                evaluation[0])),
            'accuracy': float("{:.2f}".format(
                evaluation[1])),
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


#################################################################################################################
##################################################### SHAP ######################################################

# This end point is used to explain test data after training a new model
@api_view(['POST'])
def get_prediction_shap_values(request):
    prediction_link = request.data['predictionDataLink']
    model_name = request.data['modelName']
    data_link = request.data['dataLink']
    label_name = request.data['labelName']

    host_ip_hash_string = hashlib.sha224(
        request.get_host().encode()).hexdigest()
    saving_formate = ".h5"
    saving_name = model_name + saving_formate
    saving_path = "saved_models/" + host_ip_hash_string + "/" + saving_name
    loaded_model = tf.keras.models.load_model(saving_path)

    try:
        X, y = split_x_y(data_link, label_name)
        X_train, X_test, y_train, y_test = split_train_test(
            X, y, 0.2)
    except:
        content = {'error_message': 'invalid data link!'}
        return Response(data=content, status=status.HTTP_503_SERVICE_UNAVAILABLE)

    saving_folder = "saved_models/" + host_ip_hash_string + "/"
    original_data_shape = pd.read_csv(
        saving_folder + model_name + "_data_shabe.csv")

    # med = X_test.median().values.reshape((1,X_test.shape[1]))
    kernel_explainer = shap.KernelExplainer(loaded_model, X_test)

    loaded_data = load_google_drive_data(prediction_link)

    filled_dataFrame = pd.DataFrame(
        0, index=np.arange(len(loaded_data)), columns=list(original_data_shape.columns))
    filled_dataFrame.update(loaded_data)

    shap_values_list = kernel_explainer.shap_values(filled_dataFrame.values)
    numEntries = len(shap_values_list[0])

    jsonPlotArray = []
    for pred in range(0, numEntries):
        subJsonPlotArray = []
        counter = 0
        for column in filled_dataFrame.columns:
            entry = {'name': column, 'effect': shap_values_list[0][pred][counter], 'value': int(
                filled_dataFrame.iloc[pred][column])}
            subJsonPlotArray.append(entry)
            counter = counter + 1
        jsonPlotArray.append(subJsonPlotArray)

    return JsonResponse(jsonPlotArray, safe=False)

# This Methode is used to render the model and explainer information in the explanation page
@api_view(['POST'])
def get_explainer_information(request):
    model_name = request.data['modelName']
    background_value = request.data['backgroundValue']

    host_ip_hash_string = hashlib.sha224(
        request.get_host().encode()).hexdigest()
    saving_formate = ".h5"
    saving_name = model_name + saving_formate
    saving_path = "saved_models/" + host_ip_hash_string + "/" + saving_name
    last_modified = os.path.getmtime(saving_path)
    dt_m = datetime.datetime.fromtimestamp(last_modified)
    loaded_model = tf.keras.models.load_model(saving_path)
    saving_folder = "saved_models/" + host_ip_hash_string + "/"

    # Check again if the summary Plot exists, otherwise the collapsed Summary Plot Component will not be rendered
    summary_exist = exists("saved_models/" + model_name + '_summary_plot.png')

    background_value_int = int(background_value)

    modelinfo_data_shape = pd.read_csv(
        saving_folder + model_name + "_modelinfo.csv")

    try:
        X, y = split_x_y(
            modelinfo_data_shape.columns[1], modelinfo_data_shape.columns[0])
        X_train, X_test, y_train, y_test = split_train_test(
            X, y, background_value_int/100)
    except:
        content = {'error_message': 'invalid data link!'}
        return Response(data=content, status=status.HTTP_503_SERVICE_UNAVAILABLE)

    # Get the Explainer Base Value
    med = X_test.median().values.reshape((1,X_test.shape[1]))
    kernel_explainer = shap.KernelExplainer(loaded_model, med)

    feature_string = ""
    for feature in X.columns:
        feature_string = feature_string + " " + feature

    resultDic = {
        "background_value": background_value,
        "modelName": model_name,
        "dataLink": modelinfo_data_shape.columns[1],
        "baseValue": int(kernel_explainer.expected_value),
        "labelToPredict": modelinfo_data_shape.columns[0],
        "modelFeaturesString": feature_string,
        "summaryExist": summary_exist,
        "lastModified": dt_m,
        "loss": modelinfo_data_shape.columns[2],
        "median": modelinfo_data_shape.columns[3],
        "mean": modelinfo_data_shape.columns[4]
    }

    return JsonResponse(resultDic)

# This End Point is used to fetch all saved Models and show them to the user in the drop down component 
@api_view(['GET'])
def get_model_list(request):

    # Get User unique host_ip_hash_string
    host_ip_hash_string = hashlib.sha224(
        request.get_host().encode()).hexdigest()
    
    # Collect all saved Modells by Name and return the list
    saving_folder = "saved_models/" + host_ip_hash_string + "/"
    model_list = []
    for file in os.listdir(saving_folder):
        if file.endswith(".h5"):
            file_name = file.split('.')
            clean_file_name = file_name[0].split('/')
            model_list.append(clean_file_name[-1])

    return JsonResponse({"modelList": model_list})


# This End Point is used to Fetch Model Information after choosing a Model from the DropDown Component
@api_view(['POST'])
def get_model_information(request):
    
    # Get all Model Information by Model Name
    model_name = request.data['modelName']

    host_ip_hash_string = hashlib.sha224(
        request.get_host().encode()).hexdigest()
    saving_formate = ".h5"
    saving_name = model_name + saving_formate
    saving_path = "saved_models/" + host_ip_hash_string + "/" + saving_name

    # Get model creation date or last Date the Model was modified 
    last_modified = os.path.getmtime(saving_path)
    dt_m = datetime.datetime.fromtimestamp(last_modified)
    saving_folder = "saved_models/" + host_ip_hash_string + "/"

    # Check if the Model has already a summary Plot 
    summary_exist = exists("saved_models/" + model_name + '_summary_plot.png')

    # Get Model Features and Information by Model Name
    features_csv = pd.read_csv(
        saving_folder + model_name + "_data_shabe.csv")
    modelinfo_data_shape = pd.read_csv(
        saving_folder + model_name + "_modelinfo.csv")

    # Collect Feature Names to give the user the option to ented the test data manuelly
    featureArray = []
    for feature in features_csv.columns:
        item = {"name": feature}
        featureArray.append(item)

    # Collect Feature Names to show in the Information Box after choosing a Model from the DropDown List
    feature_string = ""
    for feature in features_csv.columns:
        feature_string = feature_string + " " + feature

    # Collect all fetched Information in a Dic
    infoDic = {
        "modelName": model_name,
        "featureArray": featureArray,
        "modelFeaturesString": feature_string,
        "lastModified": dt_m,
        "summaryExist": summary_exist,
        "labelName": modelinfo_data_shape.columns[0],
        "dataLink": modelinfo_data_shape.columns[1],
        "loss": modelinfo_data_shape.columns[2],
        "median": modelinfo_data_shape.columns[3],
        "mean": modelinfo_data_shape.columns[4]
    }

    return JsonResponse(infoDic)

# This end point ist used to explain Entered or imported test data in the configuraion page
@api_view(['POST'])
def explain_model(request):
    model_name = request.data['modelName']
    background_value = request.data['backgroundValue']
    example = request.data['example']
    prediction_link = request.data['predictionDataLink']
    fExampleArray = request.data['fExampleArray']
    data_link = request.data['dataLink']
    label_name = request.data['labelName']
    calculate_summary = request.data['calculateSummary']

    print("#####################################################")
    print("model_name: ", model_name)
    print("background_value: ", background_value)
    print("example: ", example)
    print("prediction_link: ", prediction_link)
    print("fExampleArray: ", fExampleArray)
    print("data_link: ", data_link)
    print("label_name: ", label_name)
    print("calculate_summary: ", calculate_summary)
    print("#####################################################")

    host_ip_hash_string = hashlib.sha224(
        request.get_host().encode()).hexdigest()
    saving_formate = ".h5"
    saving_name = model_name + saving_formate
    saving_path = "saved_models/" + host_ip_hash_string + "/" + saving_name
    loaded_model = tf.keras.models.load_model(saving_path)
    saving_folder = "saved_models/" + host_ip_hash_string + "/"
    original_data_shape = pd.read_csv(
        saving_folder + model_name + "_data_shabe.csv")

    background_value_int = int(background_value)

    # Split the Training Data to according to the user given background_value_int.
    try:
        X, y = split_x_y(data_link, label_name)
        X_train, X_test, y_train, y_test = split_train_test(
            X, y, background_value_int/100)
    except:
        content = {'error_message': 'invalid data link!'}
        return Response(data=content, status=status.HTTP_503_SERVICE_UNAVAILABLE)

    # Calcualte the med of the X_test data for the KernelExplainer to fill the missing feature Values 
    # We can also create the Explainer using shap.KernelExplainer(loaded_model, X_test) but for more efficiency we can use the med of all background Trainingdata
    med = X_test.median().values.reshape((1,X_test.shape[1]))
    kernel_explainer = shap.KernelExplainer(loaded_model, med)

    # Check if the user wishes to Include the Summary Plot 
    # The Summary Plot is calculated only one time for each Model 
    if calculate_summary:
        shap_values = kernel_explainer.shap_values(X_train) 
        shap.summary_plot(shap_values[0], X_train, show=False)
        plt.savefig("saved_models/" + model_name + '_summary_plot.png', bbox_inches='tight')

    # If the user has choosen to enter the Test data manuelly
    if example == '1':
        cleanExampleDic = {}
        for key in fExampleArray.keys():
            itemValue = fExampleArray.get(key)
            if any(char.isdigit() for char in itemValue):
                itemValue = int(itemValue)
            cleanExampleDic[key] = itemValue

        # wrap user example in a data frame
        exampleDataFrame = pd.DataFrame(cleanExampleDic, index=[0])
        filled_dataFrame = pd.DataFrame(
            0, index=np.arange(1), columns=list(original_data_shape.columns))
        filled_dataFrame.update(exampleDataFrame)

        # Calculate the shapley Values
        try:
            print("####################### before")
            shap_values_list = kernel_explainer.shap_values(filled_dataFrame.values)
            print("####################### after")
        except:
            content = {'message': 'Entered Test Data is Invalid!'}
            return Response(data=content, status=status.HTTP_503_SERVICE_UNAVAILABLE)

        # Collect Feature names, their effect from the model base value and the one_hot_encoded feature value
        subJsonPlotArray = []
        counter = 0
        for column in filled_dataFrame.columns:
            subJsonPlotArray.append({'name': column, 'effect': shap_values_list[0][0][counter], 'value': int(
                filled_dataFrame.iloc[0][column])})
            counter = counter + 1

        return JsonResponse(subJsonPlotArray, safe=False)

    # If the user has choosen to enter the Test data from google drive
    elif example == '2':
        try:
            path = 'https://drive.google.com/uc?export=download&id=' + \
                prediction_link.split('/')[-2]
            data = pd.read_csv(path)
            # The Model does not accept any String values so we have to encode all feature values to one hot formate
            loaded_data = pd.get_dummies(data=data)
        except:
            content = {'message': 'Data Link is Invalid!'}
            return Response(data=content, status=status.HTTP_503_SERVICE_UNAVAILABLE)

        # Fill the Pandas DataFrame with the imported Test data
        filled_dataFrame = pd.DataFrame(
            0, index=np.arange(len(loaded_data)), columns=list(original_data_shape.columns))
        filled_dataFrame.update(loaded_data)

        # Calculate the shapley Values
        try:
            shap_values_list = kernel_explainer.shap_values(filled_dataFrame.values)
        except:
            content = {'message': 'Imported Test Data is Invalid!'}
            return Response(data=content, status=status.HTTP_503_SERVICE_UNAVAILABLE)
            
        numEntries = len(shap_values_list[0])

        # Collect Feature names, their effect from the model base value and the one_hot_encoded feature value
        jsonPlotArray = []
        for pred in range(0, numEntries):
            subJsonPlotArray = []
            counter = 0
            for column in filled_dataFrame.columns:
                entry = {'name': column, 'effect': shap_values_list[0][pred][counter], 'value': int(
                    filled_dataFrame.iloc[pred][column])}
                subJsonPlotArray.append(entry)
                counter = counter + 1
            jsonPlotArray.append(subJsonPlotArray)

        return JsonResponse(jsonPlotArray, safe=False)

    return