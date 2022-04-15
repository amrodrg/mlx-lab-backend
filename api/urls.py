from unicodedata import name
from django.urls import path

from . import views

urlpatterns = [
    path("", views.build_model, name="build"),
    path("evaluate", views.evaluate_model, name="evaluate"),


    ################################## SHAP ##################################

    path("shap/configure", views.explain_model, name="configure"),
    path("shap/model_information", views.get_model_information, name="model_information"),

]
