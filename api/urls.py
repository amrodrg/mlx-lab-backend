from unicodedata import name
from django.urls import path

from . import views

urlpatterns = [
    path("", views.build_model, name="build"),
    path("evaluate", views.evaluate_model, name="evaluate"),
    path("predict", views.use_model, name="predict"),
    path("check_datalink", views.check_data_link, name="check_datalink"),
    path("check_labelsname", views.check_labels_column_name,
         name="check_labelsname"),
]
