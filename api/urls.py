from django.urls import path

from . import views

urlpatterns = [
    path("", views.build_model, name="build"),
    path("evaluate", views.evaluate_model, name="evaluate"),
]
