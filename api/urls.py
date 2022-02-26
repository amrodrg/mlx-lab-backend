from django.urls import path

from . import views

urlpatterns = [
    path("", views.build_model, name="build"),
    path("explaine/", views.explaine_model, name="explaine"),
]
