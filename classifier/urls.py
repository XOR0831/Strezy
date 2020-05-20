from django.urls import path
from . import views

urlpatterns = [
    path('predict', views.predict, name='predict'),
    path('predict_class', views.predict_class_only, name='predict_class'),
    path('history', views.show_history, name='history'),
]
