from django.urls import path
from . import views

urlpatterns = [
    path('predict', views.predict, name='predict'),
    path('history', views.show_history, name='history'),
]