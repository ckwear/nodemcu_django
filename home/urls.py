from django.urls import path

from . import views

urlpatterns = [
    path('main/pannel/', views.MainPage, name='main_page'),
    path('store/', views.StorePage, name='store'),
]