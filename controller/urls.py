from django.urls import path

from . import views

urlpatterns = [
    path('nodemcu/<int:id>/', views.NodeMCUController, name='nodemcu_cnt'),
    path('user/<int:id>/', views.DeviceContorller, name='user_cnt'),
]