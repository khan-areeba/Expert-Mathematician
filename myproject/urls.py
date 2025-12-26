from django.contrib import admin
from django.urls import path
from videoapp import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index, name='index'),
    path('video_feed/', views.video_feed, name='video_feed'),
    path('get_response/', views.get_response, name='get_response'),
    path('export_to_word/', views.export_to_word_view, name='export_to_word'),
    path('speech_input/', views.speech_input_view, name='speech_input'),  # New route
]
