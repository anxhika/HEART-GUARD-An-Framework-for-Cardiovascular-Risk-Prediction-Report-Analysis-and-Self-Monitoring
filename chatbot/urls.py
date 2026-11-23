from django.urls import path
from . import views

urlpatterns = [
    path('api/chat/',         views.chat_api,     name='chatbot_api'),
    path('api/history/',      views.chat_history, name='chatbot_history'),
    path('api/new-session/',  views.new_session,  name='chatbot_new_session'),
    path('api/sessions/',     views.session_list, name='chatbot_sessions'),
]
