from django.urls import path
from . import views

app_name = 'report_analysis'

urlpatterns = [
    path('', views.report_history, name='history'),
    path('upload/', views.upload_report, name='upload'),
    path('<int:pk>/', views.report_detail, name='detail'),
    path('<int:pk>/status/', views.report_status_api, name='status_api'),
    path('<int:pk>/reprocess/', views.reprocess_report, name='reprocess'),
    path('<int:pk>/delete/', views.delete_report, name='delete'),
]
