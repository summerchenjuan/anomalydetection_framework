from django.urls import path
from . import views

app_name = 'SampleManage'
urlpatterns = [
     path('', views.index, name='index'),
     path('upload',views.upload,name='upload'),
     path('resultechart',views.resultechart,name = 'resultechart'),
     path('testform',views.testform,name='testform'),
     path('visualize',views.visualize,name='visualize'),
     path('predict',views.predict,name='predict'),
     path('<path:pathtest>/retag',views.retag,name='retag'),
     path('saveretag',views.savetag,name='saveretag')
]