from django.contrib import admin
# Register your models here.
from .models import PriSample,Metric,Nodename,ProSample,PredictMethod,DetectMethod
from django.http import HttpResponseRedirect
from django.urls import include, path

class PriSampleAdmin(admin.ModelAdmin):
    list_display = ('nodename','timestamp','label')
    list_filter = ('nodename','timestamp','label')
    search_fields = ('nodename','timestamp','label','bytes_in_value')
    #批量操作
    actions = ["delete","label_anomaly","label_normal","unlabel"]
    change_list_template = "SampleManage/sample.html"

    def label_anomaly(self,request,queryset):
        queryset.update(label = -1)

    def label_normal(self,request,queryset):
        queryset.update(label = 1)

    def delete(self,request,queryset):
        queryset.delete()

    def unlabel(self,request,queryset):
        queryset.update(label = 0)


class MetricAdmin(admin.ModelAdmin):
    list_display = ('metric',)

class NodenameAdmin(admin.ModelAdmin):
    list_display = ('nodename',)

class ProSampleAdmin(admin.ModelAdmin):
    list_display = ('nodename', 'timestamp', 'label')
    list_filter = ('nodename', 'timestamp', 'label')
    search_fields = ('nodename', 'timestamp', 'label')
    actions = ["delete"]

    def delete(self,request,queryset):
        queryset.delete()

class PredictMethodAdmain(admin.ModelAdmin):
    list_display = ('methodname','algorithm','paramvalue')

class DetectMethodAdmin(admin.ModelAdmin):
    list_display = ('methodname','algorithm','paramvalue')

#

# admin.site.register(PriSample,PriSampleAdmin)
# admin.site.register(ProSample,ProSampleAdmin)
admin.site.register(Metric,MetricAdmin)
admin.site.register(Nodename,NodenameAdmin)
admin.site.register(PredictMethod,PredictMethodAdmain)
admin.site.register(DetectMethod,DetectMethodAdmin)