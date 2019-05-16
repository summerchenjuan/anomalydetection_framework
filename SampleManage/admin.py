from django.contrib import admin
# Register your models here.
from .models import PriSample
from django.http import HttpResponseRedirect
from django.urls import include, path

class PriSampleAdmin(admin.ModelAdmin):
    list_display = ('nodename','timestamp','label')
    list_filter = ('nodename','timestamp','label')
    search_fields = ('nodename','timestamp','label')
    #批量操作
    actions = ["mark"]
    change_list_template = "SampleManage/sample.html"

    def mark(self,request,queryset):
        queryset.update(nodename='b')

#
admin.site.register(PriSample,PriSampleAdmin)

