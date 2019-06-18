from django import forms
from django.forms import ModelForm,Form
from .models import PriSample,Mulvisualize
from .models import Metric,Nodename,Algorithm,PredictMethod
from django.forms import fields
from django.forms import widgets,DateTimeInput,TextInput,DateTimeField
from django.utils import timezone
import datetime,time
from  django.contrib.admin import widgets

class PriSampleForm(ModelForm):
    class Meta:
        model = PriSample
        fields = ['nodename','timestamp','label']
        widgets = {
            'nodename': TextInput(attrs={'readonly':'readonly'}),
            'timestamp': TextInput(attrs={'readonly':'readonly'}),
        }

class DateInput(forms.DateInput):
    input_type = 'date'

class VisualizeForm(Form):
    #nodename选择
    nodelist_choices_list = []
    for value in Nodename.objects.values_list('nodename'):
        choice = (value[0],value[0])
        nodelist_choices_list.append(choice)
    nodelist_choices = tuple(nodelist_choices_list)
    nodenames = fields.CharField(
        widget = forms.widgets.Select(choices=nodelist_choices,attrs={
            'id':'nodenameid',
            'name':'nodename'
        })
    )

    #metric选择
    metric_choices_list = []
    for  value in Metric.objects.values_list('metric'):
        choice = ( value[0],value[0])
        metric_choices_list.append(choice)
    metric_choices = tuple(metric_choices_list)
    metrics = fields.CharField(
    initial='bytes_in_value',
        #widget=widgets.Select(choices=((1, 'bytes_in_value'), (2, 'bytes_out_value'),))  # 插件表现形式为下拉框
        widget = forms.widgets.Select(choices=metric_choices)
        )


class PredictVisForm(Form):
    # nodename选择
    nodelist_choices_list = []
    for value in Nodename.objects.values_list('nodename'):
        choice = (value[0], value[0])
        nodelist_choices_list.append(choice)
    nodelist_choices = tuple(nodelist_choices_list)
    nodenames = fields.CharField(
        widget=forms.widgets.Select(choices=nodelist_choices,attrs={
            'id':'nodenameid',
            'name':'nodename'
        })
    )

    # metric选择
    metric_choices_list = []
    for value in Metric.objects.values_list('metric'):
        choice = (value[0], value[0])
        metric_choices_list.append(choice)
    metric_choices = tuple(metric_choices_list)
    metrics = fields.CharField(
        initial='bytes_in_value',
        # widget=widgets.Select(choices=((1, 'bytes_in_value'), (2, 'bytes_out_value'),))  # 插件表现形式为下拉框
        widget=forms.widgets.Select(choices=metric_choices)
    )

    teststart = forms.SplitDateTimeField(
        initial = timezone.now(),
        widget = forms.widgets.SplitDateTimeWidget(attrs={'name':'teststart'},
                                                 date_attrs={'type': 'date'}, 
                                                 time_attrs={'type': 'time',
                                                             'step': '01'}))
    
    testend = forms.SplitDateTimeField(
        initial = timezone.now(),
        widget = forms.widgets.SplitDateTimeWidget(attrs={'name':'testend'},
                                                   date_attrs={'type':'date'},
                                                   time_attrs={'type':'time',
                                                               'step':'01'}
        )
    )

    predictmethod_chioce_list = []
    lists = PredictMethod.objects.values('methodname')
    for i in range(len(lists)):
        choice = (lists[i]['methodname'], lists[i]['methodname'])
        predictmethod_chioce_list.append(choice)
    predictmethod_chioce = tuple(predictmethod_chioce_list)
    predictmethod = fields.CharField(
        widget=forms.widgets.Select(
            choices=predictmethod_chioce,
            attrs={'id': 'predictmethod', 'name': 'predictmethod'}))


    # predictmethod = fields.CharField(
    #     max_length=50,
    #     widget=forms.widgets.TextInput(attrs={'id':'predictmethod','name':'predictmethod','value':'LSTM1'})
    # )


    detectmethod = fields.CharField(
        max_length=50,
        widget=forms.widgets.TextInput(attrs={'id':'detectmethod','name':'detectmethod','value':'Nsigma'})
    )

class ChoiceForm(Form):
    test1 = fields.CharField(max_length=10)
    test2 = fields.CharField(max_length=10)

class TrainForm(Form):
    nodelist_choices_list = []
    for value in Nodename.objects.values_list('nodename'):
        choice = (value[0], value[0])
        nodelist_choices_list.append(choice)
    nodelist_choices = tuple(nodelist_choices_list)
    nodename = fields.MultipleChoiceField(
        initial=[1,],
        choices=nodelist_choices,
        widget=forms.widgets.SelectMultiple(attrs={'name':'nodenames','id':'nodenamesid','class':"selectpicker" })
    )

    metric_choices_list = []
    for value in Metric.objects.values_list('metric'):
        choice = (value[0], value[0])
        metric_choices_list.append(choice)
    metric_choices = tuple(metric_choices_list)
    metrics = fields.MultipleChoiceField(
       choices= metric_choices,
        widget=forms.widgets.SelectMultiple(attrs={'name': 'metrics', 'id': 'metrics', 'class': "selectpicker"})
    )

    premetrics = fields.MultipleChoiceField(
        choices=metric_choices,
        widget=forms.widgets.SelectMultiple(attrs={'name': 'premetrics', 'id': 'premetrics', 'class': "selectpicker"})
    )

    trainstart = forms.SplitDateTimeField(
        initial=timezone.now(),
        widget=forms.widgets.SplitDateTimeWidget(attrs={'name': 'trainstart'},
                                                 date_attrs={'type': 'date'},
                                                 time_attrs={'type': 'time',
                                                             'step': '01'}))

    trainend = forms.SplitDateTimeField(
        initial=timezone.now(),
        widget=forms.widgets.SplitDateTimeWidget(attrs={'name': 'trainend'},
                                                 date_attrs={'type': 'date'},
                                                 time_attrs={'type': 'time',
                                                             'step': '01'}))

    algorithm_chioce_list = []
    lists = Algorithm.objects.filter(istrain=1).values('algorithmname')
    for i in range(len(lists)):
        choice = (i+1, lists[i]['algorithmname'])
        algorithm_chioce_list.append(choice)
    algorithm_chioce = tuple(algorithm_chioce_list)
    algorithm = fields.CharField(
        widget=forms.widgets.Select(
            choices=algorithm_chioce,
            attrs={'id':'algorithm','name':'algorithm'}))

    parameters = fields.CharField(
        max_length=500,
        widget=forms.widgets.TextInput(
            attrs={'id':'parameter','name':'parameter','value':'default'}))


class MulvisualizeForm(ModelForm):
    nodelist_choices_list = []
    for value in Nodename.objects.values_list('nodename'):
        choice = (value[0], value[0])
        nodelist_choices_list.append(choice)
    nodelist_choices = tuple(nodelist_choices_list)
    nodename = fields.MultipleChoiceField(
        choices=nodelist_choices,
        widget=forms.widgets.SelectMultiple(attrs={'name': 'nodenames', 'id': 'nodenamesid', 'class': "selectpicker"})
    )

    metriclist_choices_list = []
    for value in Metric.objects.values_list('metric'):
        choice = (value[0], value[0])
        metriclist_choices_list.append(choice)
    metriclist_choices = tuple(metriclist_choices_list)
    metric = fields.MultipleChoiceField(
        choices=metriclist_choices,
        widget=forms.widgets.SelectMultiple(attrs={'name': 'metrics', 'id': 'metricsid', 'class': "selectpicker"})
    )
    #
    # class Meta:
    #     model = Mulvisualize
    #     fields = ['dedimensionname']
    #     widgets = {
    #         'dedimensionname': TextInput(attrs={'name':'dedimensionname'}),
    #     }
