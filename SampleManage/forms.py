from django import forms
from django.forms import ModelForm,Form
from .models import PriSample,Mulvisualize
from .models import Metric,Nodename,Algorithm,PredictMethod,DetectMethod
from django.forms import fields
from django.forms import widgets,DateTimeInput,TextInput,DateTimeField
from django.utils import timezone
import datetime,time
from  django.contrib.admin import widgets
from bootstrap_datepicker_plus import DateTimePickerInput

class PriSampleForm(ModelForm):
    class Meta:
        model = PriSample
        fields = ['nodename','timestamp','label']
        widgets = {
            'nodename': TextInput(attrs={'readonly':'readonly'}),
            'timestamp': TextInput(attrs={'readonly':'readonly'}),
        }

class TagForm(Form):
    nodename = fields.CharField(
        widget=forms.widgets.TextInput(attrs={'readonly':'readonly','id':'nodename','name':'nodename'})
    )
    timestamp = forms.SplitDateTimeField(
        widget=forms.widgets.SplitDateTimeWidget(attrs={'id':'timestamp','name': 'timestamp'},
                                                 date_attrs={'type': 'date','readonly':'readonly'},
                                                 time_attrs={'type': 'time','readonly':'readonly',
                                                             'step': '01'}))
    CHOICES = [('0', 'normal'), ('1', 'abnormal')]
    mantag = forms.ChoiceField(widget=forms.RadioSelect, choices=CHOICES)



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

class VisualizeESForm(Form):
    #nodename选择
    nodelist_choices_list = []
    for value in Nodename.objects.values_list('nodename'):
        choice = (value[0],value[0])
        nodelist_choices_list.append(choice)
    nodelist_choices = tuple(nodelist_choices_list)
    nodenames = fields.CharField(
        widget = forms.widgets.Select(choices=nodelist_choices,attrs={
            'id':'nodenameid',
            'name':'nodename',
            'class': "selectpicker"
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
        widget = forms.widgets.Select(choices=metric_choices,attrs={
            'id':'metricsid',
            'name':'metrics',
            'class': "selectpicker"
        })
        )

    start = forms.SplitDateTimeField(
        initial=timezone.now(),
        widget=forms.widgets.SplitDateTimeWidget(attrs={'name': 'teststart'},
                                                 date_attrs={'type': 'date', 'class': 'form-control',
                                                             'style': 'width:220px'},
                                                 time_attrs={'type': 'time', 'class': 'form-control',
                                                             'style': 'width:220px',
                                                             'step': '01'}))

    end = forms.SplitDateTimeField(
        initial=timezone.now(),
        widget=forms.widgets.SplitDateTimeWidget(attrs={'name': 'testend'},
                                                 date_attrs={'type': 'date', 'class': 'form-control',
                                                             'style': 'width:220px'},
                                                 time_attrs={'type': 'time', 'class': 'form-control',
                                                             'style': 'width:220px',
                                                             'step': '01'}))


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
        widget=forms.widgets.SelectMultiple(attrs={'name':'nodenames','id':'nodenamesid','class':"selectpicker"})
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

    # trainstart = forms.DateField(
    #     widget=DateTimePickerInput(format='%m/%d/%Y HH:mm:ss')
    # )

    trainstart = forms.SplitDateTimeField(
        initial=timezone.now(),
        widget=forms.widgets.SplitDateTimeWidget(attrs={'name': 'trainstart','class':'form-control','style':'width:300px'},
                                                 date_attrs={'type': 'date','class':'form-control','style':'width:220px'},
                                                 time_attrs={'type': 'time','class':'form-control','style':'width:220px',
                                                             'step': '01'}))

    trainend = forms.SplitDateTimeField(
        initial=timezone.now(),
        widget=forms.widgets.SplitDateTimeWidget(attrs={'name': 'trainend'},
                                                 date_attrs={'type': 'date', 'class': 'form-control',
                                                             'style': 'width:220px'},
                                                 time_attrs={'type': 'time', 'class': 'form-control',
                                                             'style': 'width:220px',
                                                             'step': '01'}))

    algorithm_chioce_list = []
    lists = Algorithm.objects.filter(istrain=1).values('algorithmname')
    for i in range(len(lists)):
        choice = (lists[i]['algorithmname'], lists[i]['algorithmname'])
        algorithm_chioce_list.append(choice)
    algorithm_chioce = tuple(algorithm_chioce_list)
    algorithm = fields.CharField(
        widget=forms.widgets.Select(
            choices=algorithm_chioce,
            attrs={'id':'algorithm','name':'algorithm', 'class': "selectpicker"}))

    parameters = fields.CharField(
        max_length=500,
        widget=forms.widgets.TextInput(
            attrs={'id':'parameter','name':'parameter','value':'default','class':'form-control','style':'width:220px'},))


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

    class Meta:
        model = Mulvisualize
        fields = ['dedimensionname']
        widgets = {
            'dedimensionname': TextInput(attrs={'name':'dedimensionname'}),
        }

class PredictVisesForm(Form):
    nodelist_choices_list = []
    for value in Nodename.objects.values_list('nodename'):
        choice = (value[0], value[0])
        nodelist_choices_list.append(choice)
    nodelist_choices = tuple(nodelist_choices_list)
    nodename = fields.CharField(
        widget=forms.widgets.Select(choices=nodelist_choices, attrs={
            'id': 'nodenameid',
            'name': 'nodename',
             'class': "selectpicker",
        })
    )

    #modelparam = fields.CharField(widget = forms.widgets.TextInput(attrs={'id':'modelparam','name':'modelparam'}))

    metric_choices_list = []
    for value in Metric.objects.values_list('metric'):
        choice = (value[0], value[0])
        metric_choices_list.append(choice)
    metric_choices = tuple(metric_choices_list)
    # metrics = fields.MultipleChoiceField(
    #    choices= metric_choices,
    #     widget=forms.widgets.SelectMultiple(attrs={'name': 'metrics', 'id': 'metrics', 'class': "selectpicker"})
    # )
    #
    # premetrics = fields.MultipleChoiceField(
    #     choices=metric_choices,
    #     widget=forms.widgets.SelectMultiple(attrs={'name': 'premetrics', 'id': 'premetrics', 'class': "selectpicker"})
    # )

    metric = fields.CharField(
        initial='bytes_in_value',
        # widget=widgets.Select(choices=((1, 'bytes_in_value'), (2, 'bytes_out_value'),))  # 插件表现形式为下拉框
        widget=forms.widgets.Select(choices=metric_choices,attrs={'id':'metric','name':'metric','class': "selectpicker"})
    )
    predictstart = forms.SplitDateTimeField(
        initial=timezone.now(),
        widget=forms.widgets.SplitDateTimeWidget(attrs={'name': 'teststart'},
                                                 date_attrs={'type': 'date', 'class': 'form-control',
                                                             'style': 'width:220px'},
                                                 time_attrs={'type': 'time', 'class': 'form-control',
                                                             'style': 'width:220px',
                                                             'step': '01'}))


    predictend = forms.SplitDateTimeField(
        initial=timezone.now(),
        widget=forms.widgets.SplitDateTimeWidget(attrs={'name': 'testend'},
                                                 date_attrs={'type': 'date', 'class': 'form-control',
                                                             'style': 'width:220px'},
                                                 time_attrs={'type': 'time', 'class': 'form-control',
                                                             'style': 'width:220px',
                                                             'step': '01'}))

    predictmethod_chioce_list = []
    lists = PredictMethod.objects.values('methodname')
    for i in range(len(lists)):
        choice = (lists[i]['methodname'], lists[i]['methodname'])
        predictmethod_chioce_list.append(choice)
    predictmethod_chioce = tuple(predictmethod_chioce_list)
    predictmethod = fields.CharField(
        widget=forms.widgets.Select(
            choices=predictmethod_chioce,
            attrs={'id': 'predictmethod', 'name': 'predictmethod', 'class': "selectpicker"}))

    predictparam = fields.CharField(
        max_length=500,
        widget=forms.widgets.TextInput(
            attrs={
                'id': 'predictparam',
                'name': 'predictparam',
                'value': 'default',
                'class': 'form-control',
                'style': 'width:220px'}))


class DetectVisesForm(Form):
    nodelist_choices_list = []
    for value in Nodename.objects.values_list('nodename'):
        choice = (value[0], value[0])
        nodelist_choices_list.append(choice)
    nodelist_choices = tuple(nodelist_choices_list)
    nodename = fields.CharField(
        widget=forms.widgets.Select(choices=nodelist_choices, attrs={
            'id': 'nodenameid',
            'name': 'nodename',
             'class': "selectpicker",
        })
    )

    metric_choices_list = []
    for value in Metric.objects.values_list('metric'):
        choice = (value[0], value[0])
        metric_choices_list.append(choice)
    metric_choices = tuple(metric_choices_list)

    metric = fields.CharField(
        initial='bytes_in_value',
        # widget=widgets.Select(choices=((1, 'bytes_in_value'), (2, 'bytes_out_value'),))  # 插件表现形式为下拉框
        widget=forms.widgets.Select(choices=metric_choices,attrs={'id':'metric','name':'metric','class': "selectpicker"})
    )

    detectstart = forms.SplitDateTimeField(
        initial=timezone.now(),
        widget=forms.widgets.SplitDateTimeWidget(attrs={'name': 'teststart'},
                                                 date_attrs={'type': 'date', 'class': 'form-control',
                                                             'style': 'width:220px'},
                                                 time_attrs={'type': 'time', 'class': 'form-control',
                                                             'style': 'width:220px',
                                                             'step': '01'}))

    detectend = forms.SplitDateTimeField(
        initial=timezone.now(),
        widget=forms.widgets.SplitDateTimeWidget(attrs={'name': 'testend'},
                                                 date_attrs={'type': 'date', 'class': 'form-control',
                                                             'style': 'width:220px'},
                                                 time_attrs={'type': 'time', 'class': 'form-control',
                                                             'style': 'width:220px',
                                                             'step': '01'}))

    predictmethod_chioce_list = [('None','None')]
    lists = PredictMethod.objects.values('methodname')
    for i in range(len(lists)):
        choice = (lists[i]['methodname'], lists[i]['methodname'])
        predictmethod_chioce_list.append(choice)
    predictmethod_chioce = tuple(predictmethod_chioce_list)
    predictmethod = fields.CharField(
             widget=forms.widgets.Select(
                 choices=predictmethod_chioce,
                 attrs={
                     'id': 'predictmethod',
                     'name': 'predictmethod',
                     'class': "selectpicker",
                 }
             )
    )

    detectmethod_chioce_list = []
    lists = DetectMethod.objects.values('methodname')
    for i in range(len(lists)):
        choice = (lists[i]['methodname'], lists[i]['methodname'])
        detectmethod_chioce_list.append(choice)
    detectmethod_chioce = tuple(detectmethod_chioce_list)
    detectmethod = fields.CharField(
        widget=forms.widgets.Select(
            choices=detectmethod_chioce,
            attrs={
                'id': 'detectmethod',
                'name': 'detectmethod',
                'class': "selectpicker",
            }
        )
    )

    predictparam = fields.CharField(
        max_length=500,
        widget=forms.widgets.TextInput(
            attrs={'id': 'predictparam', 'name': 'predictparam', 'value': 'default', 'class': 'form-control', 'style':'width:220px'}))

    detectparam = fields.CharField(
        max_length=500,
        widget=forms.widgets.TextInput(
            attrs={'id': 'detectparam', 'name': 'detectparam', 'value': 'default', 'class': 'form-control', 'style':'width:220px'}))

