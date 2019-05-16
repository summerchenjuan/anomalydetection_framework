from django.forms import ModelForm,TextInput
from django.forms import Form
from .models import PriSample
from .models import Metric
from django.forms import fields
from django.forms import widgets

class PriSampleForm(ModelForm):
    class Meta:
        model = PriSample
        fields = ['nodename','timestamp','label']
        widgets = {
            'nodename': TextInput(attrs={'readonly':'readonly'}),
            'timestamp': TextInput(attrs={'readonly':'readonly'}),
        }

class VisualizeForm(Form):
    choices_list = []
    for  value in Metric.objects.values_list('metric'):
        choice = ( value[0],value[0])
        choices_list.append(choice)
    choices = tuple(choices_list)
    metrics = fields.CharField(
        initial='bytes_in_value',
        #widget=widgets.Select(choices=((1, 'bytes_in_value'), (2, 'bytes_out_value'),))  # 插件表现形式为下拉框
        widget = widgets.Select(choices=choices)
        )

