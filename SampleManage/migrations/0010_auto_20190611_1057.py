# Generated by Django 2.1.1 on 2019-06-11 10:57

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('SampleManage', '0009_mulvisualize'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='mulvisualize',
            name='id',
        ),
        migrations.AlterField(
            model_name='mulvisualize',
            name='dedimensionname',
            field=models.CharField(max_length=50, primary_key=True, serialize=False),
        ),
    ]
