from django.db import models

# Create your models here.
class History(models.Model):
    title = models.CharField(max_length=255)
    datetime = models.DateTimeField(auto_now=Ture)