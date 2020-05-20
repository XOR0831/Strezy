from django.db import models

# Create your models here.
class History(models.Model):
    title = models.CharField(max_length=255)
    types = models.CharField(max_length=255)
    description = models.TextField()
    datetime = models.DateTimeField(auto_now=True)
