from django.http import HttpResponse 
from django.shortcuts import render
def home(request):
    return render(request, "predictor/home.html")

def result(request):
    return render(request, "predictor/result.html")