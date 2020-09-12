import pandas as pd
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render

# Create your views here.
from ml_utils.linear_regression import linear_regression
from ml_utils.test_a import ann_model, ann


def importing_data(request):
    if request.method == "GET":
        return render(request, "files/importing_data.html")
    csv_file = request.FILES['file']
    fs = FileSystemStorage()
    filename = fs.save(csv_file.name, csv_file)
    uploaded_file_url = fs.url(filename)
    print(request.POST)
    p_x = float(request.POST.get("percentage_x",  "0.05"))
    p_y = float(request.POST.get("percentage_y", "0.15"))
    dataset = pd.read_csv("/code/media/" + filename, sep=";", decimal=".")
    data = linear_regression("/code/media/" + filename)
    print(data)
    Ann_data= ann("/code/media/" + filename, p_x, p_y)
    print(Ann_data)
    # let's check if it is a csv file
    # if not csv_file.name.endswith('.csv'):
    #     messages.error(request, 'THIS IS NOT A CSV FILE')
    return render(request, "files/importing_data.html", {"name": csv_file.name})


def dashboard_view(request):
    return render(request, "files/dashboard.html")
