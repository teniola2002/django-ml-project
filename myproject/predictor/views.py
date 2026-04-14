from django.shortcuts import render
import pickle

model = pickle.load(open("/Users/teniolaadediran/Desktop/Model/model.pkl", "rb"))
def home(request):
    return render(request, "predictor/home.html")


def result(request):
    if request.method == "POST":

        Na = float(request.POST["Na"])
        Mg = float(request.POST["Mg"])
        Al = float(request.POST["Al"])
        Si = float(request.POST["Si"])
        K = float(request.POST["K"])
        Ca = float(request.POST["Ca"])
        Ba = float(request.POST["Ba"])
        Fe = float(request.POST["Fe"])

        data = [Na, Mg, Al, Si, K, Ca, Ba, Fe]

        prediction = model.predict([data])

        return render(request, "predictor/result.html", {
            "prediction": prediction[0],   
            "data": data                  
        })

    return render(request, "predictor/result.html")
    
    