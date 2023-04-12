from flask import Flask, request, render_template

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.Pipeline.predict_pipeline import CustomData
from src.Pipeline.predict_pipeline import PredictPipeline

application = Flask(__name__)
app = application

#route for the home page
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predictdata", methods = ["GET", "POST"])
def predict_datapoint():
    if request == "GET":
        return render_template("home.html")
    else:
        data = CustomData(gender = request.form.get("gender"),
                          race_ethnicity = request.form.get("race_ethnicity"),
                          parental_level_of_education = request.form.get("parental_level_of_education"),
                          lunch = request.form.get("lunch"),
                          test_preparation_course = request.form.get("test_preparation_course"),
                          reading_score = request.form.get("reading_score"),
                          writing_score = request.form.get("writing_score"))
        
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
        print("After prediction")
        return render_template("home.html", result = round(results[0],2))
    
if __name__ == "__main__":
    app.run(host = "0.0.0.0")