from flask import Flask, render_template, request, url_for, flash, redirect
from werkzeug.exceptions import abort
import pickle
import numpy as np

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your secret key'

cos_comp = lambda x, period: np.cos(x / period * 2 * np.pi)
sin_comp = lambda x, period: np.sin(x / period * 2 * np.pi)
seasons = np.roll([s//3 for s in range(3,15)],-1)
yes_or_no = {"yes": 1, "no": 0}
weathersit_dict = {"clear": 0, "cloudy": 1, "rainy": 2}
filename = "super_puper_duper_model.sav"
loaded_model = pickle.load(open(filename, 'rb'))


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    hour = request.form['hour']
    day = request.form['day']
    month = request.form['month']
    year = request.form['year']
    holiday = request.form['holiday']
    workingday = request.form['workingday']
    weathersit = request.form['weathersit']
    humidity = request.form['humidity']
    windspeed = request.form['windspeed']
    temperature = request.form['temperature']
    atemperature = request.form['atemperature']
    weekday = request.form['weekday']

    int_features = preprocess(day, month, year, holiday, workingday, weathersit, humidity, windspeed, temperature, atemperature, hour,weekday )
    final_features = [np.array(int_features)]
    prediction = loaded_model.predict(final_features)
    output = int(prediction[0])

    return render_template('index.html', prediction_text='Count of shared bikes is :{}'.format(output))

def preprocess(day, month, year, holiday, workingday, weathersit, humidity, windspeed, temperature, atemperature, hour,weekday ):

    day_cos = cos_comp(int(day), 31)
    day_sin = sin_comp(int(day), 31)
    month_cos = cos_comp(int(month), 12)
    month_sin = sin_comp(int(month), 12)
    year = int(year) -  2011
    holiday = yes_or_no[holiday.lower()]
    workingday = yes_or_no[workingday.lower()]
    weathersit = weathersit_dict[weathersit.lower()]
    season = seasons[int(month)]
    humidity = float(humidity)/100
    windspeed = float(windspeed)/67
    temperature = float(temperature)/41
    atemperature = float(atemperature)/50
    return np.array([season, year, hour, holiday, weekday, workingday, weathersit, temperature, atemperature, humidity, windspeed, day_cos, day_sin, month_cos, month_sin])
