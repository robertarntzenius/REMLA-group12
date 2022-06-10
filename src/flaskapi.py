import joblib
import numpy as np
from flask import Flask, request, render_template, redirect
from flasgger import Swagger
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
import pandas as pd

from src.preprocessing import text_prepare

app = Flask(__name__)
app.config['SECRET_KEY'] = 'you-will-never-guess'
swagger = Swagger(app)

class QuestionForm(FlaskForm):
    question = StringField('Question', validators=[DataRequired()])
    submit = SubmitField('Predict')

# Homepage with only a string field for the StackOverflow question you want to predict the tags for
@app.route('/')
@app.route('/index')
def index():
    form = QuestionForm()
    if form.validate_on_submit():
        return redirect('/predict')
    return render_template('index.html', form=form)

# Page where the predicted tags are shown
@app.route('/predict', methods=['POST'])
def predict():
    prediction = []
    if request.method == 'POST':
        question = str(request.form.get('question'))
        processed_question = text_prepare(question)
        model = joblib.load('../output/model.joblib')
        prediction = model.predict(np.array([processed_question]).reshape(1, -1))

    print(prediction)

    return render_template('predict.html', tags=prediction)

# Testing page where the tags "java" and "c++" are always shown as predicted tags
@app.route('/dumbpredict')
def dumb_predict():
    return render_template('predict.html', tags=["java", "c++"])

if __name__ == '__main__':
    # clf = joblib.load('output/model.joblib')
    app.run(host="0.0.0.0", port=8080, debug=True)
