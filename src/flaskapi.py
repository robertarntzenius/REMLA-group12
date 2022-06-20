"""Module for flask api calls."""
# pylint: disable=R0903
import joblib
from flask import Flask, request, render_template, redirect
from flasgger import Swagger
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired

from preprocessing import text_prepare

app = Flask(__name__)
app.config['SECRET_KEY'] = 'you-will-never-guess'
swagger = Swagger(app)

class QuestionForm(FlaskForm):
    """Form for question."""
    question = StringField('Question', validators=[DataRequired()])
    submit = SubmitField('Predict')

# Homepage with only a string field for the StackOverflow question you want to predict the tags for
@app.route('/')
@app.route('/index')
def index():
    """Homepage."""
    form = QuestionForm()
    if form.validate_on_submit():
        return redirect('/predict')
    return render_template('index.html', form=form)

# Page where the predicted tags are shown
@app.route('/predict', methods=['POST'])
def predict():
    """Predict the tag of a question."""
    tags = joblib.load('output/tags.joblib')
    tfidf_vectorizer = joblib.load('output/tfidf_vectorizer.joblib')
    classifier_tfidf = joblib.load('output/classifier_tfidf.joblib')

    question = str(request.form.get('question'))
    processed_question = tfidf_vectorizer.transform([text_prepare(question)])

    prediction = classifier_tfidf.predict(processed_question)
    result = [i for (i, v) in zip(tags, prediction[0]) if v == 1]
    if not result:
        result = ["No tags"]

    return render_template('predict.html', question=question, tags=result)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
