import json

# pylint: disable=R0903
import joblib
from flasgger import Swagger
from flask import Flask, redirect, render_template, request
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, BooleanField
from wtforms.validators import DataRequired

from preprocessing import text_prepare

from metrics import MetricHandler

app = Flask(__name__)
app.config["SECRET_KEY"] = "you-will-never-guess"
swagger = Swagger(app)

metric_handler = MetricHandler()

class QuestionForm(FlaskForm):
    """Form for question."""

    question = StringField("Question", validators=[DataRequired()])
    submit = SubmitField("Predict")


class TagsAccurateForm(FlaskForm):
    tags_accurate = BooleanField()
    submit = SubmitField('Yes')

class TagsNotAccurateForm(FlaskForm):
    tags_accurate = BooleanField()
    submit = SubmitField('No')

# Homepage with only a string field for the StackOverflow question you want to predict the tags for
@app.route("/")
@app.route("/index")
def index():
    """Homepage."""
    form = QuestionForm()
    if form.validate_on_submit():
        return redirect("/predict")
    return render_template("index.html", form=form)


# Page where the predicted tags are shown
@app.route("/predict", methods=["POST"])
def predict():
    """Predict the tag of a question."""
    tags = joblib.load("output/tags.joblib")
    tfidf_vectorizer = joblib.load("output/tfidf_vectorizer.joblib")
    classifier_tfidf = joblib.load("output/classifier_tfidf.joblib")

    question = str(request.form.get("question"))
    processed_question = tfidf_vectorizer.transform([text_prepare(question)])

    prediction = classifier_tfidf.predict(processed_question)
    result = [i for (i, v) in zip(tags, prediction[0]) if v == 1]
    if not result:
        result = []

    metric_handler.new_prediction(result)

    return render_template("predict.html", question=question, tags=result)


@app.route('/feedbacksucces', methods=['POST'])
def feedbacksucces():
    #question = str(request.form.get('question'))
    tags_accurate = request.form.get('tags_accurate')

    metric_handler.feedback(tags_accurate)

    if not tags_accurate:
        suggested_tags = request.form.get('suggested_tags')
        metric_handler.suggested(suggested_tags)

    return render_template('feedbacksuccess.html')

@app.route('/feedback', methods=['POST'])
def feedback():
    question = str(request.form.get('question'))
    tags = json.loads(request.form.get('tags'))
    return render_template('feedback.html', question=question, tags=tags)

@app.route('/metrics')
def metrics():
    output = ""

    output += "# HELP number_of_predictions Total number of predictions cast\n"
    output += "# TYPE number_of_predictions counter\n"
    output += "number_of_predictions " + str(metric_handler.get_no_predictions()) + "\n\n"
    output += "# HELP correct_predictions Total number of correct predictions\n"
    output += "# TYPE correct_predictions counter\n"
    output += "correct_predictions " + str(metric_handler.get_no_correct_predictions()) + "\n\n"

    no_tags, no_suggested = metric_handler.get_no_tags()

    output += "# HELP tags_predicted Total number of predicted tags\n"
    output += "# TYPE tags_predicted counter\n"
    output += "tags_predicted " + str(no_tags) + "\n\n"
    output += "# HELP tags_suggested Total number of tags suggested\n"
    output += "# TYPE tags_suggested counter\n"
    output += "tags_suggested " + str(no_suggested) + "\n\n"

    return output

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
