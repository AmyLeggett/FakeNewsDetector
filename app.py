import numpy as np
from flask import Flask, render_template, session
from flask import request
import pickle
from sklearn import preprocessing
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer
import gunicorn

app = Flask(__name__)
# 1.21.6
# 1.24.2
class_names = ["false", "half-true", "mostly-true", "true", "barely-true", "pants-on-fire"]
explainer = LimeTextExplainer(class_names=class_names)
app.secret_key = "key"



def get_array(text):
    value = []
    model = AutoModelForSequenceClassification.from_pretrained("muhtasham/bert-small-finetuned-wnut17-ner")
    tokenizer = AutoTokenizer.from_pretrained("muhtasham/bert-small-finetuned-wnut17-ner")
    filename = 'bert-small-finetuned-wnut17-ner-6label'
    loaded_model = pickle.load(open(filename, 'rb'))
    for i in range(len(text)):
        text[i] = ''.join(text[i])
        text[i] = tokenizer.encode(text[i], return_tensors="pt")
        text[i] = model(text[i])
        text[i] = text[i].logits.detach().numpy()
        proba = loaded_model.predict_proba(text[i])
        value.append(proba.tolist()[0])
        print(loaded_model.predict(text[i]))
    print(value)
    return np.array(value)


@app.route('/', methods=["GET", "POST"])
def home():
    if request.method == "POST":
        text = request.form.get("statement")
        exp = explainer.explain_instance(text, get_array,
                                         num_features=5, num_samples=50, labels=[3])
        exp = exp.as_html()
        return render_template('exp.html', exp=exp)
    return render_template("home.html")


if __name__ == '__main__':
    app.run()


@app.route('/', methods=["GET", "POST"])
def about():
    return render_template("home.html")
