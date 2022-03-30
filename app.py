import flask
import numpy as np
import test

from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
model = test.ld_model()
@app.route("/")
def test12():
    return render_template("index.html")


@app.route('/predict',methods=['POST'])
def predict():

    raw = [int(x) for x in request.form.values()]
    print(raw)
    norm = [float(i)/sum(raw) for i in raw]
    final_features = [np.array(norm).reshape(1,8)]
    prediction = model.predict(final_features)

    output = test.pr(prediction)

    return render_template('index.html', prediction_text='Selected leaf is {} '.format(output))

if __name__ == "__main__":
    app.run(debug="True")