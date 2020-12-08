from flask import Flask, render_template, request, jsonify
import numpy as np

import pickle

model = pickle.load(open('kaggle_xfboost.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def man():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def home():
    age = int(request.form['age'])
    temperature = request.form['temperature']
    pulse = request.form['pulse']
    rr = request.form['rr']
    rhonchi = request.form['rhonchi']
    wheezes = request.form['wheezes']
    cough = request.form['cough']
    fever = request.form['fever']
    loss_of_smell = request.form['loss_of_smell']
    loss_of_taste = request.form['loss_of_taste']
    listt = [[age, temperature, pulse, rr, rhonchi, wheezes, cough, fever, loss_of_smell, loss_of_taste]];

    prediction = model.predict_proba(np.array(listt, dtype='f'))[:,0]
    pred = model.predict_proba(np.array(listt, dtype='f'))[:,1]

    p = (prediction*100)
    pp = (pred*100)

    return render_template('after.html', output='Probability To Negative: {}% Probability To Positive: {}%'.format(p, pp))

@app.route('/predict_api',methods=['POST'])
def predict_api():

    data = request.get_json()
    age = data['age']
    temperature = data['temperature']
    pulse = data['pulse']
    rr = data['rr']
    rhonchi = data['rhonchi']
    wheezes = data['wheezes']
    cough = data['cough']
    fever = data['fever']
    loss_of_smell = data['loss_of_smell']
    loss_of_taste = data['loss_of_taste']
    listt = [[age, temperature, pulse, rr, rhonchi, wheezes, cough, fever, loss_of_smell, loss_of_taste]];
    prediction = model.predict_proba(np.array(listt))
    p = prediction.tolist()
    return jsonify(p)

if __name__ == "__main__":
    app.run(debug=True, port='1080')















