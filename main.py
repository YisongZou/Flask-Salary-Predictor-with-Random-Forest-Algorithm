import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    #output = round(prediction[0], 2)
    output1 = round(prediction[0], 0)
    output = int(output1)
    return render_template('index.html', prediction_text='Gold Medal Number is {}'.format(output))

if __name__ == "__main__":
    #app.run(host='127.0.0.1', port=8080, debug=True)
    app.run()
