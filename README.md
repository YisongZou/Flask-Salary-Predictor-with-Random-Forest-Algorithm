# Flask Salary Predictor
In this project, we are going to use a random forest algorithm (or any other preferred algorithm) from scikit-learn library to help predict the salary based on your years of experience. We will use Flask as it is a very light web framework to handle the POST requests.

The dataset is from [Kaggle Years of experience and Salary dataset](https://www.kaggle.com/rohankayan/years-of-experience-and-salary-dataset)

![pic](https://github.com/YisongZou/IDS721-Final-Project/blob/main/Screen%20Shot%202021-04-22%20at%201.42.12%20AM.png)

# Architecture
![image](https://user-images.githubusercontent.com/61890131/116028539-37bec680-a60c-11eb-8527-35cf3cf1dac5.png)

The above diagram is the cloud architecture of our salary prediction system. Inside the cloud diagram we have our google cloud services listed: Storage Bucket, Compute Instance, Cloud Run, and Cloud Build. The storage bucket stores the kaggle salary dataset. The compute instances are where code files lie within the cloud platform. We update our github repo by merging feature branch into master branch or change part of the code of the website layout. This will automatically trigger Cloud Build to deploy the code updates into the production flask container.

# Model
`model.py` trains and saves the model to disk.
`model.pkl` is the model compressed in pickle format.

```python
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import requests
import json

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Train the model

# random forest model (or any other preferred algorithm)
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=20, random_state=0)

regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Saving model using pickle
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load( open('model.pkl','rb'))
print(model.predict([[1.8]]))
```

# App
`main.py` has the main function and contains all the required functions for the flask app. In the code, we have created the instance of the Flask() and loaded the model. `model.predict()` method takes input from the json request and converts it into 2D numpy array. The results are stored and returned into the variable named `output`. Finally, we used port 8080 and have set debug=True to enable debugging when necessary.

```python
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
# Load model from model.pkl
model = pickle.load(open('model.pkl', 'rb'))

# Homepage route
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

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Salary is {}'.format(output))

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8080, debug=True)
```


## How to Run the App

1) Clone the repo
```python
git clone https://github.com/YisongZou/IDS721-Final-Project.git
```
2) Setup - Install the required packages
```python
make all
```
3) Train the model
```python
python3 model.py
```
4) Run the application
```python
python3 main.py
```

## Set up Google Cloud Project
Step 1: Create new GCP project

Step 2: Check to see if the console is pointing to the correct project
```python
gcloud projects describe $GOOGLE_CLOUD_PROJECT
```

Step 3: Set working project if not correct
```python
gcloud config set project $GOOGLE_CLOUD_PROJECT
```

Step 4: Follow Step 1-4 to set up github repo and test Flask application

Step 5: In the root project of the folder, replace PROJECT-ID below with the correct GCP project-id, and build the google cloud containerized flask application
```python
gcloud builds submit --tag gcr.io/<PROJECT-ID>/app
```

Step 6: In the root folder of the project, replace PROJECT-ID below with the correct GCP project-id, and run the flask application
```python
gcloud run deploy --image gcr.io/<PROJECT-ID>/app --platform managed
```

Step 7: Paste the URL link provided on the console, in a preferred browser to run the application


## Set up Continuous Deployment (CD)
- Create a new build trigger
- Specify github repository
- Deployment specifications already available in: `cloudbuild.yaml` file
- Push a simple change; Triggered on Master branch
- View progress in [build triggers page](https://console.cloud.google.com/cloud-build/triggers)


## Load Testing

Website link with Continuous Delivery enabled.
https://final-project-311720.uc.r.appspot.com/

Loadtest code repo: https://github.com/YisongZou/IDS721-Finalproject-Locust-load-test
```Photo shows that the load test can handle uo to 1k+ requests per second```
![]https://github.com/YisongZou/IDS721-Finalproject-Locust-load-test/blob/main/IMG_1076.PNG
