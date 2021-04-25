# Flask-salary-predictor
This is project can help you predict the salary based on your years of experience.

The dataset is the 
```
Kaggle Years of experience and Salary dataset
```
https://www.kaggle.com/rohankayan/years-of-experience-and-salary-dataset
![pic](https://github.com/YisongZou/IDS721-Final-Project/blob/main/Screen%20Shot%202021-04-22%20at%201.42.12%20AM.png)
# Model
model.py trains and saves the model to the disk.
model.pkl is the pickle model 

# App
app.py has the main function and contains all the required functions for the flask app.



Procedure--
Setup
```
make all
```
Train:
```
python3 model.py
```
Run app
```
python3 main.py
```
Website link with continuous delivery enabled.
https://final-project-311720.uc.r.appspot.com/

Loadtest code repo: https://github.com/YisongZou/IDS721-Finalproject-Locust-load-test
