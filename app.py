import numpy as np
from flask import Flask, request, jsonify, render_template
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os.path
from os import path
 
import pickle
 
app = Flask(__name__)
 
@app.route('/')
def home():  
    if str(path.exists("data.pkl")) == "False":
        df = pd.read_csv('iris.csv')
        df.to_pickle("data.pkl")
    else:
        df = pd.read_pickle("data.pkl")
    rows=len(df)
    col=len(df.columns)
    return render_template('index.html',info="Dataset has {r} rows, {c} columns".format(r=rows,c=col))
 
@app.route('/add',methods=['POST'])
def add():
    if request.method == 'POST':
        row=request.form['noc']
        return render_template('add.html',rows=int(row))
    
@app.route('/append',methods=['POST'])
def append():
    rows=None
    col=None
    if request.method == 'POST':
        for x in range(len(request.form.getlist('petal_length[]'))):
            sample_data = {}
            sample_data['petal_length'] = request.form.getlist('petal_length[]')[x]
            sample_data['sepal_length'] = request.form.getlist('sepal_length[]')[x]
            sample_data['petal_width'] = request.form.getlist('petal_width[]')[x]
            sample_data['sepal_width'] = request.form.getlist('sepal_width[]')[x]
            sample_data['species'] = request.form.getlist('species[]')[x]
            df = pd.read_pickle("data.pkl")
            df = df.append(sample_data,ignore_index=True)
            df.to_pickle("data.pkl")
            rows=len(df)
            col=len(df.columns)
    return render_template('index.html',info="Dataset has {r} rows, {c} columns".format(r=rows,c=col),response="Data Successfully added to table")
    
@app.route('/train',methods=['POST'])
def train():
    df = pd.read_pickle("data.pkl")
    rows=len(df)
    model_choice = request.form['model_choice']
    x = df.iloc[:,:-1].values
    y = df.iloc[:,-1].values
    if model_choice == 'decisiontree':
        from sklearn import tree
        classifier=tree.DecisionTreeClassifier()
        classifier.fit(x,y)
        pickle.dump(classifier, open('dtree.pkl','wb'))
        with open('dtree.pkl', 'rb') as f:
            data = pickle.load(f)
    elif model_choice == 'knnmodel':
        from sklearn import neighbors
        classifier=neighbors.KNeighborsClassifier()
        classifier.fit(x,y)
        pickle.dump(classifier, open('knn.pkl','wb'))
        with open('knn.pkl', 'rb') as f:
            data = pickle.load(f)
    return render_template('index.html',info="Dataset has {} rows, 5 columns".format(rows),response="Training successful and stored in localstorage",data=data,model_choice=model_choice)
 
@app.route('/predict',methods=['POST'])
def predict():
    df = pd.read_pickle("data.pkl")
    rows=len(df)
    model_choice = request.form['model_choice']
    features = [request.form['sepal_len'],request.form['sepal_wid'],request.form['petal_len'],request.form['petal_wid']]
    final_features = [np.array(features)]
    if model_choice == 'decisiontree':
        if str(path.exists("dtree.pkl")) == "False":
            return render_template('index.html',info="Dataset has {} rows, 5 columns".format(rows),prediction_text='Please train decision tree model before testing')
        model = pickle.load(open('dtree.pkl', 'rb'))
    elif model_choice == 'knnmodel':
        if str(path.exists("knn.pkl")) == "False":
            return render_template('index.html',info="Dataset has {} rows, 5 columns".format(rows),prediction_text='Please train knn model before testing')
        model = pickle.load(open('knn.pkl', 'rb'))
            
    prediction = model.predict(final_features)
 
    return render_template('index.html',info="Dataset has {} rows, 5 columns".format(rows), features='Given [ Sepal_length, Sepal_width, Petal_length, Petal_width ]:{}'.format(features),prediction_text='Predicted Species:{}'.format(prediction),response="Prediction Successful")
       
if __name__ == "__main__":
    app.run(debug=True)
    