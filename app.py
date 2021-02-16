import numpy as np
from flask import Flask, request, render_template
import matplotlib.pyplot as plt
import pandas as pd
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

@app.route('/view',methods=['POST'])
def view():
    if request.method == 'POST':
        df = pd.read_pickle("data.pkl")
        return render_template('view.html',df_view=df.to_html())
 
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
        fig = plt.figure(figsize=(25,20))
        tree.plot_tree(classifier, feature_names=["sepal_length","sepal_width","petal_length","petal_width"]  ,class_names=["setosa","versicolor","virginica"],filled=True)
        fig.savefig("static/images/treeimg.jpg")
        return render_template('index.html',info="Dataset has {} rows, 5 columns".format(rows),response="Decisiontree model trained and stored in localstorage",data=data,model_choice=model_choice)
    elif model_choice == 'KNN':
        from sklearn import neighbors
        classifier=neighbors.KNeighborsClassifier()
        classifier.fit(x,y)
        pickle.dump(classifier, open('knn.pkl','wb'))
        with open('knn.pkl', 'rb') as f:
            data = pickle.load(f)
        return render_template('index.html',info="Dataset has {} rows, 5 columns".format(rows),response="K-NN model trained and stored in localstorage",data=data,model_choice=model_choice)
    elif model_choice == 'SVM':
        from sklearn import svm
        classifier= svm.SVC(gamma='scale')
        classifier.fit(x,y)
        pickle.dump(classifier, open('svm.pkl','wb'))
        with open('svm.pkl', 'rb') as f:
            data = pickle.load(f)
        return render_template('index.html',info="Dataset has {} rows, 5 columns".format(rows),response="SVM model trained and stored in localstorage",data=data,model_choice=model_choice)
    elif model_choice == 'LogisticRegression':
        from sklearn.linear_model import LogisticRegression
        classifier = LogisticRegression(multi_class='auto',solver='lbfgs')
        classifier.fit(x, y)
        pickle.dump(classifier, open('lr.pkl','wb'))
        with open('lr.pkl', 'rb') as f:
            data = pickle.load(f)
        return render_template('index.html',info="Dataset has {} rows, 5 columns".format(rows),response="Logistic Regression model trained and stored in localstorage",data=data,model_choice=model_choice)
    elif model_choice == 'RandomForest':
        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier()
        classifier.fit(x,y)
        pickle.dump(classifier, open('rf.pkl','wb'))
        with open('rf.pkl', 'rb') as f:
            data = pickle.load(f)
        return render_template('index.html',info="Dataset has {} rows, 5 columns".format(rows),response="Random Forest model trained and stored in localstorage",data=data,model_choice=model_choice)
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
    elif model_choice == 'KNN':
        if str(path.exists("knn.pkl")) == "False":
            return render_template('index.html',info="Dataset has {} rows, 5 columns".format(rows),prediction_text='Please train knn model before testing')
        model = pickle.load(open('knn.pkl', 'rb'))
    elif model_choice == 'SVM':
        if str(path.exists("svm.pkl")) == "False":
            return render_template('index.html',info="Dataset has {} rows, 5 columns".format(rows),prediction_text='Please train SVM model before testing')
        model = pickle.load(open('svm.pkl', 'rb'))
    elif model_choice == 'LogisticRegression':
        if str(path.exists("lr.pkl")) == "False":
            return render_template('index.html',info="Dataset has {} rows, 5 columns".format(rows),prediction_text='Please train Logistic Regression model before testing')
        model = pickle.load(open('lr.pkl', 'rb'))  
        final_features = [np.array(features).astype(float)]
    elif model_choice == 'RandomForest':
        if str(path.exists("rf.pkl")) == "False":
            return render_template('index.html',info="Dataset has {} rows, 5 columns".format(rows),prediction_text='Please train Random Forest model before testing')
        model = pickle.load(open('rf.pkl', 'rb'))    
    prediction = model.predict(final_features)
 
    return render_template('index.html',info="Dataset has {} rows, 5 columns".format(rows), features='Given [ Sepal_length, Sepal_width, Petal_length, Petal_width ]:{}'.format(features),prediction_text='{}'.format(prediction),response="Prediction ({}) Successful".format(model_choice),model_choice=model_choice)
       
if __name__ == "__main__":
    app.run(debug=True)
    