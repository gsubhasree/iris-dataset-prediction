# ML model deployment with Iris Dataset

https://prediction-on-iris-dataset.herokuapp.com/

## Requirements:
	Python(version 3.7) IDE (Anaconda recommended)
	FLASK 
	Gunicorn
	Libraries: numpy, pandas, seaborn, scikit-learn, matplotlib, pickle

## Installation & Setup:
	After installing packages in requirements and setting up virtual env,
	run this command in the directory containing code:
		python app.py
	After executing the command above, visit http://localhost:5000/ in your browser to see your app

## Problem Statement:
	To train and deploy ML classification algorithms on IRIS Dataset. 
	Algorithms used here are Logistic Regression,Decision Tree, K Nearest Neighbours, Support Vector Machine & Random Forest Classifier.

	The deployed website has the following provisions:
		Add new data over the given dataset: 
			User can input data consisting of sepal length,sepal width, petal length,petal width, and species(setosa, versicolor, virginica). 
		Train the current dataset on model of user's choice(from the 5) and retain the model
		Test the current model: 
			The species is predicted by the trained model of user's choice with the input parameters:sepal length,sepal width, petal length and petal width.
		View the dataset

## Approach:
	Exploratory Data Analysis is done on the given dataset (In main.ipynb file)
	It is found that the dataset does not have null/NaN values
	The current dataset and the models built are retained by generation of .pkl files and stored in localstorage.
	The built web app is then deployed to Heroku

## Acknowledgements:
### Installation:
Anaconda: https://docs.anaconda.com/anaconda/install/

### Resources:
ML scikit-learn classification models:
https://stackabuse.com/overview-of-classification-methods-in-python-with-scikit-learn/

Integrating ML models with flask: 
https://www.analyticsvidhya.com/blog/2020/09/integrating-machine-learning-into-web-applications-with-flask/

### Deploy to heroku:
https://hidenobu-tokuda.com/how-to-build-a-hello-world-web-application-using-flask-and-deploy-it-to-heroku/
https://stackabuse.com/deploying-a-flask-application-to-heroku/
	
	
	
		

