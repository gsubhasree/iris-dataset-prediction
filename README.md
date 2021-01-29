# ML model deployment with Iris Dataset

https://iris-flower-class-predictor.herokuapp.com/

## Requirements:
	Python IDE (Anaconda recommended)
	FLASK 
	Gunicorn
	Libraries: numpy, pandas, seaborn, scikit-learn, pickle

## Installation & Setup:
	After installing packages in requirements and setting up virtual env,
	run this command in the directory containing code:
		python app.py
	After executing the command above, visit http://localhost:5000/ in your browser to see your app

## Problem Statement:
	To train and deploy 2 ML classification algorithms:-Decision Tree and K Nearest Neighbours, on the Iris Dataset

	The deployed website has the following provisions:
		Add new data over the given dataset: 
			User can input data consisting of sepal length,sepal width, petal length,petal width, and species(setosa, versicolor, virginica). 
		Train the current dataset on model of user's choice(from the 2) and retain the model
		Test the current model: 
			The species is predicted by the trained model of user's choice with the input parameters:sepal length,sepal width, petal length and petal width.

## Approach:
	Exploratory Data Analysis is done on the given dataset
	It is found that the dataset does not have null/NaN values
	Dataset is then splitted into train and test sets,classification models: decision tree, Support Vector Machine and K Nearest Neighbours are applied and corresponding accuracies of the models are calculated
	Decision Tree and KNN are found to perform better and hence are chosen for deployment
	The current dataset and the models built are retained by generation of .pkl files
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
	
	
	
		

