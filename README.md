# Phishing URL Detection 

# Most Challenging Problem Solved

## Problem
I worked on a project where I had to detect phishing websites using machine learning.  
The goal was to collect both phishing and normal URLs, extract useful features from them, and then train models to predict if a website is fake or real.

## Why It Was Challenging
It was difficult because phishing URLs are designed to look very similar to real ones, so extracting the right features was tricky.  
Also, comparing the performance of multiple ML and deep learning models required careful testing.

## My Approach
I gathered phishing and benign URLs, cleaned the data, and extracted URL-based and content-based features.  
Then I trained different machine learning and deep neural network models on this dataset.

## Final Solution
I compared the accuracy and performance of each model and identified which model worked best for predicting phishing websites.  
This helped me build a more reliable phishing detection system.

## What I Learned
I learned how feature extraction affects model accuracy and how to compare multiple ML models effectively.  
The project also improved my understanding of phishing attacks and security-focused ML systems

<img width="1913" height="973" alt="Screenshot 2025-10-13 230416" src="https://github.com/user-attachments/assets/5687a73f-44be-48f4-ac8b-e3ee4984297d" />

<img width="1915" height="971" alt="Screenshot 2025-10-13 230239" src="https://github.com/user-attachments/assets/83030bfa-d4ad-46e4-8d2f-8db94e8563c5" />



## Objective

A phishing website is a common social engineering method that mimics trustful uniform resource locators (URLs) and webpages. The objective of this project is to train machine learning models and deep neural nets on the dataset created to predict phishing websites. Both phishing and benign URLs of websites are gathered to form a dataset and from them required URL and website content-based features are extracted. The performance level of each model is measures and compared.

## Installation
To install the required packages and libraries, run this command in the project directory after Forking and cloning this repository:
```bash
pip install -r requirements.txt
```

## Technologies Used

![](https://forthebadge.com/images/badges/made-with-python.svg)

[<img target="_blank" src="https://upload.wikimedia.org/wikipedia/commons/3/31/NumPy_logo_2020.svg" width=200>](https://numpy.org/doc/) [<img target="_blank" src="https://upload.wikimedia.org/wikipedia/commons/e/ed/Pandas_logo.svg" width=200>](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html)
[<img target="_blank" src="https://upload.wikimedia.org/wikipedia/commons/8/84/Matplotlib_icon.svg" width=100>](https://matplotlib.org/)
[<img target="_blank" src="https://scikit-learn.org/stable/_static/scikit-learn-logo-small.png" width=200>](https://scikit-learn.org/stable/) 
[<img target="_blank" src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcScq-xocLctL07Jy0tpR_p9w0Q42_rK1aAkNfW6sm3ucjFKWML39aaJPgdhadyCnEiK7vw&usqp=CAU" width=200>](https://flask.palletsprojects.com/en/2.0.x/) 

## Feature Extraction
The system starts by retrieving URLs to be checked for phishing. These URLs can be collected from user input in the webpage created. Once the URLs are obtained, the system extracts relevant features from the web pages. These features are essential for training and evaluating the machine learning models. Various features were extracted from the URL database based on Domain, HTML and Address bar of the URLs. 

## Machine Learning Models

Various machine learning models are compared and The machine learning model with high accuracy is selected which predicts whether the URL is a phishing site or not. It provides a probability score or a binary classification (phishing or not phishing) based on the trained model's decision boundary. The system categorize URLs into "phishing" or "legitimate" and the result is finally displayed on the webpage. 
#### Refer Phishingproject.ipynb for more details.

## Result

Accuracy of various model used for URL detection
<br>

<br>

||ML Model|	Accuracy|  	f1_score|	Recall|	Precision|
|---|---|---|---|---|---|
0|	Gradient Boosting Classifier|	0.974|	0.977|	0.994|	0.986|
1|	CatBoost Classifier|	        0.972|	0.975|	0.994|	0.989|
2|	Multi-layer Perceptron|	        0.969|	0.973|	0.995|	0.981|
3|	Random Forest|	                0.967|	0.971|	0.993|	0.990|
4|	Support Vector Machine|	        0.964|	0.968|	0.980|	0.965|
5|	Decision Tree|      	        0.960|	0.964|	0.991|	0.993|
6|	K-Nearest Neighbors|        	0.956|	0.961|	0.991|	0.989|
7|	Logistic Regression|        	0.934|	0.941|	0.943|	0.927|
8|	Naive Bayes Classifier|     	0.605|	0.454|	0.292|	0.997|

## Conclusion
1. The final take away form this project is to explore various machine learning models, perform Exploratory Data Analysis on phishing dataset and understanding their features. 
2. Creating this notebook helped me to learn a lot about the features affecting the models to detect whether URL is safe or not, also I came to know how to tuned model and how they affect the model performance.
3. The final conclusion on the Phishing dataset is that the some feature like "HTTTPS", "AnchorURL", "WebsiteTraffic" have more importance to classify URL is phishing URL or not.
4. Gradient Boosting Classifier currectly classify URL upto 97.4% respective classes and hence reduces the chance of malicious attachments.


