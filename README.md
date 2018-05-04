# CS-412-HW5-Mini-project
My submission to the final project in CS-412 Intro to Machine Learning. The dataset used can be downloaded from [Young People Survey dataset](https://www.kaggle.com/miroslavsabo/young-people-survey/data)

To run the project, simply run "python hw5.py".

I have made the following library imports in my program:

import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.metrics import accuracy_score

Prior to running the project, please ensure that all the imported libraries have been installed.
Also make sure the csv file "responses.csv" is in the same directory as hw5.py.
