# author: Shobhit Lamba
# e-mail: slamba4@uic.edu

# Importing the libraries
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

warnings.filterwarnings(action = 'ignore', category = DeprecationWarning)

# Part 1: Data preprocessing

data = pd.read_csv("responses.csv")
data = data[np.isfinite(data['Empathy'])]
data = data.fillna(data.mean()) 

data = data.select_dtypes(exclude = [object])

y = data['Empathy'].values
X = data.drop('Empathy', axis = 1)

# Part 2: Training and Validation

def ensemble(X_train, X_test, y_train, y_test):
    # Multinomial Naive Bayes
    clf1 = MultinomialNB()
    clf1 = clf1.fit(X_train, y_train)
    clf1 = BaggingClassifier(clf1)
    
    # Multi-layer Perceptron
    clf2 = MLPClassifier(solver = 'lbfgs', alpha = 1e-2,
                         hidden_layer_sizes = (25, 20), random_state = 1)
    clf2 = clf2.fit(X_train, y_train) 
    clf2 = BaggingClassifier(clf2)
    
    # SVC
    clf3 = SVC(C = 1.5)
    clf3 = clf3.fit(X_train, y_train)
    clf3 = BaggingClassifier(clf3)
    
    # Decision tree
    clf4 = tree.DecisionTreeClassifier(criterion = 'entropy', splitter = 'best',
                                       max_depth = 2, min_samples_split = 2, 
                                       min_samples_leaf = 1, min_weight_fraction_leaf = 0.0, 
                                       max_features = None, random_state = None, max_leaf_nodes = None, 
                                       min_impurity_decrease = 0.0, min_impurity_split = None)
    clf4 = clf4.fit(X_train, y_train)
    clf4 = BaggingClassifier(clf4)
    
    # Random Forest
    clf5 = RandomForestClassifier(max_depth = 5, random_state = 0)
    clf5 = clf5.fit(X_train, y_train)   
    
    # KNN
    clf6 = KNeighborsClassifier(n_neighbors = 3)
    clf6 = clf6.fit(X_train, y_train)
    clf6 = BaggingClassifier(clf6)
    
    # Gaussian Naive Bayes
    clf7 = GaussianNB()
    clf7 = clf7.fit(X_train, y_train)
    clf7 = BaggingClassifier(clf7)
    
    # AdaBoost Classifier
    clf8 = AdaBoostClassifier(n_estimators = 100)
    clf8 = clf8.fit(X_train, y_train)
    
    # Logistic Regression
    clf9 = LogisticRegression(random_state = 1)
    clf9 = clf9.fit(X_train, y_train)
    clf9 = BaggingClassifier(clf9)
    
    # Soft Vote ensemble
    clf = VotingClassifier(estimators = [
            ('mnb', clf1), ('mlp', clf2), ('svc', clf3), ('dt', clf4), ('rf', clf5), ('knn', clf6), ('gnb', clf7), ('ab', clf8), ('lr', clf9)],
           voting = 'hard')
    clf = clf.fit(X_train, y_train)
    
    return(accuracy_score(y_test, clf.predict(X_test)))
    
def runValidation(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.2, random_state = 1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state = 1)
    ensembled = ensemble(X_train, X_val, y_train, y_val)
    
    return ensembled

accuracy = []

for i in range(1,20): #Taking steps of 5
    X_new = SelectKBest(chi2, k = i * 5).fit_transform(X, y)
    accuracy.append(runValidation(X_new, y))
      
best_accuracy_loc = accuracy.index(max(accuracy)) + 1 
print("Best value of k:", best_accuracy_loc * 5)
print("Validation accuracy for best k:", max(accuracy))
  

# Part 3: Testing
X_new = SelectKBest(chi2, k = best_accuracy_loc * 5).fit_transform(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.2, random_state = 1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state = 1)
ensembled_accuracy = ensemble(X_train, X_test, y_train, y_test)                                                                                                                                                                                                                 
print("Test Accuracy for model:", ensembled_accuracy)

