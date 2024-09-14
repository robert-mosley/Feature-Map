from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.datasets import load_iris
import pandas as pd
import seaborn as sns
import numpy as np
import sys

print("Commands:")
print("Feature Importances: shows a bar graph of the most important features")
print("Univariate Selection: Uses statistical test to determine which features have the strongest correlation with output variable")
print("Heat Map: shows a correlation heat map of all the variables and to what degree they influence each other")
print("")

while True:
    try:
        file_name = sys.argv[1]
    except:
        file_name = input("Enter File: ")

    try:
        data = pd.read_csv(file_name)
        data = data.dropna()
        data = pd.get_dummies(data)
        break
    except Exception as e:
        print(e)
        pass

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

def findfeature_importances(feat_num):
    model = ExtraTreesClassifier()
    model.fit(X,y)
    feat_imp = pd.Series(model.feature_importances_, index=X.columns)
    feat_imp.nlargest(feat_num).plot(kind="barh")
    plt.show()

def Univariate_Selection(feat_num):
    best_score = SelectKBest(score_func=chi2, k=5)
    x = np.abs(X)
    fit = best_score.fit(x,y)
    score = pd.DataFrame(fit.scores_)
    columns = pd.DataFrame(X.columns)
    scores = pd.concat([columns, score], axis=1)
    scores.columns = ["feats", "scores"]
    print(scores.nlargest(feat_num, "scores"))

def HeatMap(feat_num):
    correlation = data.corr().nlargest(feat_num, data.columns)
    top_corr = correlation.index
    plt.figure(figsize=(20,20))
    g=sns.heatmap(data[top_corr].corr(), annot=True, cmap="RdYlGn")
    plt.show()



while True:
    cmd = input("Feature_Analyse>> ")
    if cmd == "Feature Importances":
        cmd = input("How many features would you like to be shown>> ")
        findfeature_importances(int(cmd))
    if cmd == "Univariate Selection":
        cmd = input("How many features would you like to be shown>> ")
        Univariate_Selection(int(cmd))
    if cmd == "Heat Map":
        cmd = input("How many features would you like to be shown>> ")
        HeatMap(int(cmd))
