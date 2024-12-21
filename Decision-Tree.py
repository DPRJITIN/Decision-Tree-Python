    #Step 1: Import Necessary Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

    #Step 2: Load and Prepare the Dataset
from sklearn.datasets import load_iris
#Load the dataset
iris = load_iris()
X = iris.data
y = iris.target
#Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    #Step 3: Create and Train the Decision Tree Model
#Initialise the Decision Tree Classifier
dt = DecisionTreeClassifier(random_state=42)
#Train the model
dt.fit(X_train, y_train)

    #Step 4: Make Predictions and Evaluate the Model
#Make predictions
y_pred = dt.predict(X_test)
#Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
#Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))
#Print confusion matrix
print("Confusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)
#Visualise the confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

    #Step 5: Visualise the Decision Tree
#Visualise the decision tree
plt.figure(figsize=(20,10))
plot_tree(dt, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title("Decision Tree Visualisation")
plt.show()