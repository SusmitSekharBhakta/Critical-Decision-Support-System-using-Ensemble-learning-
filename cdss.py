
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")
# load the dataset
df = pd.read_csv("PatientData.csv")
del df['Index']
df.replace(np.nan,0)
# split the data into features and target
X = df.iloc[0:,:-1].values
y = df.iloc[:,-1].values
np.nan_to_num(X, copy=False)
np.nan_to_num(y, copy=False)
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.20,random_state=0)
#print("X_test")
#print(X_test)
#  the logistic regression model
logisticReg = LogisticRegression(random_state=95,max_iter=95)
#ensemble = LogisticRegression(random_state=95,max_iter=95)
#  the decision tree model
#decisionTrClf = DecisionTreeClassifier(random_state=90)
#ensemble = DecisionTreeClassifier(random_state=90)
# the SVM model
#SVM = SVC(kernel='rbf', probability=True, random_state=100)
#ensemble = SVC(kernel='rbf', probability=True, random_state=100)
# the NB clf model
#nbClf = GaussianNB()
ensemble = GaussianNB()
#  ensemble model using all four models

'''ensemble = VotingClassifier(estimators=[('lr', logisticReg),
                                        ('dt', decisionTrClf),
                                        ('svm', SVM),
                                        ('nb', nbClf)],
                            voting='soft')'''

#ensemble = VotingClassifier(estimators=[('dt', decisionTrClf),
#                                       ('lr', logisticReg)],
#                            voting='soft')

'''ensemble = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(random_state=95,max_iter=95))
 '''
'''ensemble = BaggingClassifier(base_estimator = decisionTrClf, n_estimators = 200, random_state = 90)    #ensemble bagging'''

'''ensemble = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, random_state=100, max_features=5 )     #ensemble gradient boosting'''

'''ensemble = AdaBoostClassifier(DecisionTreeClassifier(random_state=90), n_estimators=200) #ensemble adaboost'''
#ensemble = StackingClassifier(estimators=[('dt', decisionTrClf), ('lr', logisticReg)]) #ensemble Stacking
#ensemble = GaussianNB()
# fit the ensemble model to the data
ensemble.fit(X_train, y_train)

# evaluate the model using k-fold cross-validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(ensemble, X_test, y_test, cv=5)
y_pred = ensemble.predict(X_test)
# Calculate precision and recall
precision = metrics.precision_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)
print("Precision:", precision)
print("Recall:", recall)
#print("\nClassification report:\n", classification_report(y_test, y_pred))
print("\nF1 score:", f1_score(y_test, y_pred))
print("\nAccuracy score:", accuracy_score(y_test, y_pred))
import pickle
filename = 'file.sav'
pickle.dump(ensemble, open(filename, 'wb'))
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)
print("Loaded model")
print(result)
################################################################################################################
# code for visualization
from tabulate import tabulate
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import FileLink
import csv

# Example evaluation metrics data
models = ['bagging(DT)', 'gradientboosting', 'AdaBoost(DT)', 'Stacking(dt+LR)', 'Voting(lr+dt+svm+nb)', 'Voting(lr+dt)']
accuracy = [0.9964698960580506, 0.995881545401059, 0.9966660129437145, 0.9964698960580506, 0.985095116689547, 0.9964698960580506]
precision = [0.9907251264755481,0.9865884325230512, 0.9915611814345991, 0.98989898989899, 0.9623745819397993, 0.98989898989899]
recall = [0.994077834179357, 0.9957698815566836, 0.994077834179357, 0.9949238578680203, 0.9737732656514383, 0.9949238578680203]
f1_score = [0.9923986486486487, 0.9911578947368421, 0.9928179129700041, 0.9924050632911392, 0.968040370058873, 0.9924050632911392]

# Combine the data into a list of lists
data = [models, accuracy, precision, recall, f1_score]

# Transpose the data so that each model's metrics are in a row
data = list(map(list, zip(*data)))

# Print the table using tabulate
headers = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
print(tabulate(data, headers=headers, tablefmt='orgtbl'))
# Write table to CSV file
with open('table.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])
    writer.writerows(data)

# Make table downloadable
display(FileLink('table.csv'))

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plot the accuracy, precision, recall, and f1-score as bars in the first subplot
bar_width = 0.2
x = np.arange(len(models))

ax1.bar(x - 1.5*bar_width, accuracy, bar_width, label='Accuracy')
ax1.bar(x - 0.5*bar_width, precision, bar_width, label='Precision')
ax1.bar(x + 0.5*bar_width, recall, bar_width, label='Recall')
ax1.bar(x + 1.5*bar_width, f1_score, bar_width, label='F1-Score')
ax1.set_xticks(x)
ax1.set_xticklabels(models, rotation=45, ha='right')
ax1.set_xlabel('Models')
ax1.set_ylabel('Metric Scores')
ax1.set_title('Evaluation Metrics for Different Models')
ax1.legend()

# Plot the accuracy, precision, recall, and f1-score as lines in the second subplot
ax2.plot(models, accuracy, label='Accuracy')
ax2.plot(models, precision, label='Precision')
ax2.plot(models, recall, label='Recall')
ax2.plot(models, f1_score, label='F1-Score')
ax2.set_xticklabels(models, rotation=45, ha='right')
ax2.set_xlabel('Models')
ax2.set_ylabel('Metric Scores')
ax2.set_title('Evaluation Metrics for Different Models')
ax2.legend()

# Adjust the layout and display the figure
fig.tight_layout()
plt.show()
