import numpy as np #for numeric calculation
import pandas as pd #for data anaysis
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import matplotlib.pyplot as plt  #for data visualization 
import seaborn as sns  #for data visualization
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.metrics import precision_recall_curve 
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve, auc


df = pd.read_csv("diabetes_data_upload.csv")
print(df.head())


le = preprocessing.LabelEncoder()
df = df.apply(le.fit_transform)

x=df.drop('class',axis=1)
y=df['class']
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.20,random_state=42)

model=GaussianNB();
model.fit(x_train,y_train);
y_pred=model.predict(x_test)
pred=le.inverse_transform(y_pred)
print('\nPredcited class labels')
print(pred)
#print(pred.size)
'''
n=0
for x in np.nditer(y_test):
  if x==1:
    n=n+1
print('n')
print(n)'''

accuracy=accuracy_score(y_test,y_pred)*100
print('\nAccuracy')
print(accuracy)

m=confusion_matrix(y_test,y_pred)
sns.heatmap(confusion_matrix(y_test,y_pred),cmap='rainbow',annot=True,fmt="d")                                                       
#cmap value shows color for plot can be 'coolwarm', 'inferno', 'Blues','BuPu','Greens' etc.
plt.title('Confusion matrix for gaussian naive bayes classifier')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()



print('Classification report')
print(classification_report(y_test,y_pred))


precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
# create plot
plt.plot(precision, recall, label='Precision-recall curve for Gaussian Naive bayes classifier')
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.title('Precision-recall curve for Gaussian Naive bayes classifier')
plt.legend(loc="lower left")
plt.show()


a=average_precision_score(y_test, y_pred)
print('Average Precision score')
print(a)


fpr, tpr,thresholds = roc_curve(y_test, y_pred)
ab=auc(fpr,tpr)
print ('area under roc curve')
print (ab)

def plot_ROC(y_test, y_score, n_classes=2):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr['positive'], tpr['positive'], _ = roc_curve(y_test, y_score)
    roc_auc['positive'] = auc(fpr['positive'], tpr['positive'])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    plt.figure()
    lw = 2
    plt.plot(fpr['positive'], tpr['positive'], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc['positive'])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve for Gaussian Naive bayes classifier')
    plt.legend(loc="lower right")
    plt.show()
plot_ROC(y_test, y_pred)
