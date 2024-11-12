import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
dataset = pd.read_csv("/content/LoanApproval.csv")
dataset.head()
dataset.shape
dataset.info()
dataset.describe()
pd.crosstab(dataset['Credit_History'], dataset['Loan_Status'], margins=True)
dataset['CoapplicantIncome'].hist(bins=20)
dataset.boxplot(column='LoanAmount')
dataset['LoanAmountlog'] = np.log(dataset['LoanAmount'])
dataset['LoanAmountlog'].hist(bins=20)
dataset.isnull().sum()
dataset['Gender'].fillna(dataset['Gender'].mode()[0], inplace=True)
dataset['Married'].fillna(dataset['Married'].mode()[0], inplace=True)
dataset['Dependents'].fillna(dataset['Dependents'].mode()[0], inplace=True)
dataset['Self_Employed'].fillna(dataset['Self_Employed'].mode()[0], inplace=True)
dataset['LoanAmount'] = dataset['LoanAmount'].fillna(dataset['LoanAmount'].mean())
dataset['LoanAmountlog'] = dataset['LoanAmountlog'].fillna(dataset['LoanAmountlog'].mean())
dataset['Loan_Amount_Term'].fillna(dataset['Loan_Amount_Term'].mode()[0], inplace=True)
dataset['Credit_History'].fillna(dataset['Credit_History'].mode()[0], inplace=True)
dataset.isnull().sum()
dataset['TotalIncome'] = dataset['ApplicantIncome'] + dataset['CoapplicantIncome']
dataset['TotalIncomelog'] = np.log(dataset['TotalIncome'])
dataset['TotalIncomelog'].hist(bins=20)
dataset.head()

X = dataset.iloc[:, np.r_[1:5, 9:11, 13:15]].values
y = dataset.iloc[:, 12].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(X_train)

from sklearn.preprocessing import LabelEncoder
label_encoder_X = LabelEncoder()
for i in range(0, 5):
    X_train[:, i] = label_encoder_X.fit_transform(X_train[:, i])
X_train[:, 7] = label_encoder_X.fit_transform(X_train[:, 7])
X_train

label_encoder_y = LabelEncoder()
y_train = label_encoder_y.fit_transform(y_train)
y_train

for i in range(0, 5):
    X_test[:, i] = label_encoder_X.fit_transform(X_test[:, i])
X_test[:, 7] = label_encoder_X.fit_transform(X_test[:, 7])
label_encoder_y = LabelEncoder()
y_test = label_encoder_y.fit_transform(y_test)
y_test

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)

from sklearn.tree import DecisionTreeClassifier
DTClassifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
DTClassifier.fit(X_train, y_train)
DecisionTreeClassifier(criterion='entropy', random_state=0)
y_pred = DTClassifier.predict(X_test)
y_pred

from sklearn import metrics
print('The accuracy of the decision tree is:', metrics.accuracy_score(y_pred, y_test))

from sklearn.naive_bayes import GaussianNB
NBClassifier = GaussianNB()
NBClassifier.fit(X_train, y_train)
GaussianNB()
y_pred = NBClassifier.predict(X_test)
y_pred

print('The accuracy of Naive Bayes is:', metrics.accuracy_score(y_pred, y_test))

test_data = pd.read_csv("/content/LoanApproval.csv")
test_data.head()
test_data.isnull().sum()
test_data['Gender'].fillna(test_data['Gender'].mode()[0], inplace=True)
test_data['Dependents'].fillna(test_data['Dependents'].mode()[0], inplace=True)
test_data['Self_Employed'].fillna(test_data['Self_Employed'].mode()[0], inplace=True)
test_data['Loan_Amount_Term'].fillna(test_data['Loan_Amount_Term'].mode()[0], inplace=True)
test_data['Credit_History'].fillna(test_data['Credit_History'].mode()[0], inplace=True)
test_data.isnull().sum()
test_data.boxplot(column='LoanAmount')

plt.title('Loan Amount in Rupees')  # Adding this line to set the title

test_data['LoanAmount'] = test_data['LoanAmount'].fillna(test_data['LoanAmount'].mean())
test_data['LoanAmountlog'] = np.log(test_data['LoanAmount'])
test_data.isnull().sum()
test_data['TotalIncome'] = test_data['ApplicantIncome'] + test_data['CoapplicantIncome']
test_data['TotalIncomelog'] = np.log(test_data['TotalIncome'])
test_data.head()

test = test_data.iloc[:, np.r_[1:5, 9:11, 13:15]].values

for i in range(0, 5):
    test[:, i] = label_encoder_X.fit_transform(test[:, i])
test[:, 7] = label_encoder_X.fit_transform(test[:, 7])
test = ss.fit_transform(test)

pred = NBClassifier.predict(test)
pred