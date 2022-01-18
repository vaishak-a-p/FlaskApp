# importing necessary libraries
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

adv_data = pd.read_csv('../Dataset/Social_Network_Ads.csv')

# separating independent and dependent features
X = adv_data.drop(['User ID', 'Purchased'], axis=1)
X = pd.get_dummies(X, drop_first=True)
y = adv_data['Purchased']

# scaling the data
standardisation = preprocessing.StandardScaler()
scaler = standardisation.fit(X)
X = scaler.transform(X)

# pickling scaler
pickle.dump(scaler, open('scaler.pkl', 'wb'))

# splitting into train and test dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.2)

# creating logistic regression model
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

# pickling the logistic regression model
pickle.dump(lr_model, open('logistic_regression_model.pkl', 'wb'))
