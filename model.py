import pandas as pd 
import numpy as np
import pickle



dataset = pd.read_csv("glass_data.csv")
print(dataset.head())
print(dataset.dtypes)


X = dataset[["Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]]
y = dataset["Type"]

# Splitting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)

# Training the model
from sklearn.ensemble import RandomForestClassifier
cls = RandomForestClassifier(
    criterion='entropy',
    n_estimators=300,
    random_state=42
)
cls.fit(X_train, y_train)

pickle.dump(cls, open("model.pkl", "wb"))
# Prediction
y_pred = cls.predict(X_test)

# Accuracy
print('ACCURACY is:', cls.score(X_test, y_test) * 100, '%')
