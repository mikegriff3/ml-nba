import pandas as pd
import numpy as np

df = pd.read_csv('./game_vectors_csv/11_2_2017.csv')

print df.columns

#X = np.array(df.drop(['HW', 'createdAt', 'updatedAt', 'HOME', 'VISITOR', 'L', 'PW', 'PL'], 1))
X = np.array(df[['MOV', 'W', 'SRS', 'ORtg', 'DEF_eFG_PCT', 'DRtg', 'OFF_eFG_PCT', 'oFG']])
y = np.array(df['HW'])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Fit only to the training data
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(1000,1000,1000), learning_rate='adaptive',max_iter=1000, early_stopping=True, solver='sgd')

mlp.fit(X_train,y_train)

predictions = mlp.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))

print(classification_report(y_test,predictions))
