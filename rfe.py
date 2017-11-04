import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('./game_vectors_csv/11_2_2017.csv')

print df.shape

df.drop(['createdAt', 'updatedAt', 'HOME', 'VISITOR', 'L', 'PW', 'PL'], 1, inplace=True)

print df.shape

X = np.array(df.drop(['HW'], 1))
y = np.array(df['HW'])

model = LogisticRegression()
rfe = RFE(model, 10)
fit = rfe.fit(X, y)

print "Num Features: %d" % fit.n_features_
print "Selected Features: %s" % fit.support_
print "Feature Ranking: %s" % fit.ranking_