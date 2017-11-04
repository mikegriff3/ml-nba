import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

df = pd.read_csv('./game_vectors_csv/11_2_2017.csv')

df.drop(['createdAt', 'updatedAt', 'HOME', 'VISITOR', 'L', 'PW', 'PL'], 1, inplace=True)

X = np.array(df.drop(['HW'], 1))
y = np.array(df['HW'])

print df.columns

# feature extraction
model = ExtraTreesClassifier()
model.fit(X, y)
#print model.feature_importances_

for i in range(len(model.feature_importances_)):
  print round(model.feature_importances_[i], 3) * 10, df.columns[i + 1] 

   