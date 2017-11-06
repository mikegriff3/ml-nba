import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV

import matplotlib.pylab as plt
from matplotlib.pylab import rcParams

train = pd.read_csv('./game_vectors_csv/11_2_2017.csv')
target = 'HW'

# X = np.array(df.drop(['HW', 'createdAt', 'updatedAt', 'HOME', 'VISITOR', 'L', 'PW', 'PL'], 1))
# y = np.array(df['HW'])

def modelfit(alg, train, predictors, performCV=True, printFeatureImportance=True, cv_folds=5):
  # Fit the algorithm on data
  alg.fit(train[predictors], train['HW'])

  # Predict training set
  predictions = alg.predict(train[predictors])
  predprob = alg.predict_proba(train[predictors])[:,1]

  # Perform cross-validation
  if performCV:
    cv_score = cross_validation.cross_val_score(alg, train[predictors], train['HW'], cv=cv_folds, scoring='roc_auc')

  # Print model report
  print "\nModel Report"
  print "Accuracy : %.4g" % metrics.accuracy_score(train['HW'].values, predictions)
  print "AUC Score (Train): %f" % metrics.roc_auc_score(train['HW'], predprob)

  if performCV:
    print "CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score), np.std(cv_score), np.min(cv_score), np.max(cv_score))  

  # Print Feature Importance
  if printFeatureImportance:
    feat_imp = pd.Series(alg.feature_importances_, predictors).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    plt.tight_layout()
    plt.show()

predictors = [x for x in train.columns if x not in [target, 'createdAt', 'updatedAt', 'HOME', 'VISITOR', 'L', 'PW', 'PL']]
gbm0 = GradientBoostingClassifier(random_state=10)
modelfit(gbm0, train, predictors)