import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

df = pd.read_csv('./game_vectors_csv/11_2_2017.csv')

X = pd.DataFrame()
X['FG'] = df['FG']
X['FGA'] = df['FGA']
# #X['FG_PCT'] = df['FG_PCT']
X['Three_Pointers'] = df['Three_Pointers']
X['Three_Pointers_Att'] = df['Three_Pointers_Att']
# #X['Three_Pointers_Pct'] = df['Three_Pointers_Pct']
# X['Two_Pointers'] = df['Two_Pointers']
X['Two_Pointers_Att'] = df['Two_Pointers_Att']
# #X['Two_Pointers_Pct'] = df['Two_Pointers_Pct']
X['FTM'] = df['FTM']
# X['FTA'] = df['FTA']
# #X['FT_PCT'] = df['FT_PCT']
# X['ORB'] = df['ORB']
X['DRB'] = df['DRB']
X['TRB'] = df['TRB']
#X['AST'] = df['AST']
#X['STL'] = df['STL']
# X['BLK'] = df['BLK']
#X['TOV'] = df['TOV']
# X['PF'] = df['PF']
#X['PTS'] = df['PTS']
X['HW'] = df['HW']
X['W'] = df['W']
X['MOV'] = df['MOV']
X['SOS'] = df['SOS']
X['SRS'] = df['SRS']
X['ORtg'] = df['ORtg']
X['DRtg'] = df['DRtg']
# X['PACE'] = df['PACE']
# X['FTr'] = df['FTr']
# X['Three_PAr'] = df['Three_PAr']
# X['OFF_eFG_PCT'] = df['OFF_eFG_PCT']
X['OFF_TOV_PCT'] = df['OFF_TOV_PCT']
X['ORB_PCT'] = df['ORB_PCT']
# X['OFF_FT_FGA'] = df['OFF_FT_FGA']
# X['DEF_eFG_PCT'] = df['DEF_eFG_PCT']
#X['DEF_TOV_PCT'] = df['DEF_TOV_PCT']
X['DRB_PCT'] = df['DRB_PCT']
# X['DEF_FT_FGA'] = df['DEF_FT_FGA']
X['oPTS'] = df['oPTS']
# X['oFG'] = df['oFG']
# X['oFGA'] = df['oFGA']
# X['oFGPCT'] = df['oFGPCT']
# X['o3P'] = df['o3P']
# X['o3PA'] = df['o3PA']
X['o3PCT'] = df['o3PCT']
# X['o2P'] = df['o2P']
# X['o2PA'] = df['o2PA']
# X['o2PCT'] = df['o2PCT']
# X['oFTM'] = df['oFTM']
# X['oFTA'] = df['oFTA']
# X['oFTPCT'] = df['oFTPCT']
X['oORB'] = df['oORB']
X['oDRB'] = df['oDRB']
# X['oTRB'] = df['oTRB']
# X['oAST'] = df['oAST']
# X['oSTL'] = df['oSTL']
# X['oBLK'] = df['oBLK']
# X['oTOV'] = df['oTOV']
# X['oPF'] = df['oPF']

y = X['HW']
X = X.drop(['HW'], axis=1)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print accuracy
print y_test

# example_measures = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,1,1,2,3,2,1]])
# example_measures = example_measures.reshape(len(example_measures), -1)
# prediction = clf.predict(example_measures)
# print(prediction)