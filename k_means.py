from __future__ import division
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd

df = pd.read_csv('./game_vectors_csv/11_2_2017.csv')

X = pd.DataFrame()
# X['FG'] = df['FG']
# X['FGA'] = df['FGA']
# # #X['FG_PCT'] = df['FG_PCT']
# X['Three_Pointers'] = df['Three_Pointers']
# X['Three_Pointers_Att'] = df['Three_Pointers_Att']
# # #X['Three_Pointers_Pct'] = df['Three_Pointers_Pct']
# # X['Two_Pointers'] = df['Two_Pointers']
# X['Two_Pointers_Att'] = df['Two_Pointers_Att']
# # #X['Two_Pointers_Pct'] = df['Two_Pointers_Pct']
# X['FTM'] = df['FTM']
# # X['FTA'] = df['FTA']
# # #X['FT_PCT'] = df['FT_PCT']
# # X['ORB'] = df['ORB']
# X['DRB'] = df['DRB']
# X['TRB'] = df['TRB']
# #X['AST'] = df['AST']
# #X['STL'] = df['STL']
# # X['BLK'] = df['BLK']
# #X['TOV'] = df['TOV']
# # X['PF'] = df['PF']
# #X['PTS'] = df['PTS']
# X['HW'] = df['HW']
# X['W'] = df['W']
# X['MOV'] = df['MOV']
# X['SOS'] = df['SOS']
# X['SRS'] = df['SRS']
# X['ORtg'] = df['ORtg']
# X['DRtg'] = df['DRtg']
# # X['PACE'] = df['PACE']
# # X['FTr'] = df['FTr']
# # X['Three_PAr'] = df['Three_PAr']
# # X['OFF_eFG_PCT'] = df['OFF_eFG_PCT']
# X['OFF_TOV_PCT'] = df['OFF_TOV_PCT']
# X['ORB_PCT'] = df['ORB_PCT']
# # X['OFF_FT_FGA'] = df['OFF_FT_FGA']
# # X['DEF_eFG_PCT'] = df['DEF_eFG_PCT']
# #X['DEF_TOV_PCT'] = df['DEF_TOV_PCT']
# X['DRB_PCT'] = df['DRB_PCT']
# # X['DEF_FT_FGA'] = df['DEF_FT_FGA']
# X['oPTS'] = df['oPTS']
# # X['oFG'] = df['oFG']
# # X['oFGA'] = df['oFGA']
# # X['oFGPCT'] = df['oFGPCT']
# # X['o3P'] = df['o3P']
# # X['o3PA'] = df['o3PA']
# X['o3PCT'] = df['o3PCT']
# # X['o2P'] = df['o2P']
# # X['o2PA'] = df['o2PA']
# # X['o2PCT'] = df['o2PCT']
# # X['oFTM'] = df['oFTM']
# # X['oFTA'] = df['oFTA']
# # X['oFTPCT'] = df['oFTPCT']
# X['oORB'] = df['oORB']
# X['oDRB'] = df['oDRB']
# # X['oTRB'] = df['oTRB']
# # X['oAST'] = df['oAST']
# # X['oSTL'] = df['oSTL']
# # X['oBLK'] = df['oBLK']
# # X['oTOV'] = df['oTOV']
# # X['oPF'] = df['oPF']

X = np.array(df.drop(['HW', 'createdAt', 'updatedAt', 'HOME', 'VISITOR', 'L', 'PW', 'PL'], 1).astype(float))
y = np.array(df['HW'])

clf = KMeans(n_clusters=2)
clf.fit(X)

correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    #print prediction[0], y[i]
    if prediction[0] == y[i]:
      correct += 1

length = len(X)
print correct, length
score = correct / length
print score