import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

df = pd.read_csv('./game_vectors_csv/11_2_2017.csv')

# X = pd.DataFrame()
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

# HW will be dependent variable 
X = np.array(df.drop(['HW', 'createdAt', 'updatedAt', 'HOME', 'VISITOR', 'L', 'PW', 'PL'], 1).astype(float))
y = np.array(df['HW'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#print X_train[0]

n_nodes_hl1 = 1000
n_nodes_hl2 = 1000
n_nodes_hl3 = 1000

n_classes = 2
batch_size = 100
hm_epochs = 10

x = tf.placeholder('float')
y = tf.placeholder('float')

def neural_network_model(data):
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([len(X_train[0]), n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes])),}


    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3,output_layer['weights']), output_layer['biases']

    return output

def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            i=0
            while i < len(X_train):
              start = i
              end = i+batch_size
              batch_x = np.array(X_train[start:end])
              batch_y = np.array(y_train[start:end])

              _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                            y: batch_y})
              epoch_loss += c
              i+=batch_size

            print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:X_test, y:y_test}))

train_neural_network(x)