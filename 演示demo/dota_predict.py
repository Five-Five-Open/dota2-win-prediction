# coding: utf-8

import pandas as pd
import numpy as np
import os
from pandas import Series, DataFrame
np.set_printoptions(suppress=True)
import random
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.metrics import accuracy_score

##################################################################################
# Load data
######################################## raw data file, change the path for your own !!!!##############################################
dota_data = pd.read_csv (os.getcwd() + "/new.csv")
dota_data.info ()

# Show win rate for Radiant and Dire
import matplotlib.pyplot as plt

fig = plt.figure ()
fig.set (alpha=0.2)

plt.figure (figsize=(5, 5))
# Pie chart
plt.pie (dota_data.radiant_win.value_counts (),
         autopct='%.2f%%',
         explode=[0.2, 0],
         startangle=90,
         labels=('Radiant', 'Dire'))
plt.title (u"Win Rate for Radiant and Dire")  # Title
plt.show ()

# Board rate and win rate for each hero
columns = ['radiant_1', 'radiant_2', 'radiant_3', 'radiant_4', 'radiant_5', 'dire_1', 'dire_2', 'dire_3', 'dire_4',
           'dire_5']
win = np.zeros (115)
lose = np.zeros (115)

# For radiant
for i in range (5):
    # group each radiant position by win rate
    a = dota_data.groupby ([columns[i], 'radiant_win'])
    # the win number and lose number for each radiant positon
    s = DataFrame (a.size ())
    # rearrange 'a' to be a matrix format (radiant positive versus win or lose)
    s_sum = s.unstack ()
    z = s_sum[0]

    # A Revision here
    m, x = np.shape (z)
    for j in range (m):
        if np.isnan ((z[0].values[j - 1])):
            continue
        n = (z[0].values[j - 1])
        lose[z.index[j - 1] - 1] = lose[z.index[j - 1] - 1] + n

    for j in range (m):
        if np.isnan ((z[1].values[j - 1])):
            continue
        n = (z[1].values[j - 1])
        win[z.index[j - 1] - 1] = win[z.index[j - 1] - 1] + n

# for dire
for i in range (5, 10):
    # group each dire position by win rate
    a = dota_data.groupby ([columns[i], 'radiant_win'])
    # the win number and lose number for each dire position
    s = DataFrame (a.size ())
    s_sum = s.unstack ()
    z = s_sum[0]

    # A Revison here
    m, x = np.shape (z)
    for j in range (m):
        if np.isnan ((z[0].values[j])):
            continue
        n = (z[0].values[j])
        lose[z.index[j - 1] - 1] = lose[z.index[j - 1] - 1] + n

    for j in range (m):
        if np.isnan ((z[1].values[j])):
            continue
        n = (z[1].values[j])
        win[z.index[j - 1] - 1] = win[z.index[j - 1] - 1] + n

board = Series (win + lose)
win_rate = Series (win / board)
##################################### need revision here!!!!!!!!, 21961 is sanmple instances number ###################################
board_rate = board / 21961

list = [win_rate, board_rate]
hero_rate = pd.concat (list, axis=1, )
######################################## it will generated automatically which list the win rate for each heros#########
######################################## change the path for your own !!!!##############################################
#hero_rate.to_csv('/hero_rate.csv')
np.savetxt('hero_rate.csv', hero_rate, delimiter=',',fmt='%f')

# preprocessing for raw data
def _dataset_to_features(dataset_df):
    # Initial an empty x matrix, column number is defined as twice of heros number, row number is instances number of sample
    print(dataset_df)
    x_matrix = np.zeros ((dataset_df.shape[0], 2 * 120))

    # Initial an empty y matrix, row number is instances number of sample
    y_matrix = np.zeros (dataset_df.shape[0])

    # Transfer raw data to a Numpy array
    dataset_np = dataset_df.values

    # Map each hero to one hot matrix of x for two side
    for i, row in enumerate (dataset_np):
        radiant_win = row[10]
        for j in range (5):
            # map the radiant
            x_matrix[i, row[j] - 1] = 1
            # map the dire
            x_matrix[i, row[j + 5] - 1] = 1
        # map the win rate for Radiant
        y_matrix[i] = 1 if radiant_win else 0

    return [x_matrix, y_matrix]


# drop the first column (index) of default indexs
dota_data.drop ('Unnamed: 0', axis=1, inplace=True)
#print(dota_data)
train_data = _dataset_to_features (dota_data)


x, y = train_data


x_train, x_test, y_train, y_test = train_test_split (
    x, y, test_size=0.30, random_state=42)

# Data Nomalizaton
scaler = StandardScaler ()
#print("x_train:"+x_train.shape)
scaler.fit (x_train)
x_train = scaler.transform (x_train)
#print(x_test.shape)
x_test = scaler.transform (x_test)
#print(x_test.shape)
cv = 3
model = LogisticRegression (C=0.005, random_state=42)
cv_scores = cross_val_score (model, x_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
cv_score = np.mean (cv_scores)

model = LogisticRegression (C=0.005, random_state=42)
model.fit (x_train, y_train)
#print("x_test:")

probabilities = model.predict_proba (x_test)
lr_prodict = model.predict (x_test)
print(lr_prodict)
#######################################################!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Output interface ##############################
# 4 instance of x (namely, 4 matches)is involed in this result
# print out the lose rate and win rate
#print("probabilities:")
#print (probabilities)
# print out the game result predition (win or lose)
#print("lr_prodict")
#print (lr_prodict)
#########################################################################################################################################
lr_accuracy_score = accuracy_score (y_test, lr_prodict)

mean_tpr = 0.0
mean_fpr = np.linspace (0, 1, 100)
all_tpr = []

fpr, tpr, thresholds = roc_curve (y_test, probabilities[:, 1])
mean_tpr += interp (mean_fpr, fpr, tpr)
mean_tpr[0] = 0.0
roc_auc = auc (fpr, tpr)
plt.plot (fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
plt.plot ([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
plt.xlabel ('False Positive Rate')
plt.ylabel ('True Positive Rate')
plt.title ('Receiver operating characteristic example')
plt.legend (loc="lower right")
plt.show ()
