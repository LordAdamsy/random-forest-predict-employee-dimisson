import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from time import sleep
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.metrics import precision_recall_curve

#import data
df = pd.read_csv('Modified_HR_dataset.csv')

#extract x and y
x = df.values[:, :-1]
y = df.values[:, -1]

#encode categorical data
labelencoder_x = LabelEncoder()
x = np.array([labelencoder_x.fit_transform(x[:, i]) for i in range(x.shape[1])]).T

'''
#box plot
fig,axes = plt.subplots(3, 3, figsize=(7, 10))
axes[0, 0].boxplot(x[:, 0])
axes[0, 0].set_title('satisfaction_level', size = 10)
axes[0, 1].boxplot(x[:, 1])
axes[0, 1].set_title('last_evaluation', size = 10)
axes[0, 2].boxplot(x[:, 2])
axes[0, 2].set_title('number_project', size = 10)
axes[1, 0].boxplot(x[:, 3])
axes[1, 0].set_title('average_montly_hours', size = 10)
axes[1, 1].boxplot(x[:, 4])
axes[1, 1].set_title('time_spend_company', size = 10)
axes[1, 2].boxplot(x[:, 5])
axes[1, 2].set_title('Work_accident', size = 10)
axes[2, 0].boxplot(x[:, 6])
axes[2, 0].set_title('promotion_last_5years', size = 10)
axes[2, 1].boxplot(x[:, 7])
axes[2, 1].set_title('sales', size = 10)
axes[2, 2].boxplot(x[:, 8])
axes[2, 2].set_title('salary', size = 10)
fig.tight_layout()
plt.show()
'''

#normalization and standardization
x = (x - x.mean(axis=0)) / x.std(axis=0)

#random forest class
class random_forest():
    def __init__(self, tree_num = 100, max_depth = 2, sample_num = 500, random_seed = 0):
        self.num = tree_num
        self.sample = sample_num
        self.depth = max_depth
        self.seed = random_seed
        self.trees = []
        self.idx = []
        self.rd = np.random.RandomState(self.seed)

    def fit(self, x, y):
        tbar = tqdm(range(self.num))
        idx = []
        for _ in tbar:
            tbar.set_description('Training Sub Decision Tree into Random Forest')
            tree = DTC(max_depth = self.depth, random_state = self.seed, max_features = 'sqrt')
            sample_index = self.rd.randint(0, x.shape[0], self.sample)
            tree.fit(x[sample_index], y[sample_index])
            self.trees.append(tree)
            idx.append(sample_index)
        self.idx = np.array([i for i in range(x.shape[0]) if i not in np.unique(np.array(idx))])

    def predict(self, x):
        predictions = np.zeros((x.shape[0], self.num))
        for i, tree in enumerate(self.trees):
            predictions[:, i] = tree.predict(x)
        return np.mean(predictions, axis=1)

if __name__ == '__main__':
    #split train and test data without data balance method
    rf = random_forest()
    rf.fit(x, y.astype(int))
    oob_idx = rf.idx
    y_pred = rf.predict(x[oob_idx, :])
    print('Oob-accuracy without data balance method: ', np.mean(np.round(y_pred) == y[oob_idx])) 
    precisions, recalls, _ = precision_recall_curve(y[oob_idx].astype(int), y_pred)
    plt.plot(precisions, recalls)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()

    #split train and test data with data balance method
    loop = 10
    acc = []
    for i in range(loop):    
        #split train and test data
        idx_zero = np.where(y == 0)[0]
        idx_one = np.array([i for i  in range(x.shape[0]) if i not in idx_zero])
        np.random.shuffle(idx_one)
        idx_zero = idx_zero[:idx_one.shape[0]]
        x_split_zero, x_split_one, y_split_zero, y_split_one = x[idx_zero], x[idx_one], y[idx_zero], y[idx_one]
        train_zero_x, test_zero_x, train_zero_y, test_zero_y= train_test_split(x_split_zero, y_split_zero, test_size=0.2)
        train_one_x, test_one_x, train_one_y, test_one_y= train_test_split(x_split_one, y_split_one, test_size=0.2)
        x1, y1 = np.concatenate((train_zero_x, train_one_x, test_zero_x, test_one_x), axis=0), np.concatenate((train_zero_y, train_one_y, test_zero_y, test_one_y), axis=0)

        #training
        rf = random_forest()
        rf.fit(x1, y1.astype(int))
        oob_idx = rf.idx
        y_pred = rf.predict(x1[oob_idx, :])
        print('Oob-accuracy with data balance method for ' + str(i) + '-th model: ', np.mean(np.round(y_pred) == y1[oob_idx]))
        acc.append(np.mean(np.round(y_pred) == y1[oob_idx]))
    precisions, recalls, _ = precision_recall_curve(y1[oob_idx].astype(int), y_pred)
    print('Oob-accuracy with data balance method: ', np.mean(acc))
    plt.plot(precisions, recalls)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()