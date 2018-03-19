import numpy as np
import os
# import random

def mk_train_test(train_num, seed = 30):
    np.random.seed(seed)
    DIR = "C://Users//korea//Desktop//project//sungmo//feature_30"
    list_0 = os.listdir(DIR + "//0")
    list_1 = os.listdir(DIR + "//1")

    list_0_index = np.arange(len(list_0))
    list_1_index = np.arange(len(list_1))

#Make Train0_list
    train_0_index = np.random.choice(list_0_index, train_num, replace = False)
    train_0_index = sorted(train_0_index)
    train_0_list = []
    for i in train_0_index :
       train_0_list.append(list_0[i])

#Make Train1_list
    train_1_index = np.random.choice(list_1_index, train_num, replace = False)
    train_1_index = sorted(train_1_index)
    train_1_list = []
    for i in train_1_index :
       train_1_list.append(list_1[i])

#Make Test0_list
    test_0_index = np.delete(list_0_index, train_0_index)
    test_0_list = []
    for i in test_0_index :
       test_0_list.append(list_0[i])

#Make Test1_list
    test_1_index = np.delete(list_1_index, train_1_index)
    test_1_list = []
    for i in test_1_index :
       test_1_list.append(list_1[i])

    return train_0_list , train_1_list , test_0_list, test_1_list
