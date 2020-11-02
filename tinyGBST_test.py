from tinyGBST import GBST, GBSTDataset
import tinyGBST
import time
import numpy as np
import pandas as pd


def opencsv():
    print('Load Data...')
    data1 = pd.read_csv(
        'bmj_gbst_on_loan_sample_apart_xdata_selected_xy_train.csv')
    data2 = pd.read_csv(
        'bmj_gbst_on_loan_sample_apart_xdata_selected_xy_test.csv')
    data1 = data1.drop(['appl_no', 'cust_no', 'gapdays',
                        'day', 'second_label'], axis=1)
    data2 = data2.drop(['appl_no', 'cust_no', 'gapdays',
                        'day', 'second_label'], axis=1)
    train_features = data1.values[0:, 0:-1]
    train_ys = []
    test_features = data2.values[0:, 0:-1]
    test_ys = []
    for y in data1.values[0:, -1]:
        train_ys.append(
            [-1] + ([-1 if value == 0 else 1 for value in sorted(eval(y).values())]))
    for y in data2.values[0:, -1]:
        test_ys.append(
            [-1] + ([-1 if value == 0 else 1 for value in sorted(eval(y).values())]))
    train_ys = np.array(train_ys)
    test_ys = np.array(test_ys)
    return train_features, train_ys, test_features, test_ys


def trainModel(features, ys, test_features, test_ys):
    print('Training ...')
    dataset = GBSTDataset(features, ys)
    testset = GBSTDataset(test_features, test_ys)
    params = {
        "lambda": 1,
        "learning_rate": 0.5,
        "max_depth": 5,
        "eps": 0.26}
    model = GBST()
    model.train(
        params,
        dataset,
        testset,
        num_boost_round=20,
        early_stopping_rounds=10)
    return


def loadModel():
    train_features, train_ys, test_features, test_ys = opencsv()
    trainset = GBSTDataset(train_features, train_ys)
    testset = GBSTDataset(test_features, test_ys)
    model = GBST()
    model.models = tinyGBST.load_model()
    model.best_iteration = len(model.models) - 2
    print("Auc on trainset:", model.calc_auc(trainset))
    print("Auc on testset:", model.calc_auc(testset))
    return


def survivalTree():
    sta_time = time.time()
    train_features, train_ys, test_features, test_ys = opencsv()
    print("load data finish\n")
    trainModel(train_features, train_ys, test_features, test_ys)
    print("\nfinish!")
    end_time = time.time()
    times = end_time - sta_time
    print("\n运行时间: %ss == %sm == %sh\n\n" %
          (times, times / 60, times / 60 / 60))


if __name__ == '__main__':
    survivalTree()
    loadModel()
