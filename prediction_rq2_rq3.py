import os
import json
import multiprocessing

import numpy as np
import pandas as pd

from krippendorff import krippendorff
from scipy.stats import kendalltau, spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score, matthews_corrcoef, \
    roc_auc_score, classification_report, mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import label_binarize



def evaluate(y_test, to_evaluate):
    p = precision_score(y_test, to_evaluate, average='weighted', labels=np.unique(to_evaluate))
    r = recall_score(y_test, to_evaluate, average='weighted', labels=np.unique(to_evaluate))
    f1 = f1_score(y_test, to_evaluate, average='weighted', labels=np.unique(to_evaluate))
    mcc = matthews_corrcoef(list(y_test), to_evaluate)
    labels = [0,1,2]
    try:
        auc_roc = roc_auc_score(label_binarize(list(y_test),classes=labels), label_binarize(to_evaluate,classes=labels), average='weighted', multi_class='ovr', labels=labels)
    except:
        auc_roc = roc_auc_score(list(y_test), to_evaluate, average='weighted', multi_class='ovr',
                                labels=np.unique(to_evaluate))
    alpha = krippendorff.alpha(reliability_data=[to_evaluate, list(y_test)])
    alpha_ordinal = krippendorff.alpha(reliability_data=[to_evaluate, list(y_test)],level_of_measurement='ordinal')
    kappa = cohen_kappa_score(to_evaluate, list(y_test))
    mse = mean_squared_error(to_evaluate, list(y_test))
    mae = mean_absolute_error(to_evaluate, list(y_test))
    r2 = r2_score(to_evaluate, list(y_test))
    spearman, spearmanp = spearmanr(to_evaluate, list(y_test))
    accuracy = accuracy_score(to_evaluate, list(y_test))
    tau, p_tau = kendalltau(to_evaluate, list(y_test))
    cr = classification_report(y_test, to_evaluate, labels=labels, output_dict=True)

    try:
        to_evaluate = to_evaluate.tolist()
    except:
        1
    return {
            'cr': cr,
            'alpha': alpha,
            'alpha_ordinal': alpha_ordinal,
            'kappa': kappa,
            'auc_roc': auc_roc,
            'y_test': list(y_test),
            'y_predict': to_evaluate,
            'p': p,
            'r': r,
            'f1': f1,
            'mcc': mcc,
            'tau': tau,
            'p_tau': p_tau,
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'spearman': spearman,
            'acc': accuracy
        }


def sklearn_classif():
    result_tuple = []

    for n_estimators in range(10,210,10):
        cls = RandomForestClassifier(random_state=88, n_estimators=n_estimators)
        result_tuple.append((str(n_estimators), cls))

    return result_tuple

def preprocess(smell, variation):

    base = './data/'

    df = pd.read_csv(base+ 'all_'+smell + '_merged_classname.csv').reset_index(drop=True)
    y_data = df['perception'].replace('NON-SEVERE', 0).replace('SEVERE', 2).replace('MEDIUM', 1)

    del df['class-name']
    del df['perception']
    full = list(df.columns)
    features = []

    features_selected_file = './results/fs/'+smell+'_'+variation.replace('AutoSpearman_','AS_')+'.json'
    with open(features_selected_file, 'r') as f2:
        features_selected_idx = json.load(f2)
        # features_selected_idx = f2.read()
        # features_selected_idx = features_selected_idx.replace('"','').replace('\n','').replace('    ','').replace('[','').replace(']','').split(',')
        # if len(features_selected_idx)<1: return None
        # if len(features_selected_idx)==1 and features_selected_idx[0] == '0': return None

    for idx in features_selected_idx:
        if idx == '': continue
        features.append(full[int(idx)-1])


    x_data = df[features].replace('?', 0).apply(pd.to_numeric)

    return x_data, y_data


def gen_smell(smell, as_variation, clf_tuple):

    (cls_name, cls) = clf_tuple

    result_path = './performance/rq3/raw/' + smell + '_' + as_variation + '_' + cls_name + '.json'
    if os.access(result_path,os.F_OK):
        return

    results = {'mean':{}}

    print('***' + smell + str(as_variation) + ' ' + cls_name + '***')

    x_data, y_data = preprocess(smell, as_variation)

    KF = LeaveOneOut()

    y_val_arr = []
    y_predict_arr = []

    for train_index, val_index in KF.split(x_data,y_data):

        x_train_, x_val = x_data.iloc[train_index], x_data.iloc[val_index]
        y_train_, y_val = y_data[train_index], y_data[val_index]

        y_val = int(y_val)

        cls.fit(x_train_, y_train_)

        y_pred = cls.predict(x_val)[0]


        y_val_arr.append(int(y_val))
        y_predict_arr.append(int(y_pred))

    res = evaluate(y_val_arr, y_predict_arr)

    results['mean'] = res

    with open(result_path, 'w') as f:
        json.dump(results, f, indent=4, sort_keys=True)

if __name__ == '__main__':

    res = {}
    smells = ['spaghetti-code','blob','shotgun-surgery','complex-class']
    params = []

    selections = [
        'AutoSpearman',
        'AutoSpearman_Additive',
        'AutoSpearman_Reductive'
    ]

    classifiers = sklearn_classif()

    for smell in smells:
        for as_variations in selections:
            for clf_tuple in classifiers:
                (cls_name, cls) = clf_tuple
                params.append((smell, as_variations, clf_tuple))
    print(len(params))

    p2 = multiprocessing.Pool(processes=7)
    with p2:
        res = p2.starmap_async(gen_smell, params)
        res.get()

