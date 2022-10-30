import json
import os
import pickle
from multiprocessing import Pool, freeze_support

import numpy as np
import pandas as pd
import shap
from krippendorff import krippendorff
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import label_binarize

from AutoSpearman import AutoSpearman
from figure_rq1 import process_labels


def evaluate(y_test, to_evaluate):
    f1 = f1_score(y_test, to_evaluate, average='weighted', labels=np.unique(to_evaluate))
    mcc = matthews_corrcoef(list(y_test), to_evaluate)
    labels = [0,1,2]
    try:
        auc_roc = roc_auc_score(label_binarize(list(y_test),classes=labels), label_binarize(to_evaluate,classes=labels), average='weighted', multi_class='ovr', labels=labels)
    except:
        auc_roc = roc_auc_score(list(y_test), to_evaluate, average='weighted', multi_class='ovr',
                                labels=np.unique(to_evaluate))
    alpha_ordinal = krippendorff.alpha(reliability_data=[to_evaluate, list(y_test)],level_of_measurement='ordinal')
    try:
        to_evaluate = to_evaluate.tolist()
    except:
        1
    return {
            'alpha': alpha_ordinal,
            'auc_roc': auc_roc,
            'y_test': list(y_test),
            'y_predict': to_evaluate,
            'f1': f1,
            'mcc': mcc,
        }

def predict(smell, remove, n_estimators):
    prefixes = {
        'cassandra': ['org.apache.cassandra'],
        'cayenne': ['org.apache.cayenne'],
        'cxf': ['org.apache.cxf'],
        'jena': ['org.apache.jena', 'jena.'],
        'solr-lucene': ['org.apache.lucene', 'org.apache.solr', 'org.apache.tools.bzip2r', 'org.tartarus.snowball.ext'],
        'pig': ['org.apache.pig'],
        'cdt': ['org.eclipse.cdt'],
        'jackrabbit': ['org.apache.jackrabbit'],
        'mahout': ['org.apache.mahout']
    }
    dataset='All'

    performance_path = './performance/rq1/pk_res_' + smell + '_' + dataset + '_' + str(remove) +'_'+str(n_estimators)+ '_fake.txt'
    if os.access(performance_path, os.F_OK):
        return
    print('predicting ' + smell + ' ' + str(remove))
    explanations = []
    data = pd.read_csv('./data/fake/'+ smell+'.csv')

    y = data['perception'].replace('NON-SEVERE', 0).replace('SEVERE', 2).replace('MEDIUM', 1)
    class_names = data['class-name']
    prj = None
    del data['perception']
    del data['class-name']

    features = list(data.columns)

    if remove:
        for m in 'NOCS_project,NOI_project,NOM_project,NOMNAMM_project,LOC_project,NOPK_project'.split(','):
            print(m)
            try:
                features.remove(m)
            except:
                continue
    X = data[features].replace('?', 0).replace(False,0).replace(True,1).apply(pd.to_numeric)
    columns_original = list(X.columns)

    X = AutoSpearman(X)
    cols = list(X.columns)
    res = []
    for i in cols:
        res.append(columns_original.index(i))
    print(smell)
    print(res)

    LOOCV = LeaveOneOut()
    y_val_arr = []
    y_predict_arr = []
    i = 0
    for train_index, val_index in LOOCV.split(X, y):
        i += 1
        # print(i)
        class_name = class_names[val_index].iloc[0]
        for k, v in prefixes.items():
            for vv in v:
                if class_name.startswith(vv):
                    prj = k
                    break
        explain_dict = {'smell': smell, 'dataset': dataset, 'remove': remove, 'project': prj, 'class-name': class_name}

        x_train_, x_val = X.iloc[train_index], X.iloc[val_index]
        y_train_, y_val = y[train_index], y[val_index]

        y_val = int(y_val)
        explain_dict['real'] = y_val
        clf = RandomForestClassifier(random_state=88, n_estimators=n_estimators)
        clf.fit(x_train_, y_train_)

        y_pred = clf.predict(x_val)[0]
        explain_dict['predict'] = y_pred
        explain_dict['x_val'] = x_val.values[0, :]
        explain_dict['cols'] = cols

        explainer = shap.TreeExplainer(clf)
        shap_values = explainer(x_val)
        explain_dict['shap_values'] = shap_values

        y_val_arr.append(y_val)
        y_predict_arr.append(int(y_pred))
        explanations.append(explain_dict)
    performance = evaluate(y_val_arr,y_predict_arr)
    with open(performance_path, 'w') as f2:
        json.dump(performance, f2)
    with open('./pk/pk_res_' + smell + '_' + dataset + '_' + str(remove) +'_'+str(n_estimators)+ '_fake.pk', 'wb') as f1:
        pickle.dump(explanations, f1)
    return y_val_arr, y_predict_arr, explanations

def extend(dataset):
    data = {
        'is_controller': ['manage','process','control','ctrl','command','cmd','process','proc','ui', 'drive', 'system','subsystem','parser','service'],
        'is_procedural': ['make','create','factory','exec','compute','display','view','calculate','batch','thread','cluster'],
        'is_test': ['test','junit'],
        'is_util': ['util','preprocess'],
        'is_external':  ['org.tartarus.snowball.ext','org.apache.tools.bzip2r']
    }
    for k, v in data.items():
        df = pd.DataFrame()
        for idx, row in dataset.iterrows():
            row[k] = False
            for term in v:
                if term in row['class-name'].lower():
                    row[k] = True
                    break
            df = df.append(row, ignore_index=True)
        dataset = df.copy()
    return dataset

def fake(smell):
    data = pd.read_csv('./data/all_' + smell + '_merged_classname.csv')

    for m in 'NOCS_project,NOI_project,NOM_project,NOMNAMM_project,LOC_project,NOPK_project'.split(','):
        unique_values = np.unique(data[m])
        for iuv, uv in enumerate(unique_values):
            data[m] = data[m].replace(uv, iuv)

    data.to_csv('./data/fake/' + smell + '.csv', index=False)


def RQ1_fig(smell,n_estimators):
    values = None
    x_vals = None
    dataset = 'All'
    base_values = None
    with open('./pk/pk_res_' + smell + '_' + dataset + '_False_'+str(n_estimators) + '_fake.pk', 'rb') as f1:
        explanations = pickle.load(f1)
    for explanation in explanations:
        if values is None:
            values = [explanation['shap_values'].values[:, :, 0],
                      explanation['shap_values'].values[:, :, 1],
                      explanation['shap_values'].values[:, :, 2]]
            x_vals = explanation['x_val']
            base_values = [explanation['shap_values'].base_values[:, 0],
                           explanation['shap_values'].base_values[:, 1],
                           explanation['shap_values'].base_values[:, 2]]
        else:
            for idx in [0, 1, 2]:
                values[idx] = np.vstack((values[idx], explanation['shap_values'].values[:, :, idx]))
                base_values[idx] = np.vstack(
                    (base_values[idx], explanation['shap_values'].base_values[:, idx]))
            x_vals = np.vstack((x_vals, explanation['x_val']))

    shap.summary_plot(values, features=x_vals, plot_type="bar", class_inds=[2, 1, 0],
                      color=get_cmap("tab20c"), feature_names=process_labels(explanation),
                      class_names=['NON-SEVERE', 'MEDIUM', 'SEVERE'], max_display=10, show=False)
    plt.xlabel("", fontsize=13)
    plt.savefig('./results/figs/rq1_' + smell + '_fake.pdf',
                bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    fake('blob')
    fake('complex-class')
    predict('blob',False,260)
    predict('blob',False,170)
    RQ1_fig('blob', 260)
    RQ1_fig('complex-class', 170)