import json

import pandas as pd
import shap
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import minmax_scale


def preprocess(smell, variation):

    base = './data/'

    df = pd.read_csv(base+ 'all_'+smell + '_merged_classname.csv').reset_index(drop=True)
    y_data = df['perception'].replace('NON-SEVERE', 0).replace('SEVERE', 2).replace('MEDIUM', 1)
    classnames = df['class-name']
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

    return x_data, y_data, classnames


def gen_alias_name(labels):

    mapping={"InfoGainAttributeEval":'IG',
             "SymmetricalUncertAttributeEval":'Symm',
             "_Ranker":'',
             "GainRatioAttributeEval":'Gain',
             "ChiSquaredAttributeEval":'ChiSq',
             "SignificanceAttributeEval":'Sig',
             "OneRAttributeEval":'OneR',
             "ReliefFAttributeEval":'Relief',
             "SVMAttributeEval":'SVM',
             "CorrelationAttributeEval":'Corr',
             "BestFirst":'BF',
             "WrapperSubsetEval":'Wrap',
             'AutoSpearman_MERIT_Chi2_None':'AS_Chi2',
             'AutoSpearman_MERIT_Gini_None':'AS_Gini',
             'AutoSpearman_MERIT_Infogain_None':'AS_IG',
              'AutoSpearman_MERIT_Merit_None':'AS_Merit',
             'AutoSpearman_MERIT_ReliefF_None':'AS_ReliefF',
             'AutoSpearman_None':'AutoSpearman',
              'AutoSpearman_WRAP_DT_None':'AS_Wrap_DT',
             'AutoSpearman_WRAP_KNN_None':'AS_Wrap_KNN',
             'AutoSpearman_WRAP_LR_None':'AS_Wrap_LR',
              'AutoSpearman_WRAP_NB_None':'AS_Wrap_NB',
             'AutoSpearman_WRAP_RF_None':'AS_Wrap_RF',
             'AutoSpearman_WRAP_SVM_None':'AS_Wrap_SVM',
              "bayes.NaiveBayes":'NB',
             "GreedyStepwise":'GS',
             "functions.Logistic":'Log',
             "CfsSubsetEval":'CFS',
             "ConsistencySubsetEval":'Consist',
             "lazy.IBk": 'KNN',
             "rules.JRip":'JRip',
             "AutoSpearman_None":'AutoSpearman',
             "number_public_visibility_methods": 'NPVM_type',
             "persistance": 'Pers.',
             "Avg-commit-size": 'AVG_CS',
             "avg.dev.scattering": 'DSC',
             "number-changes": 'NC',
             "ref": 'Ref.',
             "intensity": 'Intensity',

             "number-committors": 'NCOM',
             "number-fixes": 'NF',
             "Readability": 'Read.',
             'number_constructor_DefaultConstructor_methods': 'numb_constructor_DC'
             }

    res = []
    for label in labels:
        for m in mapping.keys():
            label = label.replace(m,mapping[m])
        label = label.replace('_type','').replace('DefaultConstructor_methods','DC')
        label = label.replace('number_','num_').replace('attributes','attr').replace('DefaultConstructor','DC')
        res.append(str(label))
    return res


def extract(smell, classname, extract_predicted):
    setname = 'all'
    if smell == 'shotgun-surgery':
        filename = 'reductive'
        nmd = 11+1
    elif smell == 'spaghetti-code':
        filename = 'additive'
        nmd = 3+1
    elif smell == 'complex-class':
        filename = 'reductive'
        nmd = 5+1
    else:
        filename = 'reductive'
        nmd = 10+1

    data = pd.read_csv('./results/settings/'+filename+'.csv')
    row = data[data['smell'] == smell][data['setname'] == setname][data['merge'] == True].iloc[0]
    as_variation = row['evaluator']
    param = row['param']
    cls = RandomForestClassifier(random_state=88, n_estimators=param)


    x_data, y_data, classnames = preprocess(smell, as_variation)


    y_predict_arr = []
    y_true_arr = []

    i = list(classnames).index(classname)
    train_index = []
    test_index = []
    for j in range(0, len(x_data)):
        if j != i:
            train_index.append(j)
        else:
            test_index.append(j)

    x_train, x_test = x_data.iloc[train_index], x_data.iloc[test_index]
    y_train, y_test = y_data.iloc[train_index], y_data.iloc[test_index]

    cls.fit(x_train, y_train)
    y_predict = cls.predict(x_test)
    y_predict_arr.append(int(y_predict[0]))
    y_true_arr.append(int(y_test.values[0]))

    explainer = shap.TreeExplainer(cls)
    shap_values = explainer(x_test)

    if extract_predicted:
        classes = int(y_predict[0])
    else:
        classes = 0 if y_true_arr[0] == y_predict[0] else int(y_true_arr[0])
    row = 0
    features = gen_alias_name(x_train.columns.values.tolist())
    # shap will throw an error when displaying boolean values, source code should be modified to fix it
    #  the statement
    #                 yticklabels[rng[i]] = features[order[i]] + " = " + feature_names[order[i]]
    #  should be changed to:
    #                 yticklabels[rng[i]] = str(features[order[i]]) + " = " + str(feature_names[order[i]])
    plt = shap.waterfall_plot(shap.Explanation(values=
                                               shap_values.values[:, :, classes][row],
                                               base_values=explainer.expected_value[classes],
                                               data=x_test.values[0, :],
                                               feature_names=features),
                              max_display=nmd, show=False)

    plt.show()
    plt.savefig('./results/figs/discussion/'+smell+'_'+classname+'_'+str(extract_predicted)+'.pdf', bbox_inches='tight')


extract('blob','org.apache.cxf.tools.corba.processors.idl.IDLLexer',extract_predicted=True)
extract('complex-class','org.eclipse.cdt.debug.mi.core.command.CommandFactory',extract_predicted=True)
extract('complex-class','org.eclipse.cdt.debug.mi.core.command.CommandFactory',extract_predicted=False)
extract('blob', 'org.apache.pig.test.utils.FILTERFROMFILE',extract_predicted=False)
extract('blob', 'org.apache.pig.test.pigmix.mapreduce.L3',extract_predicted=False)