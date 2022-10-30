import glob
import json
import multiprocessing
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from numpy import mean
from scipy.stats import entropy
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import recall_score
from sklearn.model_selection import LeaveOneOut

fold = 10


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



def explain(clf,values,x_train_cols):
    res = []
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer(values)
    real_x = []
    shap_values.values = shap_values.values[:, :, 1]
    shap_values.base_values = shap_values.base_values[:, 1]
    shap_values.feature_names = x_train_cols
    # shap.force_plot(explainer.expected_value[0], shap_values.values[0, :], x_test.values[0, :], matplotlib=True)
    # shap.plots._waterfall.waterfall_legacy(explainer.expected_value[0], shap_values.values.T.sum(1))
    # shap.summary_plot(shap_values.values,features=values)
    real_values = []
    for id, exp in enumerate(shap_values.values):
        real_values.append(exp)
        real_x.append(values.iloc[id])
        ttttmp = {}
        for i,name in enumerate(x_train_cols):
            ttttmp[name]=abs(exp[i])
        tmpres = {key: value for key, value in sorted(ttttmp.items(), key=lambda item: item[1])}
        res.append(tmpres)
    return explainer.expected_value, real_values, res, real_x

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
             'number_constructor_DefaultConstructor_methods': 'number_constructor_methods'
             }

    res = []
    for label in labels:
        for m in mapping.keys():
            label = label.replace(m,mapping[m])
        res.append(label)
    return res

def gen_smell_loocv(smell, as_variation, clf_tuple):
    (cls, param) = clf_tuple

    result_path_explain = './pk/explain/res_' + smell + '_' + as_variation.replace('AutoSpearman_','AS_')+'.pk'

    if os.access(result_path_explain, os.F_OK):
        return

    x_data, y_data, classname = preprocess(smell, as_variation)

    i = 0
    explain_result_dict = {
        'gain':None,
        'shap':[]
    }
    loo = LeaveOneOut()
    y_predict_arr = []
    x_test_arr = []
    y_true_arr = []

    for train_index, test_index in loo.split(x_data,y_data):
        i += 1
        print('***' + smell +'***' + str(i))
        x_train, x_test = x_data.iloc[train_index], x_data.iloc[test_index]
        y_train, y_test = y_data.iloc[train_index], y_data.iloc[test_index]
        cls.fit(x_train, y_train)
        y_predict = cls.predict(x_test)
        y_predict_arr.append(int(y_predict[0]))
        y_true_arr.append(int(y_test.values[0]))

        x_test_arr.extend(x_test.values.tolist())

        explain_result = {}
        expected_value, shap_values, shap, real_x = explain(cls, x_test, x_train.columns.values.tolist(),
                                                            )
        explain_result['shap'] = shap
        explain_result['shap_values'] = shap_values
        explain_result['expected_value'] = expected_value
        explain_result['shap_features'] = real_x
        explain_result_dict['shap'].append(explain_result)

    explain_result_dict['gain'] = dict(
        sorted(
            zip(x_data.columns.values.tolist(),
                mutual_info_classif(x_test_arr, y_predict_arr, discrete_features=False, random_state=88)
                )
            , key=lambda x: -x[1]))
    explain_result_dict['predict'] = y_predict_arr
    explain_result_dict['true'] = y_true_arr

    with open(result_path_explain, 'wb') as f11:
        pickle.dump(explain_result_dict, f11)
    return explain_result_dict

def agreement(df1,df2,num):
    unioned = set()
    rank1_df1 = set(df1[df1['rank']==1]['columns'])
    intercepted = set(rank1_df1)
    rank1_df2 = set(df2[df2['rank']==1]['columns'])
    unioned = unioned.union(rank1_df1).union(rank1_df2)
    intercepted = intercepted.intersection(rank1_df2)
    if num == 3:
        rank2_df1 = set(df1[df1['rank'] == 2]['columns'])
        rank3_df1 = set(df1[df1['rank'] == 3]['columns'])
        rank2_df2 = set(df2[df2['rank'] == 2]['columns'])
        rank3_df2 = set(df2[df2['rank'] == 3]['columns'])
        unioned = unioned.union(rank2_df1).union(rank3_df1).union(rank2_df2).union(rank3_df2)
        to_intercept = rank1_df2.union(rank2_df2).union(rank3_df2)
        intercepted = rank1_df1.union(rank2_df1).union(rank3_df1).intersection(to_intercept)
        return len(intercepted) / len(unioned), rank1_df1.union(rank2_df1).union(rank3_df1), to_intercept
    if num == 5:
        rank2_df1 = set(df1[df1['rank'] == 2]['columns'])
        rank3_df1 = set(df1[df1['rank'] == 3]['columns'])
        rank4_df1 = set(df1[df1['rank'] == 4]['columns'])
        rank5_df1 = set(df1[df1['rank'] == 5]['columns'])
        rank2_df2 = set(df2[df2['rank'] == 2]['columns'])
        rank3_df2 = set(df2[df2['rank'] == 3]['columns'])
        rank4_df2 = set(df2[df2['rank'] == 4]['columns'])
        rank5_df2 = set(df2[df2['rank'] == 5]['columns'])

        unioned = unioned.union(rank2_df1).union(rank3_df1).union(rank2_df2).union(rank3_df2).union(rank4_df2).union(rank5_df2)
        to_intercept = rank1_df2.union(rank2_df2).union(rank3_df2).union(rank4_df2).union(rank5_df2)
        intercepted = rank1_df1.union(rank2_df1).union(rank3_df1).union(rank4_df1).union(rank5_df1).intersection(to_intercept)
        return len(intercepted) / len(unioned), rank1_df1.union(rank2_df1).union(rank3_df1).union(rank4_df1).union(rank5_df1), to_intercept
    return len(intercepted) / len(unioned), rank1_df1, rank1_df2

def agreement_ig_plain(shap, gain, num):
    gain_arr = set()
    shap_arr = set()

    for i in range(0, num):
        gain_arr.add(list(gain.items())[i][0].replace('-','.').replace('-','.'))
        shap_arr.add(shap[i].replace('-', '.').replace('-', '.'))
    unioned = set()
    unioned = unioned.union(shap_arr).union(gain_arr)
    intercepted = unioned.intersection(gain_arr)
    return len(intercepted) / len(unioned), shap_arr, gain_arr


def agreement_ig(df1,gain,num):
    unioned = set()
    rank1_df1 = set(df1[df1['rank']==1]['columns'])
    to_pick = len(rank1_df1)
    gain_arr = set()
    for i in range(0, to_pick):
        gain_arr.add(gain[i].replace('-','.').replace('-','.'))
    unioned = unioned.union(rank1_df1).union(gain_arr)
    intercepted = unioned.intersection(gain_arr)

    if num == 1:
        return len(intercepted) / len(unioned),rank1_df1,gain_arr
    if num == 3:

        rank2_df1 = set(df1[df1['rank'] == 2]['columns'])

        rank3_df1 = set(df1[df1['rank'] == 3]['columns'])
        unioned = rank1_df1.union(rank2_df1).union(rank3_df1).union(gain_arr)
        to_pick = len(unioned)
        for i in range(0, to_pick):
            gain_arr.add(gain[i].replace('-', '.').replace('-', '.'))
        intercepted = rank1_df1.union(rank2_df1).union(rank3_df1).intersection(gain_arr)
        return len(intercepted) / len(unioned),rank1_df1.union(rank2_df1).union(rank3_df1),gain_arr

        # intercepted = intercepted.intersection(rank2_df1).intersection(rank3_df1).intersection([gain[1]]).intersection([gain[2]])

def agreement_ig_ig(gain1,gain2,num):
    gain1[0] = gain1[0].replace('-','.').replace('-','.')
    if num == 3:
        gain1[1] = gain1[1].replace('-', '.').replace('-', '.')
        gain1[2] = gain1[2].replace('-', '.').replace('-', '.')

        set1 = {gain1[0], gain1[1], gain1[2]}
        set2 = {gain2[0].replace('-', '.').replace('-', '.'), gain2[1].replace('-', '.').replace('-', '.'),
                gain2[2].replace('-', '.').replace('-', '.')}

        intercepted = set(set1).intersection(set2)
        unioned = set(set1).union(set2)
        return len(intercepted)/len(unioned), set1, set2
    else:
        return (1,[gain1[0]],[gain2[0]]) if gain1[0] == gain2[0] else (0,[gain1[0]],[gain2[0]])

taken = {
    'msr_rev':set(),
    'kbs':set(),
    'hybrid':set(),
    'all':set()
}

def get_email(smell,cls,i):
    candidates = []
    resarr = []
    base = 0
    taken = set()
    if smell == 'complex-class': base = 341
    if smell == 'shotgun-surgery': base = 1029
    if smell == 'spaghetti-code': base = 692-2
    i+=base
    for filename in glob.glob(
            './data/emails/' + smell + '/' + cls + "_-" + str(i) + ".txt"):
        candidates.append(filename)
        taken.add(i)
    if len(candidates)<1:
        realNumber = None
        numbers = []
        for filename in glob.glob('./data/emails/'+smell+'/'+cls+"_-*.txt"):
            numbers.append(int(filename.split('_-')[1].split('.txt')[0]))
            realNumber = int(filename.split('_-')[1].split('.txt')[0])
        if len(numbers) > 1:
            if numbers[0] not in taken:
                realNumber = numbers[0]
            else:
                realNumber = numbers[1]
        if realNumber is not None:
            taken.add(realNumber)
            candidates.append('./data/emails/' + smell + '/' + cls + "_-" + str(realNumber) + ".txt")
    for f in candidates:
        with open(f,'r', encoding='utf-8') as f1:
            res = f1.read()
            resarr.append(res.split('flaw affecting the class?')[-1].split('(3)')[0])
    return resarr


def prop(metrics_in_dict,fea_num_percentage):
    heads_sorted = []
    metrics = {'cohesion': {'C3', 'LCOM', 'TCC_type', 'LCOM5_type', 'LCOM5'}, 'coupling': {'ATFD_type', 'FANOUT_type', 'CFNAMM_type', 'CBO', 'MPC', 'CBO_type'}, 'size': {'NOMNAMM_type', 'NOCS_package', 'num_final_not_static_attributes', 'RFC_type', 'number_constructor_NotDefaultConstructor_methods', 'NOAM_type', 'number_private_visibility_attributes', 'number_final_not_static_methods', 'number_final_methods', 'NMO_type', 'WMC_type', 'NOPK_project', 'number_not_abstract_not_final_methods', 'number_public_visibility_methods', 'NOPA_type', 'NOA_type', 'NOCS_type', 'NOC_type', 'number_package_visibility_methods', 'num_static_attributes', 'number_abstract_methods', 'NOM_project', 'NOI_package', 'LOCNAMM_type', 'NOCS_project', 'number_standard_design_methods', 'RFC', 'number_package_visibility_attributes', 'number_not_final_not_static_methods', 'NOM_package', 'num_static_not_final_attributes', 'LOC_project', 'number_static_methods', 'LOC_type', 'LOC', 'number_final_static_methods', 'NOMNAMM_package', 'number_protected_visibility_attributes', 'num_final_static_attributes', 'NOI_project', 'NOII_type', 'NOMNAMM_project', 'number_private_visibility_methods', 'NOM_type', 'num_not_final_not_static_attributes', 'number_not_final_static_methods', 'LOC_package', 'num_final_attributes', 'WMC', 'number_constructor_DefaultConstructor_methods', 'number_protected_visibility_methods', 'WMCNAMM_type'}, 'complexity': {'NIM_type', 'Readability', 'WMCNAMM_type', 'WMC', 'DIT_type', 'WMC_type', 'WOC_type'}, 'changehistory': {'avg.dev.scattering', 'persistance', 'CE', 'number-changes', 'EXP', 'number-committors', 'Avg-commit-size', 'number-fixes', 'NCOM', 'OWN', 'DSC'}, 'production': {'isStatic_type', 'is_controller', 'is_external', 'is_procedural', 'is_util'}, 'testing': {'is_test'}}
    heads = list(metrics.keys())
    kvs_sorted = list(sorted(metrics_in_dict.items(), key=lambda kv: (kv[1], kv[0]), reverse=True))
    fea_num = fea_num_percentage
    kvs = []
    for idx, val in enumerate(kvs_sorted):
        if idx + 1 > fea_num:
            break
        kvs.append(val)
    res = {}
    categories = set()
    for a,i in enumerate(kvs):
        for k in heads:
            i_name,intensity = i
            if i_name in metrics[k]:
                categories.add(k)
                if i_name not in heads_sorted:
                    heads_sorted.append(i_name)
    for c in categories:
        res[c] = 1

    vals = []
    for key_v,val_v in kvs:
        vals.append(val_v)
    return res, heads_sorted, entropy(vals, base=2)

def generate_explain_csv(fea_num, dataset):
    cols = ['coupling','changehistory','cohesion','size','complexity','production','testing']
    result = './results/explanations/'+dataset+'_'+str(fea_num)+'.csv'
    if os.access(result, os.F_OK):
        return
    data = pd.read_csv('./results/settings/rq2.csv')
    annotation = pd.read_csv('./data/annotate.csv')

    excel = []

    for i, row in data.iterrows():
        smell = row['smell']
        with open('./pk/explain/expl_' +smell+ '_'+dataset+'.pkl', 'rb') as f2:
            expl = pickle.load(f2)

        for idx, inner in enumerate(expl['shap']):

            comments = ''.join(inner['txt'])
            comments = comments.replace('\n\n','\n')
            if comments == '':
                continue
            r = annotation[annotation['class']==inner['cls']][annotation['smell']==smell][annotation['comments']==comments]
            if len(r)<1:
                r =  annotation[annotation['class']==inner['cls']][annotation['smell']==smell]

            if len(inner['shap_importance'])>=fea_num:
                prop_dict, heads_sorted, entropy_val = prop(inner['shap_importance'], fea_num)
            else:
                continue
            if len(r)>1:
                r = r.iloc[0]
                multi_csv = pd.read_csv('./data/annotation.csv') # fix encoding issues of comments
                rr = multi_csv[multi_csv['comments']==comments]
                if len(rr) > 0:
                    r['comments'] = rr['comments']
            if len(r)>0:
                to_append = {
                    'smell': smell,
                    'entropy':entropy_val,
                    'class': inner['cls'],
                    'correct': inner['select_predict'] == inner['true'],
                    'features': heads_sorted,
                    'criticality': inner['true'],
                    'comments': comments,
                }
                for h in cols:
                    try:
                        to_append[h] = int(r[h])
                    except:
                        continue
            else:
                continue
            for i,j in prop_dict.items():
                to_append['selected_'+i] = j

            ok = False
            for h in cols:
                if to_append[h] ==1:
                    ok = True
                    break
            if ok and to_append['coupling'] == to_append['coupling'] and to_append['coupling'] is not None:
                excel.append(to_append)
        pd.DataFrame(excel).to_csv(result)

def explain_selected(dataset):
    if dataset == 'orig':
        file = 'rq2'
    elif dataset == 'reductive':
        file = 'reductive'
    else:
        file = 'additive'
    data = pd.read_csv('./results/settings/'+file+'.csv')

    params = []

    for i, row in data.iterrows():
        smell = row['smell']
        as_variation = row['evaluator']
        param = row['param']
        cls = RandomForestClassifier(random_state=88, n_estimators=param)
        params.append((smell, as_variation, (cls, param)))

    p3 = multiprocessing.Pool(processes=8)
    with p3:
        res = p3.starmap_async(gen_smell_loocv, params)
        res.get()

    agreement_dict = []
    for (smell, as_variation, tp) in params:

        cls, param = tp

        result_path_selection = './pk/explain/res_' + smell + '_' + as_variation.replace('AutoSpearman_','AS_') + '.pk'

        with open(result_path_selection, 'rb') as f1:
            selection = pickle.load(f1)

        selected_metrics = []
        selected_metrics_shap_importance ={}

        for metric, val in selection['gain'].items():
            selected_metrics.append(metric)
            selected_metrics_shap_importance[metric] = []

        for dict_inner in selection['shap']:
            if len(dict_inner['shap'])<1:
                for m in selected_metrics:
                    selected_metrics_shap_importance[m].append(np.nan)
                continue
            for metric, val in dict_inner['shap'][0].items():
                selected_metrics_shap_importance[metric].append(val)
        for m in selected_metrics:
            selected_metrics_shap_importance[m] = np.mean(selected_metrics_shap_importance[m])
        selected_metrics_shap_importance = list(k for (k,v) in sorted(selected_metrics_shap_importance.items(), key=lambda item: -item[1]))

        if len(selected_metrics)>3:
            val, a1, a2 = agreement_ig_plain(selected_metrics_shap_importance, selection['gain'],3)
            agreement_dict.append({'smell':smell,'A':'selected_shap','B':'selected_gain','num':3,'val':val, 'topA':a1, 'topB':a2,'selected':json.dumps(selected_metrics),'selected_num':len(selected_metrics)})
        val, a1, a2 = agreement_ig_plain(selected_metrics_shap_importance, selection['gain'],10)

        agreement_dict.append({'smell':smell,'A':'selected_shap','B':'selected_gain','num':10,'val':val, 'topA':a1, 'topB':a2,'selected':json.dumps(selected_metrics),'selected_num':len(selected_metrics)})

    pd.DataFrame(agreement_dict).to_csv('./results/agreement_gain_shap_'+dataset+'.csv')

    ##### merge explanations with developer comments

    for i, row in data.iterrows():
        smell = row['smell']
        path = './pk/explain/expl_' +smell+ '_'+dataset+'.pkl'
        if os.access(path, os.F_OK):
            continue
        as_variation = row['evaluator']
        result_path_selection = './pk/explain/res_' + smell + '_' + as_variation + '.pk'
        with open(result_path_selection, 'rb') as f1:
            selection = pickle.load(f1)
        selected_metrics = []
        selected_metrics_shap_importance = {}

        for metric, val in selection['gain'].items():
            selected_metrics.append(metric)
            selected_metrics_shap_importance[metric] = []

        base = './data/'

        df = pd.read_csv(base + 'all_' + smell + '_merged_classname.csv').reset_index(drop=True)
        classnames = df['class-name']
        expl = {
            'shap': [],
            'select_gain': selection['gain'],
        }
        skip = 0
        for i, dict_inner in enumerate(selection['shap']):
            if len(dict_inner['shap']) < 1: continue
            txt = get_email(smell, classnames.iloc[i], i)
            if len(txt) < 1:
                skip += 1
            inner = {
                'cls': classnames.iloc[i]
            }
            inner['shap_importance'] = dict_inner['shap'][0]
            inner['shap_values'] = dict_inner['shap_values'][0]
            inner['shap_features'] = dict_inner['shap_features'][0]
            inner['gain_features'] = df[selected_metrics].iloc[i]
            inner['select_predict'] = selection['predict'][i]
            inner['true'] = selection['true'][i]
            inner['txt'] = txt
            expl['shap'].append(inner)

        with open(path, 'wb') as f2:
            pickle.dump(expl, f2)


def agreement_multi(a,b):
    if len(np.unique(a))<2: return 1

    return recall_score(y_true=a,y_pred=b)
    ok = 0
    not_ok = 0
    for i,aa in enumerate(a):
        if aa == 0: continue
        if b[i] > 0:
            ok+=1
        else:
            not_ok+=1
    return ok/ (not_ok+ok) if not_ok+ok>0 else np.nan

def generate_data_for_figure(fea_num, dataset):
    result = './results/explanations/'+dataset+'_'+str(fea_num)+'.csv'
    data = pd.read_csv(result)
    heads = ['coupling','changehistory','cohesion','size','complexity','production','testing']
    smells = ['spaghetti-code','blob','shotgun-surgery','complex-class']

    outer_result = []
    for smell in smells:
        tmp = {}
        for r in data[data['smell'] == smell]['features']:
            features = eval(r)
            for feature in features:
                if tmp.get(feature) is None:
                    tmp[feature] = 1
                else:
                    tmp[feature] += 1
        features = dict(sorted(tmp.items(), key=lambda d: -d[1]))

        bd_set = data
        entropy_set = data

        set_multi_smel = bd_set[bd_set['smell'] == smell]
        entropy_set_smel = entropy_set[entropy_set['smell'] == smell]

        def extract_heads(set_multi_smel, heads):
            correct_val = []
            explained_val = []
            for i, set_multi_smell in set_multi_smel.iterrows():
                for head in heads:
                    set_multi_head = int(set_multi_smell[head])
                    to_compare = 'selected_' + head
                    set_multi_to_compare = 1 if int(set_multi_smell[to_compare]) > 0 else 0
                    correct_val.append(set_multi_head)
                    explained_val.append(set_multi_to_compare)
            if len(correct_val) < 1:
                p = np.nan
            elif len(np.unique(correct_val)) < 2:
                p = 1
            else:
                p = recall_score(
                    correct_val,
                    explained_val)
            return p

        p = extract_heads(set_multi_smel, heads)
        if p == p:
            dict_tmp = {'smell': smell, 'type': 'agreement',
                        'value': p, 'head':'all',
                        'fea_num': fea_num, 'features':features}
            bd_set = bd_set[bd_set['smell']==smell]
            outer_result.append(dict_tmp)
            dict_tmp = {'smell': smell, 'type':'entropy', 'value': mean(entropy_set_smel['entropy']),
                         'fea_num': fea_num, 'head':'all', 'features':features}
            outer_result.append(dict_tmp)
        for head in heads:
            p = extract_heads(set_multi_smel, [head])
            if p == p:
                dict_tmp = {'smell': smell, 'type': 'agreement',
                            'value': p, 'head': head,
                            'fea_num': fea_num, 'features': features}
                bd_set = bd_set[bd_set['smell'] == smell]
                outer_result.append(dict_tmp)
    return outer_result

def plot(data, filename):
    sns.set_context('paper')
    sns.set_style('darkgrid')

    data = data.replace('spaghetti-code','Spaghetti Code').replace('blob','Blob').replace('shotgun-surgery','Shotgun Surgery').replace('complex-class','Complex Class')
    data = data[data['head']=='all']
    data = data[data['fea_num']<16]
    data_agreement = data[data['type']=='agreement']
    data_entropy = data[data['type']=='entropy']

    # 设置figure_size尺寸
    plt.rcParams['savefig.dpi'] = 400  # 图片像素
    plt.rcParams['figure.dpi'] = 400  # 分辨率
    plt.rcParams['figure.figsize'] = (4,2)


    fig, ax1 = plt.subplots()
    ax1.set_xlabel('')
    ax1.set_ylabel('Recall (coverage)')
    p1 = sns.lineplot(data=data_agreement, x="fea_num", y="value", style='smell', hue='smell', ci=None, legend = True)
    ax1.tick_params(axis='y')
    ax1.set_xlim(2,16)
    plt.xticks(range(2,16,1))
    plt.yticks(np.arange(0, 1.1, step=0.1))
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    ax1.set_xlabel('')

    plt.legend(loc="lower right", fontsize='x-small', title_fontsize='x-small')
    plt.savefig('./results/figs/'+filename+'.pdf', bbox_inches='tight')
    plt.show()

    p1 = sns.lineplot(data=data_entropy, x="fea_num", y="value", style='smell', hue='smell', ci=None, legend = True, )
    plt.ylabel('Complexity (entropy)')
    plt.xlabel('')

    plt.xlim(2, 16)
    plt.xticks(range(2,16,1))
    plt.ylim(0,4)
    plt.legend(loc="lower right", fontsize='x-small', title_fontsize='x-small')

    plt.savefig('./results/figs/'+filename+'_entropy.pdf', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    dataset = 'orig' #AutoSpearman
    path = './results/explanations/'+dataset+'.csv'
    explain_selected(dataset)
    res = []
    for i in range(2,16):
        generate_explain_csv(i, dataset)
        res.extend(generate_data_for_figure(i, dataset))
    if not os.access(path, os.F_OK):
        pd.DataFrame(res).to_csv('./results/explanations/'+dataset+'.csv',index=False)
    data = pd.read_csv('./results/explanations/'+dataset+'.csv')
    plot(data, 'rq2') #generate figure for rq2

    #### prepare for rq3
    for dataset in ['reductive','additive']:
        path = './results/explanations/'+dataset+'.csv'
        explain_selected(dataset)
        res = []
        for i in range(2,16):
            generate_explain_csv(i, dataset)
