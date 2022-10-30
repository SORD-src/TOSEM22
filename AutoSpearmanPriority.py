import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from AutoSpearman import AutoSpearman

high_priority = {
    'spaghetti-code':['complexity','size'],
    'shotgun-surgery': ['coupling', 'complexity'],
    'complex-class': ['complexity'],
    'blob':['changehistory','size','testing','complexity','production']
}

metrics = {
    'complexity':'NIM_type,DIT_type,WMC,WMCNAMM_type,WOC_type,Readability'.split(','),
    'cohesion':'LCOM5,C3,TCC_type'.split(','),
    'coupling':'CBO,MPC,ATFD_type,CFNAMM_type,FANOUT_type'.split(','),
    'size':'RFC,number_private_visibility_attributes,number_protected_visibility_attributes,number_package_visibility_attributes,num_final_attributes,num_static_attributes,num_final_static_attributes,num_not_final_not_static_attributes,num_final_not_static_attributes,num_static_not_final_attributes,number_public_visibility_methods,number_private_visibility_methods,number_protected_visibility_methods,number_package_visibility_methods,number_final_methods,number_abstract_methods,number_not_abstract_not_final_methods,number_static_methods,number_final_static_methods,number_final_not_static_methods,number_not_final_static_methods,number_not_final_not_static_methods,number_standard_design_methods,number_constructor_DefaultConstructor_methods,number_constructor_NotDefaultConstructor_methods,LOC,LOC_type,NOCS_package,NOMNAMM_package,NOI_package,LOC_package,NOM_package,NOPK_project,NOCS_project,NOI_project,NOM_project,NOMNAMM_project,LOC_project,NOPA_type,NOII_type,NOAM_type,NOCS_type,NOM_type,NOA_type,NOC_type,NOMNAMM_type,NMO_type,LOCNAMM_type,WMCNAMM_type'.split(','),
    'changehistory':'Avg-commit-size,number-changes,CE,persistance,NCOM,EXP,DSC,OWN,avg.dev.scattering,number-fixes,number-committors'.split(','),
    'production':'is_controller,is_procedural,is_util,is_external,isStatic_type'.split(','),
    'testing':['is_test']
}

def extract_priority(smell, metric):
    result = 9999
    for i, v in enumerate(high_priority[smell]):
        if metric in metrics[v]:
            result = i
            break
    return result

def extract_high_priority_metrics(smell, X_AS_train):
    candidates = []
    result = []
    for metric_type in high_priority[smell]:
        candidates.extend(metrics[metric_type])
    for i in X_AS_train:
        if i in candidates:
            result.append(i)
    return result

def AutoSpearmanAdditive(X_train, smell, correlation_threshold=0.7, correlation_method='spearman', VIF_threshold=5):
    """An automated feature selection approach that address collinearity and multicollinearity.
    For more information, please kindly refer to the `paper <https://ieeexplore.ieee.org/document/8530020>`_.

    Parameters
    ----------
    X_train : :obj:`pd.core.frame.DataFrame`
        The X_train data to be processed
    correlation_threshold : :obj:`float`
        Threshold value of correalation.
    correlation_method : :obj:`str`
        Method for solving the correlation between the features.
    VIF_threshold : :obj:`int`
        Threshold value of VIF score.
    """
    protected_metrics = extract_high_priority_metrics(smell, X_train.columns)
    AS_metrics = list(X_train.columns)
    for i in protected_metrics:
        AS_metrics.remove(i)
    X_AS_train = AutoSpearman(X_train[protected_metrics])
    X_AS_train_tmp = X_AS_train.copy()
    selected_features = list(X_AS_train_tmp.columns)
    for metric in AS_metrics:
        print('(Part 1) Automatically select non-correlated metrics based on a Spearman rank correlation test')
        X_AS_train_tmp[metric] = X_train[metric]
        corrmat = X_AS_train_tmp.corr(method=correlation_method)

        # identify correlated metrics with the correlation threshold of the threshold
        highly_correlated_metrics = ((corrmat > correlation_threshold) | (corrmat < -correlation_threshold)) & (
                corrmat != 1)
        if sum(highly_correlated_metrics[metric])>0:
            del X_AS_train_tmp[metric]
            continue
        X_AS_train_tmp = add_constant(X_AS_train_tmp)

        # Calculate VIF scores
        vif_scores = pd.DataFrame([variance_inflation_factor(X_AS_train_tmp.values, i)
                                   for i in range(X_AS_train_tmp.shape[1])],
                                  index=X_AS_train_tmp.columns)
        # Prepare a final dataframe of VIF scores
        vif_scores.reset_index(inplace=True)
        vif_scores.columns = ['Feature', 'VIFscore']
        vif_scores = vif_scores.loc[vif_scores['Feature'] != 'const', :]
        vif_scores.sort_values(by=['VIFscore'], ascending=False, inplace=True)

        # Find features that have their VIF scores of above the threshold
        filtered_vif_scores = vif_scores[vif_scores['VIFscore'] >= VIF_threshold]

        # exclude the metric with the highest VIF score
        filtered_list = list(filtered_vif_scores['Feature'].head())
        if metric in filtered_list:
            del X_AS_train_tmp[metric]
            continue
        selected_features = list(X_AS_train_tmp.columns)
        selected_features.remove('const')
        X_AS_train_tmp = X_train.loc[:, selected_features]

    print('Finally,', selected_features, 'are selected.')
    all_cols = X_AS_train_tmp.columns
    X_train = X_train.loc[:, all_cols]
    return X_train

def AutoSpearmanReductive(X_train, smell, correlation_threshold=0.7, correlation_method='spearman', VIF_threshold=5):
    """An automated feature selection approach that address collinearity and multicollinearity.
    For more information, please kindly refer to the `paper <https://ieeexplore.ieee.org/document/8530020>`_.

    Parameters
    ----------
    X_train : :obj:`pd.core.frame.DataFrame`
        The X_train data to be processed
    correlation_threshold : :obj:`float`
        Threshold value of correalation.
    correlation_method : :obj:`str`
        Method for solving the correlation between the features.
    VIF_threshold : :obj:`int`
        Threshold value of VIF score.
    """
    X_AS_train = X_train.copy()
    AS_metrics = X_AS_train.columns
    count = 1

    # (Part 1) Automatically select non-correlated metrics based on a Spearman rank correlation test.
    print('(Part 1) Automatically select non-correlated metrics based on a Spearman rank correlation test')
    while True:
        corrmat = X_AS_train.corr(method=correlation_method)
        top_corr_features = corrmat.index
        abs_corrmat = abs(corrmat)

        # identify correlated metrics with the correlation threshold of the threshold
        highly_correlated_metrics = ((corrmat > correlation_threshold) | (corrmat < -correlation_threshold)) & (
                corrmat != 1)
        n_correlated_metrics = np.sum(np.sum(highly_correlated_metrics))
        if n_correlated_metrics > 0:
            # find the strongest pair-wise correlation
            find_top_corr = pd.melt(abs_corrmat, ignore_index=False)
            find_top_corr.reset_index(inplace=True)
            find_top_corr = find_top_corr[find_top_corr['value'] != 1]
            top_corr_index = find_top_corr['value'].idxmax()
            top_corr_i = find_top_corr.loc[top_corr_index, :]

            # get the 2 correlated metrics with the strongest correlation
            correlated_metric_1 = top_corr_i[0]
            correlated_metric_2 = top_corr_i[1]

            order_1 = extract_priority(smell, correlated_metric_1)
            order_2 = extract_priority(smell, correlated_metric_2)
            if order_1 == order_2:
                print('> Step', count, 'comparing between', correlated_metric_1, 'and', correlated_metric_2)

                # compute their correlation with other metrics outside of the pair
                correlation_with_other_metrics_1 = np.mean(abs_corrmat[correlated_metric_1][
                                                               [i for i in top_corr_features if
                                                                i not in [correlated_metric_1, correlated_metric_2]]])
                correlation_with_other_metrics_2 = np.mean(abs_corrmat[correlated_metric_2][
                                                               [i for i in top_corr_features if
                                                                i not in [correlated_metric_1, correlated_metric_2]]])
                print('>>', correlated_metric_1, 'has the average correlation of',
                      np.round(correlation_with_other_metrics_1, 3), 'with other metrics')
                print('>>', correlated_metric_2, 'has the average correlation of',
                      np.round(correlation_with_other_metrics_2, 3), 'with other metrics')
                # select the metric that shares the least correlation outside of the pair and exclude the other
                if correlation_with_other_metrics_1 < correlation_with_other_metrics_2:
                    exclude_metric = correlated_metric_2
                else:
                    exclude_metric = correlated_metric_1
            elif order_1 < order_2:
                exclude_metric = correlated_metric_2
            else:
                exclude_metric = correlated_metric_1
            print('>>', 'Exclude', exclude_metric)
            count = count + 1
            AS_metrics = list(set(AS_metrics) - set([exclude_metric]))
            X_AS_train = X_AS_train[AS_metrics]
        else:
            break

    print('According to Part 1 of AutoSpearman,', AS_metrics, 'are selected.')

    # (Part 2) Automatically select non-correlated metrics based on a Variance Inflation Factor analysis.
    print('(Part 2) Automatically select non-correlated metrics based on a Variance Inflation Factor analysis')

    # Prepare a dataframe for VIF
    X_AS_train = add_constant(X_AS_train)

    selected_features = X_AS_train.columns
    count = 1
    while True:
        # Calculate VIF scores
        vif_scores = pd.DataFrame([variance_inflation_factor(X_AS_train.values, i)
                                   for i in range(X_AS_train.shape[1])],
                                  index=X_AS_train.columns)
        # Prepare a final dataframe of VIF scores
        vif_scores.reset_index(inplace=True)
        vif_scores.columns = ['Feature', 'VIFscore']
        vif_scores = vif_scores.loc[vif_scores['Feature'] != 'const', :]
        vif_scores.sort_values(by=['VIFscore'], ascending=False, inplace=True)

        # Find features that have their VIF scores of above the threshold
        filtered_vif_scores = vif_scores[vif_scores['VIFscore'] >= VIF_threshold]

        # Terminate when there is no features with the VIF scores of above the threshold
        if len(filtered_vif_scores) == 0:
            break

        # exclude the metric with the highest VIF score
        filtered_list = list(filtered_vif_scores['Feature'].head())
        filtered_order = list(map(lambda i: extract_priority(smell, i), filtered_list))
        max_filtered_order = max(filtered_order)
        max_idx = filtered_order.index(max_filtered_order)
        metric_to_exclude = filtered_list[max_idx]
        # metric_to_exclude = filtered_list[0]

        print('> Step', count, '- exclude', str(metric_to_exclude))
        count = count + 1

        selected_features = list(set(selected_features) - set([metric_to_exclude]))

        X_AS_train = X_AS_train.loc[:, selected_features]

    print('Finally, according to Part 2 of AutoSpearman,', selected_features, 'are selected.')
    all_cols = X_AS_train.columns
    for col in all_cols:
        if col not in list(AS_metrics):
            all_cols = all_cols.drop(col)
    selected = all_cols
    X_train = X_train.loc[:, selected]
    return X_train

