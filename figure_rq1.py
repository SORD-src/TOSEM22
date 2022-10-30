import pickle
import numpy as np
import shap
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap

def process_labels(explanation):
    cols = explanation['cols']
    return list(map(lambda i:
                    i.replace('number_private_visibility_attributes', 'num_private_visibility_attr')
                    .replace('num_final_static_attributes', 'num_final_static_attr')
                    .replace('number_constructor_DefaultConstructor_methods', 'num_constructor_DC')
                    .replace('number_abstract_methods', 'num_abstract_methods')
                    .replace('number_package_visibility_methods', 'num_package_visibility_methods')
                    .replace('number_protected_visibility_attributes', 'num_protected_visibility_attr')
                    .replace('number_constructor_NotDefaultConstructor_methods',
                             'num_constructor_notDC')
                    .replace('number_final_not_static_methods', 'num_final_not_static_methods')
                    .replace('avg.dev.scattering', 'DSC')
                    .replace('Avg-commit-size', 'AVGCS')
                    .replace('persistance', 'Pers.')
                    .replace('_type', '')
                    .replace('number-fixes', 'NF')
                    .replace('num_final_attributes', 'num_final_attr'),
                    cols))

def RQ1_fig(smell, remove,n_estimators):
    values = None
    x_vals = None
    dataset = 'All'
    base_values = None
    with open('./pk/pk_res_' + smell + '_' + dataset + '_' + str(remove)+'_'+str(n_estimators) + '.pk', 'rb') as f1:
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
    plt.savefig('./results/figs/rq1_' + smell + '_' + str(remove) + '.pdf',
                bbox_inches="tight")
    plt.show()

remove_project = False

RQ1_fig('blob',remove_project,260)
RQ1_fig('complex-class',remove_project,170)