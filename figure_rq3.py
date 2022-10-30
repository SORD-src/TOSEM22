import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy import mean
from sklearn.metrics import recall_score


def generate_data_for_figure(fea_num, flag):
    flag = flag.replace('original','orig')
    data = pd.read_csv('./results/explanations/'+flag+'_'+str(fea_num)+'.csv')
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
                        'fea_num': fea_num, 'features':features, 'approach':flag}
            bd_set = bd_set[bd_set['smell']==smell]
            outer_result.append(dict_tmp)
            dict_tmp = {'smell': smell, 'type':'entropy', 'value': mean(entropy_set_smel['entropy']),
                         'fea_num': fea_num, 'head':'all', 'features':features, 'approach':flag}
            outer_result.append(dict_tmp)
        for head in heads:
            p = extract_heads(set_multi_smel, [head])
            if p == p:
                dict_tmp = {'smell': smell, 'type': 'agreement',
                            'value': p, 'head': head,
                            'fea_num': fea_num, 'features': features, 'approach':flag}
                bd_set = bd_set[bd_set['smell'] == smell]
                outer_result.append(dict_tmp)
    return outer_result

def plot(data):
    sns.set_context('paper')
    sns.set_style('darkgrid')

    data = data[data['head']=='all']
    data = data[data['fea_num']<16]
    data_agreement = data[data['type']=='agreement']
    data_entropy = data[data['type']=='entropy']

    # 设置figure_size尺寸
    plt.rcParams['savefig.dpi'] = 400  # 图片像素
    plt.rcParams['figure.dpi'] = 400  # 分辨率
    plt.rcParams['figure.figsize'] = (4,2)

    for smell in ['Shotgun Surgery','Complex Class','Blob','Spaghetti Code']:
        data_agreement_inner = data_agreement[data_agreement['smell']==smell]
        data_entropy_inner = data_entropy[data_entropy['smell']==smell]

        fig, ax1 = plt.subplots()
        color = 'salmon'
        ax1.set_xlabel('')
        ax1.set_ylabel('Recall (coverage)')
        p1 = sns.lineplot(data=data_agreement_inner, x="fea_num", y="value", style='approach', hue='approach', ci=None, legend = True)
        ax1.tick_params(axis='y')
        ax1.set_xlim(2,16)
        plt.xticks(range(2,16,1))
        plt.yticks(np.arange(0, 1.1, step=0.1))


        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        ax1.set_xlabel('')

        plt.legend(loc="lower right", fontsize='x-small', title_fontsize='x-small')
        plt.savefig('./results/figs/rq3_compare_'+smell+'.pdf', bbox_inches='tight')

        plt.show()

        p1 = sns.lineplot(data=data_entropy_inner, x="fea_num", y="value", style='approach', hue='approach', ci=None, legend = True)
        plt.ylabel('Complexity (entropy)')
        plt.xlabel('')

        plt.xlim(2, 16)
        plt.xticks(range(2,16,1))
        plt.ylim(0, 4)
        plt.legend(loc="lower right", fontsize='x-small', title_fontsize='x-small')

        ax = plt.gca()

        plt.savefig('./results/figs/rq3_entropy_compare_'+smell+'.pdf', bbox_inches='tight')

        plt.show()

if __name__ == '__main__':

    res = []
    path = './results/compare_rq3.csv'
    if not os.access(path,os.F_OK):
        for i in range(2,16):
            for dataset in ['additive','reductive','original']:
                res.extend(generate_data_for_figure(i,dataset))
        # # #
        pd.DataFrame(res)\
            .replace('spaghetti-code','Spaghetti Code')\
            .replace('additive','AS_Additive')\
            .replace('reductive','AS_Reductive')\
            .replace('original','AutoSpearman')\
            .replace('blob','Blob')\
            .replace('shotgun-surgery','Shotgun Surgery')\
            .replace('complex-class','Complex Class')\
            .to_csv(path,index=False)
    data = pd.read_csv(path)
    plot(data)
