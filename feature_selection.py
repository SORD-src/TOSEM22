import multiprocessing
import os
import json
import pandas as pd


from AutoSpearman import AutoSpearman
from AutoSpearmanPriority import AutoSpearmanReductive, AutoSpearmanAdditive

def get_file(smell):
    p = './data/all_'+smell+'_merged_classname.csv'
    df = pd.read_csv(p)

    del df['class-name']
    del df['perception']
    return df

def gen_smell(smell, algo):
    result_path = './results/fs/' + smell + '_'+ str(algo)+'.json'

    if os.access(result_path,  os.F_OK):
        return

    # load a dataset
    dataset = get_file(smell)
    d = pd.read_csv(dataset)
    columns_original = list(d.columns)

    d = d.replace('?', 0).replace(False,0).replace(True,1).apply(pd.to_numeric)
    arr = algo.split('_')
    if len(arr)<2:
        columns = list(AutoSpearman(d).columns)
    elif arr[1] == 'Reductive':
        columns = list(AutoSpearmanReductive(d,smell).columns)
    elif arr[1] == 'Additive':
        columns = list(AutoSpearmanAdditive(d, smell).columns)
    else:
        columns = list(AutoSpearman(d).columns)
    res = []
    for i in columns:
        res.append(columns_original.index(i))
    print(res)

    with open(result_path, 'w') as f:
        json.dump(res, f, indent=4, sort_keys=True)

if __name__ == '__main__':
    res = {}
    smells = ['spaghetti-code','blob','shotgun-surgery','complex-class']
    params = []
    methods = ['AutoSpearman', 'AS_Additive','AS_Reductive']

    for smell in smells:
        for algo in methods:
            algo_str = 'None' if algo is None else algo
            result_path = './results/fs/' + smell + '_' + algo_str +'.json'
            if os.access(result_path, os.F_OK):
                print('skip')
                continue
            params.append((smell, algo))
    print(len(params))
    p2 = multiprocessing.Pool(processes=7)
    with p2:
        res = p2.starmap_async(gen_smell, params)
        res.get()
