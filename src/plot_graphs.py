from itertools import cycle
from matplotlib import pyplot as plt
import os
import json
import numpy as np
import pickle
%matplotlib inline


def crop_json(file_path, savepath, to_crop):
    f = []
    with open(file_path) as file:
        f = json.load(file)
    with open(savepath, 'w') as save:
        json.dump(f[:to_crop + 1], save)

def generate_plot(np_stats, color_bg, color_line, marker, label, metric_dim=0):
    if len(np_stats.shape) > 2:
        np_stats = np_stats[:, :, metric_dim]

    disp = np_stats.std(axis=0)
    means = np_stats.mean(axis=0)

    x = list(range(0, means.shape[0]))

    line_thikness = 0.5
    plt.fill_between(x, means - disp, means + disp, color=color_bg, alpha=0.5)
    plt.xticks(range(0, means.shape[0] + 2, 5))
    return plt.plot(x, means, color=color_line, label=label,
                    marker=marker, linewidth=line_thikness)


def create_chart(expert_dirs, expert_names, attr_type, metric_dim=0, save_dir=None, modes=['json'], experiments=range(5)):
    os.makedirs(save_dir, exist_ok = True)
    fig = plt.figure(num=None, figsize=(8, 6),
                     dpi=200, facecolor='w', edgecolor='k')
    plt_list = []

    color_pool = cycle([('red', 'darksalmon', 'o'),
                        ('midnightblue', 'skyblue', 'v'),
                        ('g', 'lightgreen', 's'),
                        #('gold', 'palegoldenrod', '*'),
                        ('maroon', 'rosybrown', '*'),
                        ('purple', 'violet', '+'),
                        ('slategrey', 'lightgrey', '1'),
                        ('darkorange', 'wheat', 's'),
                        ('darkcyan', 'lightcyan', 'P')])

    for expert, expert_name, mode in zip(expert_dirs, expert_names, modes):
        strats = os.listdir(expert)
        for stratname in strats:
            #.ipynb_checkpoints
            if '.ipynb' in stratname:
                continue

            all_stats = []

            color = next(color_pool)
            folder_path = os.path.join(expert, stratname)
            if 'statistics0.json' not in os.listdir(folder_path):
                continue
            if mode == 'json':
                exp_files = []
                for exp in experiments:
                    exp_files.append(f'statistics{exp}.json')
                for file in sorted(os.listdir(folder_path)):
                    if file in exp_files:
                        file_path = os.path.join(folder_path, file)
                        #print(file_path)
                        try:
                            with open(file_path) as f:
                                stats = np.array([np.array(stat['f1_score']) for stat in json.load(f)])

                        except:
                            with open(file_path, 'rb') as f:
                                stats = np.array([np.array(stat['f1_entity_level']) for stat in json.load(f)])
                        all_stats.append(stats)
            else:
                for file in sorted(os.listdir(folder_path)):
                    file_path = os.path.join(folder_path, file)
                    if '.ipynb' in file:
                        continue
                    with open(file_path, 'rb') as f:
                        all_stats = np.load(f)

            chart, = generate_plot(np.array(all_stats),
                                   color_bg=color[1],
                                   color_line=color[0],
                                   marker=color[2],
                                   label=expert_name + ' ' + stratname,
                                   metric_dim=metric_dim)
            plt_list.append(chart)

    plt.legend(handles=plt_list, loc='lower right', fontsize='x-large')
    plt.ylabel('Performance, F1', fontdict={'size' : 15})
    plt.xlabel('AL iteration, #', fontdict={'size' : 15})
    plt.tick_params(labelsize=12)

    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f'{attr_type}.png'))

    plt.show()
