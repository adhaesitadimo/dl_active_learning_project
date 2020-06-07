from itertools import cycle
from matplotlib import pyplot as plt
import os
import numpy as np


def generate_plot(np_stats, color_bg, color_line, marker, label, metric_dim=0):
    if len(np_stats.shape) > 2:
        np_stats = np_stats[:, :, metric_dim]
    
    disp = np_stats.std(axis=0)
    means = np_stats.mean(axis=0)

    x = list(range(1, means.shape[0] + 1))

    line_thikness = 0.5
    plt.fill_between(x, means - disp, means + disp, color=color_bg, alpha=0.5)
    plt.xticks(range(1, means.shape[0] + 2, 2))
    return plt.plot(x, means, color=color_line, label=label, 
                    marker=marker, linewidth=line_thikness)


def create_chart(expert_dir, attr_type, metric_dim=0, save_dir=None):
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
    
    for filename in sorted(os.listdir(expert_dir)):
        noext = os.path.splitext(filename)[0]
        if '_' not in noext:
            continue
            
        f_type, f_strat = noext.split('_')
        
        if f_type != attr_type:
            continue
        
        color = next(color_pool)
        file_path = os.path.join(expert_dir, filename)
        stats = np.load(file_path)
        chart, = generate_plot(stats, 
                               color_bg=color[1], 
                               color_line=color[0], 
                               marker=color[2],
                               label=f_strat,
                               metric_dim=metric_dim)
        plt_list.append(chart)
        
    plt.legend(handles=plt_list, loc='lower right', fontsize='x-large')
    plt.ylabel('Performance, F1', fontdict={'size' : 15})
    plt.xlabel('AL iteration, #', fontdict={'size' : 15})
    plt.tick_params(labelsize=12)
    
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f'{attr_type}.png'))
        
    plt.show()
