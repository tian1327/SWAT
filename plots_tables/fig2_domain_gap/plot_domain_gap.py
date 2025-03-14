import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# matplotlib.rcParams['font.family'] = 'Times New Roman'
# matplotlib.rcParams['font.family'] = 'DejaVu Sans'

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.style'] = 'normal'
plt.rcParams['font.variant'] = 'normal'

MARKERSIZE = 12
LINEWIDTH = 3


dataset_map = {
    'dtd': 'DTD',
    'eurosat': 'EuroSAT',
    'fgvc-aircraft': 'FGVC-Aircraft',
    'flowers102': 'Flowers102',
    'semi-aves': 'Semi-Aves',
    'pets': 'Oxford Pets',
    'food': 'Food-101',
    'cars': 'Stanford Cars',
    'imagenet': 'ImageNet',
    'average': 'mean accuracy over nine datasets'
}

openclip_zeroshot_map = {
    'dtd': 53.5,
    'eurosat': 51.1,
    'fgvc-aircraft': 17.1,
    'flowers102': 68.2,
    'semi-aves': 8.4,
    'pets': 88.7,
    'food': 77.2,
    'cars': 79.2,
    'imagenet': 63.0,
    'average': 56.3
}

real_prompt_zeroshot_map = {
    'dtd': 59.2,
    'eurosat': 56.9,
    'fgvc-aircraft': 18.0,
    'flowers102': 76.0,
    'semi-aves': 43.4,
    'pets': 88.7,
    'food': 77.1,
    'cars': 80.6,
    'imagenet': 63.6,
    'average': 62.6
}

real_linear_zeroshot_map = {
    'dtd': 61.0,
    'eurosat': 51.5,
    'fgvc-aircraft': 27.3,
    'flowers102': 79.4,
    'semi-aves': 49.2,
    'pets': 89.7,
    'food': 78.0,
    'cars': 81.7,
    'imagenet': 65.5,
    'average': 64.8
}

# CMLP for 50 epochs, without WiSE-FT
crossmodal_map = {
    'dtd': [62.2, 67.2, 71.9],
    'eurosat': [74.8, 80.6, 85.2],
    'fgvc-aircraft': [25.1, 27.9, 32.4],
    'flowers102': [88.9, 92.5, 95.5],
    'semi-aves': [29.1, 38.8, 46.8],
    'pets': [88.3, 88.8, 89.1],
    'food': [76.7, 77.3, 77.5],
    'cars': [80.7, 82.7, 84.7],
    'imagenet': [63.2, 63.1, 63.1],
    'average': [65.4, 68.8, 71.8]
}

# from my experiments
clap_map = {
    'dtd': [63.0, 66.4, 69.9],
    'eurosat': [74.7, 77.4, 81.7],
    'fgvc-aircraft': [28.0, 33.6, 39.1],
    'flowers102': [90.1, 92.9, 94.8],
    'semi-aves': [34.0, 42.9, 49.2],
    'pets': [87.0, 87.8, 88.4],
    'food': [76.7, 77.5, 78.5],
    'cars': [84.9, 86.1, 87.8],
    'imagenet': [64.0, 65.6, 67.1],
    'average': [66.9, 70.0, 72.9]
}

# this is FTFS w/ CutMix
ft_fs_map = {
    'semi-aves': [48.0, 52.3, 56.5],
    'flowers102': [92.2, 95.2, 97.1],
    'fgvc-aircraft': [28.8, 35.4, 42.7],
    'eurosat': [81.8, 89.4, 94.3],
    'dtd': [66.7, 70.6, 73.4],
    'pets': [89.0, 89.6, 89.6],
    'food': [76.1, 77.0, 78.2],
    'cars': [82.5, 85.3, 87.8],
    'imagenet': [62.4, 64.8, 66.9],
    'average': [69.7, 73.3, 76.3]
}

ft_retr_map = {
    'semi-aves': 52.1,
    'flowers102': 81.6,
    'fgvc-aircraft': 48.3,
    'eurosat': 27.9,
    'dtd': 53.3,
    'pets': 90.3,
    'food': 75.7,
    'cars': 75.3,
    'imagenet': 60.9,
    'average': 62.8
}

# this is SWAT w/ T2T500 results
swat_map = {
    'semi-aves': [58.5, 61.3, 63.1],
    'flowers102': [90.6, 94.1, 96.4],
    'fgvc-aircraft': [55.7, 59.1, 62.4],
    'eurosat': [83.4, 88.7, 92.9],
    'dtd': [58.3, 62.6, 66.3],
    'pets': [91.3, 91.5, 91.6],
    'food': [77.3, 77.6, 78.3],
    'cars': [81.1, 83.5, 85.4],
    'imagenet': [65.8, 66.6, 67.6],
    'average': [73.6, 76.1, 78.2]
}

# this is SWAT+ w/ T2T500 results, finetune whole model in stage 2
swat_plus_map = {
    'semi-aves': [59.9, 62.7, 64.7],
    'flowers102': [94.2, 96.7, 98.3],
    'fgvc-aircraft': [55.6, 56.8, 60.2],
    'eurosat': [83.4, 89.7, 93.5],
    'dtd': [61.5, 67.0, 69.8],
    'pets': [91.6, 91.9, 92.2],
    'food': [77.9, 78.4, 79.1],
    'cars': [83.7, 87.0, 89.2],
    'imagenet': [66.6, 68.1, 69.3],
    'average': [74.9, 77.6, 79.6]
}


swat_improved_map = {
    'dtd': [63.5, 69.1, 72.9], # T2T10
    'cars': [83.5, 86.8, 88.6], # T2T10
    'flowers102': [91.8, 95.2, 97.0], # T2T10
    'eurosat': [84.7, 90.0, 94.0], # T2T10
}


def plot_results(dataset):
    print(f'Plotting results for {dataset} ...')

    dataset_name = dataset_map[dataset]

    # get the fewshot x values
    x = [4, 8, 16]
    x_fs = [0, 4, 8, 16]

    y_ft_fs = ft_fs_map[dataset]
    y_ft_retr = ft_retr_map[dataset]
    swat = swat_map[dataset]
    swat_plus = swat_plus_map[dataset]
    real_prompt_acc = real_prompt_zeroshot_map[dataset]
    openai_prompt_acc = openclip_zeroshot_map[dataset]
    real_linear_acc = real_linear_zeroshot_map[dataset]
    y_cmlp = crossmodal_map[dataset]
    y_clap = clap_map[dataset]

    plt.figure(figsize=(6, 4.5))

    # plot the straight line for the finetune on retrieved data
    plt.hlines(y_ft_retr, 0, 16, colors='tab:orange', linestyles='solid', label='FT on retrieved (ours)', linewidth=LINEWIDTH, alpha=0.8)


    # plot CMLP data
    plt.plot(x, y_cmlp, label="CrossModal LP (CVPR'23)", linewidth=LINEWIDTH, linestyle='solid', color='tab:blue', marker='X', markersize=MARKERSIZE, alpha=0.8)
    # plot CLAP data
    plt.plot(x, y_clap, label="CLAP (CVPR'24)", linewidth=LINEWIDTH, linestyle='solid', color='tab:olive', marker='P', markersize=MARKERSIZE, alpha=0.8)

    # plot the single point for the real prompt zero-shot accuracy
    plt.plot(0, real_prompt_acc, linestyle=None, color='tab:pink', marker='d', markersize=MARKERSIZE, alpha=1.0)

    # plot the single point for the openai prompt zero-shot accuracy
    plt.plot(0, openai_prompt_acc, linestyle=None, color='tab:gray', marker='s', markersize=MARKERSIZE, alpha=0.8)

    # plot the single point for the real linear zero-shot accuracy
    plt.plot(0, real_linear_acc, linestyle=None, color='tab:purple', marker='p', markersize=MARKERSIZE, alpha=0.8)


    plt.plot(x, y_ft_fs, label='FT on few-shot (ours)', linestyle='solid', linewidth=LINEWIDTH, color='tab:green', marker='o', markersize=MARKERSIZE, alpha=0.8)



    if dataset in swat_improved_map:
        swat_improved = swat_improved_map[dataset]
        plt.plot(x, swat, label='SWAT (ours)', linestyle='solid',linewidth=LINEWIDTH, color='tab:red', marker='^', markersize=MARKERSIZE, alpha=0.8)
        # plt.plot(x, swat_improved, label='SWAT-improved (ours)', linestyle='dashed',linewidth=LINEWIDTH, color='tab:red', marker='^', markersize=MARKERSIZE, alpha=0.8)
    else:
        plt.plot(x, swat, label='SWAT (ours)', linestyle='solid',linewidth=LINEWIDTH, color='tab:red', marker='^', markersize=MARKERSIZE, alpha=0.8)

    # SWAT+
    plt.plot(x, swat_plus, label='SWAT+ (ours)', linestyle='dashed',linewidth=LINEWIDTH, color='tab:red', marker='^', markersize=MARKERSIZE, alpha=0.8)


    plt.title(f"{dataset_name}", fontsize=20)
    plt.xlabel('shots of images per class', fontsize=19)
    plt.ylabel("test accuracy (%)", fontsize=19)
    # only show the ticks at the specified values of x
    plt.xticks(x_fs, fontsize=18)

    if dataset == 'dtd':
        plt.yticks(np.arange(40, 80, 10), fontsize=18)
    # # elif dataset == 'semi-aves':
    # #     plt.yticks(np.arange(0, 70, 10), fontsize=15)
    elif dataset == 'food':
        plt.yticks(np.arange(74, 80, 1), fontsize=18)
    # elif dataset == 'cars':
    #     plt.yticks(np.arange(65, 90, 5), fontsize=15)
    # elif dataset == 'imagenet':
    #     plt.yticks(np.arange(58, 68, 2), fontsize=15)
    # elif dataset == 'pets':
    #     plt.yticks(np.arange(84, 94, 2), fontsize=15)
    # # elif dataset == 'flowers102':
    # #     plt.yticks(np.arange(50, 100, 10), fontsize=15)
    # # elif dataset == 'fgvc-aircraft':
    # #     plt.yticks(np.arange(0, 70, 10), fontsize=15)
    # # elif dataset == 'eurosat':
    # #     plt.yticks(fontsize=15)
    # elif dataset == 'average':
    #     plt.yticks(np.arange(50, 80, 5), fontsize=15)
    else:
        plt.yticks(fontsize=18)


    # plot the legend
    # ft_retr_legend = plt.Line2D([0], [0], marker='v', color='w', markerfacecolor='tab:orange', markersize=MARKERSIZE, label='Finetune on retrieved data')
    real_prompt_legend = plt.Line2D([0], [0], marker='d', color='w', markerfacecolor='tab:pink', markersize=MARKERSIZE, label="REAL-Prompt (CVPR'24)")
    openai_prompt_legend = plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='tab:gray', markersize=MARKERSIZE, label="OpenCLIP (CVPR'23)")
    real_linear_legend = plt.Line2D([0], [0], marker='p', color='w', markerfacecolor='tab:purple', markersize=MARKERSIZE, label="REAL-Linear (CVPR'24)")

    legend_location = 'upper left' if dataset == 'fgvc-aircraft' else 'lower right'


    # Reorder legend items to make 'SWAT' appear first
    handles, labels = plt.gca().get_legend_handles_labels()

    swat_plus_index = labels.index('SWAT+ (ours)')
    handles.insert(0, handles.pop(swat_plus_index))
    labels.insert(0, labels.pop(swat_plus_index))

    swat_index = labels.index('SWAT (ours)')
    handles.insert(1, handles.pop(swat_index))
    labels.insert(1, labels.pop(swat_index))

    ft_fs_index = labels.index('FT on few-shot (ours)')
    handles.insert(2, handles.pop(ft_fs_index))
    labels.insert(2, labels.pop(ft_fs_index))

    ft_retr_index = labels.index('FT on retrieved (ours)')
    handles.insert(3, handles.pop(ft_retr_index))
    labels.insert(3, labels.pop(ft_retr_index))

    # if dataset in swat_improved_map:
    #     swat_improved_index = labels.index('SWAT-improved (ours)')
    #     handles.insert(0, handles.pop(swat_improved_index))
    #     labels.insert(0, labels.pop(swat_improved_index))

    handles = handles + [real_prompt_legend, real_linear_legend, openai_prompt_legend]
    labels = labels + ["REAL-Prompt (CVPR'24)", "REAL-Linear (CVPR'24)", "OpenCLIP (CVPR'23)"]


    # Update the legend with reordered handles and labels

    # plt.legend(handles=handles, labels=labels,
    #         loc=legend_location, prop={'size': 17}, frameon=True, facecolor='white', framealpha=1.0)

    # plt.legend(handles=plt.gca().get_legend_handles_labels()[0] + [real_linear_legend] +[real_prompt_legend] + [openai_prompt_legend]
    #             , loc=legend_location, prop={'size': 11}, frameon=True, facecolor='white', framealpha=0.3)


    plt.grid(alpha=0.2)
    plt.tight_layout()

    plt.savefig(f'results/{dataset}_plot.png', dpi=300)
    # plt.savefig(f'plots/{dataset}_plot_noswat.png', dpi=300)

    plt.clf()  # clear the current figure


if __name__ == '__main__':
    for dataset in [
        'dtd',
        'eurosat',
        'fgvc-aircraft',
        'flowers102',
        'semi-aves',
        'pets',
        'food',
        'cars',
        'imagenet',
        'average'
        ]:
        plot_results(dataset)
