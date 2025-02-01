import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NAMES_DICT = {
    'fgvc-aircraft': 'FGVC-Aircraft',
    'eurosat': 'EuroSAT',
    'dtd': 'DTD',
    'semi-aves': 'Semi-Aves',
    'flowers102': 'Flowers102',
}

MEAN_init_dict = {
    'semi-aves': 63.5,
    'flowers102': 97.2,
    'fgvc-aircraft': 58.1,
    'eurosat': 93.7,
    'dtd': 71.3, 
}


def plot_mean_std(mean_list, std_list, dataset_list):
    print(dataset_list)
    x = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    # plot the mean, std
    plt.figure(figsize=(6, 3.5))
    for mean, std, dataset in zip(mean_list, std_list, dataset_list):
        # mean_0 = MEAN_init_dict[dataset]
        # std_0 = 0.0

        # mean = [mean_0] + mean
        # std = [std_0] + std

        # plt.errorbar(x, mean, xerr=std, fmt='o', label=dataset, capsize=5)
        plt.errorbar(x, mean, yerr=std, linestyle='dashed', linewidth=1, marker='o', 
                     markersize=4, label=NAMES_DICT[dataset],
            #  color='tab:blue', 
             capsize=6, barsabove=False, 
             ecolor='tab:red', 
             capthick=1, elinewidth=2,
            #  errorevery=(1,1)
             )

    # plt.title("Stage2Acc Mean and Std", fontsize=20)
    # plt.ylabel('Dataset', fontsize=19)
    # plt.xlabel("Stage2Acc", fontsize=19)

    # plt.legend(fontsize=11, alpha=0.2)
    plt.legend(handles=plt.gca().get_legend_handles_labels()[0],
              loc="lower right", prop={'size': 14}, frameon=True, facecolor='white', framealpha=0.3) 
    
    plt.xlabel('# of total epochs for classifier retraining', fontsize=15)
    plt.ylabel("test accuracy (%)", fontsize=15)
    # plt.ylim(62, 65)

    if dataset == 'flowers102':
        y_lower = 96
        y_upper = 98
    elif dataset == 'eurosat':
        y_lower = 92
        y_upper = 94
    elif dataset == 'dtd':
        y_lower = 70
        y_upper = 72
    elif dataset == 'semi-aves':
        y_lower = 63
        y_upper = 65       
    elif dataset == 'fgvc-aircraft':
        y_lower = 62
        y_upper = 66         
    

    plt.ylim(y_lower, y_upper)

    # Set the y ticks interval to 1
    plt.yticks(np.arange(y_lower, y_upper+1, 1), fontsize=15)
    plt.xticks(np.arange(0, 110, 10), fontsize=15)

    plt.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(f'plots_no_overfit/no_overfit_{dataset}.png', dpi=300)
    plt.clf()
    print('done')




def get_mean_std(dataset):

    fn = f'results_stage2_acc_bar/{dataset}.csv'
    df = pd.read_csv(fn)
    # print(df.shape)

    # group by eps column, and calculate the mean and std value of the Stage2Acc column
    # then make into the mean column and std column
    grouped = df.groupby('eps').agg({
        'Stage2Acc': ['mean', 'std']
    }).reset_index()
    
    mean = list(grouped['Stage2Acc']['mean'].values)
    std = list(grouped['Stage2Acc']['std'].values)

    return mean, std


if __name__ == '__main__':
    dataset_list = [
        'semi-aves', 
        'flowers102', 
        'fgvc-aircraft', 
        'eurosat', 
        'dtd'
        ]

    for dataset in dataset_list:
        mean, std = get_mean_std(dataset)

        # plot
        plot_mean_std([mean], [std], [dataset])
