import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import math

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.style'] = 'normal'
plt.rcParams['font.variant'] = 'normal' 

LARGE_SIZE = 20
SMALL_SIZE = 11

NAMES_DICT = {
    'semi-aves': 'Semi-Aves',
    'flowers102': 'Flowers',    
    'fgvc-aircraft': 'Aircraft',
    'eurosat': 'EuroSAT',
    'dtd': 'DTD',
    'food101': 'Food',
    'stanford_cars': 'Cars',
    'oxford_pets': 'Pets',
    'imagenet': 'ImageNet',
}



def plot_mean_std(mean_list, std_list, dataset_list):

    x = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    # plot the mean, std
    plt.figure(figsize=(5, 3))
    for mean, std, dataset in zip(mean_list, std_list, dataset_list):
        
        # std = [s if s != 0 else 0.00001 for s in std]

        # mean = np.array(mean)
        # std = np.array(std)

        # mean = np.log(mean)
        # std = np.log(std)

        # mean = [math.log2(m) for m in mean]
        # std = [math.log2(s) for s in std]

        # plot the shadow for the std
        plt.fill_between(x, np.array(mean) - np.array(std), np.array(mean) + np.array(std), alpha=0.4)
        
        # plot the mean
        plt.plot(x, mean, 
                 linewidth=1,
                 marker='o', 
                 markersize=2, 
                 label=NAMES_DICT[dataset])
    
    plt.ylim(60,68)
    # set x tick every 10
    plt.xticks(np.arange(0, 110, 10))

    # set y tick every 5
    # plt.yticks(np.arange(80, 100, 5))


    plt.legend(handles=plt.gca().get_legend_handles_labels()[0],
              loc="lower right",
            #   bbox_to_anchor=(0.5, 0.18), 
              prop={'size': 10}, 
              ncol=3, 
              frameon=True, facecolor='white', 
              framealpha=0.6)   
        
    plt.xlabel('# of total epochs for classifier retraining', fontsize=SMALL_SIZE)
    plt.ylabel("test accuracy (%)", fontsize=SMALL_SIZE)
    plt.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(f'no_overfit_all.png', dpi=300)

    print('done')




def get_mean_std(dataset):

    # fn = f'swat_ablate_stage2_epochs/{dataset}.csv'
    fn = f'swat_ablate_stage2_epochs_trainingseed/{dataset}.csv'
    df = pd.read_csv(fn, header=0)

    # group by epoch column, and calculate the mean and std value of the Stage1Acc column
    # then make into the mean column and std column
    stats = df.groupby(df['epoch'].astype(int))['Stage1Acc'].agg(['mean', 'std']).reset_index()
    
    mean = list(stats['mean'].values)
    std = list(stats['std'].values)

    # save the mean, std into a csv file
    if not os.path.exists('mean_std'):
        os.mkdir('mean_std')
    stats.to_csv(f'mean_std/{dataset}.csv', index=False)

    return mean, std


if __name__ == '__main__':

    # dataset_list = [
    #     # 'semi-aves', 
    #     'flowers102', 
    #     # 'fgvc-aircraft', 
    #     'eurosat', 
    #     # 'dtd',
    #     'oxford_pets',
    #     # 'food101',
    #     'stanford_cars',
    #     # 'imagenet'
    #     ]

    # dataset_list = [
    #     'semi-aves', 
    #     # 'flowers102', 
    #     'fgvc-aircraft', 
    #     # 'eurosat', 
    #     'dtd',
    #     # 'oxford_pets',
    #     # 'food101',
    #     # 'stanford_cars',
    #     # 'imagenet'
    #     ]    
    
    dataset_list = [
        'semi-aves', 
        # 'flowers102', 
        'fgvc-aircraft', 
        # 'eurosat', 
        'dtd',
        # 'oxford_pets',
        # 'food101',
        # 'stanford_cars',
        # 'imagenet'
        ] 

    mean_list = []
    std_list = []
    for dataset in dataset_list:
        mean, std = get_mean_std(dataset)

        # plot the mean, std
        mean_list.append(mean)
        std_list.append(std)

    # plot
    plot_mean_std(mean_list, std_list, dataset_list)
