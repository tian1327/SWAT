import json
import sys


if __name__ == '__main__':

    # second argument is method, either 'finetune' or 'cutmix'
    # third argument is data_source, either 'fewshot' or 'retrieved' or 'mixed'
    # fourth argument is shots, 4/8/16
    # fifth argument is seed, 1/2/3

    if len(sys.argv) != 5:
        print('Usage: python plot_classifier_weights.py <method> <data_source> <shots> <seed>')
        exit()

    method = sys.argv[1]
    data_source = sys.argv[2]
    shots = sys.argv[3]
    seed = sys.argv[4]

    datasets = [
        'semi-aves',
        'flowers102',
        'fgvc-aircraft',
        'eurosat',
        'dtd',
        'oxford_pets',
        'food101',
        'stanford_cars',
        'imagenet'
    ]
    
    acc_list = []
    for dataset in datasets:

        epochs = 10 if dataset == 'imagenet' else 50

        if method == 'finetune' and data_source == 'fewshot':    
            # folder = f'../output/FTFS_vitb32/output_{dataset}'
            raise NotImplementedError

        elif method == 'cutmix' and data_source == 'fewshot':                
            if dataset == 'oxford_pets':
                dataset = 'oxfordpets'
            folder = f'../../output/FTFS-cutmix_vitb32/output_{dataset}'

        elif method == 'finetune' and data_source == 'retrieved':
            folder = f'../../output/FT_retrieved_vitb32/output_{dataset}'

        elif method == 'finetune' and data_source == 'fewshot+retrieved':
            folder = f'../../output/FT_mixed_vitb32/output_{dataset}'

        elif (method == 'cutmix' or method == 'swat') and data_source == 'fewshot+retrieved': # SWAT
            folder = f'../../output/swat_vitb32_T2T500/output_{dataset}'
        
        else:
            raise NotImplementedError
                
        method_final = 'cutmix' if method == 'swat' else method

        path = f'{folder}/{dataset}_{method_final}_{data_source}_REAL-Prompt_{shots}shots_seed{seed}_{epochs}eps/'

        # stage 1 / 2 test scores
        test_score1_path = path+'stage1_test_scores.json'
        test_score2_path = path+'stage2_test_scores.json'

        # load test scores
        with open(test_score1_path, 'r') as f:
            test_scores1 = json.load(f)

        with open(test_score2_path, 'r') as f:
            test_scores2 = json.load(f)
        
        if method == 'finetune' and data_source == 'fewshot':    
            # folder = f'../output/FTFS_vitb32/output_{dataset}'
            raise NotImplementedError

        elif method == 'cutmix' and data_source == 'fewshot':                
            acc = test_scores1['acc']

        elif method == 'finetune' and data_source == 'retrieved':
            acc = test_scores1['acc']

        elif method == 'finetune' and data_source == 'fewshot+retrieved':
            acc = test_scores1['acc']

        elif method == 'cutmix' and data_source == 'fewshot+retrieved': 
            acc = test_scores1['acc']
        
        elif method == 'swat' and data_source == 'fewshot+retrieved': # SWAT
            acc = test_scores2['acc']
        
        else:
            raise NotImplementedError
        
        acc = acc*100
        acc_list.append(round(acc,1))
    
    # calculate the mean of acc_list and append to the end of acc_list
    mean = sum(acc_list)/len(acc_list)
    acc_list.append(round(mean,1))
    
    acc_list = [str(x) for x in acc_list]

    output = " ".join(acc_list)
    print(output)
    

