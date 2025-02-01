# import torch
# from utils.models import MyLinear
# import matplotlib.pyplot as plt
import json
import sys


# def plot_norm(sorted_retr_ct, dataset, filename):

#     # get the norm values in the order of the sorted_retr_ct
#     norm1 = []
#     norm2 = []
#     for i, info in sorted_retr_ct.items():
#         norm1.append(info['norm1'])
#         norm2.append(info['norm2'])
#     # set the plot size
#     plt.figure(figsize=(10, 5))
#     plt.plot(norm1, alpha=0.8, label='Before probing', linewidth=2)
#     plt.plot(norm2, alpha=0.8, label='After probing', linewidth=2)
#     if dataset == "Semi-Aves":
#         plt.ylim(0.6, 1.8)
#     elif dataset == "Flowers102":
#         plt.ylim(0.8, 2.2)
#     elif dataset == "FGVC-Aircraft":
#         plt.ylim(1.0, 1.8)
#     elif dataset == "EuroSAT":
#         plt.ylim(1.0, 1.2)
#     elif dataset == "DTD":
#         plt.ylim(1.0, 1.4)
#     plt.xticks(fontsize=16)
#     plt.yticks(fontsize=16)
#     plt.xlabel('Classes sorted by decreasing number of training images', fontsize=19)
#     plt.ylabel('Norm', fontsize=19)
#     plt.legend(fontsize=19)
#     plt.title(f'Norm of per-class weights of learned classifier - {dataset}',fontsize=20)
#     plt.tight_layout()
#     plt.savefig(f'norm_weights/{filename}.png', dpi=600)
#     plt.clf()


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
        # 'flowers102',
        # 'fgvc-aircraft',
        # 'eurosat',
        # 'dtd',
        # 'oxford_pets',
        # 'food101',
        # 'stanford_cars',
        # 'imagenet'
    ]

    # dataset_name_map = {
    #     'semi-aves': 'Semi-Aves',
    #     'flowers102': 'Flowers102',
    #     'fgvc-aircraft': 'FGVC-Aircraft',
    #     'eurosat': 'EuroSAT',
    #     'dtd': 'DTD'
    # }

    for dataset in datasets:
        # print(f'\n{dataset}, {method}, {data_source}, {shots}, {seed}')

        # get the retrived + fewshot data count per class
        fewshot_text = f'../../data/{dataset}/fewshot{shots}_seed1.txt'
        retr_txt = f'../../data/{dataset}/T2T500.txt'

        # get number of classes from the fewshot text file
        with open(fewshot_text, 'r') as f:
            lines = f.readlines()
        classes_set = set()
        for line in lines:
            class_id = line.strip().split(' ')[1]
            classes_set.add(class_id)
        num_classes = len(classes_set)
        # print(f'# of classes: {num_classes}')

        retr_ct = {}
        for i in range(num_classes):
            retr_ct[str(i)] = {'ct': 0, 'norm1': 0.0, 'norm2': 0.0}

        # get the training data count per class
        with open(retr_txt, 'r') as f:
            lines = f.readlines()

        for line in lines:
            class_id = line.strip().split(' ')[1]
            retr_ct[class_id]['ct'] += 1

        # print(retr_ct)

        epochs = 10 if dataset == 'imagenet' else 50

        if method == 'finetune' and data_source == 'fewshot':
            # folder = f'../output/FTFS_vitb32/output_{dataset}'
            # raise NotImplementedError
            pass

        elif method == 'cutmix' and data_source == 'fewshot':
            if dataset == 'oxford_pets':
                dataset = 'oxfordpets'
            folder = f'../../output/FTFS-cutmix_vitb32/output_{dataset}'

        elif method == 'finetune' and data_source == 'retrieved':
            folder = f'../../output/FT_retrieved_vitb32/output_{dataset}'

        elif method == 'finetune' and data_source == 'fewshot+retrieved':
            folder = f'../../output/FT_mixed_vitb32/output_{dataset}'

        elif method == 'cutmix' and data_source == 'fewshot+retrieved': # SWAT
            folder = f'../../output/swat_vitb32_T2T500/output_{dataset}'

        else:
            raise NotImplementedError


        # path = f'{folder}/{dataset}_{method}_{data_source}_REAL-Prompt_{shots}shots_seed{seed}_{epochs}eps/'

        # path = "../../output/swat_ablate_stage2/output_imagenet/imagenet_finetune_fewshot_REAL-Prompt_16shots_seed1_10eps/"
        path = "../../output/swat_ablate_stage2/output_semi-aves/semi-aves_finetune_fewshot_REAL-Prompt_16shots_seed1_10eps/"


        # stage 1 / 2 test scores
        test_score1_path = path+'stage1_test_scores.json'
        test_score2_path = path+'stage1_test_scores.json'
        # test_score2_path = path+'stage2_test_scores.json'

        # load test scores
        with open(test_score1_path, 'r') as f:
            test_scores1 = json.load(f)

        with open(test_score2_path, 'r') as f:
            test_scores2 = json.load(f)

        for i in range(num_classes):
            retr_ct[str(i)]['test_acc1'] = test_scores1['per_class_recall'][str(i)]
            retr_ct[str(i)]['test_acc2'] = test_scores2['per_class_recall'][str(i)]

        # # Load the checkpoint
        # ckpt1 = torch.load(ckpt1_path, map_location='cpu')
        # ckpt2 = torch.load(ckpt2_path, map_location='cpu')

        # classifier_head = MyLinear(inp_dim=512, num_classes=num_classes, bias=False)

        avg_acc1 = test_scores1['acc']*100
        avg_acc2 = test_scores2['acc']*100

        # classifier_head.load_state_dict(ckpt1['head'])
        # weights1 = classifier_head.linear.weight.data
        # norm1 = weights1.norm(dim=1)

        # # assign the norm of each class to retr_ct dict
        # for i in range(num_classes):
        #     retr_ct[str(i)]['norm1'] = norm1[i].item()

        # classifier_head.load_state_dict(ckpt2['head'])
        # weights2 = classifier_head.linear.weight.data
        # norm2 = weights2.norm(dim=1)
        # for i in range(num_classes):
        #     retr_ct[str(i)]['norm2'] = norm2[i].item()

        # sort the retr_ct dict by ct
        sorted_retr_ct = dict(sorted(retr_ct.items(), key=lambda x: x[1]['ct'], reverse=True))
        # print(sorted_retr_ct)

        test_acc1 = []
        test_acc2 = []
        for i, info in sorted_retr_ct.items():
            test_acc1.append(info['test_acc1'])
            test_acc2.append(info['test_acc2'])

        # print the average of first 90% of the classes test_acc1
        head_len = int(0.9*num_classes)
        head_acc1 = test_acc1[:head_len]
        tail_acc1 = test_acc1[head_len:]
        avg_head_acc1 = sum(head_acc1)/len(head_acc1)*100
        avg_tail_acc1 = sum(tail_acc1)/len(tail_acc1)*100

        # print the average of first 90% of the classes test_acc2
        head_acc2 = test_acc2[:head_len]
        tail_acc2 = test_acc2[head_len:]
        avg_head_acc2 = sum(head_acc2)/len(head_acc2)*100
        avg_tail_acc2 = sum(tail_acc2)/len(tail_acc2)*100

        print(f"{dataset}, {num_classes}, {method}, {data_source}, {shots}, {seed}, {round(avg_head_acc1,1)}, {round(avg_tail_acc1,1)}, {round(avg_acc1,1)}, {round(avg_head_acc2,1)}, {round(avg_tail_acc2,1)}, {round(avg_acc2,1)}")

        # print(f'before probing: {round(avg_head_acc1, 1)}, {round(avg_tail_acc1, 1)}, {round(avg_acc1, 1)}')
        # print(f'after probing: {round(avg_head_acc2, 1)}, {round(avg_tail_acc2, 1)}, {round(avg_acc2, 1)}')

        # filename = f'{dataset}_{method}_{data_source}_{shots}shots_seed1'
        # plot_norm(sorted_retr_ct, dataset_name_map[dataset], filename)