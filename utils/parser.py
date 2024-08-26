import argparse
import os
from utils.extras import get_class_num_list, str2bool
import yaml


def parse_args():

    parser = argparse.ArgumentParser(description='Arguments for script.')
    
    # logging
    parser.add_argument('--log_mode', type=str, default='both', choices=['console', 'file', 'both'], help='where to log.')
    parser.add_argument('--folder', type=str, default='output', help='Folder for saving output.')
    parser.add_argument('--prefix', type=str, default=None, help='Prefix for Log file Name.')

    # model
    parser.add_argument('--model_cfg', type=str, default='vitb32_openclip_laion400m', 
                        choices=['vitb32_openclip_laion400m', 'vitb16_openclip_laion400m',
                                 'vitb32_openclip_laion2b', 'rn50_openclip_openai',
                                 'vitb32_clip', 'vitb16_clip', 'rn50_clip'
                                 ],
                        help='ViT Transformer arch.')
    # parser.add_argument('--resume_path', type=str, help='Model path to resume training for.')
    parser.add_argument('--model_path', default=None, type=str, help='Model path to start training from.')    

    # prompt
    parser.add_argument('--prompt_name', type=str, default='most_common_name',
                        choices=['most_common_name', 'most_common_name_REAL', 'name', 'name-80prompts',
                                 'c-name', 's-name', 't-name', 'f-name', 'c-name-80prompts'], help='names for prompts.')
    parser.add_argument('--use_attribute', default=False, type=str2bool, help='Use attribute when making prompts.')

    # dataset
    parser.add_argument('--dataset', type=str, default='semi-aves', 
                        choices=['semi-inat-2021', 'semi-aves', 'flowers102', 'cub2011', 'imagenet_1k',
                                 'fgvc-aircraft', 'dtd', 'eurosat',
                                 'dtd_selected'
                                 ], 
                        help='Dataset name.')
    
    # retrieval
    parser.add_argument('--database', type=str, default='LAION400M', help='Database from which images are mined.')

    # training data
    parser.add_argument('--data_source', type=str, default='fewshot', 
                        choices=['fewshot', 'retrieved', 'fewshot+retrieved', 'dataset-cls', 
                                 'ltrain', 'ltrain+val', 'ltrain+val+unlabeled', 
                                 'ltrain+val+unlabeled+retrieved',
                                 'fewshot+unlabeled', 'fewshot+retrieved+unlabeled'], 
                        help='training data source.')
    parser.add_argument('--shots', type=int, default=16, help='number of shots for fewshot data')
    # parser.add_argument('--fewshot_split', type=str, default='fewshotX.txt', help='fewshot file name.')
    parser.add_argument('--retrieval_split', type=str, default='T2T500+T2I0.25.txt', help='retrieval file name.')
    parser.add_argument('--unlabeled_split', type=str, default='u_train_in_oracle.txt', help='unlabeled in domain data file name.')
    parser.add_argument('--val_split', type=str, default='fewshotX.txt', help='val file name.')
    parser.add_argument('--test_split', type=str, default='test.txt', help='test file name.')
    parser.add_argument('--seed', type=int, default=1, help='Random seeds for different splits.')
    parser.add_argument('--training_seed', type=int, default=1, help='Random seeds for training.') # this is used for stage 2 probing loss error bars

    # training
    parser.add_argument('--method', type=str, default='finetune', choices=['zeroshot','probing', 'finetune', 'finetune-mixed',
                                                                           'finetune-multitask', 'CMLP',
                                                                            'mixup',  'mixup-fs', 'cutmix', 'cutmix-fs',
                                                                            'resizemix', 'dataset-cls',
                                                                            'saliencymix', 'attentivemix', 'CMO',
                                                                            'FLYP', 'fixmatch'], 
                        help='Method for training.')
    parser.add_argument('--cls_init', type=str, default='REAL-Prompt', choices=['random', 'text', 'REAL-Prompt', 'REAL-Linear'], 
                        help='Initialize the classifier head in different ways.')

    parser.add_argument('--mix_prob', type=float, default=0.5, help='Mixing probability, i.e. use mixing strategy or not. Option applied to all mixing methods.')    
    parser.add_argument('--mixup_alpha', type=float, default=1.0, help='Mixup alpha for Beta distribution.')
    parser.add_argument('--skip_stage2', action='store_true', help='Set to skip stage 2 probing')

    # attentive mix
    parser.add_argument('--attentive_threshold', type=float, default=0.85, help='Threshold for heatmap binary mask.')
    parser.add_argument('--attentive_name', type=str, default='c-name', choices=['general', 'c-name', 's-name'], help='Which name to use for GEM localization.')

    parser.add_argument('--cutmix_beta', type=float, default=1.0, help='cutmix beta for Beta distribution. 1.0 means uniform distribution.')   
    # parser.add_argument('--cross_modal', default=False, type=str2bool, help='cross-modal adaptation.')
    parser.add_argument('--recal_prompt', action='store_true', help='Recalculate the prompt embedding or not.')
    parser.add_argument('--recal_fea', action='store_true', help='re-run feature extraction.')
    parser.add_argument('--pre_extracted', default=False, type=str2bool, help='use pre-extracted features.')
    parser.add_argument('--locked_text', action='store_true', help='Set to freeze the text encoder during training.')
    parser.add_argument('--freeze_visual', default=False, type=str2bool, help='Freeze the visual encoder during training.')
    parser.add_argument('--tau_norm', default=True, type=str2bool, help='try tau normalization, select best tau on val set.')  
    
    # CMO
    parser.add_argument('--cmo_alpha', type=float, default=1.0, help='alpha for CMO weights scaling for minority classes.')

    # fixmatch
    parser.add_argument('--mu', type=int, default=1, help='number of times of the batch size for few-shot data.')
    parser.add_argument('--threshold', type=float, default=0.95, help='confidence threshold to retain the pseudo-labels.')
    parser.add_argument('--lambda_u', type=float, default=1.0, help='weight for the consistency loss.')

    # parser.add_argument('--resume_epochs', type=int, default=0, help='resume training from a checkpoint of kth epoch.')
    parser.add_argument('--check_zeroshot', action='store_true', help='check zeroshot acc.')
    parser.add_argument('--zeroshot_only', action='store_true', help='run zeroshot only.')
    # parser.add_argument('--probe_again', default=True, type=str2bool, help='run stage 2 probing again after finetuning')  
    # parser.add_argument('--train_till_converge', default=False, type=str2bool, help='training until converge.')   
    parser.add_argument('--early_stop', default=False, type=str2bool, help='use val set for early stopping.')    
    parser.add_argument('--epochs', type=int, default=0, help='number of epochs to train the model')
    parser.add_argument('--stop_epochs', type=int, default=200, help='number of epochs to stop the training of the model')
    # parser.add_argument('--patience', type=int, default=20, help='Patience for early stopping in epochs.')
    
    # optimization
    parser.add_argument('--num_workers', type=int, default=8, help='Num of workers.') 
    parser.add_argument('--start_validation', type=int, default=0, help='Start validation after x iterations.')    
    # parser.add_argument('--val_every_iter', type=int, default=10, help='Validate every x iterations.')
    parser.add_argument('--lr_classifier', type=float, default=1e-4, help='Learning rate for the classifier head.')
    parser.add_argument('--lr_backbone', type=float, default=1e-6, help='Learning rate for the visual encoder.')
    parser.add_argument('--lr_projector', type=float, default=None, help='Learning rate for the visual and text projector.')    
    parser.add_argument('--wd', type=float, default=1e-2, help='weight decay for model.')
    parser.add_argument('--bsz', type=int, default=32, help='Batch Size')
    parser.add_argument('--optim', type=str, default='AdamW', choices=['AdamW', 'SGD'], help='type of optimizer to use.')
    parser.add_argument('--temperature', type=float, default=0.07, help='Logit Scale for training')
    parser.add_argument('--alpha', type=float, default=0.5, help='mixing ratio for WiSE-FT, alpha=1.0 means no WiSE-FT ensembling.')

    # loss
    parser.add_argument('--loss_name', type=str, default='CE', choices=['CE', 'WeightedCE', 'Focal', 'BalancedSoftmax'], help='type of loss function to use.')
    parser.add_argument('--dataset_wd', type=float, default=1.0, help='weight decay for dataset classification loss.')
    parser.add_argument('--fewshot_weight', type=float, default=1.0, help='fewshot weights for WeightedCE.')
    parser.add_argument('--focal_alpha', type=float, default=1.0, help='alpha for Focal loss.')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='gamma for Focal loss.')  

    # save
    parser.add_argument('--save_ckpt', default=False, type=str2bool, help='Save model checkpoints or not.')
    parser.add_argument('--save_freq', type=int, default=10, help='Save Frequency in epoch.')

    # other
    parser.add_argument('--utrain', type=str, default=None, help='filepath to the unlabeled data with pseudo-labels')
    
    args = parser.parse_args()

    # read the dataset and retrieved path from the config.yml file
    with open('config.yml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        args.dataset_path = config['dataset_path']
        args.retrieved_path = config['retrieved_path']

    if args.method == 'zeroshot':
        args.check_zeroshot = True
        args.zeroshot_only = True
        args.skip_stage2 = True

    # adjust prompt_name based on cls_init
    if args.cls_init == 'REAL-Prompt' or args.cls_init == 'REAL-Linear':
        args.prompt_name = 'most_common_name'
    elif args.cls_init == 'text':
        args.prompt_name = 'name'
    elif args.cls_init == 'random':
        args.prompt_name = 'most_common_name'        

    if args.method == "probing" or args.method == "CMLP":
        args.freeze_visual = True
        # args.pre_extracted = True # because stage 2 has to recalculate the feature using stage 2 model 
    else:
        args.freeze_visual = False
        args.pre_extracted = False

    if not args.freeze_visual:
        assert args.pre_extracted==False, \
            'visual encoder not frozen, pre-extracted features are not compatible.'

    if args.model_path is not None:
        assert args.pre_extracted==False, 'reloading a trained model, pre-extracted features are not compatible.'
    
    if args.method == 'CMLP' or args.method == 'finetune-mixed':
        args.bsz = int(args.bsz / 2)
 
 
    #---------- adjust the train and val split based on shot, seed, data_source
    args.fewshot_data = [[f'fewshot{args.shots}_seed{args.seed}.txt'], [os.path.join(args.dataset_path, args.dataset)]]
    args.retrieval_data = [[args.retrieval_split], [os.path.join(args.retrieved_path, args.dataset)]]
    args.val_split = [[f'fewshot{args.shots}_seed{args.seed}.txt'], [os.path.join(args.dataset_path, args.dataset)]]
    args.test_split = [['test.txt'], [os.path.join(args.dataset_path, args.dataset)]]

    if args.data_source == 'fewshot':
        args.train_split = [[f'fewshot{args.shots}_seed{args.seed}.txt'], [os.path.join(args.dataset_path, args.dataset)]]
    
    elif args.data_source == 'retrieved':
        args.train_split = [[args.retrieval_split], [os.path.join(args.retrieved_path, args.dataset)]]
    
    elif args.data_source == 'fewshot+retrieved':
        args.train_split = [[f'fewshot{args.shots}_seed{args.seed}.txt', args.retrieval_split], 
                            [os.path.join(args.dataset_path, args.dataset), os.path.join(args.retrieved_path, args.dataset)]]
    
    elif args.data_source == 'fewshot+unlabeled':
        args.train_split = [[f'fewshot{args.shots}_seed{args.seed}.txt', args.unlabeled_split], 
                            [os.path.join(args.dataset_path, args.dataset), os.path.join(args.dataset_path, args.dataset)]]

    elif args.data_source == 'fewshot+retrieved+unlabeled':
        args.train_split = [[f'fewshot{args.shots}_seed{args.seed}.txt', args.retrieval_split, args.unlabeled_split], 
                            [os.path.join(args.dataset_path, args.dataset), os.path.join(args.retrieved_path, args.dataset), 
                             os.path.join(args.dataset_path, args.dataset)]]

    elif args.data_source == 'ltrain':
        args.train_split = [[f'ltrain.txt'], [os.path.join(args.dataset_path, args.dataset)]]
        args.val_split = [[f'test.txt'], [os.path.join(args.dataset_path, args.dataset)]] # use test set as val set
        args.early_stop = True

    elif args.data_source == 'ltrain+val':
        args.train_split = [[f'ltrain+val.txt'], [os.path.join(args.dataset_path, args.dataset)]]
        args.val_split = [[f'test.txt'], [os.path.join(args.dataset_path, args.dataset)]] # use test set as val set
        args.early_stop = True
    
    elif args.data_source == 'ltrain+val+unlabeled':
        args.train_split = [[f'ltrain+val.txt', args.unlabeled_split], [os.path.join(args.dataset_path, args.dataset), 
                                                                        os.path.join(args.dataset_path, args.dataset)]]
        args.val_split = [[f'test.txt'], [os.path.join(args.dataset_path, args.dataset)]] # use test set as val set
        args.early_stop = True
    
    elif args.data_source == 'ltrain+val+unlabeled+retrieved':
        args.train_split = [[f'ltrain+val.txt', args.unlabeled_split, args.retrieval_split], 
                            [os.path.join(args.dataset_path, args.dataset), os.path.join(args.dataset_path, args.dataset), 
                             os.path.join(args.retrieved_path, args.dataset)]]
        args.val_split = [[f'test.txt'], [os.path.join(args.dataset_path, args.dataset)]] # use test set as val set
        args.early_stop = True

    elif args.data_source == 'dataset-cls':
        args.train_split = [['dataset_train.txt'], ['']] # note here the second element for the path is empty, just for dataset classification
        args.val_split = [['dataset_val.txt'], ['']]
        args.test_split = [['dataset_test.txt'], ['']]
    else:
        raise NotImplementedError


    # adjust train_split for fixmatch
    if args.method == 'fixmatch':
        # args.train_split = [[f'fewshot{args.shots}_seed{args.seed}.txt'], 
        #                     [os.path.join(args.dataset_path, args.dataset)]]
        
        args.train_split = [[f'ltrain+val.txt'], 
                            [os.path.join(args.dataset_path, args.dataset)]]

        # note here, we add the labeled data to the unlabeled split based on the original implementation
        # args.u_train_split = [[f'fewshot{args.shots}_seed{args.seed}.txt', args.unlabeled_split], 
        #                     [os.path.join(args.dataset_path, args.dataset), os.path.join(args.dataset_path, args.dataset)]]
 
        args.u_train_split = [['ltrain+val.txt', args.unlabeled_split], 
                            [os.path.join(args.dataset_path, args.dataset), os.path.join(args.dataset_path, args.dataset)]]
 

        # args.u_train_split = [[args.unlabeled_split], 
        #                     [os.path.join(args.dataset_path, args.dataset)]]


    # adjust folder
    args.folder = f'{args.folder}/output_{args.dataset}'

    # build cls_num_list
    args.dataset_root = f'data/{args.dataset}'
    # test_file = os.path.join(args.dataset_root, args.test_split[0][0])
    # cls_num_list = get_class_num_list(test_file)
    # args.cls_num_list = cls_num_list

    return args