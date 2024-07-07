import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import json
from utils.models import MyLinear, save_model_ckpt, save_test_scores, save_best_model#, save_head_weights
import time
import numpy as np
from utils import features
from utils.parser import parse_args
from utils.logger import get_logger
from utils.optimizers import get_optimizer
from utils.extras import transform
from testing import validate, validate_multitask, validate_dataset
from testing import calculate_scores
from utils.extras import get_engine#, cal_hard_avg_acc, cal_easy_avg_acc
from utils.datasets.dataset_utils import NUM_CLASSES_DICT, load_dataset, TensorDataset, TextTensorDataset
import random
from utils.prompt_templates import prompt_maker
from utils.optimizers import get_warmup_scheduler
import copy
from utils.features import extract_test_feats
from utils.losses import FocalLoss, WeightedCELoss, BalancedSoftmaxLoss
from utils.methods import train_flyp
import torch.nn.functional as F
import cv2
# from gem import create_gem_model
# import pickle


def set_logger(args):
      
    # case_name
    case_name = f'{args.prefix+"_" if args.prefix else ""}{args.dataset}_{args.method}_{args.data_source}_{args.cls_init}_{args.shots}shots_seed{args.seed}_{args.epochs}eps'

    # setup path
    output_dir = os.path.join(args.folder, f'{case_name}')
    if not os.path.exists(f'{output_dir}'):
        os.makedirs(f'{output_dir}')

    ckpt_path = f'{output_dir}/model_ckpts'
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    if not os.path.exists(f'{args.dataset_root}/pre_extracted/'):
        os.makedirs(f'{args.dataset_root}/pre_extracted/')
        print(f'Created directory: {args.dataset_root}/pre_extracted/')

    ## setup logger
    logger = get_logger(f'{output_dir}', 'main', args.log_mode)
    logger.info('logging started')
    logger.info(f'case_name: {case_name}')

    # print args
    for arg in vars(args):
        logger.info(f'{arg} = {getattr(args, arg)}')    

    loss_logger = open(f'{output_dir}/loss.csv', 'w')  
    loss_logger.write(f'Epoch,Iter,Train_loss,Val_loss,Val_acc,Test_acc\n')

    # print train, val, test split
    logger.info(f'train_split: {args.train_split}')
    logger.info(f'val_split: {args.val_split}')
    logger.info(f'test_split: {args.test_split}')

    # device 
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {args.device}")
    if torch.cuda.is_available():
        logger.info(f'Number of GPUs available: {torch.cuda.device_count()}')
    
    args.output_dir = output_dir
    args.ckpt_path = ckpt_path

    return logger, loss_logger


def set_model(args, logger):

    model, preprocess, tokenizer = get_engine(model_cfg=args.model_cfg, device=args.device)
    logger.info(f'Loaded model: {args.model_cfg}')
    # if torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model)

    return model, preprocess, tokenizer

def set_prompt(args, model, tokenizer, logger):

    prompt_tensors_dict, text_prompts_dict, \
        tokenized_text_prompts_dict = get_prompts_tensors(args, model, tokenizer, logger) 
       
    prompt_tensors = prompt_tensors_dict[args.prompt_name]
    text_prompts = text_prompts_dict[args.prompt_name]
    tokenized_text_prompts = tokenized_text_prompts_dict[args.prompt_name]    

    return prompt_tensors, text_prompts, tokenized_text_prompts, prompt_tensors_dict


def set_classifier(args, prompt_tensors, logger):

    if args.method == "dataset-cls":
        num_class = 2 # binary classification, retrieved 0 or fewshot 1
        classifier_head = MyLinear(inp_dim=512, num_classes=num_class, bias=False)    
        logger.info(f'Initialized classifier head with random weights. Num of classes: {num_class}')

    elif args.cls_init == 'REAL-Prompt' or args.cls_init == 'REAL-Linear' or args.cls_init == 'text':
        weights = features.prompt_sampler(prompt_tensors, sample_by='mean')
        classifier_head = MyLinear(weights=weights)
        logger.info(f'Initialized classifier head with text embedding. weights.shape: {weights.shape}')

    elif args.cls_init == 'random':
        num_class = NUM_CLASSES_DICT[args.dataset]
        classifier_head = MyLinear(inp_dim=512, num_classes=num_class, bias=False)    
        logger.info(f'Initialized classifier head with random weights. Num of classes: {num_class}')
    else:
        raise NotImplementedError(f'Classifier head {args.cls_init} not implemented.')

    classifier_head.to(args.device)
    return classifier_head
 

def extract_dataloader(args, best_model, split, fea_path, bsz=128):

    # extract features using the best model
    # logger.info(f'Extracting features ...')
    dataset = load_dataset(dataset_root=args.dataset_root, 
                                split=split,
                                preprocess=preprocess,
                                tokenized_text_prompts=tokenized_text_prompts,
                                )
    dataloader = DataLoader(dataset, batch_size=bsz, 
                            shuffle=False, num_workers=args.num_workers, drop_last=False)

    features = extract_test_feats(best_model, dataloader=dataloader)
    torch.save(features, fea_path)
    # logger.info(f'Saved features to {fea_path}')    

    dataset = TensorDataset(pre_extracted_path=fea_path, device=args.device)
    # logger.info(f'Loaded pre-extracted features from: {fea_path}')
    dataloader = DataLoader(dataset, batch_size=bsz, shuffle=False, drop_last=False, num_workers=0) 

    return dataloader

def extract_train_dataloader(args, logger, best_model, split, fea_path, bsz=128):

    # extract features using the best model
    # logger.info(f'Extracting features ...')
    dataset = load_dataset(dataset_root=args.dataset_root, 
                                split=split,
                                preprocess=preprocess,
                                tokenized_text_prompts=tokenized_text_prompts,
                                pl_list=None)
    dataloader = DataLoader(dataset, batch_size=bsz, pin_memory=True,
                            shuffle=True, num_workers=args.num_workers, drop_last=True)

    features = extract_test_feats(best_model, dataloader=dataloader)
    torch.save(features, fea_path)
    # logger.info(f'Saved features to {fea_path}')    

    dataset = TensorDataset(pre_extracted_path=fea_path, device=args.device)
    dataloader = DataLoader(dataset, batch_size=bsz, shuffle=True, drop_last=True, num_workers=0) 

    return dataloader

def get_dataloader_preextracted(args, pre_extract_train_fea_path, pre_extract_val_fea_path, 
                                pre_extract_test_fea_path, device):

    train_dataset = TensorDataset(pre_extracted_path=pre_extract_train_fea_path, device=device)
    logger.info(f'Loaded pre-extracted train features from: {pre_extract_train_fea_path}')
    train_loader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=True, drop_last=True, num_workers=0)

    val_dataset = TensorDataset(pre_extracted_path=pre_extract_val_fea_path, device=device)
    logger.info(f'Loaded pre-extracted val features from: {pre_extract_val_fea_path}')
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, drop_last=False, num_workers=0)

    test_dataset = TensorDataset(pre_extracted_path=pre_extract_test_fea_path, device=device)
    logger.info(f'Loaded pre-extracted test features from: {pre_extract_test_fea_path}')
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, drop_last=False, num_workers=0)    

    return train_loader, val_loader, test_loader


def get_dataloader(args, train_split, val_split, test_split, tokenized_text_prompts, preprocess, utrain_labels=None):

    train_dataset = load_dataset(dataset_root=args.dataset_root, 
                                split=train_split,                                                                                                       
                                preprocess=transform(224, 'train'),
                                tokenized_text_prompts=tokenized_text_prompts,
                                )

    val_dataset = load_dataset(dataset_root=args.dataset_root, 
                                split=val_split, 
                                preprocess=preprocess,
                                tokenized_text_prompts=tokenized_text_prompts,
                                )

    test_dataset = load_dataset(dataset_root=args.dataset_root, 
                                split=test_split, preprocess=preprocess,
                                tokenized_text_prompts=tokenized_text_prompts,
                                )        

    train_loader = DataLoader(train_dataset, batch_size=args.bsz, pin_memory=True,
                            shuffle=True, drop_last=True, num_workers=args.num_workers)

    val_loader = DataLoader(val_dataset, batch_size=128, drop_last=False, pin_memory=True,
                            shuffle=False, num_workers=args.num_workers)

    test_loader = DataLoader(test_dataset, batch_size=128, drop_last=False, pin_memory=True,
                            shuffle=False, num_workers=args.num_workers)

    return train_loader, val_loader, test_loader

def get_retrieve_fewshot_dataloader(args, retrieve_split, fewshot_split, tokenized_text_prompts, preprocess, utrain_labels=None):

    train_dataset_retr = load_dataset(dataset_root=args.dataset_root, 
                                split=retrieve_split,                                                                                                       
                                preprocess=transform(224, 'train'),
                                tokenized_text_prompts=tokenized_text_prompts,
                                pl_list=utrain_labels,
                                )

    train_dataset_fs = load_dataset(dataset_root=args.dataset_root, 
                                split=fewshot_split,                                                                                                       
                                preprocess=transform(224, 'train'),
                                tokenized_text_prompts=tokenized_text_prompts,
                                pl_list=utrain_labels,
                                )

    train_dataloader_retr = DataLoader(train_dataset_retr, batch_size=args.bsz, 
                            shuffle=True, drop_last=True, num_workers=args.num_workers)

    train_dataloader_fewshot = DataLoader(train_dataset_fs, batch_size=args.bsz, 
                            shuffle=True, drop_last=True, num_workers=args.num_workers)

    return train_dataloader_retr, train_dataloader_fewshot


def pre_extract_feature(args, model, tokenized_text_prompts, preprocess):

    pre_extract_train_fea_path = f'{args.dataset_root}/pre_extracted/{args.dataset}_{args.model_cfg}_{args.seed}_train_features.pth'
    pre_extract_val_fea_path = f'{args.dataset_root}/pre_extracted/{args.dataset}_{args.model_cfg}_{args.seed}_val_features.pth'
    pre_extract_test_fea_path = f'{args.dataset_root}/pre_extracted/{args.dataset}_{args.model_cfg}_{args.seed}_test_features.pth'

    if args.recal_fea or not os.path.exists(pre_extract_train_fea_path):
        train_dataset = load_dataset(dataset_root=args.dataset_root, 
                                    split=args.train_split, 
                                    preprocess=transform(224, 'train'),
                                    tokenized_text_prompts=tokenized_text_prompts,
                                    pl_list=None)
        train_loader = DataLoader(train_dataset, batch_size=128, 
                                    shuffle=False, num_workers=args.num_workers, drop_last=False)

        train_features = extract_test_feats(model, dataloader=train_loader)
        torch.save(train_features, pre_extract_train_fea_path)
        logger.info(f'Extracted train features to {pre_extract_train_fea_path}')

    if args.recal_fea or not os.path.exists(pre_extract_val_fea_path):
        val_dataset = load_dataset(dataset_root=args.dataset_root, 
                                    split=args.val_split,                                    
                                    preprocess=preprocess,
                                    tokenized_text_prompts=tokenized_text_prompts,
                                    pl_list=None)
        val_loader = DataLoader(val_dataset, batch_size=128, 
                                    shuffle=False, num_workers=args.num_workers, drop_last=False)

        val_features = extract_test_feats(model, dataloader=val_loader)
        torch.save(val_features, pre_extract_val_fea_path)
        logger.info(f'Extracted val features to {pre_extract_val_fea_path}')
    
    if args.recal_fea or not os.path.exists(pre_extract_test_fea_path):
        test_dataset = load_dataset(dataset_root=args.dataset_root, 
                                    split=args.test_split,
                                    preprocess=preprocess,
                                    tokenized_text_prompts=tokenized_text_prompts,
                                    pl_list=None)
        test_loader = DataLoader(test_dataset, batch_size=128, 
                                    shuffle=False, num_workers=args.num_workers, drop_last=False)

        test_features = extract_test_feats(model, dataloader=test_loader)
        torch.save(test_features, pre_extract_test_fea_path)
        logger.info(f'Extracted test features to {pre_extract_test_fea_path}')
    
    return pre_extract_train_fea_path, pre_extract_val_fea_path, pre_extract_test_fea_path


def get_prompts_tensors(args, model, tokenizer, logger):

    dataset_root = f'{args.retrieved_path}/{args.dataset}'
    metric_fn = f'{dataset_root}/{args.dataset}_metrics-{args.database.upper()}.json' 
    with open(metric_fn, 'r') as f:
        metrics = json.load(f)
    logger.info(f'Loaded metrics from: {metric_fn}')
    logger.info(f'len(metrics): {len(metrics)}')

    prompts_dir = os.path.join(args.retrieved_path, args.dataset, 'pre_extracted/')
    if not os.path.exists(prompts_dir):
        os.makedirs(prompts_dir)
        logger.info(f'Created directory: {prompts_dir}')

    prompt_tensors_dict= {}
    text_prompts_dict = {}
    tokenized_text_prompts_dict = {}
    # for label_type in ['s-name', 'c-name', 't-name', 'f-name', 'c-name_attribute', 'c-name-80prompts']:
    for label_type in [args.prompt_name]:
        prompt_tensors_filename = f"{prompts_dir}{args.dataset}_{args.model_cfg}_{label_type}_prompt_tensors.pth"
        text_prompts_filename = f"{prompts_dir}{args.dataset}_{args.model_cfg}_{label_type}_text_prompts.pth"
        tokenized_text_prompts_filename = f"{prompts_dir}{args.dataset}_{args.model_cfg}_{label_type}_tokenized_text_prompts.pth"

        if not args.recal_prompt and os.path.exists(tokenized_text_prompts_filename):
            logger.info(f'Loading prompt tensors from {prompt_tensors_filename}')
            prompt_tensors = torch.load(prompt_tensors_filename)
            prompt_tensors_dict[label_type] = prompt_tensors

            text_prompts = torch.load(text_prompts_filename)
            text_prompts_dict[label_type] = text_prompts

            tokenized_text_prompts = torch.load(tokenized_text_prompts_filename)
            tokenized_text_prompts_dict[label_type] = tokenized_text_prompts

        else:
            logger.info(f'Calculating prompt tensors for {label_type} ...')
            text_prompts = prompt_maker(metrics, args.dataset, label_type)
            # text_prompts is a dict with key=class_id, value={'corpus': prompt_lst}
            text_prompts_dict[label_type] = text_prompts
            torch.save(text_prompts, text_prompts_filename)
            logger.info(f'Saved text prompts to {text_prompts_filename}')

            # tokenize the text_prompts first in case of finetune needed
            tokenized_text_prompts = features.get_text_features(model, text_prompts, tokenize=tokenizer, operation='tokenize')
            tokenized_text_prompts_dict[label_type] = tokenized_text_prompts
            torch.save(tokenized_text_prompts, tokenized_text_prompts_filename)
            logger.info(f'Saved tokenized text prompts to {tokenized_text_prompts_filename}')

            prompt_tensors = features.get_text_features(model, text_prompts, tokenizer, 'encode')
            prompt_tensors_dict[label_type] = prompt_tensors
            torch.save(prompt_tensors, prompt_tensors_filename)
            logger.info(f'Saved prompt tensors to {prompt_tensors_filename}')
    
    return prompt_tensors_dict, text_prompts_dict, tokenized_text_prompts_dict


def get_text_dataloader(args, prompt_tensors, device):

    text_dataset = TextTensorDataset(prompt_tensors, device) 
    text_dataloader = DataLoader(text_dataset, batch_size=args.bsz, shuffle=True, 
                                num_workers=0, drop_last=True)
    return text_dataloader
    

def set_dataloaders(args, model, tokenized_text_prompts, preprocess, logger):    

    # pre-extracted features
    if args.pre_extracted:
        train_fea_path, val_fea_path, test_fea_path = pre_extract_feature(args, model, tokenized_text_prompts, preprocess)
    
    # dataset
    if args.utrain:
        logger.info(f'Train with labeled and unlabeled data.')
        with open(args.utrain, 'r') as f:
            utrain_labels = f.readlines()
        logger.info(f'Load utrain data with pseudo-labels from: {args.utrain}')
    else:
        utrain_labels = None        

    if args.pre_extracted:
        train_loader, val_loader, test_loader = get_dataloader_preextracted(args, train_fea_path, 
                                                                                val_fea_path, 
                                                                                test_fea_path, args.device)
    else:
        train_loader, val_loader, test_loader = get_dataloader(args, args.train_split, args.val_split, args.test_split,
                                                                    tokenized_text_prompts, preprocess, utrain_labels)
    logger.info(f'len(train_loader): {len(train_loader)}')
    logger.info(f'len(val_loader): {len(val_loader)}')
    logger.info(f'len(test_loader): {len(test_loader)}')
    
    # for mixup-fs two dataloaders are needed, one for retreived data, one for few-shot data
    if args.method == 'mixup-fs' or args.method == 'finetune-mixed' or args.method == 'cutmix-fs':
        # train_dataloader_retrieve, train_dataloader_fewshot = get_retrieve_fewshot_dataloader(args, 'real_T2T500_T2I0.25.txt', 'fewshot15.txt',
        #                                                                                       tokenized_text_prompts, preprocess, utrain_labels)

        _, train_dataloader_fewshot = get_retrieve_fewshot_dataloader(args, [args.retrieval_split], args.fewshot_split,
                                                                    tokenized_text_prompts, preprocess, utrain_labels)       
         
        # logger.info(f'len(train_dataloader_retrieve): {len(train_dataloader_retrieve)}')
        logger.info(f'len(train_dataloader_fewshot): {len(train_dataloader_fewshot)}')
        train_loader = (train_loader, train_dataloader_fewshot)

    elif args.method == 'CMO':

        train_dataset = load_dataset(dataset_root=args.dataset_root, 
                                split=args.train_split,                                                                                                       
                                preprocess=transform(224, 'train'),
                                tokenized_text_prompts=tokenized_text_prompts,
                                )

        cls_num_list = args.cls_num_list
        cls_weight = 1.0 / (np.array(cls_num_list) ** args.cmo_alpha)
        # cls_weight = cls_weight / np.sum(cls_weight) * len(cls_num_list)
        cls_weight = cls_weight / np.sum(cls_weight)

        samples_weight = np.array([cls_weight[t] for t in train_dataset.targets])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        # print('samples_weight.shape:', samples_weight.shape)
        weighted_sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight),
                                                                  replacement=True)
        
        weighted_train_loader = DataLoader(train_dataset, batch_size=args.bsz, pin_memory=True,
                                            # shuffle=True, 
                                            drop_last=True, 
                                            num_workers=args.num_workers, sampler=weighted_sampler)
        
        train_loader = (train_loader, weighted_train_loader)    

    return train_loader, val_loader, test_loader


def set_text_dataloader(args, prompt_tensors, prompt_tensors_dict):

    logger.info(f'Cross-modal adaptation: train with {args.prompt_name} prompts.')
    if args.use_attribute:
        logger.info(f'Use attribute when making prompts.')
        text_dataloader = get_text_dataloader(args, prompt_tensors_dict['c-name_attribute'], args.device)
    else:
        text_dataloader = get_text_dataloader(args, prompt_tensors, args.device)

    return text_dataloader


def set_loss(args):
    if args.loss_name == 'CE':
        loss = nn.CrossEntropyLoss()
    elif args.loss_name == 'WeightedCE':
        loss = WeightedCELoss(fewshot_weight=args.fewshot_weight)        
    elif args.loss_name == 'Focal':
        loss = FocalLoss(gamma=args.focal_gamma, alpha=args.focal_alpha)
    elif args.loss_name == 'BalancedSoftmax':
        loss = BalancedSoftmaxLoss(cls_num_list=args.cls_num_list)
    else:
        raise NotImplementedError(f'Loss {args.loss_name} not implemented.')

    args.loss = loss

    return loss


def lock_text_tower(model):

    for m in [model.transformer, model.token_embedding, model.positional_embedding, model.ln_final, model.text_projection]:
        if type(m) is nn.Parameter:
            m.requires_grad = False
        else:
            for p in m.parameters():
                p.requires_grad = False


def set_params(args, model, classifier_head, logger, dataset_classifier_head=None):

    params_classifier = [{'params': classifier_head.parameters(), 'lr': args.lr_classifier}]
    params_visual = [{'params': model.visual.parameters(), 'lr': args.lr_backbone}]
    # params_transformer = [{'params': model.transformer.parameters(), 'lr': args.lr_backbone}]
    if dataset_classifier_head is not None:
        params_dataset_classifier = [{'params': dataset_classifier_head.parameters(), 'lr': args.lr_classifier}]

    if args.method == "zeroshot":
        logger.info('zeroshot only.')
        for param in model.parameters():
            param.requires_grad = False
        params = params_classifier # place holder
        logit_scale = torch.tensor([4.60517]).to(device=args.device)

    elif args.method == "probing" or args.method == "CMLP":
        logger.info('Freezing the visual encoder.')
        for param in model.parameters():
            param.requires_grad = False
        params = params_classifier
        logit_scale = torch.tensor([4.60517]).to(device=args.device) 
        # 4.60517 = np.log(100) = np.log(1 / 0.01), 0.01 is the temperature

    elif args.method == "finetune" or args.method == "finetune-multitask" or \
        args.method == "finetune-mixed" or args.method == "dataset-cls" or \
        args.method == "mixup" or args.method == "mixup-fs" or \
        args.method == "cutmix" or args.method == "cutmix-fs" or args.method == "resizemix" or \
        args.method == "saliencymix" or args.method == "attentivemix" or \
        args.method == "CMO":

        logger.info('Training the visual encoder and linear head.')

        lock_text_tower(model)
        params = params_classifier + params_visual
        if args.method == "finetune-multitask":
            params = params + params_dataset_classifier

        logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / args.temperature)) # ln(1/0.07)=2.65926
        params.append({'params': [logit_scale], 'lr': args.lr_classifier})
    
    elif args.method == "FLYP":
        logger.info('Training the visual encoder and text encoder.')

        if args.locked_text:
            logger.info('Freezing the text encoder.')
            # from REACT's lock_text_tower()
            lock_text_tower(model)
            # params = params_visual
        """
        # check if the model.visual.proj and model.text_projection are in the params
        for name, param in model.named_parameters():
            print(name)
        """        

        if args.lr_projector is None:
            args.lr_projector = args.lr_backbone

        # here we first set the projectors to be frozen to exclude them from the params list
        model.visual.proj.requires_grad = False
        model.text_projection.requires_grad = False

        total_params = list(model.parameters())
        params_list = [p for p in total_params if p.requires_grad]
        params = [{'params': params_list, 'lr': args.lr_backbone}]
        
        # set a different learning rate for the model.visual.proj and model.text_projection
        model.visual.proj.requires_grad = True
        model.text_projection.requires_grad = True
        params_visual_proj = {'params': [model.visual.proj], 'lr': args.lr_projector}
        params.append(params_visual_proj)
        params_text_proj = {'params': [model.text_projection], 'lr': args.lr_projector}
        params.append(params_text_proj)

        logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / args.temperature)) # ln(1/0.07)=2.65926
        params.append({'params': [logit_scale], 'lr': args.lr_classifier})        

    else:
        raise NotImplementedError(f'Method {args.method} not implemented.')

    args.logit_scale = logit_scale

    return params, logit_scale


def set_optimizer(args, params, train_loader):

    optimizer = get_optimizer(params, optim_type=args.optim, wd=args.wd)
    total_iter = len(train_loader) * args.epochs
    base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iter, eta_min=1e-9)
    warmup_lr = 1e-5 if args.lr_backbone > 5e-5 else 1e-6
    scheduler = get_warmup_scheduler(optimizer=optimizer, scheduler=base_scheduler, warmup_iter=50, warmup_lr=warmup_lr)
    
    return optimizer, scheduler, total_iter


def get_batch(args, image_loader, text_loader, image_loader_iter, text_loader_iter, model):
           
    # image feature
    if image_loader_iter is not None:
        try:
            inputs, labels, tokenized_text, source = next(image_loader_iter)
        except StopIteration:
            image_loader_iter = iter(image_loader)
            inputs, labels, tokenized_text, source = next(image_loader_iter)
        images = inputs.to(args.device)
        labels = labels.to(args.device).long()  

        if not args.freeze_visual:
            image_features = model.encode_image(images)
            image_feature = image_features / image_features.norm(dim=-1, keepdim=True) # Normalization.
        else:
            image_feature = images # already normalized
    else:
        image_feature = None

    # text feature
    if text_loader_iter is not None:
        try:
            text, text_label = next(text_loader_iter)
        except StopIteration:
            text_loader_iter = iter(text_loader)
            text, text_label = next(text_loader_iter)
        text_feature = text
    else:
        text_feature = None        

    # concatenate
    if image_feature is not None and text_feature is not None:
        feature = torch.cat([image_feature, text_feature], dim=0)
        label = torch.cat([labels, text_label], dim=0)
    elif image_feature is not None:
        feature = image_feature
        label = labels
    elif text_feature is not None:
        feature = text_feature
        label = text_label
    else:
        raise ValueError("Both image_feature and text_feature are None")        
    
    return feature, label


def load_model(args, logger, model, test_loader=None, classifier_head=None):

    logger.info(f'Loading model from: {args.model_path}')
    ckpt = torch.load(args.model_path)

    # for WSFT ensembled model
    # model.load_state_dict(ckpt['wsft_backbone'])
    # classifier_head.load_state_dict(ckpt['wsft_head'])

    if 'clip' in ckpt: 
        model.load_state_dict(ckpt['clip'])

        classifier_head.load_state_dict(ckpt['head'])
        # classifier_head.load_state_dict(ckpt['best_tau_head'])
        # classifier_head.load_state_dict(ckpt['wsft_head'])

        # file_path = 'output/FT-CE_fewshot15+real_t2t500-40eps_probing_semi-aves_vitb32_openclip_laion400m_c-name/best_tau_head_weights.pth'
        # best_tau_head = torch.load(file_path)
        # classifier_head.load_state_dict(best_tau_head)    

        # print('ckpt[epoch]:', ckpt['epoch'])
        # print('ckpt[num_iter]:', ckpt['num_iter'])
        # print('ckpt[best_val_acc]:', round(ckpt['best_val_acc'], 3))
        # print('ckpt[best_epoch]:', ckpt['best_epoch'])
        # print('ckpt[best_iter]:', ckpt['best_iter'])
        # print('ckpt[val_acc]:', round(ckpt['val_acc']), 3)
        # print('ckpt[test_acc]:', ckpt['test_acc'])
        logger.info(f'ckpt[test_acc]: {ckpt["test_acc"]}')
        # print('ckpt[best_tau]:', ckpt['best_tau'])
        # print('ckpt[best_tau_test_acc]:', ckpt['best_tau_test_acc'])
        # print('ckpt[wsft_test_acc]:', ckpt['wsft_test_acc'])    

    elif 'model' in ckpt: # for SuperContrastive ckpt
        # model.load_state_dict(ckpt['model']) 
        """
        # Missing key(s) in state_dict: "positional_embedding", "text_projection", 
        # "logit_scale", "token_embedding.weight", "ln_final.weight", "ln_final.bias". 
        """

        # load only the visual encoder weights, and keep others the same
        model.load_state_dict(ckpt['model'], strict=False)
        # here we initialize the classifier head with the zeroshot head weights
        classifier_head = classifier_head
        print('ckpt[epoch]:', ckpt['epoch'])
    else:
        print('ckpt.keys():', ckpt.keys())
        classifier_head.load_state_dict(ckpt['best_tau_head'])
        # raise ValueError('No model weights found in the checkpoint.')
  

    del ckpt

    # print(type(classifier_head))
    # print the weight shape of the classifier head
    # learned_head_weights = classifier_head.linear.weight.data
    # print('learned_head_weights:', learned_head_weights.shape)
    
    if test_loader is not None:
        model_test_acc, _, _ = validate(args,data_loader=test_loader, model=model, 
                                        logger=logger,
                                        loss=args.loss, logit_scale=args.logit_scale, 
                                        classifier_head=classifier_head, 
                                        dataset=args.dataset, 
                                        device=args.device,
                                        pre_extracted=args.pre_extracted, 
                                        )
        logger.info(f"Loaded Model Test Acc: {round(model_test_acc, 3)}")        

    # loaded_model = copy.deepcopy(model)
    # loaded_head = copy.deepcopy(classifier_head)
    
    # return loaded_model, loaded_head

def run_zeroshot(args, test_loader, model, logger, loss, logit_scale, classifier_head):
    if args.method == 'dataset-cls':
        zs_test_acc, zs_loss, zs_confusion_matrix = validate_dataset(args,data_loader=test_loader, model=model, logger=logger,
                                                            loss=args.loss, logit_scale=logit_scale,
                                                            classifier_head=classifier_head, show_confusion_matrix=True,
                                                            dataset=args.dataset, 
                                                            output_dir=args.output_dir, device=args.device,
                                                            pre_extracted=args.pre_extracted, 
                                                            )
    else:
        zs_test_acc, zs_loss, zs_confusion_matrix = validate(args,data_loader=test_loader, model=model, logger=logger,
                                                            loss=args.loss, logit_scale=logit_scale,
                                                            classifier_head=classifier_head, show_confusion_matrix=True,
                                                            dataset=args.dataset, 
                                                            output_dir=args.output_dir, device=args.device,
                                                            pre_extracted=args.pre_extracted, 
                                                            )
    logger.info(f"+++++ Zero-shot Test Acc: {round(zs_test_acc, 3)}")
    # zs_scores = calculate_scores(zs_confusion_matrix)
    # save_test_scores(zs_scores, zs_confusion_matrix, output_dir, 'zeroshot_test')


def train_CMLP(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader, reload_model=False, text_dataloader=None):
    """ Train the model with Cross-Entropy Loss, linear probing"""

    if reload_model:
        load_model(args, logger, model, test_loader, classifier_head)
        # Here we reextract the test dataloader for fast testing after training, tau normalization, and WiSE-FT
        new_train_fea_path = f'{args.dataset_root}/pre_extracted/{args.dataset}_{args.model_cfg}_{args.seed}_train_features_new.pth'
        new_val_fea_path = f'{args.dataset_root}/pre_extracted/{args.dataset}_{args.model_cfg}_{args.seed}_val_features_new.pth'
        new_test_fea_path = f'{args.dataset_root}/pre_extracted/{args.dataset}_{args.model_cfg}_{args.seed}_test_features_new.pth'
        
        train_loader = extract_train_dataloader(args, logger, model, args.train_split, new_train_fea_path, args.bsz)
        val_loader = extract_dataloader(args, model, args.val_split, new_val_fea_path)
        test_loader = extract_dataloader(args, model, args.test_split, new_test_fea_path)
        logger.info(f'Extracted train, val, test dataloader for stage 2 training.')
        # reset the pre_extracted flag
        args.pre_extracted = True
        logger.info(f'Reset args.pre_extracted: {args.pre_extracted}')

    logger.info(f"Start Training (cross-modal linear probing) ......")

    logit_scale = args.logit_scale
    loss = args.loss
    optimizer = args.optimizer
    scheduler = args.scheduler

    assert text_dataloader is not None, 'text_dataloader is None.'
    assert args.pre_extracted, 'args.pre_extracted is False.'

    model.eval()
    classifier_head.train()

    best_records = {}    
    best_val_acc = -1
    num_iter = 0

    text_loader = text_dataloader
    text_loader_iter = iter(text_loader) 

    for epoch in range(1, args.epochs+1):
        
        train_loss_sum = 0
        for inputs, labels, text, source in train_loader:
            num_iter += 1
            images = inputs.to(args.device)
            labels = labels.to(args.device).long()

            image_feature = images

            try:
                text, text_label = next(text_loader_iter)
            except StopIteration:
                text_loader_iter = iter(text_loader)
                text, text_label = next(text_loader_iter)
            
            # concatenate image and text features
            combined_feature = torch.cat([image_feature, text], dim=0)
            combined_labels = torch.cat([labels, text_label], dim=0)

            logits = classifier_head(combined_feature)
            logits = logits * logit_scale.exp()
            total_loss = loss(logits, combined_labels)
            train_loss = total_loss.item()
            train_loss_sum += train_loss             
                
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step() # update learning rate for each iteration
        
        # validate after 1 epoch          
        val_acc, val_loss, confusion_matrix = validate(args, data_loader=val_loader, model=model, logger=logger, 
                                                        loss=args.loss, logit_scale=logit_scale,
                                                        classifier_head=classifier_head, show_confusion_matrix=True,
                                                        dataset=args.dataset, 
                                                        output_dir=args.output_dir, device=args.device,
                                                        pre_extracted=args.pre_extracted, 
                                                        )
        scores = calculate_scores(confusion_matrix)        
    
        if (args.early_stop or epoch == args.epochs) and val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_logit_scale = copy.deepcopy(logit_scale)
            best_epoch = epoch
            best_iter = num_iter
            best_scores = copy.deepcopy(scores)
            best_confusion_matrix = copy.deepcopy(confusion_matrix)
            best_head = copy.deepcopy(classifier_head)
            best_model = copy.deepcopy(model)

            # save into the best_records
            best_records['best_val_acc'] = best_val_acc
            best_records['best_logit_scale'] = best_logit_scale
            best_records['best_epoch'] = best_epoch
            best_records['best_iter'] = best_iter
            best_records['best_scores'] = best_scores
            best_records['best_confusion_matrix'] = best_confusion_matrix

        # test after 1 epoch
        test_acc, _, _ = validate(args,data_loader=test_loader, model=model, logger=logger, 
                                loss=args.loss, logit_scale=logit_scale,
                                classifier_head=classifier_head,
                                dataset=args.dataset, 
                                output_dir=args.output_dir, device=args.device,
                                pre_extracted=args.pre_extracted, 
                                )                
        
        train_loss_avg = train_loss_sum / len(train_loader)
        loss_logger.write(f'{epoch},{num_iter},{round(train_loss_avg, 6)},{round(val_loss, 6)},{round(val_acc, 6)},{round(test_acc, 6)}\n')
        loss_logger.flush()
        logger.info(f"Epoch {int(epoch)}, Iter {num_iter}, Trn Loss: {round(train_loss_avg, 6)}, Val Loss: {round(val_loss, 6)}, Val Acc: {round(val_acc, 3)}, Test Acc: {round(test_acc, 3)}")          

        ## save model checkpoints every X epochs
        if args.save_ckpt and (num_iter % args.save_freq == 0 or epoch == args.epochs):            
            model_path = save_model_ckpt(args, best_records,                    
                                        model, classifier_head, optimizer, scheduler, logit_scale,
                                         val_acc, epoch, num_iter)
            logger.info(f'Model ckpt saved to: {model_path}')    
    
    logger.info(f'Probing done.')

    return best_model, best_head, best_records, best_logit_scale, val_loader, test_loader


def train_probing(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader, reload_model=False, text_dataloader=None):
    """ Train the model with Cross-Entropy Loss, linear probing"""

    if reload_model:
        load_model(args, logger, model, test_loader, classifier_head)
        # Here we reextract the test dataloader for fast testing after training, tau normalization, and WiSE-FT
        new_train_fea_path = f'{args.dataset_root}/pre_extracted/{args.dataset}_{args.model_cfg}_{args.seed}_train_features_new.pth'
        new_val_fea_path = f'{args.dataset_root}/pre_extracted/{args.dataset}_{args.model_cfg}_{args.seed}_val_features_new.pth'
        new_test_fea_path = f'{args.dataset_root}/pre_extracted/{args.dataset}_{args.model_cfg}_{args.seed}_test_features_new.pth'
        
        train_loader = extract_train_dataloader(args, logger, model, args.train_split, new_train_fea_path, args.bsz)
        val_loader = extract_dataloader(args, model, args.val_split, new_val_fea_path)
        test_loader = extract_dataloader(args, model, args.test_split, new_test_fea_path)
        logger.info(f'Extracted train, val, test dataloader for stage 2 training.')
        # reset the pre_extracted flag
        args.pre_extracted = True
        logger.info(f'Reset args.pre_extracted: {args.pre_extracted}')

    logger.info(f"Start Training (linear probing) ......")

    logit_scale = args.logit_scale
    loss = args.loss
    optimizer = args.optimizer
    scheduler = args.scheduler

    model.eval()
    classifier_head.train()

    best_records = {}    
    best_val_acc = -1
    num_iter = 0
    for epoch in range(1, args.epochs+1):
        
        train_loss_sum = 0
        for inputs, labels, text, source in train_loader:
            num_iter += 1
            images = inputs.to(args.device)
            labels = labels.to(args.device).long()

            if not args.pre_extracted:
                image_features = model.encode_image(images)
                image_feature = image_features / image_features.norm(dim=-1, keepdim=True)
            else:
                image_feature = images
            
            logits = classifier_head(image_feature)
            logits = logits * logit_scale.exp()
            total_loss = loss(logits, labels)
            train_loss = total_loss.item()
            train_loss_sum += train_loss             
                
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step() # update learning rate for each iteration
        
        # validate after 1 epoch          
        val_acc, val_loss, confusion_matrix = validate(args, data_loader=val_loader, model=model, logger=logger, 
                                                        loss=args.loss, logit_scale=logit_scale,
                                                        classifier_head=classifier_head, show_confusion_matrix=True,
                                                        dataset=args.dataset, 
                                                        output_dir=args.output_dir, device=args.device,
                                                        pre_extracted=args.pre_extracted, 
                                                        )
        scores = calculate_scores(confusion_matrix)        
    
        if (args.early_stop or epoch == args.epochs) and val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_logit_scale = copy.deepcopy(logit_scale)
            best_epoch = epoch
            best_iter = num_iter
            best_scores = copy.deepcopy(scores)
            best_confusion_matrix = copy.deepcopy(confusion_matrix)
            best_head = copy.deepcopy(classifier_head)
            best_model = copy.deepcopy(model)

            # save into the best_records
            best_records['best_val_acc'] = best_val_acc
            best_records['best_logit_scale'] = best_logit_scale
            best_records['best_epoch'] = best_epoch
            best_records['best_iter'] = best_iter
            best_records['best_scores'] = best_scores
            best_records['best_confusion_matrix'] = best_confusion_matrix

        # test after 1 epoch
        test_acc, _, _ = validate(args,data_loader=test_loader, model=model, logger=logger, 
                                loss=args.loss, logit_scale=logit_scale,
                                classifier_head=classifier_head,
                                dataset=args.dataset, 
                                output_dir=args.output_dir, device=args.device,
                                pre_extracted=args.pre_extracted, 
                                )                
        
        train_loss_avg = train_loss_sum / len(train_loader)
        loss_logger.write(f'{epoch},{num_iter},{round(train_loss_avg, 6)},{round(val_loss, 6)},{round(val_acc, 6)},{round(test_acc, 6)}\n')
        loss_logger.flush()
        logger.info(f"Epoch {int(epoch)}, Iter {num_iter}, Trn Loss: {round(train_loss_avg, 6)}, Val Loss: {round(val_loss, 6)}, Val Acc: {round(val_acc, 3)}, Test Acc: {round(test_acc, 3)}")          

        ## save model checkpoints every X epochs
        if args.save_ckpt and (num_iter % args.save_freq == 0 or epoch == args.epochs):            
            model_path = save_model_ckpt(args, best_records,                    
                                        model, classifier_head, optimizer, scheduler, logit_scale,
                                         val_acc, epoch, num_iter)
            logger.info(f'Model ckpt saved to: {model_path}')    
    
    logger.info(f'Probing done.')

    return best_model, best_head, best_records, best_logit_scale, val_loader, test_loader


def train_dataset(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader):
    """ 
    Train the model with Cross-Entropy Loss, finetuning visual encoder and classifier
    For dataset classification
    """
    
    logger.info(f"Start standard finetuning ......")

    model.eval() if args.freeze_visual else model.train()
    classifier_head.train()

    logit_scale = args.logit_scale
    loss = args.loss
    optimizer = args.optimizer
    scheduler = args.scheduler
    
    val_loss = -1
    val_acc = -1
    test_acc = -1
    best_val_acc = -1
    best_epoch = -1
    best_iter = -1
    best_model = model
    best_head = classifier_head
    best_logit_scale = logit_scale
    best_records = {}
    num_iter = 0

    for epoch in range(1, args.epochs+1):        
        train_loss_sum = 0
        for inputs, labels, text, source in train_loader:
            num_iter += 1
            images = inputs.to(args.device)
            # labels = labels.to(args.device).long()
            labels = source.to(args.device).long() # use the source as labels
            # source = source.to(args.device)  
            
            image_features = model.encode_image(images)
            image_feature = image_features / image_features.norm(dim=-1, keepdim=True)
            
            logits = classifier_head(image_feature)
            logits = logits * logit_scale.exp()
            total_loss = loss(logits, labels)  
            train_loss = total_loss.item()
            train_loss_sum += train_loss             
                
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step() # update learning rate for each iteration
        
        # validate after 1 epoch
        # if (args.early_stop and num_iter >= args.start_validation) or \
        #     epoch == args.epochs:
        if args.early_stop or epoch == args.epochs:            
            val_acc, val_loss, confusion_matrix = validate_dataset(args, data_loader=val_loader, model=model, logger=logger, 
                                                        loss=args.loss, logit_scale=logit_scale,
                                                        classifier_head=classifier_head, show_confusion_matrix=True,
                                                        dataset=args.dataset, 
                                                        output_dir=args.output_dir, device=args.device,
                                                        pre_extracted=args.pre_extracted, 
                                                        )
            scores = calculate_scores(confusion_matrix)        
                        
        # check if val acc has improved, here i use the val_acc rather than val_loss,
        # as we might modify the loss function for confusing pairs
        if (args.early_stop or epoch == args.epochs) and val_acc >= best_val_acc:
            # logger.info(f'Val Acc improved from {round(best_val_acc, 3)} to {round(val_acc, 3)}')
            best_val_acc = val_acc
            best_logit_scale = copy.deepcopy(logit_scale)
            best_epoch = epoch
            best_iter = num_iter
            best_scores = copy.deepcopy(scores)
            best_confusion_matrix = copy.deepcopy(confusion_matrix)
            best_head = copy.deepcopy(classifier_head)
            best_model = copy.deepcopy(model)

            # save into the best_records
            best_records['best_val_acc'] = best_val_acc
            best_records['best_logit_scale'] = best_logit_scale
            best_records['best_epoch'] = best_epoch
            best_records['best_iter'] = best_iter
            best_records['best_scores'] = best_scores
            best_records['best_confusion_matrix'] = best_confusion_matrix

        # test after 1 epoch
        if args.early_stop or epoch == args.epochs:
            test_acc, _, _ = validate_dataset(args,data_loader=test_loader, model=model, logger=logger, 
                                loss=args.loss, logit_scale=logit_scale,
                                classifier_head=classifier_head,
                                dataset=args.dataset, 
                                output_dir=args.output_dir, device=args.device,
                                pre_extracted=args.pre_extracted, 
                                )                
        
        train_loss_avg = train_loss_sum / len(train_loader)
        loss_logger.write(f'{epoch},{num_iter},{round(train_loss_avg, 6)},{round(val_loss, 6)},{round(val_acc, 6)},{round(test_acc, 6)}\n')
        loss_logger.flush()
        logger.info(f"Epoch {int(epoch)}, Iter {num_iter}, Trn Loss: {round(train_loss_avg, 6)}, Val Loss: {round(val_loss, 6)}, Val Acc: {round(val_acc, 3)}, Test Acc: {round(test_acc, 3)}")          

        ## save model checkpoints every X epochs
        if args.save_ckpt and (num_iter % args.save_freq == 0 or epoch == args.epochs):            
            model_path = save_model_ckpt(args, best_records,                    
                                        model, classifier_head, optimizer, scheduler, logit_scale,
                                         val_acc, epoch, num_iter)
            logger.info(f'Model ckpt saved to: {model_path}')  

        if epoch == args.stop_epochs:
            break  
    
    logger.info(f'Training done.')
    logger.info(f'Best val Acc: {round(best_val_acc, 3)} at epoch {best_epoch}, iter {best_iter}')

    return best_model, best_head, best_records, best_logit_scale 


def train_ce(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader):
    """ Train the model with Cross-Entropy Loss, finetuning visual encoder and classifier"""
    
    logger.info(f"Start standard finetuning ......")

    model.eval() if args.freeze_visual else model.train()
    classifier_head.train()

    logit_scale = args.logit_scale
    loss = args.loss
    optimizer = args.optimizer
    scheduler = args.scheduler
    
    val_loss = -1
    val_acc = -1
    test_acc = -1
    best_val_acc = -1
    best_epoch = -1
    best_iter = -1
    best_model = model
    best_head = classifier_head
    best_logit_scale = logit_scale
    best_records = {}
    num_iter = 0

    for epoch in range(1, args.epochs+1):        
        train_loss_sum = 0
        for inputs, labels, text, source in train_loader:
            num_iter += 1
            images = inputs.to(args.device)
            labels = labels.to(args.device).long()
            source = source.to(args.device)  
            
            image_features = model.encode_image(images)
            image_feature = image_features / image_features.norm(dim=-1, keepdim=True)
            
            logits = classifier_head(image_feature)
            logits = logits * logit_scale.exp()
            if args.loss_name == 'WeightedCE':
                total_loss = loss(logits, labels, source) # for WeightedCE, needs to input the source
            else:
                total_loss = loss(logits, labels)  
            train_loss = total_loss.item()
            train_loss_sum += train_loss             
                
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step() # update learning rate for each iteration
        
        # validate after 1 epoch
        # if (args.early_stop and num_iter >= args.start_validation) or \
        #     epoch == args.epochs:
        if args.early_stop or epoch == args.epochs:            
            val_acc, val_loss, confusion_matrix = validate(args, data_loader=val_loader, model=model, logger=logger, 
                                                        loss=args.loss, logit_scale=logit_scale,
                                                        classifier_head=classifier_head, show_confusion_matrix=True,
                                                        dataset=args.dataset, 
                                                        output_dir=args.output_dir, device=args.device,
                                                        pre_extracted=args.pre_extracted, 
                                                        )
            scores = calculate_scores(confusion_matrix)        
                        
        # check if val acc has improved, here i use the val_acc rather than val_loss,
        # as we might modify the loss function for confusing pairs
        if (args.early_stop or epoch == args.epochs) and val_acc >= best_val_acc:
            # logger.info(f'Val Acc improved from {round(best_val_acc, 3)} to {round(val_acc, 3)}')
            best_val_acc = val_acc
            best_logit_scale = copy.deepcopy(logit_scale)
            best_epoch = epoch
            best_iter = num_iter
            best_scores = copy.deepcopy(scores)
            best_confusion_matrix = copy.deepcopy(confusion_matrix)
            best_head = copy.deepcopy(classifier_head)
            best_model = copy.deepcopy(model)

            # save into the best_records
            best_records['best_val_acc'] = best_val_acc
            best_records['best_logit_scale'] = best_logit_scale
            best_records['best_epoch'] = best_epoch
            best_records['best_iter'] = best_iter
            best_records['best_scores'] = best_scores
            best_records['best_confusion_matrix'] = best_confusion_matrix

        # test after 1 epoch
        if args.early_stop or epoch == args.epochs:
            test_acc, _, _ = validate(args,data_loader=test_loader, model=model, logger=logger, 
                                loss=args.loss, logit_scale=logit_scale,
                                classifier_head=classifier_head,
                                dataset=args.dataset, 
                                output_dir=args.output_dir, device=args.device,
                                pre_extracted=args.pre_extracted, 
                                )                
        
        train_loss_avg = train_loss_sum / len(train_loader)
        loss_logger.write(f'{epoch},{num_iter},{round(train_loss_avg, 6)},{round(val_loss, 6)},{round(val_acc, 6)},{round(test_acc, 6)}\n')
        loss_logger.flush()
        logger.info(f"Epoch {int(epoch)}, Iter {num_iter}, Trn Loss: {round(train_loss_avg, 6)}, Val Loss: {round(val_loss, 6)}, Val Acc: {round(val_acc, 3)}, Test Acc: {round(test_acc, 3)}")          

        ## save model checkpoints every X epochs
        if args.save_ckpt and (num_iter % args.save_freq == 0 or epoch == args.epochs):            
            model_path = save_model_ckpt(args, best_records,                    
                                        model, classifier_head, optimizer, scheduler, logit_scale,
                                         val_acc, epoch, num_iter)
            logger.info(f'Model ckpt saved to: {model_path}')  

        if epoch == args.stop_epochs:
            break  
    
    logger.info(f'Training done.')
    logger.info(f'Best val Acc: {round(best_val_acc, 3)} at epoch {best_epoch}, iter {best_iter}')

    return best_model, best_head, best_records, best_logit_scale 


def train_ce_mixed(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader):
    """ 
    Train the model with Cross-Entropy Loss, finetuning visual encoder and classifier.
    Half batch from retrieved data, half batch from few-shot data.
    """

    train_loader, train_dataloader_fs = train_loader
    train_loader_fs = iter(train_dataloader_fs)

    logger.info(f"Start Training ......")

    model.eval() if args.freeze_visual else model.train()
    classifier_head.train()
    
    best_val_acc = -1
    best_epoch = -1
    best_iter = -1
    best_model = model
    best_head = classifier_head
    best_logit_scale = logit_scale
    best_records = {}
    num_iter = 0
    for epoch in range(1, args.epochs+1):
        
        train_loss_sum = 0
        # train for 1 epoch
        for inputs, labels, text, source in train_loader:
            num_iter += 1
            images = inputs.to(args.device)
            labels = labels.to(args.device).long()
            source = source.to(args.device)  

            # get a batch of few-shot data, handle the case when the few-shot data is exhausted, just loop back
            try:
                inputs_fs, labels_fs, text_fs, source_fs = next(train_loader_fs)
            except StopIteration:
                train_loader_fs = iter(train_dataloader_fs)
                inputs_fs, labels_fs, text_fs, source_fs = next(train_loader_fs)
            images_fs = inputs_fs.to(args.device)
            labels_fs = labels_fs.to(args.device).long()
            source_fs = source_fs.to(args.device)

            # concatenate the retrieved data and few-shot data
            images = torch.cat([images, images_fs], dim=0)
            labels = torch.cat([labels, labels_fs], dim=0)
            source = torch.cat([source, source_fs], dim=0)

            image_features = model.encode_image(images)
            image_feature = image_features / image_features.norm(dim=-1, keepdim=True)
            
            logits = classifier_head(image_feature)
            logits = logits * logit_scale.exp()
            if args.loss_name == 'WeightedCE':
                total_loss = loss(logits, labels, source) # for WeightedCE, needs to input the source
            else:
                total_loss = loss(logits, labels)  
            train_loss = total_loss.item()
            train_loss_sum += train_loss             
                
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step() # update learning rate for each iteration
        
        # validate after 1 epoch
        # if (args.early_stop and num_iter >= args.start_validation) or \
        #     epoch == args.epochs:            
        val_acc, val_loss, confusion_matrix = validate(args, data_loader=val_loader, model=model, logger=logger, 
                                                        loss=args.loss, logit_scale=logit_scale,
                                                        classifier_head=classifier_head, show_confusion_matrix=True,
                                                        dataset=args.dataset, 
                                                        output_dir=args.output_dir, device=args.device,
                                                        pre_extracted=args.pre_extracted, 
                                                        )
        scores = calculate_scores(confusion_matrix)        
                        
        # check if val acc has improved, here i use the val_acc rather than val_loss,
        # as we might modify the loss function for confusing pairs
        if (args.early_stop or epoch == args.epochs) and val_acc >= best_val_acc:
            # logger.info(f'Val Acc improved from {round(best_val_acc, 3)} to {round(val_acc, 3)}')
            best_val_acc = val_acc
            best_logit_scale = copy.deepcopy(logit_scale)
            best_epoch = epoch
            best_iter = num_iter
            best_scores = copy.deepcopy(scores)
            best_confusion_matrix = copy.deepcopy(confusion_matrix)
            best_head = copy.deepcopy(classifier_head)
            best_model = copy.deepcopy(model)

            # save into the best_records
            best_records['best_val_acc'] = best_val_acc
            best_records['best_logit_scale'] = best_logit_scale
            best_records['best_epoch'] = best_epoch
            best_records['best_iter'] = best_iter
            best_records['best_scores'] = best_scores
            best_records['best_confusion_matrix'] = best_confusion_matrix

        # test after 1 epoch
        test_acc, _, _ = validate(args,data_loader=test_loader, model=model, logger=logger, 
                                loss=args.loss, logit_scale=logit_scale,
                                classifier_head=classifier_head,
                                dataset=args.dataset, 
                                output_dir=args.output_dir, device=args.device,
                                pre_extracted=args.pre_extracted, 
                                )                
        
        train_loss_avg = train_loss_sum / len(train_loader)
        loss_logger.write(f'{epoch},{num_iter},{round(train_loss_avg, 6)},{round(val_loss, 6)},{round(val_acc, 6)},{round(test_acc, 6)}\n')
        loss_logger.flush()
        logger.info(f"Epoch {int(epoch)}, Iter {num_iter}, Trn Loss: {round(train_loss_avg, 6)}, Val Loss: {round(val_loss, 6)}, Val Acc: {round(val_acc, 3)}, Test Acc: {round(test_acc, 3)}")          

        ## save model checkpoints every X epochs
        if args.save_ckpt and (num_iter % args.save_freq == 0 or epoch == args.epochs):            
            model_path = save_model_ckpt(args, best_records,                    
                                        model, classifier_head, optimizer, scheduler, logit_scale,
                                         val_acc, epoch, num_iter)
            logger.info(f'Model ckpt saved to: {model_path}')  

        if epoch == args.stop_epochs:
            break  
    
    logger.info(f'Training done.')
    logger.info(f'Best val Acc: {round(best_val_acc, 3)} at epoch {best_epoch}, iter {best_iter}')

    return best_model, best_head, best_records, best_logit_scale 

def train_ce_multitask(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader, dataset_classifier_head):
    """ Train the model with Cross-Entropy Loss, finetuning visual encoder and classifier"""

    logger.info(f"Start Training ......")

    model.eval() if args.freeze_visual else model.train()
    classifier_head.train()
    dataset_classifier_head.train()
    
    best_val_acc = -1
    best_epoch = -1
    best_iter = -1
    best_model = model
    best_head = classifier_head
    best_logit_scale = logit_scale
    best_records = {}
    num_iter = 0
    dataset_weight = 1.0
    for epoch in range(1, args.epochs+1):
        # dataset_weight *= args.dataset_wd # decay the dataset weight for each epoch
        dataset_weight = args.dataset_wd 

        train_loss_sum = 0
        dataset_loss_sum = 0
        # train for 1 epoch
        for inputs, labels, text, source in train_loader:
            num_iter += 1
            images = inputs.to(args.device)
            labels = labels.to(args.device).long()
            source = source.to(args.device)  
            
            image_features = model.encode_image(images)
            image_feature = image_features / image_features.norm(dim=-1, keepdim=True)
            
            logits = classifier_head(image_feature)
            logits = logits * logit_scale.exp()
            if args.loss_name == 'WeightedCE':
                total_loss = loss(logits, labels, source) # for WeightedCE, needs to input the source
            else:
                total_loss = loss(logits, labels)
              
            train_loss = total_loss.item()
            train_loss_sum += train_loss

            # add the dataset classification loss
            dataset_logits = dataset_classifier_head(image_feature)
            # dataset_logits = dataset_logits * logit_scale.exp()
            dataset_loss = loss(dataset_logits, source)
            dataset_loss_sum += dataset_loss.item()

            multitask_loss = total_loss + dataset_loss * dataset_weight
                
            optimizer.zero_grad()
            multitask_loss.backward()
            optimizer.step()
            scheduler.step() # update learning rate for each iteration
        
        # validate after 1 epoch
        # if (args.early_stop and num_iter >= args.start_validation) or \
        #     epoch == args.epochs:            
        val_acc, val_loss, confusion_matrix, dataset_val_acc, dataset_confusion_matrix = validate_multitask(args, data_loader=val_loader, model=model, logger=logger, 
                                                        loss=args.loss, logit_scale=logit_scale,
                                                        classifier_head=classifier_head, dataset_classifier_head=dataset_classifier_head,
                                                        show_confusion_matrix=True,
                                                        dataset=args.dataset, 
                                                        output_dir=args.output_dir, device=args.device,
                                                        pre_extracted=args.pre_extracted, 
                                                        )
        scores = calculate_scores(confusion_matrix)
        # dataset_scores = calculate_scores(dataset_confusion_matrix)        
                        
        # check if val acc has improved, here i use the val_acc rather than val_loss,
        # as we might modify the loss function for confusing pairs
        if (args.early_stop or epoch == args.epochs) and val_acc >= best_val_acc:
            # logger.info(f'Val Acc improved from {round(best_val_acc, 3)} to {round(val_acc, 3)}')
            best_val_acc = val_acc
            best_logit_scale = copy.deepcopy(logit_scale)
            best_epoch = epoch
            best_iter = num_iter
            best_scores = copy.deepcopy(scores)
            best_confusion_matrix = copy.deepcopy(confusion_matrix)
            best_head = copy.deepcopy(classifier_head)
            best_model = copy.deepcopy(model)

            # save into the best_records
            best_records['best_val_acc'] = best_val_acc
            best_records['best_logit_scale'] = best_logit_scale
            best_records['best_epoch'] = best_epoch
            best_records['best_iter'] = best_iter
            best_records['best_scores'] = best_scores
            best_records['best_confusion_matrix'] = best_confusion_matrix

        # test after 1 epoch
        test_acc, _, _, dataset_test_acc, _ = validate_multitask(args,data_loader=test_loader, model=model, logger=logger, 
                                loss=args.loss, logit_scale=logit_scale,
                                classifier_head=classifier_head, dataset_classifier_head=dataset_classifier_head,
                                dataset=args.dataset, 
                                output_dir=args.output_dir, device=args.device,
                                pre_extracted=args.pre_extracted, 
                                )                
        
        train_loss_avg = train_loss_sum / len(train_loader)
        dataset_loss_avg = dataset_loss_sum / len(train_loader)

        loss_logger.write(f'{epoch},{num_iter},{round(train_loss_avg, 6)},{round(val_loss, 6)},{round(val_acc, 6)},{round(test_acc, 6)}\n')
        loss_logger.flush()
        logger.info(f"Epoch {int(epoch)}, Iter {num_iter}, Trn Loss: {round(train_loss_avg, 6)}, Dataset Loss: {round(dataset_loss_avg, 6)}, weight: {round(dataset_weight, 3)}, Val Loss: {round(val_loss, 6)}, Val Acc: {round(val_acc, 3)}, Dataset Val Acc: {round(dataset_val_acc, 3)}, Test Acc: {round(test_acc, 3)}, Dataset Test Acc: {round(dataset_test_acc, 3)}")  

        ## save model checkpoints every X epochs
        if args.save_ckpt and (num_iter % args.save_freq == 0 or epoch == args.epochs):            
            model_path = save_model_ckpt(args, best_records,                    
                                        model, classifier_head, optimizer, scheduler, logit_scale,
                                         val_acc, epoch, num_iter)
            logger.info(f'Model ckpt saved to: {model_path}')  

        if epoch == args.stop_epochs:
            break  
    
    logger.info(f'Training done.')
    logger.info(f'Best val Acc: {round(best_val_acc, 3)} at epoch {best_epoch}, iter {best_iter}')

    return best_model, best_head, best_records, best_logit_scale 

def mixup_data(x, y, alpha=1.0, mix_prob=0.5):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    # samply a number uniformly from [0, 1], 
    # this includes the clean/unmixed images to the training process
    flag = torch.rand(1).item()
    if flag <= mix_prob: # do mixup
        lam = lam
    else: # do not mixup
        lam = 1.0

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam

def mixup_data_fs(x_retr, y_retr, x_fs, y_fs, alpha=1.0, mix_prob=0.5):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    # samply a number uniformly from [0, 1], 
    # this includes the clean/unmixed images to the training process
    flag = torch.rand(1).item()
    if flag <= mix_prob:
        lam = 0.0 # set to 0.0 to use few-shot data only
    else: # do not mixup
        lam = 1.0

    mixed_x = lam * x_retr + (1.0 - lam) * x_fs
    y_a, y_b = y_retr, y_fs

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)

def mixup_criterion_lam_list(criterion, pred, y_a, y_b, lam_list):
    # each value in lam_list is the lambda value for each image in the batch
    return sum([lam_list[i] * criterion(pred[i], y_a[i]) + (1.0 - lam_list[i]) * criterion(pred[i], y_b[i]) for i in range(len(lam_list))])

def train_mixup(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader):
    """ 
    Train the model with Cross-Entropy Loss, finetuning visual encoder and classifier
    Use Mixup method to augment the training data
    """

    logger.info(f"Start Training mixup ......")

    model.eval() if args.freeze_visual else model.train()
    classifier_head.train()
    
    logit_scale = args.logit_scale
    loss = args.loss
    optimizer = args.optimizer
    scheduler = args.scheduler

    best_val_acc = -1
    best_epoch = -1
    best_iter = -1
    best_model = model
    best_head = classifier_head
    best_logit_scale = logit_scale
    best_records = {}
    num_iter = 0
    for epoch in range(1, args.epochs+1):
        
        train_loss_sum = 0
        # train for 1 epoch
        for inputs, labels, text, source in train_loader:
            num_iter += 1
            images = inputs.to(args.device)
            labels = labels.to(args.device).long()
            # source = source.to(args.device)

            # apply the mixup strategy
            images, targets_a, targets_b, lam = mixup_data(images, labels, alpha=args.mixup_alpha, mix_prob=args.mix_prob)
            
            image_features = model.encode_image(images)
            image_feature = image_features / image_features.norm(dim=-1, keepdim=True)
            
            logits = classifier_head(image_feature)
            logits = logits * logit_scale.exp()

            # mixup loss
            total_loss = mixup_criterion(loss, logits, targets_a, targets_b, lam)

            train_loss = total_loss.item()
            train_loss_sum += train_loss             
                
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step() # update learning rate for each iteration
        
        # validate after 1 epoch
        # if (args.early_stop and num_iter >= args.start_validation) or \
        #     epoch == args.epochs:            
        val_acc, val_loss, confusion_matrix = validate(args, data_loader=val_loader, model=model, logger=logger, 
                                                        loss=args.loss, logit_scale=logit_scale,
                                                        classifier_head=classifier_head, show_confusion_matrix=True,
                                                        dataset=args.dataset, 
                                                        output_dir=args.output_dir, device=args.device,
                                                        pre_extracted=args.pre_extracted, 
                                                        )
        scores = calculate_scores(confusion_matrix)        
                        
        # check if val acc has improved, here i use the val_acc rather than val_loss,
        # as we might modify the loss function for confusing pairs
        if (args.early_stop or epoch == args.epochs) and val_acc >= best_val_acc:
            # logger.info(f'Val Acc improved from {round(best_val_acc, 3)} to {round(val_acc, 3)}')
            best_val_acc = val_acc
            best_logit_scale = copy.deepcopy(logit_scale)
            best_epoch = epoch
            best_iter = num_iter
            best_scores = copy.deepcopy(scores)
            best_confusion_matrix = copy.deepcopy(confusion_matrix)
            best_head = copy.deepcopy(classifier_head)
            best_model = copy.deepcopy(model)

            # save into the best_records
            best_records['best_val_acc'] = best_val_acc
            best_records['best_logit_scale'] = best_logit_scale
            best_records['best_epoch'] = best_epoch
            best_records['best_iter'] = best_iter
            best_records['best_scores'] = best_scores
            best_records['best_confusion_matrix'] = best_confusion_matrix

        # test after 1 epoch
        test_acc, _, _ = validate(args,data_loader=test_loader, model=model, logger=logger, 
                                loss=args.loss, logit_scale=logit_scale,
                                classifier_head=classifier_head,
                                dataset=args.dataset, 
                                output_dir=args.output_dir, device=args.device,
                                pre_extracted=args.pre_extracted, 
                                )                
        
        train_loss_avg = train_loss_sum / len(train_loader)
        loss_logger.write(f'{epoch},{num_iter},{round(train_loss_avg, 6)},{round(val_loss, 6)},{round(val_acc, 6)},{round(test_acc, 6)}\n')
        loss_logger.flush()
        logger.info(f"Epoch {int(epoch)}, Iter {num_iter}, Trn Loss: {round(train_loss_avg, 6)}, Val Loss: {round(val_loss, 6)}, Val Acc: {round(val_acc, 3)}, Test Acc: {round(test_acc, 3)}")          

        ## save model checkpoints every X epochs
        if args.save_ckpt and (num_iter % args.save_freq == 0 or epoch == args.epochs):            
            model_path = save_model_ckpt(args, best_records,                    
                                        model, classifier_head, optimizer, scheduler, logit_scale,
                                         val_acc, epoch, num_iter)
            logger.info(f'Model ckpt saved to: {model_path}')  

        if epoch == args.stop_epochs:
            break  
    
    logger.info(f'Training done.')
    logger.info(f'Best val Acc: {round(best_val_acc, 3)} at epoch {best_epoch}, iter {best_iter}')

    return best_model, best_head, best_records, best_logit_scale 


def train_mixup_fs(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader):
    """ 
    Train the model with Cross-Entropy Loss, finetuning visual encoder and classifier
    Use Mixup method to augment the training data
    """
    train_loader, train_dataloader_fs = train_loader
    train_loader_fs = iter(train_dataloader_fs)

    logger.info(f"Start Training ......")

    model.eval() if args.freeze_visual else model.train()
    classifier_head.train()
    
    best_val_acc = -1
    best_epoch = -1
    best_iter = -1
    best_model = model
    best_head = classifier_head
    best_logit_scale = logit_scale
    best_records = {}
    num_iter = 0
    for epoch in range(1, args.epochs+1):
        
        train_loss_sum = 0
        # train for 1 epoch
        for inputs, labels, text, source in train_loader: # this is still the mixed data
            num_iter += 1
            images = inputs.to(args.device)
            labels = labels.to(args.device).long()
            
            # get a batch of few-shot data, handle the case when the few-shot data is exhausted, just loop back
            try:
                inputs_fs, labels_fs, text_fs, source_fs = next(train_loader_fs)
            except StopIteration:
                train_loader_fs = iter(train_dataloader_fs)
                inputs_fs, labels_fs, text_fs, source_fs = next(train_loader_fs)
            images_fs = inputs_fs.to(args.device)
            labels_fs = labels_fs.to(args.device).long()

            # apply the mixup strategy
            images, targets_a, targets_b, lam = mixup_data_fs(images, labels, images_fs, labels_fs,
                                                           alpha=args.mixup_alpha, mix_prob=args.mix_prob)
            
            image_features = model.encode_image(images)
            image_feature = image_features / image_features.norm(dim=-1, keepdim=True)
            
            logits = classifier_head(image_feature)
            logits = logits * logit_scale.exp()

            # mixup loss
            total_loss = mixup_criterion(loss, logits, targets_a, targets_b, lam)

            train_loss = total_loss.item()
            train_loss_sum += train_loss             
                
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step() # update learning rate for each iteration
        
        # validate after 1 epoch
        # if (args.early_stop and num_iter >= args.start_validation) or \
        #     epoch == args.epochs:            
        val_acc, val_loss, confusion_matrix = validate(args, data_loader=val_loader, model=model, logger=logger, 
                                                        loss=args.loss, logit_scale=logit_scale,
                                                        classifier_head=classifier_head, show_confusion_matrix=True,
                                                        dataset=args.dataset, 
                                                        output_dir=args.output_dir, device=args.device,
                                                        pre_extracted=args.pre_extracted, 
                                                        )
        scores = calculate_scores(confusion_matrix)        
                        
        # check if val acc has improved, here i use the val_acc rather than val_loss,
        # as we might modify the loss function for confusing pairs
        if (args.early_stop or epoch == args.epochs) and val_acc >= best_val_acc:
            # logger.info(f'Val Acc improved from {round(best_val_acc, 3)} to {round(val_acc, 3)}')
            best_val_acc = val_acc
            best_logit_scale = copy.deepcopy(logit_scale)
            best_epoch = epoch
            best_iter = num_iter
            best_scores = copy.deepcopy(scores)
            best_confusion_matrix = copy.deepcopy(confusion_matrix)
            best_head = copy.deepcopy(classifier_head)
            best_model = copy.deepcopy(model)

            # save into the best_records
            best_records['best_val_acc'] = best_val_acc
            best_records['best_logit_scale'] = best_logit_scale
            best_records['best_epoch'] = best_epoch
            best_records['best_iter'] = best_iter
            best_records['best_scores'] = best_scores
            best_records['best_confusion_matrix'] = best_confusion_matrix

        # test after 1 epoch
        test_acc, _, _ = validate(args,data_loader=test_loader, model=model, logger=logger, 
                                loss=args.loss, logit_scale=logit_scale,
                                classifier_head=classifier_head,
                                dataset=args.dataset, 
                                output_dir=args.output_dir, device=args.device,
                                pre_extracted=args.pre_extracted, 
                                )                
        
        train_loss_avg = train_loss_sum / len(train_loader)
        loss_logger.write(f'{epoch},{num_iter},{round(train_loss_avg, 6)},{round(val_loss, 6)},{round(val_acc, 6)},{round(test_acc, 6)}\n')
        loss_logger.flush()
        logger.info(f"Epoch {int(epoch)}, Iter {num_iter}, Trn Loss: {round(train_loss_avg, 6)}, Val Loss: {round(val_loss, 6)}, Val Acc: {round(val_acc, 3)}, Test Acc: {round(test_acc, 3)}")          

        ## save model checkpoints every X epochs
        if args.save_ckpt and (num_iter % args.save_freq == 0 or epoch == args.epochs):            
            model_path = save_model_ckpt(args, best_records,                    
                                        model, classifier_head, optimizer, scheduler, logit_scale,
                                         val_acc, epoch, num_iter)
            logger.info(f'Model ckpt saved to: {model_path}')  

        if epoch == args.stop_epochs:
            break  
    
    logger.info(f'Training done.')
    logger.info(f'Best val Acc: {round(best_val_acc, 3)} at epoch {best_epoch}, iter {best_iter}')

    return best_model, best_head, best_records, best_logit_scale 


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def train_cutmix(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader):
    """ 
    Train the model with Cross-Entropy Loss, finetuning visual encoder and classifier
    Use CutMix method to augment the training data
    """

    logger.info(f"Start Training cutmix ......")

    model.eval() if args.freeze_visual else model.train()
    classifier_head.train()
    
    logit_scale = args.logit_scale
    loss = args.loss
    optimizer = args.optimizer
    scheduler = args.scheduler

    val_loss = -1
    val_acc = -1
    test_acc = -1
    best_val_acc = -1
    best_epoch = -1
    best_iter = -1
    best_model = model
    best_head = classifier_head
    best_logit_scale = logit_scale
    best_records = {}
    num_iter = 0
    for epoch in range(1, args.epochs+1):
        
        train_loss_sum = 0
        for inputs, labels, text, source in train_loader:
            num_iter += 1
            images = inputs.to(args.device, non_blocking=True)
            labels = labels.to(args.device, non_blocking=True).long()

            # apply the cutmix strategy
            r = np.random.rand(1)
            if args.cutmix_beta > 0 and r < args.mix_prob:
                # generate mixed sample
                lam = np.random.beta(args.cutmix_beta, args.cutmix_beta)
                rand_index = torch.randperm(images.size()[0]).cuda()
                target_a = labels
                target_b = labels[rand_index]
                bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
                images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
                # adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))                
            else:
                target_a = labels
                target_b = labels
                lam = 1.0
            
            image_features = model.encode_image(images)
            image_feature = image_features / image_features.norm(dim=-1, keepdim=True)
            
            logits = classifier_head(image_feature)
            logits = logits * logit_scale.exp()

            # mixup loss
            total_loss = mixup_criterion(loss, logits, target_a, target_b, lam)

            train_loss = total_loss.item()
            train_loss_sum += train_loss             
                
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step() # update learning rate for each iteration
        
        # validate after 1 epoch
        if args.early_stop or epoch == args.epochs:     
            val_acc, val_loss, confusion_matrix = validate(args, data_loader=val_loader, model=model, logger=logger, 
                                                        loss=args.loss, logit_scale=logit_scale,
                                                        classifier_head=classifier_head, show_confusion_matrix=True,
                                                        dataset=args.dataset, 
                                                        output_dir=args.output_dir, device=args.device,
                                                        pre_extracted=args.pre_extracted, 
                                                        )
            scores = calculate_scores(confusion_matrix)        
                        
        # check if val acc has improved, here i use the val_acc rather than val_loss,
        # as we might modify the loss function for confusing pairs
        if (args.early_stop or epoch == args.epochs) and val_acc >= best_val_acc:
            # logger.info(f'Val Acc improved from {round(best_val_acc, 3)} to {round(val_acc, 3)}')
            best_val_acc = val_acc
            best_logit_scale = copy.deepcopy(logit_scale)
            best_epoch = epoch
            best_iter = num_iter
            best_scores = copy.deepcopy(scores)
            best_confusion_matrix = copy.deepcopy(confusion_matrix)
            best_head = copy.deepcopy(classifier_head)
            best_model = copy.deepcopy(model)

            # save into the best_records
            best_records['best_val_acc'] = best_val_acc
            best_records['best_logit_scale'] = best_logit_scale
            best_records['best_epoch'] = best_epoch
            best_records['best_iter'] = best_iter
            best_records['best_scores'] = best_scores
            best_records['best_confusion_matrix'] = best_confusion_matrix

        # test after 1 epoch
        if args.early_stop or epoch == args.epochs: 
            test_acc, _, _ = validate(args,data_loader=test_loader, model=model, logger=logger, 
                                loss=args.loss, logit_scale=logit_scale,
                                classifier_head=classifier_head,
                                dataset=args.dataset, 
                                output_dir=args.output_dir, device=args.device,
                                pre_extracted=args.pre_extracted, 
                                )                
        
        train_loss_avg = train_loss_sum / len(train_loader)
        loss_logger.write(f'{epoch},{num_iter},{round(train_loss_avg, 6)},{round(val_loss, 6)},{round(val_acc, 6)},{round(test_acc, 6)}\n')
        loss_logger.flush()
        logger.info(f"Epoch {int(epoch)}, Iter {num_iter}, Trn Loss: {round(train_loss_avg, 6)}, Val Loss: {round(val_loss, 6)}, Val Acc: {round(val_acc, 3)}, Test Acc: {round(test_acc, 3)}")          

        ## save model checkpoints every X epochs
        if args.save_ckpt and (num_iter % args.save_freq == 0 or epoch == args.epochs):            
            model_path = save_model_ckpt(args, best_records,                    
                                        model, classifier_head, optimizer, scheduler, logit_scale,
                                         val_acc, epoch, num_iter)
            logger.info(f'Model ckpt saved to: {model_path}')  

        if epoch == args.stop_epochs:
            break  
    
    logger.info(f'Training done.')
    logger.info(f'Best val Acc: {round(best_val_acc, 3)} at epoch {best_epoch}, iter {best_iter}')

    return best_model, best_head, best_records, best_logit_scale


def train_cutmix_fs(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader):
    """ 
    Train the model with Cross-Entropy Loss, finetuning visual encoder and classifier
    Use CutMix method to augment the training data
    two dataloader, one from the mixed, one from the few-shot only
    """
    
    train_loader, train_dataloader_fs = train_loader
    train_loader_fs = iter(train_dataloader_fs)

    logger.info(f"Start Training cutmix-fs ......")

    model.eval() if args.freeze_visual else model.train()
    classifier_head.train()
    
    logit_scale = args.logit_scale
    loss = args.loss
    optimizer = args.optimizer
    scheduler = args.scheduler

    best_val_acc = -1
    best_epoch = -1
    best_iter = -1
    best_model = model
    best_head = classifier_head
    best_logit_scale = logit_scale
    best_records = {}
    num_iter = 0
    for epoch in range(1, args.epochs+1):
        
        train_loss_sum = 0
        # train for 1 epoch
        for inputs, labels, text, source in train_loader:
            num_iter += 1
            images = inputs.to(args.device)
            labels = labels.to(args.device).long()

            # get a batch of few-shot data, handle the case when the few-shot data is exhausted, just loop back
            try:
                inputs_fs, labels_fs, text_fs, source_fs = next(train_loader_fs)
            except StopIteration:
                train_loader_fs = iter(train_dataloader_fs)
                inputs_fs, labels_fs, text_fs, source_fs = next(train_loader_fs)
            images_fs = inputs_fs.to(args.device)
            labels_fs = labels_fs.to(args.device).long()

            # apply the cutmix strategy
            r = np.random.rand(1)
            if args.cutmix_beta > 0 and r < args.mix_prob:
                # generate mixed sample
                lam = np.random.beta(args.cutmix_beta, args.cutmix_beta)
                target_a = labels
                target_b = labels_fs
                bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
                images[:, :, bbx1:bbx2, bby1:bby2] = images_fs[:, :, bbx1:bbx2, bby1:bby2]
                # adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))                
            else:
                target_a = labels
                target_b = labels_fs
                lam = 1.0
            
            image_features = model.encode_image(images)
            image_feature = image_features / image_features.norm(dim=-1, keepdim=True)
            
            logits = classifier_head(image_feature)
            logits = logits * logit_scale.exp()

            # mixup loss
            total_loss = mixup_criterion(loss, logits, target_a, target_b, lam)

            train_loss = total_loss.item()
            train_loss_sum += train_loss             
                
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step() # update learning rate for each iteration
        
        # validate after 1 epoch
        # if (args.early_stop and num_iter >= args.start_validation) or \
        #     epoch == args.epochs:            
        val_acc, val_loss, confusion_matrix = validate(args, data_loader=val_loader, model=model, logger=logger, 
                                                        loss=args.loss, logit_scale=logit_scale,
                                                        classifier_head=classifier_head, show_confusion_matrix=True,
                                                        dataset=args.dataset, 
                                                        output_dir=args.output_dir, device=args.device,
                                                        pre_extracted=args.pre_extracted, 
                                                        )
        scores = calculate_scores(confusion_matrix)        
                        
        # check if val acc has improved, here i use the val_acc rather than val_loss,
        # as we might modify the loss function for confusing pairs
        if (args.early_stop or epoch == args.epochs) and val_acc >= best_val_acc:
            # logger.info(f'Val Acc improved from {round(best_val_acc, 3)} to {round(val_acc, 3)}')
            best_val_acc = val_acc
            best_logit_scale = copy.deepcopy(logit_scale)
            best_epoch = epoch
            best_iter = num_iter
            best_scores = copy.deepcopy(scores)
            best_confusion_matrix = copy.deepcopy(confusion_matrix)
            best_head = copy.deepcopy(classifier_head)
            best_model = copy.deepcopy(model)

            # save into the best_records
            best_records['best_val_acc'] = best_val_acc
            best_records['best_logit_scale'] = best_logit_scale
            best_records['best_epoch'] = best_epoch
            best_records['best_iter'] = best_iter
            best_records['best_scores'] = best_scores
            best_records['best_confusion_matrix'] = best_confusion_matrix

        # test after 1 epoch
        test_acc, _, _ = validate(args,data_loader=test_loader, model=model, logger=logger, 
                                loss=args.loss, logit_scale=logit_scale,
                                classifier_head=classifier_head,
                                dataset=args.dataset, 
                                output_dir=args.output_dir, device=args.device,
                                pre_extracted=args.pre_extracted, 
                                )                
        
        train_loss_avg = train_loss_sum / len(train_loader)
        loss_logger.write(f'{epoch},{num_iter},{round(train_loss_avg, 6)},{round(val_loss, 6)},{round(val_acc, 6)},{round(test_acc, 6)}\n')
        loss_logger.flush()
        logger.info(f"Epoch {int(epoch)}, Iter {num_iter}, Trn Loss: {round(train_loss_avg, 6)}, Val Loss: {round(val_loss, 6)}, Val Acc: {round(val_acc, 3)}, Test Acc: {round(test_acc, 3)}")          

        ## save model checkpoints every X epochs
        if args.save_ckpt and (num_iter % args.save_freq == 0 or epoch == args.epochs):            
            model_path = save_model_ckpt(args, best_records,                    
                                        model, classifier_head, optimizer, scheduler, logit_scale,
                                         val_acc, epoch, num_iter)
            logger.info(f'Model ckpt saved to: {model_path}')  

        if epoch == args.stop_epochs:
            break  
    
    logger.info(f'Training done.')
    logger.info(f'Best val Acc: {round(best_val_acc, 3)} at epoch {best_epoch}, iter {best_iter}')

    return best_model, best_head, best_records, best_logit_scale 


def train_CMO(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader):
    """ 
    Train the model with Cross-Entropy Loss, finetuning visual encoder and classifier
    Use CMO method to augment the training data
    """
    train_loader, weighted_train_loader = train_loader
    inverse_iter = iter(weighted_train_loader)

    logger.info(f"Start Training CMO ......")

    model.eval() if args.freeze_visual else model.train()
    classifier_head.train()

    logit_scale = args.logit_scale
    loss = args.loss
    optimizer = args.optimizer
    scheduler = args.scheduler    
    
    best_val_acc = -1
    best_epoch = -1
    best_iter = -1
    best_model = model
    best_head = classifier_head
    best_logit_scale = logit_scale
    best_records = {}
    num_iter = 0
    for epoch in range(1, args.epochs+1):
        
        train_loss_sum = 0
        # train for 1 epoch
        for inputs, labels, text, source in train_loader:

            try:
                inputs2, targets2, text2, source2 = next(inverse_iter)
            except:
                inverse_iter = iter(weighted_train_loader)
                inputs2, targets2, text2, source2 = next(inverse_iter)
            
            inputs2 = inputs2.to(args.device)
            targets2 = targets2.to(args.device).long()

            num_iter += 1
            images = inputs.to(args.device)
            labels = labels.to(args.device).long()

            # apply the cutmix strategy
            r = np.random.rand(1)
            if args.cutmix_beta > 0 and r < args.mix_prob:
                # generate mixed sample
                lam = np.random.beta(args.cutmix_beta, args.cutmix_beta)              
                bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
                images[:, :, bbx1:bbx2, bby1:bby2] = inputs2[:, :, bbx1:bbx2, bby1:bby2]            

                # adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))                
                target_a = labels
                target_b = targets2

            else:
                target_a = labels
                target_b = labels
                lam = 1.0
            
            image_features = model.encode_image(images)
            image_feature = image_features / image_features.norm(dim=-1, keepdim=True)
            
            logits = classifier_head(image_feature)
            logits = logits * logit_scale.exp()

            # mixup loss
            total_loss = mixup_criterion(loss, logits, target_a, target_b, lam)

            train_loss = total_loss.item()
            train_loss_sum += train_loss             
                
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step() # update learning rate for each iteration
        
        # validate
        if args.early_stop or epoch == args.epochs:          
            val_acc, val_loss, confusion_matrix = validate(args, data_loader=val_loader, model=model, logger=logger, 
                                                        loss=args.loss, logit_scale=logit_scale,
                                                        classifier_head=classifier_head, show_confusion_matrix=True,
                                                        dataset=args.dataset, 
                                                        output_dir=args.output_dir, device=args.device,
                                                        pre_extracted=args.pre_extracted, 
                                                        )
            scores = calculate_scores(confusion_matrix)        
                        
        # check if val acc has improved, here i use the val_acc rather than val_loss,
        # as we might modify the loss function for confusing pairs
        if (args.early_stop or epoch == args.epochs) and val_acc >= best_val_acc:
            # logger.info(f'Val Acc improved from {round(best_val_acc, 3)} to {round(val_acc, 3)}')
            best_val_acc = val_acc
            best_logit_scale = copy.deepcopy(logit_scale)
            best_epoch = epoch
            best_iter = num_iter
            best_scores = copy.deepcopy(scores)
            best_confusion_matrix = copy.deepcopy(confusion_matrix)
            best_head = copy.deepcopy(classifier_head)
            best_model = copy.deepcopy(model)

            # save into the best_records
            best_records['best_val_acc'] = best_val_acc
            best_records['best_logit_scale'] = best_logit_scale
            best_records['best_epoch'] = best_epoch
            best_records['best_iter'] = best_iter
            best_records['best_scores'] = best_scores
            best_records['best_confusion_matrix'] = best_confusion_matrix

        # test after 1 epoch
        if args.early_stop or epoch == args.epochs: 
            test_acc, _, _ = validate(args,data_loader=test_loader, model=model, logger=logger, 
                                loss=args.loss, logit_scale=logit_scale,
                                classifier_head=classifier_head,
                                dataset=args.dataset, 
                                output_dir=args.output_dir, device=args.device,
                                pre_extracted=args.pre_extracted, 
                                )                
        
        train_loss_avg = train_loss_sum / len(train_loader)
        loss_logger.write(f'{epoch},{num_iter},{round(train_loss_avg, 6)},{round(val_loss, 6)},{round(val_acc, 6)},{round(test_acc, 6)}\n')
        loss_logger.flush()
        logger.info(f"Epoch {int(epoch)}, Iter {num_iter}, Trn Loss: {round(train_loss_avg, 6)}, Val Loss: {round(val_loss, 6)}, Val Acc: {round(val_acc, 3)}, Test Acc: {round(test_acc, 3)}")          

        ## save model checkpoints every X epochs
        if args.save_ckpt and (num_iter % args.save_freq == 0 or epoch == args.epochs):            
            model_path = save_model_ckpt(args, best_records,                    
                                        model, classifier_head, optimizer, scheduler, logit_scale,
                                         val_acc, epoch, num_iter)
            logger.info(f'Model ckpt saved to: {model_path}')  

        if epoch == args.stop_epochs:
            break  
    
    logger.info(f'Training done.')
    logger.info(f'Best val Acc: {round(best_val_acc, 3)} at epoch {best_epoch}, iter {best_iter}')

    return best_model, best_head, best_records, best_logit_scale 

def saliency_bbox(img, lam):
    size = img.size()
    W = size[1]
    H = size[2]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # initialize OpenCV's static fine grained saliency detector and compute the saliency map
    img_arr = img.cpu().numpy().transpose(1, 2, 0)
    img_arr = (img_arr - img_arr.min()) / (img_arr.max() - img_arr.min())
    img_arr = (img_arr * 255).astype(np.uint8)
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    (success, saliencyMap) = saliency.computeSaliency(img_arr)
    saliencyMap = (saliencyMap * 255).astype("uint8")

    maximum_indices = np.unravel_index(np.argmax(saliencyMap, axis=None), saliencyMap.shape)
    x = maximum_indices[0]
    y = maximum_indices[1]

    # centered around the peak saliency
    bbx1 = np.clip(x - cut_w // 2, 0, W)
    bby1 = np.clip(y - cut_h // 2, 0, H)
    bbx2 = np.clip(x + cut_w // 2, 0, W)
    bby2 = np.clip(y + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def train_saliencymix(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader):
    """ 
    Train the model with Cross-Entropy Loss, finetuning visual encoder and classifier
    Use SaliencyMix method to augment the training data
    """

    logger.info(f"Start Training ......")

    model.eval() if args.freeze_visual else model.train()
    classifier_head.train()
    
    best_val_acc = -1
    best_epoch = -1
    best_iter = -1
    best_model = model
    best_head = classifier_head
    best_logit_scale = logit_scale
    best_records = {}
    num_iter = 0
    for epoch in range(1, args.epochs+1):
        
        train_loss_sum = 0
        # train for 1 epoch
        for inputs, labels, text, source in train_loader:
            num_iter += 1
            images = inputs.to(args.device)
            labels = labels.to(args.device).long()

            # apply the saliencymix strategy
            r = np.random.rand(1)
            if args.cutmix_beta > 0 and r < args.mix_prob:
                # generate mixed sample
                lam = np.random.beta(args.cutmix_beta, args.cutmix_beta)
                rand_index = torch.randperm(images.size()[0]).cuda()
                target_a = labels
                target_b = labels[rand_index]
                
                # old implementation
                bbx1, bby1, bbx2, bby2 = saliency_bbox(images[rand_index[0]], lam)
                images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]

                # adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))                
            else:
                target_a = labels
                target_b = labels
                lam = 1.0
            
            image_features = model.encode_image(images)
            image_feature = image_features / image_features.norm(dim=-1, keepdim=True)
            
            logits = classifier_head(image_feature)
            logits = logits * logit_scale.exp()

            # mixup loss
            total_loss = mixup_criterion(loss, logits, target_a, target_b, lam)

            train_loss = total_loss.item()
            train_loss_sum += train_loss             
                
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step() # update learning rate for each iteration
        
        # validate after 1 epoch
        # if (args.early_stop and num_iter >= args.start_validation) or \
        #     epoch == args.epochs:            
        val_acc, val_loss, confusion_matrix = validate(args, data_loader=val_loader, model=model, logger=logger, 
                                                        loss=args.loss, logit_scale=logit_scale,
                                                        classifier_head=classifier_head, show_confusion_matrix=True,
                                                        dataset=args.dataset, 
                                                        output_dir=args.output_dir, device=args.device,
                                                        pre_extracted=args.pre_extracted, 
                                                        )
        scores = calculate_scores(confusion_matrix)        
                        
        # check if val acc has improved, here i use the val_acc rather than val_loss,
        # as we might modify the loss function for confusing pairs
        if (args.early_stop or epoch == args.epochs) and val_acc >= best_val_acc:
            # logger.info(f'Val Acc improved from {round(best_val_acc, 3)} to {round(val_acc, 3)}')
            best_val_acc = val_acc
            best_logit_scale = copy.deepcopy(logit_scale)
            best_epoch = epoch
            best_iter = num_iter
            best_scores = copy.deepcopy(scores)
            best_confusion_matrix = copy.deepcopy(confusion_matrix)
            best_head = copy.deepcopy(classifier_head)
            best_model = copy.deepcopy(model)

            # save into the best_records
            best_records['best_val_acc'] = best_val_acc
            best_records['best_logit_scale'] = best_logit_scale
            best_records['best_epoch'] = best_epoch
            best_records['best_iter'] = best_iter
            best_records['best_scores'] = best_scores
            best_records['best_confusion_matrix'] = best_confusion_matrix

        # test after 1 epoch
        test_acc, _, _ = validate(args,data_loader=test_loader, model=model, logger=logger, 
                                loss=args.loss, logit_scale=logit_scale,
                                classifier_head=classifier_head,
                                dataset=args.dataset, 
                                output_dir=args.output_dir, device=args.device,
                                pre_extracted=args.pre_extracted, 
                                )                
        
        train_loss_avg = train_loss_sum / len(train_loader)
        loss_logger.write(f'{epoch},{num_iter},{round(train_loss_avg, 6)},{round(val_loss, 6)},{round(val_acc, 6)},{round(test_acc, 6)}\n')
        loss_logger.flush()
        logger.info(f"Epoch {int(epoch)}, Iter {num_iter}, Trn Loss: {round(train_loss_avg, 6)}, Val Loss: {round(val_loss, 6)}, Val Acc: {round(val_acc, 3)}, Test Acc: {round(test_acc, 3)}")          

        ## save model checkpoints every X epochs
        if args.save_ckpt and (num_iter % args.save_freq == 0 or epoch == args.epochs):            
            model_path = save_model_ckpt(args, best_records,                    
                                        model, classifier_head, optimizer, scheduler, logit_scale,
                                         val_acc, epoch, num_iter)
            logger.info(f'Model ckpt saved to: {model_path}')  

        if epoch == args.stop_epochs:
            break  
    
    logger.info(f'Training done.')
    logger.info(f'Best val Acc: {round(best_val_acc, 3)} at epoch {best_epoch}, iter {best_iter}')

    return best_model, best_head, best_records, best_logit_scale 

def saliency_bbox_list(imgs, lam):
    corners_list = []
    for img in imgs:
        bbx1, bby1, bbx2, bby2 = saliency_bbox(img, lam)
        corners_list.append([bbx1, bby1, bbx2, bby2])
    return corners_list

def train_saliencymix2(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader):
    """ 
    Train the model with Cross-Entropy Loss, finetuning visual encoder and classifier
    Use SaliencyMix method to augment the training data
    +++++ here we compute the saliency for each image and apply the saliency to the image +++++
    """

    logger.info(f"Start Training saliencymix2 ......")

    model.eval() if args.freeze_visual else model.train()
    classifier_head.train()
    
    logit_scale = args.logit_scale
    loss = args.loss
    optimizer = args.optimizer
    scheduler = args.scheduler
    
    best_val_acc = -1
    best_epoch = -1
    best_iter = -1
    best_model = model
    best_head = classifier_head
    best_logit_scale = logit_scale
    best_records = {}
    num_iter = 0
    for epoch in range(1, args.epochs+1):
        
        train_loss_sum = 0
        # train for 1 epoch
        for inputs, labels, text, source in train_loader:
            num_iter += 1
            images = inputs.to(args.device)
            labels = labels.to(args.device).long()

            # apply the saliencymix strategy
            r = np.random.rand(1)
            if args.cutmix_beta > 0 and r < args.mix_prob:
                # generate mixed sample
                lam = np.random.beta(args.cutmix_beta, args.cutmix_beta)
                rand_index = torch.randperm(images.size()[0]).cuda()
                target_a = labels
                target_b = labels[rand_index]
                shuffled_images = images[rand_index]

                corners_list = saliency_bbox_list(shuffled_images, lam)
                lam_list = []
                
                for i in range(images.size(0)):
                    bbx1, bby1, bbx2, bby2 = corners_list[i]
                    images[i, :, bbx1:bbx2, bby1:bby2] = shuffled_images[i, :, bbx1:bbx2, bby1:bby2]
                    # adjust lambda to exactly match pixel ratio
                    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))                
                    lam_list.append(lam)

            else:
                target_a = labels
                target_b = labels
                lam_list = [1.0] * images.size(0)
            
            image_features = model.encode_image(images)
            image_feature = image_features / image_features.norm(dim=-1, keepdim=True)
            
            logits = classifier_head(image_feature)
            logits = logits * logit_scale.exp()

            # mixup loss
            total_loss = mixup_criterion_lam_list(loss, logits, target_a, target_b, lam_list)

            train_loss = total_loss.item()
            train_loss_sum += train_loss             
                
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step() # update learning rate for each iteration
        
        # validate after 1 epoch
        # if (args.early_stop and num_iter >= args.start_validation) or \
        #     epoch == args.epochs:
        if args.early_stop or epoch == args.epochs:             
            val_acc, val_loss, confusion_matrix = validate(args, data_loader=val_loader, model=model, logger=logger, 
                                                        loss=args.loss, logit_scale=logit_scale,
                                                        classifier_head=classifier_head, show_confusion_matrix=True,
                                                        dataset=args.dataset, 
                                                        output_dir=args.output_dir, device=args.device,
                                                        pre_extracted=args.pre_extracted, 
                                                        )
            scores = calculate_scores(confusion_matrix)        
                        
        # check if val acc has improved, here i use the val_acc rather than val_loss,
        # as we might modify the loss function for confusing pairs
        if (args.early_stop or epoch == args.epochs) and val_acc >= best_val_acc:
            # logger.info(f'Val Acc improved from {round(best_val_acc, 3)} to {round(val_acc, 3)}')
            best_val_acc = val_acc
            best_logit_scale = copy.deepcopy(logit_scale)
            best_epoch = epoch
            best_iter = num_iter
            best_scores = copy.deepcopy(scores)
            best_confusion_matrix = copy.deepcopy(confusion_matrix)
            best_head = copy.deepcopy(classifier_head)
            best_model = copy.deepcopy(model)

            # save into the best_records
            best_records['best_val_acc'] = best_val_acc
            best_records['best_logit_scale'] = best_logit_scale
            best_records['best_epoch'] = best_epoch
            best_records['best_iter'] = best_iter
            best_records['best_scores'] = best_scores
            best_records['best_confusion_matrix'] = best_confusion_matrix

        # test after 1 epoch
        test_acc, _, _ = validate(args,data_loader=test_loader, model=model, logger=logger, 
                                loss=args.loss, logit_scale=logit_scale,
                                classifier_head=classifier_head,
                                dataset=args.dataset, 
                                output_dir=args.output_dir, device=args.device,
                                pre_extracted=args.pre_extracted, 
                                )                
        
        train_loss_avg = train_loss_sum / len(train_loader)
        loss_logger.write(f'{epoch},{num_iter},{round(train_loss_avg, 6)},{round(val_loss, 6)},{round(val_acc, 6)},{round(test_acc, 6)}\n')
        loss_logger.flush()
        logger.info(f"Epoch {int(epoch)}, Iter {num_iter}, Trn Loss: {round(train_loss_avg, 6)}, Val Loss: {round(val_loss, 6)}, Val Acc: {round(val_acc, 3)}, Test Acc: {round(test_acc, 3)}")          

        ## save model checkpoints every X epochs
        if args.save_ckpt and (num_iter % args.save_freq == 0 or epoch == args.epochs):            
            model_path = save_model_ckpt(args, best_records,                    
                                        model, classifier_head, optimizer, scheduler, logit_scale,
                                         val_acc, epoch, num_iter)
            logger.info(f'Model ckpt saved to: {model_path}')  

        if epoch == args.stop_epochs:
            break  
    
    logger.info(f'Training done.')
    logger.info(f'Best val Acc: {round(best_val_acc, 3)} at epoch {best_epoch}, iter {best_iter}')

    return best_model, best_head, best_records, best_logit_scale 


def train_resizemix(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader):
    """ 
    Train the model with Cross-Entropy Loss, finetuning visual encoder and classifier
    Use ResizeMix method to augment the training data
    """

    logger.info(f"Start Training resizemix ......")

    model.eval() if args.freeze_visual else model.train()
    classifier_head.train()
    
    logit_scale = args.logit_scale
    loss = args.loss
    optimizer = args.optimizer
    scheduler = args.scheduler    

    best_val_acc = -1
    best_epoch = -1
    best_iter = -1
    best_model = model
    best_head = classifier_head
    best_logit_scale = logit_scale
    best_records = {}
    num_iter = 0
    for epoch in range(1, args.epochs+1):
        
        train_loss_sum = 0
        # train for 1 epoch
        for inputs, labels, text, source in train_loader:
            num_iter += 1
            images = inputs.to(args.device)
            labels = labels.to(args.device).long()

            # apply the resizemix strategy
            r = np.random.rand(1)
            if r < args.mix_prob:
                # generate mixed sample
                # uniform sampling the lambda from [0.1, 0.8]
                tau = np.random.uniform(0.1, 0.8)

                rand_index = torch.randperm(images.size()[0]).cuda()
                shuffled_images = images[rand_index]
                # print('shuffled_images.size()', shuffled_images.size())
                # resize the shuffled_images to a smaller size of the original images, with scale tau
                resized_images = F.interpolate(shuffled_images, scale_factor=tau, mode='bilinear', align_corners=False)
                # print('resized_images.size()', resized_images.size())

                # get the size of the resized_images
                resized_w = resized_images.size()[-1]
                resized_h = resized_images.size()[-2]

                # get the random position to paste the resized_images
                pos_x = np.random.randint(0, images.size()[-1] - resized_w)
                pos_y = np.random.randint(0, images.size()[-2] - resized_h)

                # paste the resized_images to the original images
                images[:, :, pos_x:pos_x+resized_w, pos_y:pos_y+resized_h] = resized_images  

                # adjust lambda to exactly match pixel ratio
                lam = 1.0 - (resized_w * resized_h / (images.size()[-1] * images.size()[-2]))             

                # labels
                target_a = labels
                target_b = labels[rand_index]      
            else:
                target_a = labels
                target_b = labels
                lam = 1.0
            
            image_features = model.encode_image(images)
            image_feature = image_features / image_features.norm(dim=-1, keepdim=True)
            
            logits = classifier_head(image_feature)
            logits = logits * logit_scale.exp()

            # mixup loss, lam belongs to target_a
            total_loss = mixup_criterion(loss, logits, target_a, target_b, lam)

            train_loss = total_loss.item()
            train_loss_sum += train_loss             
                
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step() # update learning rate for each iteration
        
        # validate after 1 epoch
        # if (args.early_stop and num_iter >= args.start_validation) or \
        #     epoch == args.epochs:            
        val_acc, val_loss, confusion_matrix = validate(args, data_loader=val_loader, model=model, logger=logger, 
                                                        loss=args.loss, logit_scale=logit_scale,
                                                        classifier_head=classifier_head, show_confusion_matrix=True,
                                                        dataset=args.dataset, 
                                                        output_dir=args.output_dir, device=args.device,
                                                        pre_extracted=args.pre_extracted, 
                                                        )
        scores = calculate_scores(confusion_matrix)        
                        
        # check if val acc has improved, here i use the val_acc rather than val_loss,
        # as we might modify the loss function for confusing pairs
        if (args.early_stop or epoch == args.epochs) and val_acc >= best_val_acc:
            # logger.info(f'Val Acc improved from {round(best_val_acc, 3)} to {round(val_acc, 3)}')
            best_val_acc = val_acc
            best_logit_scale = copy.deepcopy(logit_scale)
            best_epoch = epoch
            best_iter = num_iter
            best_scores = copy.deepcopy(scores)
            best_confusion_matrix = copy.deepcopy(confusion_matrix)
            best_head = copy.deepcopy(classifier_head)
            best_model = copy.deepcopy(model)

            # save into the best_records
            best_records['best_val_acc'] = best_val_acc
            best_records['best_logit_scale'] = best_logit_scale
            best_records['best_epoch'] = best_epoch
            best_records['best_iter'] = best_iter
            best_records['best_scores'] = best_scores
            best_records['best_confusion_matrix'] = best_confusion_matrix

        # test after 1 epoch
        test_acc, _, _ = validate(args,data_loader=test_loader, model=model, logger=logger, 
                                loss=args.loss, logit_scale=logit_scale,
                                classifier_head=classifier_head,
                                dataset=args.dataset, 
                                output_dir=args.output_dir, device=args.device,
                                pre_extracted=args.pre_extracted, 
                                )                
        
        train_loss_avg = train_loss_sum / len(train_loader)
        loss_logger.write(f'{epoch},{num_iter},{round(train_loss_avg, 6)},{round(val_loss, 6)},{round(val_acc, 6)},{round(test_acc, 6)}\n')
        loss_logger.flush()
        logger.info(f"Epoch {int(epoch)}, Iter {num_iter}, Trn Loss: {round(train_loss_avg, 6)}, Val Loss: {round(val_loss, 6)}, Val Acc: {round(val_acc, 3)}, Test Acc: {round(test_acc, 3)}")          

        ## save model checkpoints every X epochs
        if args.save_ckpt and (num_iter % args.save_freq == 0 or epoch == args.epochs):            
            model_path = save_model_ckpt(args, best_records,                    
                                        model, classifier_head, optimizer, scheduler, logit_scale,
                                         val_acc, epoch, num_iter)
            logger.info(f'Model ckpt saved to: {model_path}')  

        if epoch == args.stop_epochs:
            break  
    
    logger.info(f'Training done.')
    logger.info(f'Best val Acc: {round(best_val_acc, 3)} at epoch {best_epoch}, iter {best_iter}')

    return best_model, best_head, best_records, best_logit_scale 


def get_GEM_heatmap(gem_model, images, texts):
    heatmap_list = gem_model.batched_forward(images, texts)
    return heatmap_list


def saliency_bbox_gem(heatmap, lam):
    # convert heatmap from size [1, W, H] to [W, H]
    # detach heatmap to cpu first
    # print('heatmap.size()', heatmap.size())
    heatmap= heatmap.squeeze(0).cpu()

    # print('heatmap.size()', heatmap.size())
    # convert heatmap from torch tensor to numpy array
    heatmap = heatmap.detach().numpy()

    size = heatmap.shape
    W = size[0]
    H = size[1]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # use heatmap asthe saliencymap
    maximum_indices = np.unravel_index(np.argmax(heatmap, axis=None), heatmap.shape)
    x = maximum_indices[0]
    y = maximum_indices[1]

    # centered around the peak saliency
    bbx1 = np.clip(x - cut_w // 2, 0, W)
    bby1 = np.clip(y - cut_h // 2, 0, H)
    bbx2 = np.clip(x + cut_w // 2, 0, W)
    bby2 = np.clip(y + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def saliency_bbox_list_gem(shuffled_images, lam, gem_model, texts):

    corners_list = []
    heatmap_list = get_GEM_heatmap(gem_model, shuffled_images, texts)
    # print(shuffled_images[0].size())
    # print(heatmap_list[0].size())
    for heatmap in heatmap_list:
        bbx1, bby1, bbx2, bby2 = saliency_bbox_gem(heatmap, lam)
        corners_list.append([bbx1, bby1, bbx2, bby2])
    return corners_list

def train_attentivemix(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader):
    """ 
    Train the model with Cross-Entropy Loss, finetuning visual encoder and classifier
    Use AttentiveMix method to augment the training data
    """

    # initialize GEM model
    model_name = 'ViT-B/32'
    pretrained = 'laion400m_e32'
    gem_model = create_gem_model(model_name=model_name, pretrained=pretrained, device=args.device)
    gem_model.eval()
    logger.info(f'GEM model loaded from {model_name} {pretrained}')  
    threshold = args.attentive_threshold

    # get the label names dict
    metric_fn = f'{args.dataset_root}/id_scname_dict.json' 
    with open(metric_fn, 'r') as f:
        metrics = json.load(f)

    logger.info(f"Start Training ......")

    model.eval() if args.freeze_visual else model.train()
    classifier_head.train()
    
    best_val_acc = -1
    best_epoch = -1
    best_iter = -1
    best_model = model
    best_head = classifier_head
    best_logit_scale = logit_scale
    best_records = {}
    num_iter = 0
    for epoch in range(1, args.epochs+1):
        
        train_loss_sum = 0
        # train for 1 epoch
        for inputs, labels, text, source in train_loader:
            num_iter += 1
            images = inputs.to(args.device)
            labels = labels.to(args.device).long()

            # apply the attentivemix strategy
            r = np.random.rand(1)
            if r < args.mix_prob:

                # generate mixed sample
                rand_index = torch.randperm(images.size()[0]).cuda()
                target_a = labels
                target_b = labels[rand_index]
                shuffled_images = images[rand_index]

                # use general domain name here
                if args.attentive_name == 'general':
                    texts = [['bird'] for i in range(images.size()[0])]
                elif args.attentive_name == 'c-name':
                    # use c-name to get the heatmap
                    texts = []
                    for label in target_b:
                        texts.append([metrics[str(label.item())][1]])
                elif args.attentive_name == 's-name':
                    texts = []
                    for label in target_b:
                        texts.append([metrics[str(label.item())][0]])
                # print(target_b)
                # print(texts)

                heatmap_list = get_GEM_heatmap(gem_model, shuffled_images, texts)
                
                # get the binary_mask_list using the threshold
                binary_mask_list = []
                for heatmap in heatmap_list:
                    binary_mask = (heatmap > threshold).int()
                    binary_mask_list.append(binary_mask)
                
                # build the new image by attentively mixing the images usingt he binary_mask
                lam_list = []
                for i, binary_mask in enumerate(binary_mask_list):
                    images[i] = images[i] * (1 - binary_mask) + shuffled_images[i] * binary_mask
                    lam = 1.0 - binary_mask.sum() / (images.size()[-1] * images.size()[-2])
                    lam_list.append(lam)     
            else:
                target_a = labels
                target_b = labels
                lam_list = [1.0] * images.size(0)
            
            # print('lam_list', lam_list)

            image_features = model.encode_image(images)
            image_feature = image_features / image_features.norm(dim=-1, keepdim=True)
            
            logits = classifier_head(image_feature)
            logits = logits * logit_scale.exp()

            # mixup loss, lam belongs to target_a
            total_loss = mixup_criterion_lam_list(loss, logits, target_a, target_b, lam_list)

            train_loss = total_loss.item()
            train_loss_sum += train_loss             
                
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step() # update learning rate for each iteration
        
        # validate after 1 epoch
        # if (args.early_stop and num_iter >= args.start_validation) or \
        #     epoch == args.epochs:            
        val_acc, val_loss, confusion_matrix = validate(args, data_loader=val_loader, model=model, logger=logger, 
                                                        loss=args.loss, logit_scale=logit_scale,
                                                        classifier_head=classifier_head, show_confusion_matrix=True,
                                                        dataset=args.dataset, 
                                                        output_dir=args.output_dir, device=args.device,
                                                        pre_extracted=args.pre_extracted, 
                                                        )
        scores = calculate_scores(confusion_matrix)        
                        
        # check if val acc has improved, here i use the val_acc rather than val_loss,
        # as we might modify the loss function for confusing pairs
        if (args.early_stop or epoch == args.epochs) and val_acc >= best_val_acc:
            # logger.info(f'Val Acc improved from {round(best_val_acc, 3)} to {round(val_acc, 3)}')
            best_val_acc = val_acc
            best_logit_scale = copy.deepcopy(logit_scale)
            best_epoch = epoch
            best_iter = num_iter
            best_scores = copy.deepcopy(scores)
            best_confusion_matrix = copy.deepcopy(confusion_matrix)
            best_head = copy.deepcopy(classifier_head)
            best_model = copy.deepcopy(model)

            # save into the best_records
            best_records['best_val_acc'] = best_val_acc
            best_records['best_logit_scale'] = best_logit_scale
            best_records['best_epoch'] = best_epoch
            best_records['best_iter'] = best_iter
            best_records['best_scores'] = best_scores
            best_records['best_confusion_matrix'] = best_confusion_matrix

        # test after 1 epoch
        test_acc, _, _ = validate(args,data_loader=test_loader, model=model, logger=logger, 
                                loss=args.loss, logit_scale=logit_scale,
                                classifier_head=classifier_head,
                                dataset=args.dataset, 
                                output_dir=args.output_dir, device=args.device,
                                pre_extracted=args.pre_extracted, 
                                )                
        
        train_loss_avg = train_loss_sum / len(train_loader)
        loss_logger.write(f'{epoch},{num_iter},{round(train_loss_avg, 6)},{round(val_loss, 6)},{round(val_acc, 6)},{round(test_acc, 6)}\n')
        loss_logger.flush()
        logger.info(f"Epoch {int(epoch)}, Iter {num_iter}, Trn Loss: {round(train_loss_avg, 6)}, Val Loss: {round(val_loss, 6)}, Val Acc: {round(val_acc, 3)}, Test Acc: {round(test_acc, 3)}")          

        ## save model checkpoints every X epochs
        if args.save_ckpt and (num_iter % args.save_freq == 0 or epoch == args.epochs):            
            model_path = save_model_ckpt(args, best_records,                    
                                        model, classifier_head, optimizer, scheduler, logit_scale,
                                         val_acc, epoch, num_iter)
            logger.info(f'Model ckpt saved to: {model_path}')  

        if epoch == args.stop_epochs:
            break  
    
    logger.info(f'Training done.')
    logger.info(f'Best val Acc: {round(best_val_acc, 3)} at epoch {best_epoch}, iter {best_iter}')

    return best_model, best_head, best_records, best_logit_scale 


def train_attentivemix2(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader):
    """ 
    Train the model with Cross-Entropy Loss, finetuning visual encoder and classifier
    Use AttentiveMix method to augment the training data
    Use rectangular patches
    """

    # initialize GEM model
    model_name = 'ViT-B/32'
    pretrained = 'laion400m_e32'
    gem_model = create_gem_model(model_name=model_name, pretrained=pretrained, device=args.device)
    gem_model.eval()
    logger.info(f'GEM model loaded from {model_name} {pretrained}')  
    threshold = args.attentive_threshold

    # get the label names dict
    metric_fn = f'{args.dataset_root}/id_scname_dict.json' 
    with open(metric_fn, 'r') as f:
        metrics = json.load(f)

    logger.info(f"Start Training ......")

    model.eval() if args.freeze_visual else model.train()
    classifier_head.train()
    
    best_val_acc = -1
    best_epoch = -1
    best_iter = -1
    best_model = model
    best_head = classifier_head
    best_logit_scale = logit_scale
    best_records = {}
    num_iter = 0
    for epoch in range(1, args.epochs+1):
        
        train_loss_sum = 0
        # train for 1 epoch
        for inputs, labels, text, source in train_loader:
            num_iter += 1
            images = inputs.to(args.device)
            labels = labels.to(args.device).long()

            # apply the attentivemix strategy
            r = np.random.rand(1)
            if args.cutmix_beta > 0 and r < args.mix_prob:
                # generate mixed sample
                lam = np.random.beta(args.cutmix_beta, args.cutmix_beta)
                rand_index = torch.randperm(images.size()[0]).cuda()
                target_a = labels
                target_b = labels[rand_index]
                shuffled_images = images[rand_index]

                # build texts for GEM
                if args.attentive_name == 'general':
                    texts = [['bird'] for i in range(images.size()[0])]
                elif args.attentive_name == 'c-name':
                    texts = []
                    for label in target_b:
                        texts.append([metrics[str(label.item())][1]])
                elif args.attentive_name == 's-name':
                    texts = []
                    for label in target_b:
                        texts.append([metrics[str(label.item())][0]])
                else:
                    raise NotImplementedError

                corners_list = saliency_bbox_list_gem(shuffled_images, lam, gem_model, texts)
                lam_list = []
                
                for i in range(images.size(0)):
                    bbx1, bby1, bbx2, bby2 = corners_list[i]
                    images[i, :, bbx1:bbx2, bby1:bby2] = shuffled_images[i, :, bbx1:bbx2, bby1:bby2]
                    # adjust lambda to exactly match pixel ratio
                    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))                
                    lam_list.append(lam)            
                
            else:
                target_a = labels
                target_b = labels
                lam_list = [1.0] * images.size(0)
            
            # print('lam_list', lam_list)

            image_features = model.encode_image(images)
            image_feature = image_features / image_features.norm(dim=-1, keepdim=True)
            
            logits = classifier_head(image_feature)
            logits = logits * logit_scale.exp()

            # mixup loss, lam belongs to target_a
            total_loss = mixup_criterion_lam_list(loss, logits, target_a, target_b, lam_list)

            train_loss = total_loss.item()
            train_loss_sum += train_loss             
                
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step() # update learning rate for each iteration
        
        # validate after 1 epoch
        # if (args.early_stop and num_iter >= args.start_validation) or \
        #     epoch == args.epochs:            
        val_acc, val_loss, confusion_matrix = validate(args, data_loader=val_loader, model=model, logger=logger, 
                                                        loss=args.loss, logit_scale=logit_scale,
                                                        classifier_head=classifier_head, show_confusion_matrix=True,
                                                        dataset=args.dataset, 
                                                        output_dir=args.output_dir, device=args.device,
                                                        pre_extracted=args.pre_extracted, 
                                                        )
        scores = calculate_scores(confusion_matrix)        
                        
        # check if val acc has improved, here i use the val_acc rather than val_loss,
        # as we might modify the loss function for confusing pairs
        if (args.early_stop or epoch == args.epochs) and val_acc >= best_val_acc:
            # logger.info(f'Val Acc improved from {round(best_val_acc, 3)} to {round(val_acc, 3)}')
            best_val_acc = val_acc
            best_logit_scale = copy.deepcopy(logit_scale)
            best_epoch = epoch
            best_iter = num_iter
            best_scores = copy.deepcopy(scores)
            best_confusion_matrix = copy.deepcopy(confusion_matrix)
            best_head = copy.deepcopy(classifier_head)
            best_model = copy.deepcopy(model)

            # save into the best_records
            best_records['best_val_acc'] = best_val_acc
            best_records['best_logit_scale'] = best_logit_scale
            best_records['best_epoch'] = best_epoch
            best_records['best_iter'] = best_iter
            best_records['best_scores'] = best_scores
            best_records['best_confusion_matrix'] = best_confusion_matrix

        # test after 1 epoch
        test_acc, _, _ = validate(args,data_loader=test_loader, model=model, logger=logger, 
                                loss=args.loss, logit_scale=logit_scale,
                                classifier_head=classifier_head,
                                dataset=args.dataset, 
                                output_dir=args.output_dir, device=args.device,
                                pre_extracted=args.pre_extracted, 
                                )                
        
        train_loss_avg = train_loss_sum / len(train_loader)
        loss_logger.write(f'{epoch},{num_iter},{round(train_loss_avg, 6)},{round(val_loss, 6)},{round(val_acc, 6)},{round(test_acc, 6)}\n')
        loss_logger.flush()
        logger.info(f"Epoch {int(epoch)}, Iter {num_iter}, Trn Loss: {round(train_loss_avg, 6)}, Val Loss: {round(val_loss, 6)}, Val Acc: {round(val_acc, 3)}, Test Acc: {round(test_acc, 3)}")          

        ## save model checkpoints every X epochs
        if args.save_ckpt and (num_iter % args.save_freq == 0 or epoch == args.epochs):            
            model_path = save_model_ckpt(args, best_records,                    
                                        model, classifier_head, optimizer, scheduler, logit_scale,
                                         val_acc, epoch, num_iter)
            logger.info(f'Model ckpt saved to: {model_path}')  

        if epoch == args.stop_epochs:
            break  
    
    logger.info(f'Training done.')
    logger.info(f'Best val Acc: {round(best_val_acc, 3)} at epoch {best_epoch}, iter {best_iter}')

    return best_model, best_head, best_records, best_logit_scale 

def run_tau_normalization(args, best_head, best_model, val_loader, test_loader, logit_scale, logger):

    best_tau_head = copy.deepcopy(best_head)
    best_tau = 0.0
    best_tau_val_acc = 0.0
    best_tau_test_acc = 0.0
    if args.tau_norm:
        logger.info(f"Check Tau Normalization ......") 
        tau_list = np.arange(0.0, 2.2, 0.2).tolist()
        for tau in tau_list:
            tau_head = copy.deepcopy(best_head)
            tau_head.linear.weight.data /= torch.pow(tau_head.linear.weight.data.norm(dim=-1, keepdim=True), tau) 
            # does not affect FLYP because head is already normalized, thus the norm=1
            
            # check on val set
            tau_val_acc, _, _ = validate(args,data_loader=val_loader, 
                                        model=best_model, logger=logger, 
                                        loss=args.loss, logit_scale=logit_scale,
                                        classifier_head=tau_head,
                                        dataset=args.dataset, 
                                        output_dir=args.output_dir, device=args.device,
                                        pre_extracted=True, 
                                        )
            # check on test set
            tau_test_acc, _, tau_test_confusion_matrix = validate(args,data_loader=test_loader,
                                            model=best_model, logger=logger, 
                                            loss=args.loss, logit_scale=logit_scale,
                                            show_confusion_matrix=True,
                                            classifier_head=tau_head, 
                                            dataset=args.dataset, 
                                            output_dir=args.output_dir, device=args.device,
                                            pre_extracted=True, 
                                            )
            logger.info(f"Tau: {round(tau,2)}, Val Acc: {round(tau_val_acc, 3)}, Test Acc: {round(tau_test_acc, 3)}")
            if tau_val_acc > best_tau_val_acc:
                best_tau = tau
                best_tau_val_acc = tau_val_acc
                best_tau_test_acc = tau_test_acc
                best_tau_head = copy.deepcopy(tau_head)
                best_tau_scores = calculate_scores(tau_test_confusion_matrix)
                best_tau_confusion_matrix = copy.deepcopy(tau_test_confusion_matrix)

        logger.info(f"+++++ Best Tau: {best_tau}, Val Acc: {round(best_tau_val_acc, 3)}, Test Acc: {round(best_tau_test_acc, 3)}")
        # save_test_scores(best_tau_scores, best_tau_confusion_matrix, args.output_dir, 'best_tau_test')
        # save_head_weights(best_tau_head, output_dir, 'best_tau')
    
    return best_tau_head, best_tau, best_tau_test_acc


def ensemble_model(best_model, zeroshot_model, alpha):
    """Ensemble the best_model and zeroshot_model"""

    wsft_model = copy.deepcopy(best_model)
    # Load models
    zeroshot = zeroshot_model
    finetuned = best_model
    theta_0 = zeroshot.state_dict()
    theta_1 = finetuned.state_dict()

    # make sure checkpoints are compatible
    assert set(theta_0.keys()) == set(theta_1.keys())

    # interpolate between checkpoints with mixing coefficient alpha
    theta = {
        key: (1.0-alpha) * theta_0[key] + alpha * theta_1[key]
        for key in theta_0.keys()
    }

    # update the model acccording to the new weights
    wsft_model.load_state_dict(theta)

    return wsft_model

def run_wsft(args, best_model, best_head, test_loader, zeroshot_model, zeroshot_weights, logit_scale, logger, alpha=0.5):

    learned_head_weights = best_head.linear.weight.data.to(args.device)
    wsft_head_weights = alpha * learned_head_weights + (1.0 - alpha) * zeroshot_weights
    wsft_head = MyLinear(weights=wsft_head_weights)
    wsft_head.to(args.device)
    logger.info(f'WiSE-FT classifier done. alpha: {alpha}')
    if args.freeze_visual:
        wsft_model = best_model
    else:
        # ensemble the best_model and zeroshot_model
        wsft_model = ensemble_model(best_model, zeroshot_model, alpha)
        logger.info(f'WiSE-FT model done. alpha: {alpha}')

    wsft_test_acc, _, _ = validate(args,data_loader=test_loader, 
                                                        model=wsft_model, 
                                                        classifier_head=wsft_head, # here use the wsft_head
                                                        logger=logger,
                                                        loss=args.loss, logit_scale=logit_scale, 
                                                        show_confusion_matrix=False,
                                                        dataset=args.dataset, 
                                                        output_dir=args.output_dir, device=args.device,
                                                        pre_extracted=args.pre_extracted,                                                    
                                                        )
    logger.info(f"+++++ WiSE-FT Test Acc: {round(wsft_test_acc, 3)}")
    # wsft_test_scores = calculate_scores(wsft_test_confusion_matrix)
    # save_test_scores(wsft_test_scores, wsft_test_confusion_matrix, args.output_dir, 'wsft_test')
    # save_head_weights(wsft_head, output_dir, 'wsft')

    return wsft_model, wsft_head, wsft_test_acc

def run_wsft_alpha(args, best_model, best_head, val_loader, test_loader, zeroshot_model, zeroshot_head, logit_scale, logger, step=0.1):
    logger.info(f"Checking WSFT ......")

    ensemble_val_acc = []
    ensemble_test_acc = []
    learned_head_weights = best_head.linear.weight.data.to(args.device)
    zeroshot_weights = zeroshot_head.linear.weight.data.to(args.device)
    best_alpha = 0.0
    best_wsft_test_acc = 0.0
    best_wsft_val_acc = 0.0
    best_wsft_head = best_head
    best_wsft_model = best_model
    # for alpha in np.arange(0.0, 1.0+step, step):
    for alpha in [0.5]:
        
        wsft_head_weights = alpha * learned_head_weights + (1.0 - alpha) * zeroshot_weights
        wsft_head = MyLinear(weights=wsft_head_weights)
        wsft_head.to(args.device)

        # wsft_head = best_head # use the best_head, do not ensemble the head
        # wsft_head = zeroshot_head # use the zeroshot_head, do not ensemble the head

        if args.freeze_visual:
            wsft_model = best_model
        else:
            # ensemble the best_model and zeroshot_model
            wsft_model = ensemble_model(best_model, zeroshot_model, alpha)

        wsft_val_acc, _, _ = validate(args,data_loader=val_loader, 
                                        model=wsft_model, 
                                        classifier_head=wsft_head, # here use the wsft_head
                                        logger=logger,
                                        loss=args.loss, logit_scale=logit_scale, 
                                        show_confusion_matrix=False,
                                        dataset=args.dataset, 
                                        output_dir=args.output_dir, device=args.device,
                                        pre_extracted=args.pre_extracted,                                                    
                                        )
        
        wsft_test_acc, _, _ = validate(args,data_loader=test_loader, 
                                        model=wsft_model, 
                                        classifier_head=wsft_head, # here use the wsft_head
                                        logger=logger,
                                        loss=args.loss, logit_scale=logit_scale, 
                                        show_confusion_matrix=False,
                                        dataset=args.dataset, 
                                        output_dir=args.output_dir, device=args.device,
                                        pre_extracted=args.pre_extracted,                                                    
                                        )
        ensemble_val_acc.append(wsft_val_acc)
        ensemble_test_acc.append(wsft_test_acc)
        logger.info(f"Alpha:{round(alpha, 3)}, Val Acc: {round(wsft_val_acc, 3)}, Test Acc: {round(wsft_test_acc, 3)}")
        if wsft_val_acc > best_wsft_val_acc:
            best_wsft_val_acc = wsft_val_acc
            best_wsft_test_acc = wsft_test_acc
            best_alpha = alpha
            best_wsft_head = copy.deepcopy(wsft_head)
            best_wsft_model = copy.deepcopy(wsft_model)
    
    logger.info(f"+++++ Best Alpha: {round(best_alpha, 2)}, Val Acc: {round(best_wsft_val_acc, 3)}, Test Acc: {round(best_wsft_test_acc, 3)}")
    # print(f'ensemble_val_acc', ensemble_val_acc)
    # print(f'ensemble_test_acc', ensemble_test_acc)
    
    return best_wsft_model, best_wsft_head, best_wsft_test_acc


def run_stage1_finetuning():

    # dataloaders
    train_loader, val_loader, test_loader = set_dataloaders(args, model, tokenized_text_prompts, preprocess, logger)
    text_dataloader = set_text_dataloader(args, prompt_tensors, prompt_tensors_dict) if args.method == 'CMLP' else None
    test_loader_copy = copy.deepcopy(test_loader)

    loss = set_loss(args) # depend on method
    params, logit_scale = set_params(args, model, classifier_head, logger) # depend on method
    optimizer, scheduler, total_iter = set_optimizer(args, params, train_loader)
    
    args.loss = loss
    args.logit_scale = logit_scale
    args.optimizer = optimizer
    args.scheduler = scheduler

    # check zeroshot acc
    if args.check_zeroshot or args.method == 'zeroshot':
        logger.info(f"Check Zero-shot Acc ......")
        run_zeroshot(args, test_loader, model, logger, loss, logit_scale, classifier_head)
    if args.zeroshot_only or args.method == 'zeroshot':
        exit()    

    reload_model = True if args.model_path else False
    #---------- Training
    if args.method == 'probing':         
        best_model, best_head, best_records, best_logit_scale, val_loader, test_loader = train_probing(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader, reload_model, text_dataloader)
    elif args.method == 'dataset-cls':
        best_model, best_head, best_records, best_logit_scale = train_dataset(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader)    
    elif args.method == 'CMLP': # cross modal linear probing         
        best_model, best_head, best_records, best_logit_scale, val_loader, test_loader = train_CMLP(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader, False, text_dataloader)
    elif args.method == 'finetune':
        best_model, best_head, best_records, best_logit_scale = train_ce(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader)    
    elif args.method == 'finetune-mixed': # half batch is retrieved, half batch is fewshot
        best_model, best_head, best_records, best_logit_scale = train_ce_mixed(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader)    
    elif args.method == 'finetune-multitask': # 1 backbone 2 output heads
        best_model, best_head, best_records, best_logit_scale = train_ce_multitask(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader, dataset_classifier_head)          
    elif args.method == 'mixup': # random mixup
        best_model, best_head, best_records, best_logit_scale = train_mixup(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader)
    elif args.method == 'mixup-fs': # mix retrieved with few-shot
        best_model, best_head, best_records, best_logit_scale = train_mixup_fs(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader)        
    elif args.method == 'cutmix': # cutmix
        best_model, best_head, best_records, best_logit_scale = train_cutmix(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader)  
    elif args.method == 'cutmix-fs': # cutmix with few-shot data
        best_model, best_head, best_records, best_logit_scale = train_cutmix_fs(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader)            
    elif args.method == 'CMO': # CMO
        best_model, best_head, best_records, best_logit_scale = train_CMO(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader)        
    elif args.method == 'resizemix': # resizemix
        best_model, best_head, best_records, best_logit_scale = train_resizemix(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader)            
    elif args.method == 'saliencymix': # saliencymix
        #----- paper code, use first image saliency for entire batch
        # best_model, best_head, best_records, best_logit_scale = train_saliencymix(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader)        
        #----- my code, use individual image saliency for each image in the batch
        best_model, best_head, best_records, best_logit_scale = train_saliencymix2(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader)                    
    elif args.method == 'attentivemix': # attentivemix
        # irregular binary mask
        # best_model, best_head, best_records, best_logit_scale = train_attentivemix(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader)                
        
        # rectangular patches as SaliencyMix2
        best_model, best_head, best_records, best_logit_scale = train_attentivemix2(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader)

    elif args.method == 'FLYP':
        best_model, best_head, best_records, best_logit_scale = train_flyp(args, logger, loss_logger, model, tokenizer,
                                                                            train_loader, val_loader, test_loader, text_prompts)
    elif args.method == 'SupContrastive':
        best_model, best_head, best_records, best_logit_scale = train_supervised_contrastive(args, logger, loss_logger, model, classifier_head,
                                                                                             logit_scale, loss, optimizer, scheduler,
                                                                                             train_loader, val_loader, test_loader)
    elif args.method == 'BalancedContrastive':
        best_model, best_head, best_records, best_logit_scale = train_balanced_contrastive(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader)
    else:
        raise NotImplementedError(f"Method {args.method} not implemented.")

    if args.method == 'dataset-cls':
        exit()

    #---------- Test the wsft, cannot preextract feature, as the model backbone weights is ensembled 
    # wsft_backbone, wsft_head, wsft_test_acc = run_wsft(args, best_model, best_head, test_loader, zeroshot_model, zeroshot_weights, best_logit_scale, logger)
    wsft_backbone, wsft_head, wsft_test_acc = run_wsft_alpha(args, best_model, best_head, val_loader, test_loader, zeroshot_model, zeroshot_head, best_logit_scale, logger)

    # Here we re-extract the val, test dataloader after training, for fast checking of tau normalization
    new_val_fea_path = f'{args.dataset_root}/pre_extracted/{args.dataset}_{args.model_cfg}_{args.seed}_val_features_new.pth'
    new_test_fea_path = f'{args.dataset_root}/pre_extracted/{args.dataset}_{args.model_cfg}_{args.seed}_test_features_new.pth'
    val_loader = extract_dataloader(args, best_model, args.val_split, new_val_fea_path)
    test_loader = extract_dataloader(args, best_model, args.test_split, new_test_fea_path)
    logger.info(f'Extracted val, test dataloader for fast testing after training.')

    #---------- Testing 
    test_acc, test_loss, test_confusion_matrix = validate(args,data_loader=test_loader, 
                                                        model=best_model, 
                                                        classifier_head=best_head, 
                                                        logger=logger,
                                                        loss=args.loss, logit_scale=best_logit_scale, 
                                                        show_confusion_matrix=True,
                                                        dataset=args.dataset, 
                                                        output_dir=args.output_dir, device=args.device,
                                                        pre_extracted=True,  
                                                        )
    test_scores = calculate_scores(test_confusion_matrix)
    logger.info(f"+++++ Test Acc: {round(test_acc, 3)}")
    save_test_scores(test_scores, test_confusion_matrix, args.output_dir, 'test')
    # save_head_weights(best_head, args.output_dir, 'best_val')

    #---------- Tau normalization
    best_tau_head, best_tau, best_tau_test_acc = run_tau_normalization(args, best_head, best_model, val_loader, test_loader, best_logit_scale, logger)

    # calculate the hardest 30 class average acc
    # hard_avg_acc = cal_hard_avg_acc(test_scores)
    # easy_avg_acc = cal_easy_avg_acc(test_scores)
    # logger.info(f"Hard Avg Acc: {round(hard_avg_acc*100, 3)}")
    # logger.info(f"Easy Avg Acc: {round(easy_avg_acc*100, 3)}")

    # print the logit_scale
    logger.info(f"logit_scale: {round(logit_scale.item(), 8)}")
    logger.info(f"best_logit_scale: {round(best_logit_scale.item(), 8)}")

    #----------- save stage 2 best model
    best_model_path = save_best_model(args, best_records,
                                    best_model, best_head, best_logit_scale,
                                    test_acc, best_tau, best_tau_test_acc, wsft_test_acc,
                                    best_tau_head, wsft_backbone, wsft_head, stage=2)
    logger.info(f'Stage 2 Best Model saved to: {best_model_path}')

    return test_acc, best_model_path, test_loader_copy


def run_stage2_probing(stage1_best_model_path, test_loader):

    #---------- Run stage 2 probing on the best model
    logger.info(f"Run stage 2 Probing ......")

    # load the stage 2 best model
    args.model_path = stage1_best_model_path
    # logger.info(f'Load the stage 2 best model from: {args.model_path}')
    load_model(args, logger, model, test_loader, classifier_head)  

    # re-extract the train_loader, val_loader, test_loader
    new_fewshot_fea_path = f'{args.dataset_root}/pre_extracted/{args.dataset}_{args.model_cfg}_{args.seed}_fewshot_features_new.pth'
    new_test_fea_path = f'{args.dataset_root}/pre_extracted/{args.dataset}_{args.model_cfg}_{args.seed}_test_features_new.pth'
    
    train_loader = extract_train_dataloader(args, logger, model, args.fewshot_split, new_fewshot_fea_path, args.bsz)
    val_loader = train_loader 
    test_loader = extract_dataloader(args, model, args.test_split, new_test_fea_path)
    text_dataloader = None
    logger.info(f'Extracted train, val, test dataloader for stage 2 training.')
    
    # reset the pre_extracted flag
    args.method = 'probing'
    args.pre_extracted = True
    logger.info(f'Reset args.pre_extracted: {args.pre_extracted}')
    args.epochs = 10
    args.early_stop = False

    # Imporatnt! Need to reset the params, optimizer, scheduler, loss, logit_scale
    loss = set_loss(args)
    params, logit_scale = set_params(args, model, classifier_head, logger) # depend on method
    optimizer, scheduler, total_iter = set_optimizer(args, params, train_loader)
    
    args.loss = loss
    args.logit_scale = logit_scale
    args.optimizer = optimizer
    args.scheduler = scheduler      

    #---------- Training
    best_model, best_head, best_records, _, _, _ = train_probing(args, logger, loss_logger, model, classifier_head, 
                                                                 train_loader, val_loader, test_loader, 
                                                                 reload_model=False, text_dataloader=text_dataloader)

    # test the best model after probing
    test_acc, test_loss, test_confusion_matrix = validate(args,data_loader=test_loader, 
                                                        model=best_model, 
                                                        classifier_head=best_head, 
                                                        logger=logger,
                                                        loss=args.loss, logit_scale=args.logit_scale, 
                                                        show_confusion_matrix=True,
                                                        dataset=args.dataset, 
                                                        output_dir=args.output_dir, device=args.device,
                                                        pre_extracted=True,  
                                                        )
    test_scores = calculate_scores(test_confusion_matrix)
    logger.info(f"+++++ stage 2 Test Acc: {round(test_acc, 3)}")
    save_test_scores(test_scores, test_confusion_matrix, args.output_dir, 'test', stage=3)        

    #----------- save stage 2 best model
    best_model_path = save_best_model(args, best_records,
                                    best_model, best_head, logit_scale,
                                    test_acc, best_tau=None, best_tau_test_acc=-1, wsft_test_acc=-1,
                                    best_tau_head=None, wsft_backbone=None, wsft_head=None, stage=3)
    
    logger.info(f'stage 2 Best Model saved to: {best_model_path}')

    return test_acc, best_model_path



if __name__ == '__main__':

    program_start = time.time()
    args = parse_args()
    logger, loss_logger = set_logger(args)

    # set the random seed
    random.seed(args.training_seed)
    torch.manual_seed(args.training_seed)
    np.random.seed(args.training_seed)  

    # load model
    model, preprocess, tokenizer = set_model(args, logger)
    zeroshot_model = copy.deepcopy(model)

    # make prompts 
    prompt_tensors, text_prompts, tokenized_text_prompts, prompt_tensors_dict = set_prompt(args, model, tokenizer, logger)

    # make classifier head
    classifier_head = set_classifier(args, prompt_tensors, logger)
    zeroshot_head = copy.deepcopy(classifier_head)
    classifier_head.to(args.device)

    # dataset classification head
    # if args.method == 'finetune-multitask':
    #     dataset_classifier_head = MyLinear(inp_dim=512, num_classes=2, bias=False)
    #     dataset_classifier_head.to(args.device)
    # else:
    #     dataset_classifier_head = None    
    

    # run finetuning for stage 1
    stage1_acc, stage1_best_model_path, test_loader = run_stage1_finetuning()
    stage1_method = args.method # record here, as in stage 2 method will be updated to probing

    # run probing for stage 2
    if not args.skip_stage2:
        stage2_acc, stage2_best_model_path = run_stage2_probing(stage1_best_model_path, test_loader)
    else:
        logger.info(f"Skip stage 2 Probing.")
        stage2_acc = -1
        stage2_best_model_path = 'None'

    loss_logger.close()
    program_end = time.time()
    logger.info(f"Total time: {round(program_end-program_start, 1)} s.")

    result_summary = f'{args.dataset},{stage1_method},{args.data_source},{args.cls_init},{args.shots},{args.seed},{args.retrieval_split},{round(stage1_acc,1)},{round(stage2_acc,1)}'
    logger.info(f'{result_summary}')
    print(f'{result_summary}')