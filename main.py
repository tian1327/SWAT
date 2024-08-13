import os
import torch
from utils.models import set_model, set_classifier, MyLinear, save_test_scores, save_best_model#, save_head_weights
import time
import numpy as np
from utils.parser import parse_args
from utils.logger import set_logger
from testing import validate, load_model
from testing import calculate_scores
from utils.datasets.dataset_utils import NUM_CLASSES_DICT
from utils.prompt import set_prompt
import copy
from utils.losses import set_loss
import torch.nn.functional as F
import cv2
from utils.training import set_training_seed, train_probing, run_zeroshot, train_CMLP, \
    train_dataset_cls, train_ce, train_cutmix, train_flyp, train_ce_mixed, train_fixmatch, \
    train_ce_multitask, train_mixup, train_mixup_fs, train_cutmix_fs, train_resizemix, \
    train_saliencymix2, train_attentivemix2, train_CMO, train_supervised_contrastive, train_balanced_contrastive
from utils.dataloader import extract_train_dataloader, extract_dataloader, set_dataloaders, set_text_dataloader
from utils.optimizers import set_optimizer, set_params
# from gem import create_gem_model
# import pickle


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

        logger.info(f"+++++ Best Tau: {round(best_tau,1)}, Val Acc: {round(best_tau_val_acc, 3)}, Test Acc: {round(best_tau_test_acc, 3)}")
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


def run_stage1_finetuning(args, logger, model, preprocess, tokenized_text_prompts):

    # dataloaders
    train_loader, val_loader, test_loader = set_dataloaders(args, model, tokenized_text_prompts, preprocess, logger)
    text_dataloader = set_text_dataloader(args, logger, prompt_tensors, prompt_tensors_dict) if args.method == 'CMLP' else None
    test_loader_copy = copy.deepcopy(test_loader)

    loss = set_loss(args)
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
        best_model, best_head, best_records, \
            best_logit_scale, val_loader, test_loader = train_probing(args, logger, loss_logger, model, classifier_head, \
                                                                      train_loader, val_loader, test_loader, reload_model)
    
    elif args.method == 'dataset-cls':
        best_model, best_head, best_records, best_logit_scale = train_dataset_cls(args, logger, loss_logger, model, classifier_head, \
                                                                                  train_loader, val_loader, test_loader)    
    
    elif args.method == 'CMLP': # cross modal linear probing         
        best_model, best_head, best_records, \
            best_logit_scale, val_loader, test_loader = train_CMLP(args, logger, loss_logger, model, classifier_head, \
                                                                   preprocess, tokenized_text_prompts, \
                                                                   train_loader, val_loader, test_loader, False, text_dataloader)
    
    elif args.method == 'finetune':
        best_model, best_head, \
            best_records, best_logit_scale = train_ce(args, logger, loss_logger, model, classifier_head, \
                                                      train_loader, val_loader, test_loader)    
    
    elif args.method == 'finetune-mixed': # half batch is retrieved, half batch is fewshot
        best_model, best_head, \
            best_records, best_logit_scale = train_ce_mixed(args, logger, loss_logger, model, classifier_head, \
                                                            train_loader, val_loader, test_loader)

    elif args.method == 'fixmatch': # bs is labeled, bs*mu is unlabeled
        best_model, best_head, \
            best_records, best_logit_scale = train_fixmatch(args, logger, loss_logger, model, classifier_head, \
                                                            train_loader, val_loader, test_loader)       

    elif args.method == 'finetune-multitask': # 1 backbone 2 output heads

        best_model, best_head, \
            best_records, best_logit_scale = train_ce_multitask(args, logger, loss_logger, model, classifier_head, \
                                                                train_loader, val_loader, test_loader, dataset_classifier_head)          
    
    elif args.method == 'mixup': # random mixup
        best_model, best_head, \
            best_records, best_logit_scale = train_mixup(args, logger, loss_logger, model, classifier_head, \
                                                         train_loader, val_loader, test_loader)
    
    elif args.method == 'mixup-fs': # mix retrieved with few-shot
        best_model, best_head, \
            best_records, best_logit_scale = train_mixup_fs(args, logger, loss_logger, model, classifier_head, \
                                                             train_loader, val_loader, test_loader)        
    
    elif args.method == 'cutmix': # cutmix
        best_model, best_head, \
            best_records, best_logit_scale = train_cutmix(args, logger, loss_logger, model, classifier_head, \
                                                          train_loader, val_loader, test_loader)  
    
    elif args.method == 'cutmix-fs': # cutmix with few-shot data
        best_model, best_head, \
            best_records, best_logit_scale = train_cutmix_fs(args, logger, loss_logger, model, classifier_head, \
                                                             train_loader, val_loader, test_loader)            
    
    elif args.method == 'CMO': # CMO
        best_model, best_head, \
            best_records, best_logit_scale = train_CMO(args, logger, loss_logger, model, classifier_head, \
                                                       train_loader, val_loader, test_loader)        
    
    elif args.method == 'resizemix': # resizemix
        best_model, best_head, \
            best_records, best_logit_scale = train_resizemix(args, logger, loss_logger, model, classifier_head, \
                                                             train_loader, val_loader, test_loader)            
    
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
    wsft_backbone, wsft_head, wsft_test_acc = run_wsft_alpha(args, best_model, best_head, val_loader, \
                                                             test_loader, zeroshot_model, zeroshot_head, \
                                                            best_logit_scale, logger)

    # Here we re-extract the val, test dataloader after training, for fast checking of tau normalization
    new_val_fea_path = f'{args.dataset_root}/pre_extracted/{args.dataset}_{args.model_cfg}_{args.seed}_val_features_new.pth'
    new_test_fea_path = f'{args.dataset_root}/pre_extracted/{args.dataset}_{args.model_cfg}_{args.seed}_test_features_new.pth'
    val_loader = extract_dataloader(args, best_model, args.val_split, new_val_fea_path, preprocess, tokenized_text_prompts)
    test_loader = extract_dataloader(args, best_model, args.test_split, new_test_fea_path, preprocess, tokenized_text_prompts)
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
    best_tau_head, best_tau, best_tau_test_acc = run_tau_normalization(args, best_head, best_model, val_loader, \
                                                                       test_loader, best_logit_scale, logger)

    # print the logit_scale
    logger.info(f"logit_scale: {round(logit_scale.item(), 8)}")
    logger.info(f"best_logit_scale: {round(best_logit_scale.item(), 8)}")

    #----------- save stage 2 best model
    best_model_path = save_best_model(args, best_records,
                                    best_model, best_head, best_logit_scale,
                                    test_acc, best_tau, best_tau_test_acc, wsft_test_acc,
                                    best_tau_head, wsft_backbone, wsft_head, stage=1)
    logger.info(f'Stage 1 Best Model saved to: {best_model_path}')

    # remove the extracted features
    os.remove(new_val_fea_path)
    os.remove(new_test_fea_path)

    return test_acc, best_model_path, test_loader_copy



def run_stage2_probing(stage1_best_model_path, test_loader):

    logger.info(f"Run stage 2 classifier retraining ......")

    args.model_path = stage1_best_model_path
    load_model(args, logger, model, test_loader, classifier_head)  

    # re-extract the train_loader, val_loader, test_loader
    new_fewshot_fea_path = f'{args.dataset_root}/pre_extracted/{args.dataset}_{args.model_cfg}_{args.seed}_fewshot_features_new.pth'
    new_test_fea_path = f'{args.dataset_root}/pre_extracted/{args.dataset}_{args.model_cfg}_{args.seed}_test_features_new.pth'
    
    train_loader = extract_train_dataloader(args, model, args.fewshot_data, new_fewshot_fea_path, 
                                            preprocess, tokenized_text_prompts, args.bsz)
    val_loader = train_loader 
    test_loader = extract_dataloader(args, model, args.test_split, new_test_fea_path,
                                     preprocess, tokenized_text_prompts)
    logger.info(f'Extracted train, val, test dataloader for stage 2 training.')
    
    # reset the pre_extracted flag
    args.method = 'probing'
    args.pre_extracted = True
    logger.info(f'Reset args.pre_extracted: {args.pre_extracted}')
    args.epochs = 10
    args.early_stop = False

    # Imporatnt! Need to reset the params, optimizer, scheduler, loss, logit_scale
    loss = set_loss(args)
    params, logit_scale = set_params(args, model, classifier_head, logger) # depending on method
    optimizer, scheduler, total_iter = set_optimizer(args, params, train_loader)
    
    args.loss = loss
    args.logit_scale = logit_scale
    args.optimizer = optimizer
    args.scheduler = scheduler      

    #---------- Training
    best_model, best_head, best_records, _, _, _ = train_probing(args, logger, loss_logger, model, classifier_head, 
                                                                 train_loader, val_loader, test_loader, 
                                                                 reload_model=False)

    # test the best model after probing
    test_acc, test_loss, test_confusion_matrix = validate(args,data_loader=test_loader, 
                                                        model=best_model, 
                                                        classifier_head=best_head, 
                                                        logger=logger,
                                                        loss=args.loss, 
                                                        logit_scale=args.logit_scale, 
                                                        show_confusion_matrix=True,
                                                        dataset=args.dataset, 
                                                        output_dir=args.output_dir, 
                                                        device=args.device,
                                                        pre_extracted=True,  
                                                        )
    test_scores = calculate_scores(test_confusion_matrix)
    logger.info(f"+++++ stage 2 Test Acc: {round(test_acc, 3)}")
    save_test_scores(test_scores, test_confusion_matrix, args.output_dir, 'test', stage=2)        

    #----------- save stage 2 best model
    best_model_path = save_best_model(args, best_records,
                                    best_model, best_head, logit_scale,
                                    test_acc, best_tau=None, best_tau_test_acc=-1, wsft_test_acc=-1,
                                    best_tau_head=None, wsft_backbone=None, wsft_head=None, stage=2)
    
    logger.info(f'stage 2 Best Model saved to: {best_model_path}')

    # remove the extracted features
    os.remove(new_fewshot_fea_path)
    os.remove(new_test_fea_path)

    return test_acc, best_model_path



if __name__ == '__main__':

    program_start = time.time()
    args = parse_args()
    logger, loss_logger = set_logger(args)
    set_training_seed(args)

    # load model
    model, preprocess, tokenizer = set_model(args, logger)
    zeroshot_model = copy.deepcopy(model)

    # make prompts 
    prompt_tensors, text_prompts, \
    tokenized_text_prompts, prompt_tensors_dict = set_prompt(args, model, tokenizer, logger)

    # make classifier head
    classifier_head = set_classifier(args, prompt_tensors, logger)
    zeroshot_head = copy.deepcopy(classifier_head)
    classifier_head.to(args.device) 

    # run finetuning for stage 1
    stage1_acc, stage1_best_model_path, test_loader = run_stage1_finetuning(args, logger, model, preprocess, tokenized_text_prompts)
    stage1_method = args.method # record method here, as in stage 2 method will be updated to probing

    # run probing for stage 2
    if not args.skip_stage2:
        stage2_acc, stage2_best_model_path = run_stage2_probing(stage1_best_model_path, test_loader)
    else:
        logger.info(f"Skip stage 2 Probing.")
        stage2_acc = -1
        stage2_best_model_path = 'None'

    loss_logger.close()
    program_end = time.time()
    logger.info(f"Total time: {round((program_end-program_start)/60, 1)} mins.")

    result_summary = f'{args.dataset},{stage1_method},{args.data_source},{args.cls_init},{args.shots},{args.seed},{args.retrieval_split},{round(stage1_acc,1)},{round(stage2_acc,1)}'
    logger.info(f'{result_summary}')
    print(f'{result_summary}')