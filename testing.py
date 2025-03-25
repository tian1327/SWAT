import os
import torch
import time
import argparse
from utils.logger import get_logger
from utils.extras import get_engine
from torch.utils.data import DataLoader
from utils.models import MyLinear
from utils import features
from torchmetrics import ConfusionMatrix
# from torchmetrics import Accuracy
import numpy as np
import pickle
from time import time
import random
from utils.datasets.dataset_utils import NUM_CLASSES_DICT, load_dataset, TensorDataset
from utils.prompt import prompt_maker
from utils.features import extract_test_feats
from utils.datasets.imagenet_1k import ImageNet1KDataset, ImageNetAdvDataset, ImageNetRenDataset, ImageNetSketchDataset, indices_in_1k_adv, indices_in_1k_ren
from tqdm import tqdm

def test_imagenet_ood(args, model, classifier_head, preprocess, test_loader, reload_model=False):

    if reload_model:
        load_model(args, args.logger, model, test_loader, classifier_head)
        # load_model(args, args.logger, model, None, classifier_head)
    logger = args.logger

    BATCH_SIZE = 512

    acc_list = []
    for dataset in [
        'imagenet_v2',
        'imagenet_sketch',
        'imagenet_adv',
        'imagenet_ren',
        ]:
        logger.info(f'Testing on: {dataset}')

        if dataset == 'imagenet_v2':
            val_dataset = ImageNet1KDataset(transform=preprocess, dataset_root=f'{args.dataset_path}/imagenet_v2')
            val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=args.num_workers)
            logger.info(f'len(val_dataloader): {len(val_dataloader)}')
            acc = validate_simple(args, val_dataloader, model, classifier_head)

        elif dataset == 'imagenet_sketch':
            val_dataset = ImageNetSketchDataset(transform=preprocess, dataset_root=f'{args.dataset_path}/imagenet_sketch/sketch')
            val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=args.num_workers)
            logger.info(f'len(val_dataloader): {len(val_dataloader)}')
            acc = validate_simple(args, val_dataloader, model, classifier_head)

        elif dataset == 'imagenet_adv':
            val_dataset = ImageNetAdvDataset(transform=preprocess, dataset_root=f'{args.dataset_path}/imagenet_adv/imagenet-a')
            val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=args.num_workers)
            logger.info(f'len(val_dataloader): {len(val_dataloader)}')
            acc = validate_simple(args, val_dataloader, model, classifier_head, indices_in_1k_adv)

        elif dataset == 'imagenet_ren':
            val_dataset = ImageNetRenDataset(transform=preprocess, dataset_root=f'{args.dataset_path}/imagenet_ren/imagenet-r')
            val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=args.num_workers)
            logger.info(f'len(val_dataloader): {len(val_dataloader)}')
            acc = validate_simple(args, val_dataloader, model, classifier_head, indices_in_1k_ren)

        else:
            raise ValueError('Unknown dataset.')

        acc_list.append(acc)
        args.logger.info(f'{dataset}, Test Acc: {round(acc, 3)}')

    # average the acc
    avg_acc = np.mean(acc_list)
    args.logger.info(f'Average OOD Test Acc: {round(avg_acc, 3)}')



def load_model(args, logger, model, test_loader=None, classifier_head=None):

    logger.info(f'Loading model from: {args.model_path}')
    ckpt = torch.load(args.model_path)


    # model.load_state_dict(ckpt['wsft_backbone'])
    # classifier_head.load_state_dict(ckpt['wsft_head'])

    if 'clip' in ckpt:

        #----- load normal model
        model.load_state_dict(ckpt['clip'])
        classifier_head.load_state_dict(ckpt['head'])

        #----- load WSFT ensembled model
        # model.load_state_dict(ckpt['wsft_backbone'])
        # classifier_head.load_state_dict(ckpt['wsft_head'])

        logger.info(f'ckpt[test_acc]: {ckpt["test_acc"]}')
        logger.info(f'ckpt[wsft_test_acc]: {ckpt["wsft_test_acc"]}')

        #----- load tau-normalized head
        # classifier_head.load_state_dict(ckpt['best_tau_head'])
        # classifier_head.load_state_dict(ckpt['wsft_head'])

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


    if test_loader is not None:
        model_test_acc, _, _ = validate(args, data_loader=test_loader, model=model,
                                        logger=logger,
                                        loss=args.loss, logit_scale=args.logit_scale,
                                        classifier_head=classifier_head,
                                        dataset=args.dataset,
                                        device=args.device,
                                        pre_extracted=args.pre_extracted,
                                        )
        logger.info(f"Loaded Model Test Acc: {round(model_test_acc, 3)}")


def calculate_scores(confusion_matrix):

    # the diagonal of the confusion matrix is the number of correct predictions for each class
    # along the rows, we have the predicted labels, along the columns, we have the ground truth labels

    # the sum of each row of the confusion matrix is the total number of instances for each true class
    # divide the diagonal by the sum of each row is the same as TP / (TP + FN), which is the recall

    scores = {}
    num_class = confusion_matrix.shape[0]

    scores['acc'] = np.trace(confusion_matrix) / np.sum(confusion_matrix)
    # print('sum of confusion_matrix: ', np.sum(confusion_matrix))

    # calculate the avg class accuracy
    class_accuracy = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)
    avg_class_accuracy = class_accuracy.mean()*100
    scores['avg_class_accuracy'] = avg_class_accuracy # this is the micro accuracy, which would be different from the macro accuracy as in test_acc
    # print('avg_class_accuracy: ', avg_class_accuracy)

    # calculate the per-class recall, precision and f1 score
    recall = dict()
    precision = dict()
    f1_score = dict()

    for i in range(num_class):
        tp = confusion_matrix[i, i]
        fn = np.sum(confusion_matrix[i, :]) - tp
        fp = np.sum(confusion_matrix[:, i]) - tp
        # print('tp, fn, fp: ', tp, fn, fp)

        if tp+fn == 0:
            recall[i] = 0.0
        else:
            recall[i] = tp / (tp + fn)

        if tp+fp == 0:
            precision[i] = 0.0
        else:
            precision[i] = tp / (tp + fp)

        if tp == 0:
            f1_score[i] = 0
        else:
            f1_score[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])

        # print('recall[i], precision[i], f1_score[i]: ', recall[i], precision[i], f1_score[i])

    scores['per_class_recall'] = recall
    scores['per_class_precision'] = precision
    scores['per_class_f1score'] = f1_score

    return scores


def validate_multitask(args, data_loader, model, logger, loss, logit_scale, classifier_head=None,
             dataset_classifier_head=None, show_confusion_matrix = False, device='cuda',
             dataset='semi-aves', output_dir='output',
             predict_labels=False, predict_split='u_train', pre_extracted=False):

    model.eval()
    if classifier_head:
        classifier_head.eval()
    if dataset_classifier_head:
        dataset_classifier_head.eval()

    val_acc = 0
    val_count = 0

    dataset_val_acc = 0
    dataset_val_count = 0

    if show_confusion_matrix:
        num_classes = classifier_head.num_classes
        confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=num_classes)
        # along the rows, we have the predicted labels, along the columns, we have the ground truth labels

        dataset_confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=2)

    with torch.no_grad():
        predicted_labels = []
        dataset_predicted_labels = []
        max_logits = []
        val_loss_batch = []
        for i, val_data in enumerate(data_loader):
            inputs, labels, texts, source = val_data

            if not pre_extracted:
                images = inputs.to(device)
                labels = labels.long()
                image_features = model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
            else:
                image_features = inputs.to(device)
                labels = labels.cpu().long()

            if classifier_head:
                logit = classifier_head(image_features)
                dataset_logit = dataset_classifier_head(image_features)
            else:
                logit, _ = model(images, texts)
                # similarity between text and image, this is wrong?

            max_logits.append(torch.max(logit, dim=1).values.cpu().numpy())

            pred = torch.argmax(logit, dim=1).cpu()
            predicted_labels.append(pred.numpy())

            val_acc += torch.sum(pred == labels).item()
            val_count += labels.size(0)

            # dataset classifier
            dataset_pred = torch.argmax(dataset_logit, dim=1).cpu()
            dataset_predicted_labels.append(dataset_pred.numpy())

            dataset_val_acc += torch.sum(dataset_pred == source).item()
            dataset_val_count += source.size(0)

            if show_confusion_matrix:
                confusion_matrix.update(pred, labels)
                dataset_confusion_matrix.update(dataset_pred, source)

            # val loss
            logits = logit * logit_scale.exp()
            logits = logits.cpu()
            if args.loss == "WeightedCE":
                loss_batch = loss(logits, labels, source)
            else:
                loss_batch = loss(logits, labels)
            val_loss_batch.append(loss_batch.item())

    # average class validation accuracy
    val_acc = (val_acc/val_count)*100
    dataset_val_acc = (dataset_val_acc/dataset_val_count)*100

    # average validation loss
    val_loss = np.mean(val_loss_batch)


    if show_confusion_matrix:
        confusion_matrix = confusion_matrix.compute().numpy()
        dataset_confusion_matrix = dataset_confusion_matrix.compute().numpy()
        return val_acc, val_loss, confusion_matrix, dataset_val_acc, dataset_confusion_matrix

    return val_acc, val_loss, None, dataset_val_acc, None

def validate_dataset(args, data_loader, model, logger, loss, logit_scale, classifier_head=None,
             show_confusion_matrix = False, device='cuda',
             dataset='semi-aves', output_dir='output',
             predict_labels=False, predict_split='u_train', pre_extracted=False):

    model.eval()
    if classifier_head:
        classifier_head.eval()

    val_acc = 0
    val_count = 0

    if show_confusion_matrix:
        num_classes = classifier_head.num_classes
        confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=num_classes)
        # along the rows, we have the predicted labels, along the columns, we have the ground truth labels

    with torch.no_grad():
        predicted_labels = []
        max_logits = []
        val_loss_batch = []
        for i, val_data in enumerate(data_loader):
            inputs, labels, texts, source = val_data
            inputs = inputs.to(device)
            # labels = labels.long().cuda()
            labels = source.long().cuda() # use the source as the labels for dataset classification

            if not pre_extracted:
                image_features = model.encode_image(inputs)
                image_features /= image_features.norm(dim=-1, keepdim=True)
            else:
                image_features = inputs

            if classifier_head:
                logit = classifier_head(image_features)
            else:
                logit, _ = model(inputs, texts)

            # val loss
            logits = logit * logit_scale.exp()
            logits = logits.cuda()
            if args.loss_name == "WeightedCE":
                loss_batch = loss(logits, labels, source)
            else:
                loss_batch = loss(logits, labels)
            val_loss_batch.append(loss_batch.item())

            labels = labels.cpu()
            max_logits.append(torch.max(logit, dim=1).values.cpu().numpy())
            pred = torch.argmax(logit, dim=1).cpu()
            predicted_labels.append(pred.numpy())

            val_acc += torch.sum(pred == labels).item()
            val_count += labels.size(0)

            if show_confusion_matrix:
                confusion_matrix.update(pred, labels)

    # average class validation accuracy
    val_acc = (val_acc/val_count)*100

    # average validation loss
    val_loss = np.mean(val_loss_batch)

    if predict_labels:
        predicted_labels = np.concatenate(predicted_labels)
        print('predict_labels.shape: ', predicted_labels.shape)
        predicted_labels = predicted_labels.tolist()

        max_logits = np.concatenate(max_logits)
        print('max_logits.shape: ', max_logits.shape)
        max_logits = max_logits.tolist()

        # save the predicted labels to a text file
        predicted_label_file = f'{output_dir}/{dataset}_{predict_split}_predicted_labels.txt'
        with open(predicted_label_file, 'w') as f:
            for item, logit in zip(predicted_labels, max_logits):
                f.write("%s %s\n" % (item, logit))
        logger.info(f'Predicted labels saved to: {predicted_label_file}')

    if show_confusion_matrix:
        confusion_matrix = confusion_matrix.compute().numpy()
        return val_acc, val_loss, confusion_matrix

    return val_acc, val_loss, None



def validate_simple(args, data_loader, model, classifier_head, indices_in_1k=None):

    model.eval()
    classifier_head.eval()

    val_acc = 0
    val_count = 0

    with torch.no_grad():
        # for i, val_data in tqdm(enumerate(data_loader)):
        for i, val_data in enumerate(data_loader):

            inputs, labels = val_data
            inputs = inputs.to(args.device)
            # labels = labels.long().cuda()

            image_features = model.encode_image(inputs)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logit = classifier_head(image_features)

            if indices_in_1k:
                logit = logit[:, indices_in_1k]

            # labels = labels.cpu()
            pred = torch.argmax(logit, dim=1).cpu()
            val_acc += torch.sum(pred == labels).item()
            val_count += labels.size(0)

    # average class validation accuracy
    val_acc = (val_acc/val_count)*100

    return val_acc



def validate(args, data_loader, model, logger, loss, logit_scale, classifier_head = None,
             show_confusion_matrix = False, device='cuda',
             dataset='semi-aves', output_dir='output',
             predict_labels=False, predict_split='u_train', pre_extracted=False):

    model.eval()
    if classifier_head:
        classifier_head.eval()

    val_acc = 0
    val_count = 0

    if show_confusion_matrix:
        num_classes = classifier_head.num_classes
        confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=num_classes)
        # along the rows, we have the predicted labels, along the columns, we have the ground truth labels

    with torch.no_grad():
        predicted_labels = []
        max_logits = []
        val_loss_batch = []
        for i, val_data in enumerate(data_loader):
            inputs, labels, texts, source = val_data
            inputs = inputs.to(device)
            labels = labels.long().cuda()

            if not pre_extracted:
                image_features = model.encode_image(inputs)
                image_features /= image_features.norm(dim=-1, keepdim=True)
            else:
                image_features = inputs

            if classifier_head:
                logit = classifier_head(image_features)
            else:
                logit, _ = model(inputs, texts)

            # val loss
            logits = logit * logit_scale.exp()
            logits = logits.cuda()
            if args.loss_name == "WeightedCE":
                loss_batch = loss(logits, labels, source)
            else:
                loss_batch = loss(logits, labels)
            val_loss_batch.append(loss_batch.item())

            labels = labels.cpu()
            max_logits.append(torch.max(logit, dim=1).values.cpu().numpy())
            pred = torch.argmax(logit, dim=1).cpu()
            predicted_labels.append(pred.numpy())

            val_acc += torch.sum(pred == labels).item()
            val_count += labels.size(0)

            if show_confusion_matrix:
                confusion_matrix.update(pred, labels)

    # average class validation accuracy
    val_acc = (val_acc/val_count)*100

    # average validation loss
    val_loss = np.mean(val_loss_batch)

    if predict_labels:
        predicted_labels = np.concatenate(predicted_labels)
        print('predict_labels.shape: ', predicted_labels.shape)
        predicted_labels = predicted_labels.tolist()

        max_logits = np.concatenate(max_logits)
        print('max_logits.shape: ', max_logits.shape)
        max_logits = max_logits.tolist()

        # save the predicted labels to a text file
        predicted_label_file = f'{output_dir}/{dataset}_{predict_split}_predicted_labels.txt'
        with open(predicted_label_file, 'w') as f:
            for item, logit in zip(predicted_labels, max_logits):
                f.write("%s %s\n" % (item, logit))
        logger.info(f'Predicted labels saved to: {predicted_label_file}')

    if show_confusion_matrix:
        confusion_matrix = confusion_matrix.compute().numpy()
        return val_acc, val_loss, confusion_matrix

    return val_acc, val_loss, None



def validate_topK(data_loader, model, prompt_vectors, logger, device='cuda',
                  dataset='semi-inat-2021', show_confusion_matrix= True, k = 3):

    with torch.no_grad():
        model.eval()
        correct, wrong, val_acc = 0, 0, 0
        val_count = 0

        if show_confusion_matrix:
            num_classes = len(prompt_vectors) # For now.
            confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=num_classes)
        for i, val_data in enumerate(data_loader):
            if dataset == 'semi-inat-2021':
                inputs, labels, l_target_k, l_target_p, l_target_c, l_target_o, l_target_f, l_target_g = val_data
            else:
                inputs, labels = val_data

            images = inputs.to(device)
            labels = labels.to(device).long()
            bsz = labels.shape[0]
            #print(bsz)
            logits = torch.zeros(num_classes, bsz)

            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            start = time.time()
            for j in range(num_classes):
                desc_prompts = prompt_tensors[j]['all']
                k = desc_prompts.shape[0]
                if (desc_prompts.shape[0] > 2 ):
                    k = 3
                desc_prompts = desc_prompts.to(device)
                desc_prompts = desc_prompts.squeeze()
                cosine_sim = image_features @ desc_prompts.t()
                top_k = cosine_sim.topk(k=k, dim=-1).values
                logits[j] = top_k.mean(dim=-1)

            #print(time.time()-start)
            logits = logits.to(device)
            pred = torch.argmax(logits, dim=0)
            val_acc += torch.sum(pred == labels).item()
            val_count += labels.size(0)
            if show_confusion_matrix:
                preds = pred.cpu()
                labels = labels.cpu()
                confusion_matrix.update(preds, labels)

            images.cpu()
        val_acc = (val_acc/val_count)*100

        print(f'Top 1 validation accuracy: {val_acc}')
        logger.info(f'Top 1 validation accuracy: {val_acc}')
        quit()
        if show_confusion_matrix:
            return val_acc, confusion_matrix
    return val_acc


if __name__ == '__main__':

    start = time()
    parser = argparse.ArgumentParser(description='Arguments for script.')
    parser.add_argument('--model_cfg', type=str, default='vitb32_openclip_laion400m',
                        choices=['vitb32_openclip_laion400m', 'vitb32_openclip_laion2b',
                                 'vitb32_clip', 'vitb16_clip'],
                        help='ViT Transformer arch.')
    parser.add_argument('--model_ckpt', type=str, default=None, help='model ckpt for testing.')
    parser.add_argument('--prompt_name', type=str, default='translated-name',
                        choices=['c-name', 's-name', 't-name', 'f-name'], help='names for prompts.')
    parser.add_argument('--recal_prompt', action='store_true', help='Recalculate the prompt embedding or not.')
    parser.add_argument('--log_mode', type=str, default='both', choices=['console', 'file', 'both'], help='where to log.')
    parser.add_argument('--dataset', type=str, default='semi-aves',
                        choices=['semi-inat-2021', 'semi-aves', 'flowers102', 'cub2011', 'imagenet_1k'],
                        help='Dataset name.')
    parser.add_argument('--dataset_root', type=str, default='data/semi-aves/', help='Root of Dataset.')

    parser.add_argument('--pre_extracted', default=True, help='use pre-extracted features.')
    parser.add_argument('--text_classifier_head',default=True, help='Initialize the classifier head with text embedding or not.')
    parser.add_argument('--tau', type=float, default=0, help='Tau for Normalization.')
    parser.add_argument('--seed', type=int, default=0, help='Seed for random number generator.')
    parser.add_argument('--prefix', type=str, default='Testing', help='Prefix for Log file Name.')
    parser.add_argument('--predict', type=str, default=None, help='unlabeled data file to predict.')
    parser.add_argument('--folder', type=str, default='output/', help='Folder for saving output.')

    args = parser.parse_args()

    #---------- init
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # case_name
    case_name = f'{args.prefix}_testing_{args.dataset}_{args.model_cfg}_{args.prompt_name}'

    dataset_root = args.dataset_root
    output_dir = os.path.join(args.folder, f'{case_name}')
    if not os.path.exists(f'{output_dir}'):
        os.makedirs(f'{output_dir}')

    log_path = f'{output_dir}/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    ## setup logger
    logger = get_logger(log_path, case_name, args.log_mode)
    logger.info('logging started')
    logger.info(f'case_name: {case_name}')

    # print args
    for arg in vars(args):
        logger.info(f'{arg} = {getattr(args, arg)}')

    #---------- load model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f'Number of GPUs available: {torch.cuda.device_count()}')

    model, preprocess, tokenizer = get_engine(model_cfg=args.model_cfg, device=device)
    logger.info(f'Loaded model: {args.model_cfg}')

    #---------- make prompt tensors
    metric_fn = f'{dataset_root}/prompts/s-names_prompts.json'
    with open(metric_fn, 'r') as f:
        metrics = json.load(f)

    prompts_dir = os.path.join(dataset_root, 'pre_extracted/')
    if not os.path.exists(prompts_dir):
        os.makedirs(prompts_dir)
        logger.info(f'Created directory: {prompts_dir}')

    prompt_tensors_dict= {}
    text_prompts_dict = {}
    tokenized_text_prompts_dict = {}
    for label_type in ['s-name', 'c-name', 't-name', 'f-name']:
        prompt_tensors_filename = f"{prompts_dir}{args.model_cfg}_{label_type}_prompt_tensors.pth"
        text_prompts_filename = f"{prompts_dir}{args.model_cfg}_{label_type}_text_prompts.pth"
        tokenized_text_prompts_filename = f"{prompts_dir}{args.model_cfg}_{label_type}_tokenized_text_prompts.pth"

        if not args.recal_prompt and os.path.exists(prompt_tensors_filename):
            logger.info(f'Loading prompt tensors from {prompt_tensors_filename}')
            prompt_tensors = torch.load(prompt_tensors_filename)
            prompt_tensors_dict[label_type] = prompt_tensors

            text_prompts = torch.load(text_prompts_filename)
            text_prompts_dict[label_type] = text_prompts

            tokenized_text_prompts = torch.load(tokenized_text_prompts_filename)
            tokenized_text_prompts_dict[label_type] = tokenized_text_prompts

        else:
            logger.info(f'Calculating prompt tensors for {label_type} ...')
            text_prompts = prompt_maker(metrics=metrics, dataset_name=args.dataset, name_type=label_type)
            text_prompts_dict[label_type] = text_prompts
            torch.save(text_prompts, text_prompts_filename)
            logger.info(f'Saved text prompts to {text_prompts_filename}')

            # tokenize the text_prompts first in case of finetune needed
            tokenized_text_prompts = features.get_text_features(model, text_prompts, tokenize = tokenizer, operation='tokenize')
            tokenized_text_prompts_dict[label_type] = tokenized_text_prompts
            torch.save(tokenized_text_prompts, tokenized_text_prompts_filename)
            logger.info(f'Saved tokenized text prompts to {tokenized_text_prompts_filename}')

            prompt_tensors = features.get_text_features(model, text_prompts, tokenizer, 'encode')
            prompt_tensors_dict[label_type] = prompt_tensors
            torch.save(prompt_tensors, prompt_tensors_filename)
            logger.info(f'Saved prompt tensors to {prompt_tensors_filename}')

    prompt_tensors = prompt_tensors_dict[args.prompt_name]
    text_prompts = text_prompts_dict[args.prompt_name]
    tokenized_text_prompts = tokenized_text_prompts_dict[args.prompt_name]

    #---------- pre-extract test features
    pre_extract_test_fea_path = f'{dataset_root}/pre_extracted/{args.dataset}_{args.model_cfg}_test_features.pth'

    if not os.path.exists(pre_extract_test_fea_path):
        logger.info(f'Pre-extracting test features ...')
        test_dataset = load_dataset(dataset=args.dataset, dataset_root=dataset_root,
                                    split='test',
                                    preprocess=preprocess,
                                    tokenized_text_prompts=tokenized_text_prompts,
                                    pl_list=None, return_text=False)
        test_loader = DataLoader(test_dataset, batch_size=128,
                                    shuffle=False, num_workers=8, drop_last=False)

        test_features = extract_test_feats(model, dataloader=test_loader)
        torch.save(test_features, pre_extract_test_fea_path)
        logger.info(f'Saved test features to {pre_extract_test_fea_path}')

    #---------- load classifier head
    if args.text_classifier_head: # initialize the classifier head with text embedding
        weights = features.prompt_sampler(prompt_tensors, sample_by='mean')
        print('weights.shape: ', weights.shape)
        classifier_head = MyLinear(weights=weights)
        logger.info('Initialized classifier head with text embedding.')

    else: # random init
        num_class = NUM_CLASSES_DICT[args.dataset]
        logger.info(f'Number of classes: {num_class}')
        classifier_head = MyLinear(inp_dim=512, num_classes=num_class, bias=False)
        logger.info('Initialized classifier head with random weights.')

    classifier_head.to(device)

    #---------- load model checkpoint
    if args.model_ckpt:
        model_ckpt = f'{args.model_ckpt}'
        ckpt = torch.load(model_ckpt)
        print('Load model ckpt from: ', model_ckpt)
        print('ckpt[epoch]: ', ckpt['epoch'])
        print('ckpt[iter]:', ckpt['iter'])
        print('ckpt[best_acc]: ', ckpt['best_acc'])

        model.load_state_dict(ckpt['clip'])
        logger.info('Loaded CLIP from checkpoint.')

        classifier_head.load_state_dict(ckpt['head'])
        classifier_head.to(device)
        logger.info('Loaded classifier head from checkpoint.')

    else:
        logger.info('Use default model weights.')

    #---------- load data
    if not args.pre_extracted:
        val_dataset = load_dataset(args.dataset, dataset_root, 'test', preprocess, tokenized_text_prompts)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=8, drop_last=False)
    else:
        val_dataset = TensorDataset(pre_extracted_path=pre_extract_test_fea_path, device=device)
        logger.info(f'Loaded pre-extracted test features from: {pre_extract_test_fea_path}')
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, drop_last=False, num_workers=0)

    #---------- tau normalization
    if args.tau != 0:
        classifier_head.linear.weight.data /= torch.pow(classifier_head.linear.weight.data.norm(dim=-1, keepdim=True), args.tau)
        logger.info('TAU normalization: tau =', args.tau)
    else:
        logger.info('No TAU normalization.')

    #---------- testing
    val_acc, confusion_matrix = validate(data_loader=val_loader, model=model, classifier_head=classifier_head,
                                         logger=logger, show_confusion_matrix = True, dataset=args.dataset,
                                         pre_extracted=args.pre_extracted,
                                         )

    scores = calculate_scores(confusion_matrix)
    logger.info(f"Validation Accuracy: {round(val_acc, 1)}")

    file_path = f'{output_dir}/{args.prefix}_{args.dataset}_testing_scores.json'
    with open(file_path, 'w') as f:
        json.dump(scores, f, indent=4)
    logger.info(f'Saved scores to: {file_path}')

    file_path = f'{output_dir}/{args.prefix}_{args.dataset}_testing_confusion_matrix.pkl'
    with open(file_path, 'wb') as f:
        pickle.dump(confusion_matrix, f)
    logger.info(f'Saved confusion matrix to: {file_path}')


    # predict labels for unlabeled data
    if args.predict:
        u_train_dataset = load_unlabeled_dataset(args.dataset, dataset_root, preprocess, tokenized_text_prompts)

        logger.info('Predicting labels for unlabeled data: ', args.predict)
        u_train_dataloader = DataLoader(u_train_dataset, batch_size=128,
                                        shuffle=False, num_workers=8, drop_last=False)

        u_train_acc, confusion_matrix = validate(data_loader=u_train_dataloader, model=model, logger=logger,
                                                classifier_head=classifier_head,
                                                show_confusion_matrix=False, # set to false due to -1 is the fake label
                                                dataset=args.dataset,
                                                predict_labels=True,
                                                predict_split=args.predict
                                                )

    logger.info(f'Total time taken: {round(time()-start)} seconds.')