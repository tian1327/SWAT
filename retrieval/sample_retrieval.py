import sys
sys.path.append("..")  # Adds higher directory to python modules path.
from utils.extras import get_engine
from PIL import Image
import numpy as np
import os
import torch
from utils import features
from utils.logger import get_logger
from utils.models import MyLinear
import json
import pickle
import shutil
import random
import argparse
from collections import defaultdict
from time import time
# from pre_extract_features import pre_extract_split
from utils.prompt import prompt_maker
# from torchvision.transforms import Compose, RandomHorizontalFlip
# import torch.nn.functional as F
# from scipy.stats import entropy
# import math
from extract_mined_feature import ROOT, CAPTION_MAP_DICT, MINED_DATASET_ROOT_DICT


# def filter_images
def clip_filter(model, dataloader, tokenizer,threshold: float, K: int, device = 'cuda', root_dir = ''):
    # 1. apply CLIP in batches.
    # 2. calculate image embeddings.
    # 3. calculate hfl prompt embedding -> filter images based on this ( > 0.3 and K images). 
    model.eval()
    similarity_scores = []
    labels_list = []
    file_names_list = []
    print('Calculating Cosine Similarity')
    start = time.time()

    for i, data in enumerate(dataloader):
        imgs, labels, texts, file_names = data
        imgs = imgs.to(device)
        text_tokens = tokenizer(texts)
        text_tokens = text_tokens.to(device) # HFL prompt. - Ideally a better name, better prompt.
        with torch.no_grad():
            img_feats = model.encode_image(imgs)
            text_feats = model.encode_text(text_tokens)
        img_feats /= img_feats.norm(dim=-1, keepdim=True)
        text_feats /= text_feats.norm(dim=-1, keepdim=True)
        similarity = img_feats @ text_feats.t()
        similarity_diag = torch.diag(similarity)
        similarity_scores = similarity_scores + similarity_diag.cpu().tolist()
        file_names_list = file_names_list + list(file_names)
        labels_list = labels_list + labels.tolist()
        print('Batch:',i, 'time:', time.time()-start)
    similarity_scores = list(zip(similarity_scores, labels_list, file_names_list))
    similarity_grouped = {} # maintain a label grouped score.
    for tup in similarity_scores:
        similarity, label, file_name = tup
        if not tup[1] in similarity_grouped:
            similarity_grouped[label] = []
        similarity_grouped[label].append((similarity, file_name))
    for key in similarity_grouped.keys():
        similarity_grouped[key] = sorted(similarity_grouped[key], key=lambda item: item[0])
    filter_and_remove(similarity_grouped, threshold=threshold, K=K, root_dir=root_dir)
    # sorted_similarity_scores = sorted(similarity_scores, key=lambda x:(x[1], x[0]))

def clip_zero_shot(model, dataloader, head, device='cuda', root_dir='', show_confusion_matrix=False):
    model.eval()
    start = time.time()
    correct_preds = []
    file_names_list =[]
    labels_list = []

    if show_confusion_matrix:
        num_classes = head.num_classes
        confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=num_classes)

    for i, data in enumerate(dataloader):
        imgs, labels, _, file_names = data
        imgs = imgs.to(device)
        labels = labels.to(device).long()
        with torch.no_grad():
            img_feats = model.encode_image(imgs)
            img_feats_norm = img_feats / img_feats.norm(dim=-1, keepdim=True)
            logits = head(img_feats_norm)
        pred = torch.argmax(logits, dim=1).long()
        true_preds = pred == labels
        correct_preds = correct_preds + true_preds.cpu().tolist()
        file_names_list = file_names_list + list(file_names)
        labels_list = labels_list + labels.cpu().tolist()
        
        if show_confusion_matrix:
            preds = pred.cpu()
            labels = labels.cpu()
            confusion_matrix.update(preds, labels)
            
        print('Batch:',i, 'time:', time.time()-start)
    zero_shot_filtered = list(zip(correct_preds, labels_list, file_names_list))
    zero_shot_grouped = {} # maintain a label grouped score.
    for tup in zero_shot_filtered:
        correct_pred, label, file_name = tup
        if not label in zero_shot_grouped:
            zero_shot_grouped[label] = []

        zero_shot_grouped[label].append((correct_pred, file_name))

    filtered_meta_data = {}
    file_output = []
    for item in zero_shot_grouped.items():
        class_idx, samples = item
        if not class_idx in filtered_meta_data:
            filtered_meta_data[class_idx] = []
        for (correct_pred, file_path) in samples:
            if correct_pred: filtered_meta_data[class_idx].append((correct_pred, file_path))
        file_paths = [file_path for (_,file_path) in filtered_meta_data[class_idx]]
        folder_dir = os.path.join(root_dir, str(class_idx))
        file_list = os.listdir(folder_dir)
        ct = 0
        for file in file_list:
            file_path = os.path.join(folder_dir, file)
            if not file_path in file_paths:
                ct+=1
                os.remove(file_path)
        print(f'{class_idx}-{len(file_paths)}-{ct}')
        if len(file_paths) < ct and len(file_paths) < 100:
            file_output.append(f'{class_idx}-{len(file_paths)}-{ct}')
    with open('./problematic.txt', 'w') as f:
        f.write('\n'.join(file_output))
    
    if show_confusion_matrix:
        print('Saving confusion matrix.')
        confusion_matrix = confusion_matrix.compute().numpy()
        with open('./retrieved_confusion_matrix.pkl', 'wb') as f:
            pickle.dump(confusion_matrix, f)


def filter_and_remove(similarity_grouped: dict, threshold = 0.3, soft_thresold = 0.25, K=4, root_dir = ''):
    filtered_meta_data = {}
    file_output = []
    for item in similarity_grouped.items():
        class_idx, samples = item
        if not class_idx in filtered_meta_data:
            filtered_meta_data[class_idx] = []
        for (similarity, file_path) in samples:
            if similarity >= threshold and len(filtered_meta_data[class_idx]) < K:
                filtered_meta_data[class_idx].append((similarity, file_path))
        file_paths = [file_path for (_,file_path) in filtered_meta_data[class_idx]]
        folder_dir = os.path.join(root_dir, str(class_idx))
        file_list = os.listdir(folder_dir)
        for file in file_list:
            file_path = os.path.join(folder_dir, file)
            if not file_path in file_paths:
                os.remove(file_path)
                
    ct = 0
    for key in filtered_meta_data.keys():
        if len(filtered_meta_data[key]) < K:
            ct+=1
            file_output.append(f'{key} - {filtered_meta_data[key]}')
        if len(filtered_meta_data[key]) == 0:
            print(key, len(filtered_meta_data[key]))
    with open('./multi_stage.txt', 'w') as f:
        f.write('\n'.join(file_output))
    print(ct)


def clip_i2t_similarity(model, head, preprocess, root_folder, duplicates_dict=None):
    classes = os.listdir(root_folder)
    model = model.cuda()
    i2t_rankings = {}
    start = time.time()
    for cls in classes:
        if cls not in i2t_rankings:
            i2t_rankings[cls] = []
        duplicates = duplicates_dict[cls] # Handles both de-duplication and not considering duplication.
        class_prompt = head.linear.weight[int(cls)]
        source_folder = os.path.join(root_folder, cls)
        file_list = [os.path.join(source_folder, file) for file in os.listdir(source_folder)]
        img_tensors = [preprocess(Image.open(file)).unsqueeze(0) for file in file_list]
        img_tensors = torch.cat(img_tensors, dim=0).cuda()
        with torch.no_grad():
            img_embeddings = model.encode_image(img_tensors)
        img_embeddings /= img_embeddings.norm(dim=-1, keepdim=True)
        similarity = img_embeddings @ class_prompt.t()
        similarity = similarity.cpu().tolist()
        combined = sorted(list(zip(file_list, similarity)), key=lambda x: x[1], reverse=True)

        
        for i, (file_path, similarity) in enumerate(combined):
            # base_name = os.path.basename(file_path)
            if i==200:
                break
            if (len(combined) < 200 and similarity > 0.25 or len(combined) >=200) and file_path not in duplicates:
                i2t_rankings[cls].append(file_path)
                # shutil.copy(file_path, os.path.join(destinition_folder, base_name))
        print('Done for:', cls, time.time()-start)
    with open('./i2t_rankings_deduped.json', 'w') as f:
        f.write(json.dumps(i2t_rankings, indent=4))


def remove_near_duplicates(model, preprocess, root_folder, do_deduplication=True):

    classes = os.listdir(root_folder)
    model = model.cuda()
    # Store duplicates for reference later.
    start = time.time()
    duplicate_images_dict = defaultdict(set)
    if not do_deduplication:
        return duplicate_images_dict

    for cls in classes:
        source_folder = os.path.join(root_folder, cls)
        file_list = sorted(os.listdir(source_folder), key= lambda x: int(x.split('.')))
        img_tensors = [preprocess(Image.open(os.path.join(source_folder, file))).unsqueeze(0) for file in file_list]
        img_tensors = torch.cat(img_tensors, dim=0).cuda()
        with torch.no_grad():
            img_embeddings = model.encode_image(img_tensors) # not doing this, used the preextracted image embeddings.

        img_embeddings /= img_embeddings.norm(dim=-1, keepdim=True)

        similarity = img_embeddings @ img_embeddings.t()
        similarity = similarity.cpu().numpy()
        upper_triangle = np.triu(similarity, k=1)
        i_indices, j_indices = np.where(upper_triangle > 0.9)
        to_remove = set(j_indices)
        unique_images = [i for i in range(len(file_list)) if i not in to_remove]

        for file in file_list:
            file_id = file.split('.')[0]
            if int(file_id) not in unique_images:
                duplicate_images_dict[cls].add(os.path.join(source_folder, file))
        print(f'For {cls} - fraction of unqiue images: {len(unique_images) / len(file_list)} - time: {time.time() - start}')

    return duplicate_images_dict


def remove_near_duplicates2(pre_extracted_feats):

    print('\nRemoving near duplicates ......')

    classes = list(pre_extracted_feats.keys())
    classes = sorted(classes, key=lambda x: int(x))

    duplicate_images_dict = defaultdict(set)
    dup_images_fraction = []
    for cls in classes:        
        file_list = pre_extracted_feats[cls]['file_paths']
        if file_list == None:
            print(f'class {cls} has no images. Continue')
            continue

        img_embeddings = pre_extracted_feats[cls]['feats']
        
        similarity = img_embeddings @ img_embeddings.t()
        similarity = similarity.cpu().numpy()
        upper_triangle = np.triu(similarity, k=1)
        i_indices, j_indices = np.where(upper_triangle > 0.9)
        # print('i_indices:', i_indices)
        # print('j_indices:', j_indices)
        # stop
                                                                                   
        to_remove = set(j_indices)
        for file in file_list:
            # print('file:', file)
            file_id = file.split('/')[-1].split('.')[0]
            if int(file_id) in to_remove:
                duplicate_images_dict[cls].add(file)

        dup_images_fraction.append(len(to_remove) / len(file_list))

    avg_dup_images_fraction = sum(dup_images_fraction) / len(dup_images_fraction)
    # print('duplicate image fraction per class', dup_images_fraction)
    # print(f'Average duplication fraction: {round(avg_dup_images_fraction,2)}')

    return duplicate_images_dict, dup_images_fraction, avg_dup_images_fraction


def zeroshot_clip_img_filter(model, preprocess, root_folder, pre_extracted_feats=None, head=None):

    print('\nZeroshot CLIP image filtering ......')
    filtered_images_dict = defaultdict(set)
    classes = os.listdir(root_folder)
    # filter out the files whcih are not folders
    classes = [cls for cls in classes if os.path.isdir(os.path.join(root_folder, cls))]
    classes = sorted(classes, key=lambda x: int(x))

    # Store duplicates for reference later.
    start = time()
    unique_images_fraction = []
    for cls in classes:
        source_folder = os.path.join(root_folder, cls)
        
        if pre_extracted_feats is not None:
            file_list = pre_extracted_feats[cls]['file_paths']
            img_embeddings = pre_extracted_feats[cls]['feats']
        else:
            raise ValueError('Pre-extracted features are required for zeroshot filtering.')
        
        # classify the images using the zero shot classifier.
        logits = head(img_embeddings)
        preds = torch.argmax(logits, dim=1).long()

        # record the index when the prediction is not the same as the class index
        i_indices = []
        for i in range(len(preds)):
            if preds[i] != int(cls):
                i_indices.append(i)
                # print(f'preds[{i}]:', preds[i])
                                                                                   
        to_remove = set(i_indices)
        unique_images = [i for i in range(len(file_list)) if i not in to_remove]

        for file in file_list:
            # print('file:', file)
            file_id = file.split('/')[-1].split('.')[0]
            if int(file_id) in to_remove:
                filtered_images_dict[cls].add(file)
        
        # if cls == '0':
            # print('filtered_images_dict[cls]', filtered_images_dict[cls])
        
        # print(f'For {cls} - fraction of unqiue images: {len(unique_images) / len(file_list)}')
        unique_images_fraction.append(len(unique_images) / len(file_list))
    
    avg_fraction = sum(unique_images_fraction) / len(unique_images_fraction)
    print(f'Average unique images: {round(avg_fraction, 4)}')
    print(f'Done zeroshot img filtering in {round(time() - start)} seconds.')

    return filtered_images_dict


"""
    Function to calculate image x text ranking.
"""
def cal_t2i_similarity(class_prompt, img_embeddings):

    class_prompt = class_prompt.cuda()
    similarity = img_embeddings.cuda() @ class_prompt.t()

    # Calculate the average similarity across alternate names.
    if similarity.shape[-1] > 1:
        similarity = torch.mean(similarity, dim=-1)
        # print('similarity.shape:', similarity.shape)
        # stop
    class_prompt = class_prompt.cpu()
    img_embeddings = img_embeddings.cpu()

    result = similarity.squeeze().cpu().tolist()
    # if results is just a single value for the case of only 1 image, convert it to a list
    if isinstance(result, float):
        result = [result]

    return result


def i2i_similarity(mean_embedding, img_embeddings):
    
    # convert mean_embedding from a numpy array to a tensor of shape (1, 512)
    mean_embedding = torch.from_numpy(mean_embedding).unsqueeze(0).cuda()
    similarity = img_embeddings.cuda() @ mean_embedding.t()
    result = similarity.squeeze().cpu().tolist()
    # if results is just a single value for the case of only 1 image, convert it to a list
    if isinstance(result, float):
        result = [result]

    return result


def i2i_similarity_p2p(fewshot_embedding, img_embeddings, mode):

    # convert the fewshot_embedding from a list of array to 2D array
    fewshot_embedding = np.stack(fewshot_embedding)

    fewshot_embedding = torch.from_numpy(fewshot_embedding).cuda()
    similarity = img_embeddings.cuda() @ fewshot_embedding.t()
    # print('similarity.shape:', similarity.shape)
    if mode == 'min':
        # take the minimum similarity across the fewshot embeddings in the last dimension.
        sim, _ = torch.min(similarity, dim=-1)
    elif mode == 'max':
        sim, _ = torch.max(similarity, dim=-1)
    elif mode == 'mean':
        sim = torch.mean(similarity, dim=-1)
    else:
        raise ValueError('Invalid mode type.')
    # print('sim.shape:', sim.shape)
    result = sim.squeeze().cpu().tolist()
    # print('result:', result)
    
    # if results is just a single value for the case of only 1 image, convert it to a list
    if isinstance(result, float):
        result = [result]

    return result
    

def t2t_similarity(class_prompt, caption_embeddings):

    class_prompt = class_prompt.cuda()
    similarity = caption_embeddings.cuda() @ class_prompt.t()

    # Calculate the average similarity across alternate names.
    if similarity.shape[-1] > 1:
        similarity = torch.mean(similarity, dim=-1)
        # print('similarity.shape:', similarity.shape)
        # stop

    class_prompt = class_prompt.cpu()
    caption_embeddings = caption_embeddings.cpu()

    result = similarity.squeeze().cpu().tolist()
    # if results is just a single value for the case of only 1 image, convert it to a list
    if isinstance(result, float):
        result = [result]

    return result

"""
    Mode can be name, most_common_name and all alternate labels.
"""
def get_class_prompts(metrics, class_idx, name_type='name', dataset='imagenet_1k'):

    class_prompts = make_per_class_prompts(metrics=metrics, class_idx=class_idx, name_type=name_type, dataset=args.dataset)
    # class_prompt is a dict of dict, key = 0, 1, 2 or more for alternates.
    # print('class_prompts:', class_prompts)
    # print('len(class_prompts):', len(class_prompts))

    prompt_tensors = features.get_text_features(model, class_prompts, logger=None, data_source = 'All')
    # print('len(prompt_tensors)=', len(prompt_tensors))
    # prompt_tensors is a list of dict, each dict has two keys: 'all' and 'mean'.

    prompt_embeddings = features.prompt_sampler(prompt_tensors, logger=None, sample_by='mean')
    # print('prompt_embeddings.shape:', prompt_embeddings.shape)
    # stop

    return prompt_embeddings


def add_to_split(caption_map, filtered_list, sampled_list, 
        cls:int, mined_split: dict, imgpaths_sim_zip: list, 
                num_samples: int, 
                threshold: float = 0, 
                duplicates_dict: defaultdict = defaultdict(set),
                filtered_images_dict: defaultdict = defaultdict(set)):

    ct = 0
    feature_list = []
    label_list = []
    file_list = []
    for i, (file_path, similarity, embedding) in enumerate(imgpaths_sim_zip):
        if ct == num_samples:
            break

        if (similarity >= threshold) and \
            (file_path not in duplicates_dict[str(cls)]) and \
            (file_path not in filtered_images_dict[str(cls)]):

            feature_list.append(embedding)
            label_list.append(int(cls))
            file_list.append(file_path)
            ct += 1 # count how many images are added to the split.

            caption = check_caption(caption_map, file_path)
            info = f'{round(similarity,4)}/{threshold}, {file_path}, {caption}'
            sampled_list.append(info)
        else:
            caption = check_caption(caption_map, file_path)
            info = f'{round(similarity,4)}/{threshold}, {file_path}, {caption}'
            filtered_list.append(info)
    
    # concatenate the feature_list and label_list
    if len(feature_list) == 0:
        # print('feature_list is empty. Break.')
        return ct
    else:
        feature_tensor = torch.stack(feature_list)
        labels_tensor = torch.tensor(label_list)
        mined_split['feature_list'].append(feature_tensor)
        mined_split['label_list'].append(labels_tensor)
        mined_split['file_list'].append(file_list)

        return ct #return the added count for the current class       


def check_caption(caption_map, img_path):
    img_cls = img_path.split('/')[-2]
    img_id = img_path.split('/')[-1].split('.')[0]
    img_caption = caption_map[img_cls][img_id]
    # print(f'Image class: {img_cls}, Image ID: {img_id}, Image caption: {img_caption}')
    return img_caption

def add_t2t_ranked_t2i_tshd_to_split(caption_map,
                filtered_list: list, sampled_list: list,
                cls:int, mined_split: dict, imgpaths_sim_zip: list, 
                num_samples: int, 
                threshold: float = 0, 
                duplicates_dict: defaultdict = defaultdict(set),
                filtered_images_dict: defaultdict = defaultdict(set),
                t2i_threshold = 0.25 # note the T2I threshold is 0.25 by default
                ):

    ct = 0
    feature_list = []
    label_list = []
    file_list = [] 

    for i, (file_path, similarity, embedding, t2i_sim) in enumerate(imgpaths_sim_zip):
        if ct == num_samples:
            break

        if (similarity >= threshold) and \
            (t2i_sim >= t2i_threshold) and \
            (file_path not in duplicates_dict[str(cls)]) and \
            (file_path not in filtered_images_dict[str(cls)]):

            feature_list.append(embedding)
            label_list.append(int(cls))
            file_list.append(file_path)
            ct += 1 # count how many images are added to the split.

            caption = check_caption(caption_map, file_path)
            info = f'{round(similarity,4)}/{threshold}, {round(t2i_sim, 4)}/{t2i_threshold}, {file_path}, {caption}'
            sampled_list.append(info)
        else:
            caption = check_caption(caption_map, file_path)
            info = f'{round(similarity,4)}/{threshold}, {round(t2i_sim, 4)}/{t2i_threshold}, {file_path}, {caption}'
            filtered_list.append(info)
    
    # concatenate the feature_list and label_list
    if len(feature_list) == 0:
        # print('feature_list is empty. Break.')
        return ct
    else:
        feature_tensor = torch.stack(feature_list)
        labels_tensor = torch.tensor(label_list)
        mined_split['feature_list'].append(feature_tensor)
        mined_split['label_list'].append(labels_tensor)
        mined_split['file_list'].append(file_list)

        return ct #return the added count for the current class 

"""
def add_to_split(cls:int, mined_split: dict, imgpaths_sim_zip: list, num_samples: int, 
                label_name:str, threshold: float = 0, duplicates_dict: defaultdict = defaultdict(set),
                filtered_images_dict: defaultdict = defaultdict(set)):

    ct = 0
    candidate_set = set()
    # print('len(imgpaths_sim_zip):', len(imgpaths_sim_zip))
    # print('num_samples:', num_samples)

    looping_ct = 0
    while ct < num_samples:
        # print('class, ct:', cls, ct)

        if looping_ct < len(imgpaths_sim_zip):
            idx = looping_ct
        else:
            if len(candidate_set) == 0:
                print('candidate_set is empty. Break.')
                break
            # randomly sample from the candidate set.
            idx = random.sample(candidate_set, 1)[0]
            # print('idx:', idx)

        (file_path, similarity) = imgpaths_sim_zip[idx]
        # print('CLS:', cls)
        # print('similarity:', similarity)
        # print('duplicates_dict[cls]', duplicates_dict[str(cls)])
        # print('file_path', file_path)
        # stop

        if similarity >= threshold and \
            file_path not in duplicates_dict[str(cls)] and \
            file_path not in filtered_images_dict[str(cls)]:

            if idx not in candidate_set:
                candidate_set.add(idx)
                # print('candidate_set:', candidate_set)
            mined_split['train']['data'].append({'impath': file_path, 'label': int(cls), 'classname': label_name})
            ct += 1 # count how many images are added to the split.
        
        looping_ct += 1

    return ct #return the added count for the current class      
"""


"""
    This function is used to randomly sample data.
"""
def random_sampler(args, logger,
                    prompt_tensors, num_samples, 
                    threshold, pre_extracted_feats,
                    duplicates_dict: defaultdict = defaultdict(set), 
                    filtered_images_dict: defaultdict = defaultdict(set),
                    tail_head=False):

    caption_map_path = CAPTION_MAP_DICT[args.dataset]
    with open(caption_map_path, 'rb') as f:
        caption_map = pickle.load(f)

    classes = list(pre_extracted_feats.keys())
    classes = sorted(classes, key=lambda x: int(x))

    mined_split = {'feature_list': [], 'label_list': [], 'file_list': []}
    tail_ct = 0
    num_imgs_sampled_dict = {}
    filtered_list = []
    sampled_list = []        
    for cls in classes:
        img_embeddings = None
        file_list = pre_extracted_feats[cls]['file_paths']
        if file_list == None:
            print(f'class {cls} has no images. Continue')
            num_imgs_sampled_dict[cls] = 0
            continue

        img_embeddings = pre_extracted_feats[cls]['feats']
        similarity = [1. for _ in range(len(file_list))]

        # Randomly sample above a certain threshold.
        if threshold !=0:            
            class_prompt = prompt_tensors[cls]['mean']
            # change class_prompt to a tensor of shape (1, 512)
            class_prompt = class_prompt.unsqueeze(0)
            similarity = cal_t2i_similarity(class_prompt, img_embeddings)
        
        # print(sum(similarity)/len(similarity), cls, len(similarity))

        embedding_list = [img_embeddings[i] for i in range(len(img_embeddings))]        
        path_sim_zip = list(zip(file_list, similarity, embedding_list))
        random.shuffle(path_sim_zip)
        
        # for tail classes (w/ less than 100 images), we sample above a certain threshold.
        if tail_head:
            if len(path_sim_zip) < num_samples:
                tail_ct += 1
                print(f'class {cls} has less than {num_samples} images, use threshold {threshold}.')
            else:
                threshold = 0

        num_sampled_img = add_to_split(caption_map, filtered_list, sampled_list, 
                                        int(cls), mined_split, path_sim_zip, 
                                        num_samples, threshold, duplicates_dict, filtered_images_dict)
        num_imgs_sampled_dict[cls] = num_sampled_img
    
    if tail_head:
        print(f'Number of tail classes: {tail_ct}')
    
    # save the filtered_list to a txt file.
    logger.info(f'len(filtered_list): {len(filtered_list)}')
    with open(f'{args.output_folder}/{args.prefix}_filtered_list.txt', 'w') as f:
        f.write('\n'.join(filtered_list))
    
    # save the sampled_list to a txt file.
    logger.info(f'len(sampled_list): {len(sampled_list)}')
    with open(f'{args.output_folder}/{args.prefix}_sampled_list.txt', 'w') as f:
        f.write('\n'.join(sampled_list))    

    return mined_split, num_imgs_sampled_dict


def random_sampler_i2i(dataset, num_samples, threshold, pre_extracted_feats,
                    duplicates_dict: defaultdict = defaultdict(set), 
                    filtered_images_dict: defaultdict = defaultdict(set),
                    tail_head=False):

    # hardcode for semi-aves
    import pickle
    with open('../CLIP-SSL/data/semi-aves/fewshot15_mean_features.pkl', 'rb') as f:
        mean_fea = pickle.load(f)

    classes = list(pre_extracted_feats.keys())       
    # sort the classes by int()
    classes = sorted(classes, key=lambda x: int(x))

    mined_split = {'feature_list': [], 'label_list': [], 'file_list': []}
    tail_ct = 0
    num_imgs_sampled_dict = {}
    for cls in classes:
        img_embeddings = None
        file_list = pre_extracted_feats[cls]['file_paths']
        if file_list == None:
            print(f'class {cls} has no images. Continue')
            num_imgs_sampled_dict[cls] = 0
            continue
        img_embeddings = pre_extracted_feats[cls]['feats']

        similarity = [1. for _ in range(len(file_list))]
        # Randomly sample above a certain threshold.
        if threshold !=0:            
            mean_embedding = mean_fea[int(cls)]
            # calculate the i2i similarity between each image and the mean embedding
            similarity = i2i_similarity(mean_embedding, img_embeddings)
            
        # print(sum(similarity)/len(similarity), cls, len(similarity))

        embedding_list = [img_embeddings[i] for i in range(len(img_embeddings))]
        # print('len(embedding_list):', len(embedding_list))
        
        path_sim_zip = list(zip(file_list, similarity, embedding_list))
        random.shuffle(path_sim_zip)
        
        # for tail classes (w/ less than 100 images), we sample above a certain threshold.
        if tail_head:
            if len(path_sim_zip) < num_samples:
                tail_ct += 1
                print(f'class {cls} has less than {num_samples} images, use threshold {threshold}.')
            else:
                threshold = 0

        num_sampled_img = add_to_split(int(cls), mined_split, path_sim_zip, num_samples,
                                        threshold, duplicates_dict, filtered_images_dict)

        num_imgs_sampled_dict[cls] = num_sampled_img
    
    if tail_head:
        print(f'Number of tail classes: {tail_ct}')

    return mined_split, num_imgs_sampled_dict


def t2t_ranked_sampler(args, logger, prompt_tensors, num_samples,
                        threshold, pre_extracted_feats, 
                        duplicates_dict: defaultdict = defaultdict(set),
                        filtered_images_dict: defaultdict = defaultdict(set),
                        ):
    
    caption_map_path = CAPTION_MAP_DICT[args.dataset]
    with open(caption_map_path, 'rb') as f:
        caption_map = pickle.load(f)

    classes = list(pre_extracted_feats.keys())
    classes = sorted(classes, key=lambda x: int(x))

    mined_split = {'feature_list': [], 'label_list': [], 'file_list': []}
    num_imgs_sampled_dict = {}
    filtered_list = []
    sampled_list = []    
    for cls in classes:
        file_list = pre_extracted_feats[cls]['file_paths']
        if file_list == None:
            logger.info(f'class {cls} has no images. Continue')
            continue
        
        img_embeddings = pre_extracted_feats[cls]['feats']
        caption_embeddings = pre_extracted_feats[cls]['caption_feats']
        class_prompt = prompt_tensors[cls]['mean']
        class_prompt = class_prompt.unsqueeze(0)
    
        similarity = t2t_similarity(class_prompt, caption_embeddings)
        embedding_list = [img_embeddings[i] for i in range(len(img_embeddings))]
        path_sim_zip = sorted(list(zip(file_list, similarity, embedding_list)), key=lambda x: x[1], reverse=True)

        num_sampled_img = add_to_split(caption_map, filtered_list, sampled_list, 
                                       int(cls), mined_split, path_sim_zip,
                                       num_samples, threshold, duplicates_dict, filtered_images_dict)
        num_imgs_sampled_dict[cls] = num_sampled_img

    # save the filtered_list to a txt file.
    logger.info(f'len(filtered_list): {len(filtered_list)}')
    with open(f'{args.output_folder}/{args.prefix}_filtered_list.txt', 'w') as f:
        f.write('\n'.join(filtered_list))
    
    # save the sampled_list to a txt file.
    logger.info(f'len(sampled_list): {len(sampled_list)}')
    with open(f'{args.output_folder}/{args.prefix}_sampled_list.txt', 'w') as f:
        f.write('\n'.join(sampled_list))

    return mined_split, num_imgs_sampled_dict


def t2t_ranked_t2i_tshd_sampler(args, logger, prompt_tensors, num_samples,
                        threshold, pre_extracted_feats, 
                        duplicates_dict: defaultdict = defaultdict(set),
                        filtered_images_dict: defaultdict = defaultdict(set),
                        ):
    
    caption_map_path = CAPTION_MAP_DICT[args.dataset]
    with open(caption_map_path, 'rb') as f:
        caption_map = pickle.load(f)

    classes = list(pre_extracted_feats.keys())
    classes = sorted(classes, key=lambda x: int(x))

    mined_split = {'feature_list': [], 'label_list': [], 'file_list': []}
    num_imgs_sampled_dict = {}
    filtered_list = []
    sampled_list = []
    for cls in classes:
        img_embeddings = None

        file_list = pre_extracted_feats[cls]['file_paths']
        if file_list == None:
            logger.info(f'class {cls} has no images. Continue')
            continue
        
        img_embeddings = pre_extracted_feats[cls]['feats']
        caption_embeddings = pre_extracted_feats[cls]['caption_feats']
        class_prompt = prompt_tensors[cls]['mean']
        class_prompt = class_prompt.unsqueeze(0)
        
        similarity = t2t_similarity(class_prompt, caption_embeddings)
        embedding_list = [img_embeddings[i] for i in range(len(img_embeddings))]
        t2i_similarity = cal_t2i_similarity(class_prompt, img_embeddings)
        path_sim_zip = sorted(list(zip(file_list, similarity, embedding_list, t2i_similarity)), 
                              key=lambda x: x[1], reverse=True)

        num_sampled_img = add_t2t_ranked_t2i_tshd_to_split(caption_map, filtered_list, sampled_list, int(cls), 
                                                            mined_split, path_sim_zip, num_samples, 
                                                            threshold, duplicates_dict, filtered_images_dict)
        num_imgs_sampled_dict[cls] = num_sampled_img

    # save the filtered_list to a txt file.
    logger.info(f'len(filtered_list): {len(filtered_list)}')
    with open(f'{args.output_folder}/filtered_list.txt', 'w') as f:
        f.write('\n'.join(filtered_list))
    
    # save the sampled_list to a txt file.
    logger.info(f'len(sampled_list): {len(sampled_list)}')
    with open(f'{args.output_folder}/sampled_list.txt', 'w') as f:
        f.write('\n'.join(sampled_list))

    return mined_split, num_imgs_sampled_dict





def t2t_rank_i2t_tshd_sampler(args, logger, prompt_tensors, num_samples,
                        threshold, pre_extracted_feats, 
                        duplicates_dict: defaultdict = defaultdict(set),
                        filtered_images_dict: defaultdict = defaultdict(set),
                        ):
    
    fewshot_fea = get_fewshot_features(args.dataset)

    caption_map_path = CAPTION_MAP_DICT[args.dataset]
    with open(caption_map_path, 'rb') as f:
        caption_map = pickle.load(f)

    classes = list(pre_extracted_feats.keys())
    classes = sorted(classes, key=lambda x: int(x))

    mined_split = {'feature_list': [], 'label_list': [], 'file_list': []}
    num_imgs_sampled_dict = {}
    filtered_list = []
    sampled_list = []
    for cls in classes:
        img_embeddings = None

        file_list = pre_extracted_feats[cls]['file_paths']
        if file_list == None:
            logger.info(f'class {cls} has no images. Continue')
            continue
        
        img_embeddings = pre_extracted_feats[cls]['feats']
        caption_embeddings = pre_extracted_feats[cls]['caption_feats']
        class_prompt = prompt_tensors[cls]['mean']
        class_prompt = class_prompt.unsqueeze(0)
        
        similarity = t2t_similarity(class_prompt, caption_embeddings)
        embedding_list = [img_embeddings[i] for i in range(len(img_embeddings))]
        
        # t2i_similarity = cal_t2i_similarity(class_prompt, img_embeddings)
        fewshot_embedding = fewshot_fea[int(cls)]
        # this is actually I2T similarity
        t2i_similarity = i2i_similarity_p2p(fewshot_embedding, caption_embeddings, 'max') 


        path_sim_zip = sorted(list(zip(file_list, similarity, embedding_list, t2i_similarity)), 
                              key=lambda x: x[1], reverse=True)

        num_sampled_img = add_t2t_ranked_t2i_tshd_to_split(caption_map, filtered_list, sampled_list, int(cls), 
                                                            mined_split, path_sim_zip, num_samples, 
                                                            threshold, duplicates_dict, filtered_images_dict)
        num_imgs_sampled_dict[cls] = num_sampled_img

    # save the filtered_list to a txt file.
    logger.info(f'len(filtered_list): {len(filtered_list)}')
    with open(f'{args.output_folder}/filtered_list.txt', 'w') as f:
        f.write('\n'.join(filtered_list))
    
    # save the sampled_list to a txt file.
    logger.info(f'len(sampled_list): {len(sampled_list)}')
    with open(f'{args.output_folder}/sampled_list.txt', 'w') as f:
        f.write('\n'.join(sampled_list))

    return mined_split, num_imgs_sampled_dict


def t2t_rank_i2i_tshd_sampler(args, logger, prompt_tensors, num_samples,
                        threshold, pre_extracted_feats, 
                        duplicates_dict: defaultdict = defaultdict(set),
                        filtered_images_dict: defaultdict = defaultdict(set),
                        ):
    
    fewshot_fea = get_fewshot_features(args.dataset)

    caption_map_path = CAPTION_MAP_DICT[args.dataset]
    with open(caption_map_path, 'rb') as f:
        caption_map = pickle.load(f)

    classes = list(pre_extracted_feats.keys())
    classes = sorted(classes, key=lambda x: int(x))

    mined_split = {'feature_list': [], 'label_list': [], 'file_list': []}
    num_imgs_sampled_dict = {}
    filtered_list = []
    sampled_list = []
    for cls in classes:
        img_embeddings = None

        file_list = pre_extracted_feats[cls]['file_paths']
        if file_list == None:
            logger.info(f'class {cls} has no images. Continue')
            continue
        
        img_embeddings = pre_extracted_feats[cls]['feats']
        caption_embeddings = pre_extracted_feats[cls]['caption_feats']
        class_prompt = prompt_tensors[cls]['mean']
        class_prompt = class_prompt.unsqueeze(0)
        
        similarity = t2t_similarity(class_prompt, caption_embeddings)
        embedding_list = [img_embeddings[i] for i in range(len(img_embeddings))]
        
        fewshot_embedding = fewshot_fea[int(cls)]
        # this is actually I2I similarity
        t2i_similarity = i2i_similarity_p2p(fewshot_embedding, img_embeddings, 'max') 


        path_sim_zip = sorted(list(zip(file_list, similarity, embedding_list, t2i_similarity)), 
                              key=lambda x: x[1], reverse=True)

        num_sampled_img = add_t2t_ranked_t2i_tshd_to_split(caption_map, filtered_list, sampled_list, int(cls), 
                                                            mined_split, path_sim_zip, num_samples, 
                                                            threshold, duplicates_dict, filtered_images_dict,
                                                            t2i_threshold = 0.65, # this is the threshold for I2I similarity
                                                            )
        num_imgs_sampled_dict[cls] = num_sampled_img

    # save the filtered_list to a txt file.
    logger.info(f'len(filtered_list): {len(filtered_list)}')
    with open(f'{args.output_folder}/filtered_list.txt', 'w') as f:
        f.write('\n'.join(filtered_list))
    
    # save the sampled_list to a txt file.
    logger.info(f'len(sampled_list): {len(sampled_list)}')
    with open(f'{args.output_folder}/sampled_list.txt', 'w') as f:
        f.write('\n'.join(sampled_list))

    return mined_split, num_imgs_sampled_dict



# point to mean of the 15 fewshot images.
def i2i_ranked_sampler(logger, num_samples,
                        threshold, pre_extracted_feats, 
                        duplicates_dict: defaultdict = defaultdict(set)):

    # hardcode for semi-aves
    import pickle
    with open('../CLIP-SSL/data/semi-aves/fewshot15_mean_features.pkl', 'rb') as f:
        mean_fea = pickle.load(f)
            
    classes = list(pre_extracted_feats.keys())
    # sort the classes
    classes = sorted(classes, key=lambda x: int(x))

    mined_split = {'feature_list': [], 'label_list': [], 'file_list': []}
    logger.info('T2T Ranked sampling ......')
    for cls in classes:
        img_embeddings = None

        file_list = pre_extracted_feats[cls]['file_paths']
        if file_list == None:
            logger.info(f'class {cls} has no images. Continue')
            continue
        
        img_embeddings = pre_extracted_feats[cls]['feats']

        if len(file_list) <= 10:
            logger.info(f'class {cls} has at {len(file_list)} images. ')

        mean_embedding = mean_fea[int(cls)]
        # calculate the i2i similarity between each image and the mean embedding
        similarity = i2i_similarity(mean_embedding, img_embeddings)
        embedding_list = [img_embeddings[i] for i in range(len(img_embeddings))]

        path_sim_zip = sorted(list(zip(file_list, similarity, embedding_list)), key=lambda x: x[1], reverse=True)

        add_to_split(int(cls), mined_split, path_sim_zip, num_samples, threshold, duplicates_dict)

    return mined_split

def get_fewshot_features(dataset):

    fewshot_fea = dict()
    # extract feature by class using the saved feature tensor when probing the model.
    fn = f'../data/{dataset}/pre_extracted/{dataset}_probing_vitb32_openclip_laion400m_1_train_features.pth'
    fea = torch.load(fn)
    img_fea = fea['image_features']
    labels = fea['labels']
    for i, label in enumerate(labels):
        label = label.item()
        # print('label:', label)
        if label not in fewshot_fea:
            fewshot_fea[label] = []
        fewshot_fea[label].append(img_fea[i])
    assert len(fewshot_fea[0]) == 16 # since we use 16 fewshot images 
    print('len(fewshot_fea):', len(fewshot_fea))

    return fewshot_fea

def i2i_ranked_sampler_p2p(args, logger, prompt_tensors, num_samples,
                        threshold, pre_extracted_feats,                         
                        duplicates_dict: defaultdict = defaultdict(set),
                        filtered_images_dict: defaultdict = defaultdict(set)
                        ):

    # # hardcode for semi-aves
    # import pickle
    # with open('../CLIP-SSL/data/semi-aves/fewshot15_classid_to_features.pkl', 'rb') as f:
    #     fewshot_fea = pickle.load(f)
    
    fewshot_fea = get_fewshot_features(args.dataset)
            
    caption_map_path = CAPTION_MAP_DICT[args.dataset]
    with open(caption_map_path, 'rb') as f:
        caption_map = pickle.load(f)

    classes = list(pre_extracted_feats.keys())
    classes = sorted(classes, key=lambda x: int(x))

    mined_split = {'feature_list': [], 'label_list': [], 'file_list': []}
    num_imgs_sampled_dict = {}
    filtered_list = []
    sampled_list = []        
    for cls in classes:
        file_list = pre_extracted_feats[cls]['file_paths']
        if file_list == None:
            logger.info(f'class {cls} has no images. Continue')
            continue
        
        img_embeddings = pre_extracted_feats[cls]['feats']
        caption_embeddings = pre_extracted_feats[cls]['caption_feats']
        class_prompt = prompt_tensors[cls]['mean']
        class_prompt = class_prompt.unsqueeze(0)

        fewshot_embedding = fewshot_fea[int(cls)]
        # calculate the i2i similarity between each image and the mean embedding
        # similarity = i2i_similarity_p2p(fewshot_embedding, img_embeddings, 'min')
        # similarity = i2i_similarity_p2p(fewshot_embedding, img_embeddings, 'max')
        similarity = i2i_similarity_p2p(fewshot_embedding, img_embeddings, 'mean')

        embedding_list = [img_embeddings[i] for i in range(len(img_embeddings))]

        path_sim_zip = sorted(list(zip(file_list, similarity, embedding_list)), key=lambda x: x[1], reverse=True)

        num_sampled_img = add_to_split(caption_map, filtered_list, sampled_list, 
                                int(cls), mined_split, path_sim_zip,
                                num_samples, threshold, duplicates_dict, filtered_images_dict)
        num_imgs_sampled_dict[cls] = num_sampled_img

    # save the filtered_list to a txt file.
    logger.info(f'len(filtered_list): {len(filtered_list)}')
    with open(f'{args.output_folder}/{args.prefix}_filtered_list.txt', 'w') as f:
        f.write('\n'.join(filtered_list))
    
    # save the sampled_list to a txt file.
    logger.info(f'len(sampled_list): {len(sampled_list)}')
    with open(f'{args.output_folder}/{args.prefix}_sampled_list.txt', 'w') as f:
        f.write('\n'.join(sampled_list))

    return mined_split, num_imgs_sampled_dict


def i2t_rank_sampler(args, logger, prompt_tensors, num_samples,
                        threshold, pre_extracted_feats,                         
                        duplicates_dict: defaultdict = defaultdict(set),
                        filtered_images_dict: defaultdict = defaultdict(set)
                        ):
    
    fewshot_fea = get_fewshot_features(args.dataset)
            
    caption_map_path = CAPTION_MAP_DICT[args.dataset]
    with open(caption_map_path, 'rb') as f:
        caption_map = pickle.load(f)

    classes = list(pre_extracted_feats.keys())
    classes = sorted(classes, key=lambda x: int(x))

    mined_split = {'feature_list': [], 'label_list': [], 'file_list': []}
    num_imgs_sampled_dict = {}
    filtered_list = []
    sampled_list = []        
    for cls in classes:
        file_list = pre_extracted_feats[cls]['file_paths']
        if file_list == None:
            logger.info(f'class {cls} has no images. Continue')
            continue
        
        img_embeddings = pre_extracted_feats[cls]['feats']
        caption_embeddings = pre_extracted_feats[cls]['caption_feats']
        class_prompt = prompt_tensors[cls]['mean']
        class_prompt = class_prompt.unsqueeze(0)

        fewshot_embedding = fewshot_fea[int(cls)]
        
        # note here we use caption_embeddings 
        similarity = i2i_similarity_p2p(fewshot_embedding, caption_embeddings, 'mean') 
        
        embedding_list = [img_embeddings[i] for i in range(len(img_embeddings))]

        path_sim_zip = sorted(list(zip(file_list, similarity, embedding_list)), key=lambda x: x[1], reverse=True)

        num_sampled_img = add_to_split(caption_map, filtered_list, sampled_list, 
                                int(cls), mined_split, path_sim_zip,
                                num_samples, threshold, duplicates_dict, filtered_images_dict)
        num_imgs_sampled_dict[cls] = num_sampled_img

    # save the filtered_list to a txt file.
    logger.info(f'len(filtered_list): {len(filtered_list)}')
    with open(f'{args.output_folder}/{args.prefix}_filtered_list.txt', 'w') as f:
        f.write('\n'.join(filtered_list))
    
    # save the sampled_list to a txt file.
    logger.info(f'len(sampled_list): {len(sampled_list)}')
    with open(f'{args.output_folder}/{args.prefix}_sampled_list.txt', 'w') as f:
        f.write('\n'.join(sampled_list))

    return mined_split, num_imgs_sampled_dict


# def i2t_tshd_sampler(args, logger, prompt_tensors, num_samples,
#                         threshold, pre_extracted_feats,                         
#                         duplicates_dict: defaultdict = defaultdict(set),
#                         filtered_images_dict: defaultdict = defaultdict(set)
#                         ):
    
#     fewshot_fea = get_fewshot_features(args.dataset)
            
#     caption_map_path = CAPTION_MAP_DICT[args.dataset]
#     with open(caption_map_path, 'rb') as f:
#         caption_map = pickle.load(f)

#     classes = list(pre_extracted_feats.keys())
#     classes = sorted(classes, key=lambda x: int(x))

#     mined_split = {'feature_list': [], 'label_list': [], 'file_list': []}
#     num_imgs_sampled_dict = {}
#     filtered_list = []
#     sampled_list = []        
#     for cls in classes:
#         file_list = pre_extracted_feats[cls]['file_paths']
#         if file_list == None:
#             logger.info(f'class {cls} has no images. Continue')
#             continue
        
#         img_embeddings = pre_extracted_feats[cls]['feats']
#         caption_embeddings = pre_extracted_feats[cls]['caption_feats']
#         class_prompt = prompt_tensors[cls]['mean']
#         class_prompt = class_prompt.unsqueeze(0)

#         fewshot_embedding = fewshot_fea[int(cls)]
#         # calculate the i2i similarity between each image and the mean embedding
#         # similarity = i2i_similarity_p2p(fewshot_embedding, img_embeddings, 'min')
#         # similarity = i2i_similarity_p2p(fewshot_embedding, img_embeddings, 'max')
#         # similarity = i2i_similarity_p2p(fewshot_embedding, img_embeddings, 'mean')

#         similarity = i2i_similarity_p2p(fewshot_embedding, caption_embeddings, 'mean') # note here we use caption_embeddings 

#         embedding_list = [img_embeddings[i] for i in range(len(img_embeddings))]

#         path_sim_zip = sorted(list(zip(file_list, similarity, embedding_list)), key=lambda x: x[1], reverse=True)

#         num_sampled_img = add_to_split(caption_map, filtered_list, sampled_list, 
#                                 int(cls), mined_split, path_sim_zip,
#                                 num_samples, threshold, duplicates_dict, filtered_images_dict)
#         num_imgs_sampled_dict[cls] = num_sampled_img

#     # save the filtered_list to a txt file.
#     logger.info(f'len(filtered_list): {len(filtered_list)}')
#     with open(f'{args.output_folder}/{args.prefix}_filtered_list.txt', 'w') as f:
#         f.write('\n'.join(filtered_list))
    
#     # save the sampled_list to a txt file.
#     logger.info(f'len(sampled_list): {len(sampled_list)}')
#     with open(f'{args.output_folder}/{args.prefix}_sampled_list.txt', 'w') as f:
#         f.write('\n'.join(sampled_list))

#     return mined_split, num_imgs_sampled_dict

def t2i_ranked_sampler(args, logger, 
                    prompt_tensors, num_samples,
                    threshold, pre_extracted_feats, 
                    duplicates_dict: defaultdict = defaultdict(set),
                    filtered_images_dict: defaultdict = defaultdict(set),
                    ):
    
    caption_map_path = CAPTION_MAP_DICT[args.dataset]
    with open(caption_map_path, 'rb') as f:
        caption_map = pickle.load(f)

    classes = list(pre_extracted_feats.keys())
    classes = sorted(classes, key=lambda x: int(x))

    mined_split = {'feature_list': [], 'label_list': [], 'file_list': []}
    num_imgs_sampled_dict = {}
    filtered_list = []
    sampled_list = []        
    for cls in classes:
        file_list = pre_extracted_feats[cls]['file_paths']
        if file_list == None:
            logger.info(f'class {cls} has no images. Continue')
            continue
        
        img_embeddings = pre_extracted_feats[cls]['feats']
        caption_embeddings = pre_extracted_feats[cls]['caption_feats']
        class_prompt = prompt_tensors[cls]['mean']
        class_prompt = class_prompt.unsqueeze(0)        
        
        similarity = cal_t2i_similarity(class_prompt, img_embeddings)
        embedding_list = [img_embeddings[i] for i in range(len(img_embeddings))]
        path_sim_zip = sorted(list(zip(file_list, similarity, embedding_list)), key=lambda x: x[1], reverse=True)

        num_sampled_img = add_to_split(caption_map, filtered_list, sampled_list, 
                                int(cls), mined_split, path_sim_zip,
                                num_samples, threshold, duplicates_dict, filtered_images_dict)
        num_imgs_sampled_dict[cls] = num_sampled_img

    # save the filtered_list to a txt file.
    logger.info(f'len(filtered_list): {len(filtered_list)}')
    with open(f'{args.output_folder}/{args.prefix}_filtered_list.txt', 'w') as f:
        f.write('\n'.join(filtered_list))
    
    # save the sampled_list to a txt file.
    logger.info(f'len(sampled_list): {len(sampled_list)}')
    with open(f'{args.output_folder}/{args.prefix}_sampled_list.txt', 'w') as f:
        f.write('\n'.join(sampled_list))

    return mined_split, num_imgs_sampled_dict


def get_avg_crossentropy(img_embeddings):
    # img_embeddings is a torch tensor of shape(num_images, 512)
    # print('img_embeddings.shape:', img_embeddings.shape)
    img_embeddings = img_embeddings.cuda()

    # here we calculate the crossentropy value of each img embedding w.r.t. all other img embeddings.
    # loop through each img
    avg_crossentropy = []
    crossentropy_dict = {}
    for i in range(img_embeddings.shape[0]):
        crossentropy = []
        for j in range(img_embeddings.shape[0]):
            if i == j:
                continue
            else:
                if i < j:
                    key = f'{i}-{j}'
                else:
                    key = f'{j}-{i}'
                if key in crossentropy_dict:
                    entropy = crossentropy_dict[key]
                else:
                    # calculate crossentropy between image i and image j, which is row i and row j in img_embeddings
                    entropy = F.cross_entropy(img_embeddings[i].unsqueeze(0), img_embeddings[j].unsqueeze(0))
                    # entropy = entropy.cpu().item()
                    crossentropy_dict[key] = entropy

                crossentropy.append(entropy)

        avg = sum(crossentropy) / len(crossentropy)
        avg_crossentropy.append(avg)
    
    # print('len(avg_crossentropy):', len(avg_crossentropy))
    # print('max of avg_crossentropy:', max(avg_crossentropy))
    # print('min of avg_crossentropy:', min(avg_crossentropy))
    # print('avg_crossentropy[:10]:', avg_crossentropy[:10])
    # print('mean of avg_crossentropy:', sum(avg_crossentropy) / len(avg_crossentropy))

    img_embeddings = img_embeddings.cpu()

    return avg_crossentropy


def crossentropy_sampler(root_folder, metrics, num_samples=100, model=None, preprocess=None, 
                    threshold=0, name_type='name', pre_extracted_feats = None, 
                    duplicates_dict: defaultdict = defaultdict(set)):
    
    classes = os.listdir(root_folder)
    # sort classes by names, where names are 0 to 999
    classes = sorted(classes, key=lambda x: int(x))
    # print('classes:', classes)

    mined_split = {'train': {'data': []}}
    print('crossentropy sampling ......')
    for cls in classes:
        # print('cls:', cls)
        source_folder = os.path.join(root_folder, cls)
        img_embeddings = None
        if pre_extracted_feats is not None:
            file_list = pre_extracted_feats[cls]['file_paths']
            # fix the file_list
            # file_list = [file_path.replace('imagenet_1k', 'imagenet_1k_mined') for file_path in file_list]
            img_embeddings = pre_extracted_feats[cls]['feats']

            # print('file_list:', len(file_list))
            # print('file_list[0]:', file_list[0])
            # print('img_embeddings.shape', img_embeddings.shape)

        else:
            file_list = [os.path.join(source_folder, file) for file in os.listdir(source_folder)]
            exit()

        # class_prompt = get_class_prompts(metrics=metrics, class_idx=int(cls), name_type=name_type, dataset=args.dataset)
        
        avg_crossentropy = get_avg_crossentropy(img_embeddings)

        path_sim_zip = sorted(list(zip(file_list, avg_crossentropy)), key=lambda x: x[1], reverse=True) # sort by crossentropy from high to low
        add_to_split(int(cls), mined_split, path_sim_zip, num_samples, threshold, duplicates_dict)

    return mined_split


def totalentropy_sampler(root_folder, metrics, num_samples=100, model=None, preprocess=None, 
                    threshold=0, name_type='name', pre_extracted_feats = None, 
                    duplicates_dict: defaultdict = defaultdict(set), head=None):
    
    classes = os.listdir(root_folder)
    # sort classes by names, where names are 0 to 999
    classes = sorted(classes, key=lambda x: int(x))
    # print('classes:', classes)

    if head is None:
        print('head is None, exiting ......')
        exit()

    mined_split = {'feature_list': [], 'label_list': []}
    print('totalentropy sampling ......')
    for cls in classes:
        # print('cls:', cls)
        source_folder = os.path.join(root_folder, cls)
        img_embeddings = None
        if pre_extracted_feats is not None:
            file_list = pre_extracted_feats[cls]['file_paths']
            # fix the file_list
            # file_list = [file_path.replace('imagenet_1k', 'imagenet_1k_mined') for file_path in file_list]
            img_embeddings = pre_extracted_feats[cls]['feats']

            # print('file_list:', len(file_list))
            # print('file_list[0]:', file_list[0])
            # print('img_embeddings.shape', img_embeddings.shape)

        else:
            file_list = [os.path.join(source_folder, file) for file in os.listdir(source_folder)]
            exit()

        # calclate the logits by feeding the img embeddings to the head
        logits = head(img_embeddings.cuda())
        # print('logits.shape:', logits.shape)

        # softmax the logits
        softmax = torch.nn.Softmax(dim=1)
        probs = softmax(logits)
        # print('probs.shape:', probs.shape)

        # calculate the entropy of each image
        entropy1 = -torch.sum(probs * torch.log(probs), dim=1)
        # print('entropy1.shape:', entropy1.shape)
        # print('entropy1:', entropy1)

        # convert probs to a numpy array in cpu
        # probs = probs.cpu()
        # probs = probs.detach().numpy()
        # entropy2 = entropy(probs, axis=1)
        # print('entropy2.shape:', entropy2.shape)
        # print('entropy2[:10]:', entropy2[:10])

        path_sim_zip = sorted(list(zip(file_list, entropy1)), key=lambda x: x[1], reverse=True)
        add_to_split(int(cls), mined_split, path_sim_zip, num_samples, threshold, duplicates_dict)

    return mined_split

def transform_extracted_fea(pre_extracted_feats):
    # post-process the pre_extracted_feats (a dict containing "image_features": img_feature_tensor, 
    # "labels": labels_tensor, "filepath": file_path_list, "caption_features": caption_feature_tensor)
    # the result is a dict of {class: {'file_paths': file_paths, 'feats': feats, 'caption_feats': caption_feats}}

    result_dict = {}
    collection_dict = {}
    num_rows = pre_extracted_feats['image_features'].shape[0]
    for i in range(num_rows):
        img_embedding = pre_extracted_feats['image_features'][i]
        label = str(pre_extracted_feats['labels'][i].item())
        file_path = pre_extracted_feats['filepath'][i]
        caption_embedding = pre_extracted_feats['caption_features'][i]

        if label not in collection_dict:
            collection_dict[label] = {'file_paths': [], 'feats': [], 'caption_feats': []}

        collection_dict[label]['file_paths'].append(file_path)
        collection_dict[label]['feats'].append(img_embedding)
        collection_dict[label]['caption_feats'].append(caption_embedding)
    
    print('len(collection_dict):', len(collection_dict))

    # convert the list of feats to a tensor
    for label in collection_dict:
        result_dict[label] = {'file_paths': collection_dict[label]['file_paths'], 
                                'feats': torch.stack(collection_dict[label]['feats']),
                                'caption_feats': torch.stack(collection_dict[label]['caption_feats'])}
    return result_dict


def cal_prompt_tensors(args, logger, model, tokenizer, metrics, dataset_root):

    prompts_dir = os.path.join('../data', args.dataset, 'prompts/')
    if not os.path.exists(prompts_dir):
        os.makedirs(prompts_dir)
        logger.info(f'Created directory: {prompts_dir}')    
    
    prompt_tensors_dict= {}
    text_prompts_dict = {}
    # for label_type in ['name', 'most_common_name', 'alternates']:
    for label_type in [args.prompt_name]:
        prompt_tensors_filename = f"{prompts_dir}{args.dataset}_{args.model_cfg}_{label_type}_prompt_tensors.pth"
        text_prompts_filename = f"{prompts_dir}{args.dataset}_{args.model_cfg}_{label_type}_text_prompts.pth"
        # tokenized_text_prompts_filename = f"{prompts_dir}{args.dataset}_{args.model_cfg}_{label_type}_tokenized_text_prompts.pth"

        if not args.recal_prompt and os.path.exists(prompt_tensors_filename):
            prompt_tensors_dict[label_type] = torch.load(prompt_tensors_filename)
            text_prompts_dict[label_type] = torch.load(text_prompts_filename)
            logger.info(f'Loaded prompt tensors from: {prompt_tensors_filename}')
            logger.info(f'Loaded text prompts from: {text_prompts_filename}')
        else:
            # make text_prompts which is a dict of dict, key is class idx, value is a dict of prompts
            text_prompts = prompt_maker(metrics=metrics, dataset_name=args.dataset, name_type=label_type)
            text_prompts_dict[label_type] = text_prompts    

            logger.info(f'Calculating prompt tensors for {label_type} ...')
            prompt_tensors = features.get_text_features(model, text_prompts, tokenizer, 'encode')
            prompt_tensors_dict[label_type] = prompt_tensors

            # prompt_tensors is a list of dicts, each dict has 'mean': mean_embedding of 80 prompts 
            torch.save(prompt_tensors, prompt_tensors_filename)
            logger.info(f'Saved prompt tensors to: {prompt_tensors_filename}')    

            torch.save(text_prompts, text_prompts_filename)
            logger.info(f'Saved text prompts to: {text_prompts_filename}')    

    return prompt_tensors_dict, text_prompts_dict


def save_sample_file_list(args, final_file_list, label_tensor):
    fn = f'{args.output_folder}/{args.prefix}.txt'
    with open(fn, 'w') as f:
        for i, file_path in enumerate(final_file_list):
            label = label_tensor[i].item()
            f.write(f'{file_path} {label} {0}\n') # 0 means retrieved, 1 means few-shot
    logger.info(f'Saved file_list to: {fn}')

    # copy the file to ../data/{args.dataset}/
    shutil.copy(fn, f'../data/{args.dataset}/')
    logger.info(f'Copied file to: ../data/{args.dataset}/')

    return fn

def sampling(args, logger, model, preprocess, metrics, dataset_root):

    pre_extracted_feats_fn = f'{dataset_root}/{args.dataset}_{args.model_cfg}_mined.pth'
    if os.path.exists(pre_extracted_feats_fn):
        pre_extracted_feats = torch.load(pre_extracted_feats_fn)
        logger.info(f'Loaded pre-extracted mined features from: {pre_extracted_feats_fn}')
    else:
        logger.info(f'Error: Pre-extracted features not found. {pre_extracted_feats_fn}')
        raise NotImplementedError
        
    pre_extracted_feats = transform_extracted_fea(pre_extracted_feats)

    #---------- Zeroshot CLIP image filtering.
    if not args.zeroshot_img_filter: # not doing zeroshot image filtering
        logger.info('No zeroshot image filtering!')
        filtered_images_dict = defaultdict(set)
    else:
        logger.info('Doing zeroshot image filtering!')
        zeroshot_weights = features.prompt_sampler(prompt_tensors_dict[args.prompt_name], logger, sample_by='mean')            
        head = MyLinear(weights=zeroshot_weights, bias=False)
        head.to(device=device)
        head.eval()
        filtered_images_dict = zeroshot_clip_img_filter(model=model, preprocess=preprocess, 
                                                        root_folder=root_folder, 
                                                        pre_extracted_feats=pre_extracted_feats,
                                                        head=head
                                                        )
                                                        
    #---------- Image De-duplication.
    if not args.image_dedup: # not doing image deduplication
        logger.info('No image deduplication!')
        duplicate_images_dict = defaultdict(set)
    else:
        logger.info('Doing image deduplication!')
        duplicate_images_dict, dup_images_fraction, avg_dup_images_fraction = remove_near_duplicates2(pre_extracted_feats)
        logger.info(f'dup_images_fraction: {dup_images_fraction}')
        logger.info(f'avg_dup_images_fraction: {avg_dup_images_fraction}')

    #---------- T2I Ranking / Random Sampling 
    # - T2T ranking
    # - random sampling from images above a T2I threshold
    # - T2I ranking, ranking and filtering with a T2I threshold
    # - crossentropy ranking, filtering with a value threshold 
    # - totalentropy ranking, filtering with a value threshold 

    logger.info(f'Sampling method: {args.sampling_method}, sampling number: {args.num_samples}, sampling threshold: {args.sampling_threshold}')
    if args.sampling_method == 'Random':
        mined_split, num_imgs_sampled_dict = random_sampler(args, logger,
                                                            prompt_tensors=prompt_tensors_dict[args.prompt_name], 
                                                            num_samples=args.num_samples, 
                                                            threshold=0.0, 
                                                            pre_extracted_feats=pre_extracted_feats,
                                                            duplicates_dict=duplicate_images_dict,
                                                            filtered_images_dict=filtered_images_dict,
                                                            tail_head=False, # set to False to not using head and tail different strategy
                                                            )

    elif args.sampling_method == 'Random-I2I':
        mined_split, num_imgs_sampled_dict = random_sampler_i2i(dataset=args.dataset,                                        
                                            num_samples=args.num_samples, 
                                            threshold=args.sampling_threshold, 
                                            pre_extracted_feats=pre_extracted_feats,
                                            duplicates_dict=duplicate_images_dict,
                                            filtered_images_dict=filtered_images_dict,
                                            tail_head=False, 
                                            )

    elif args.sampling_method == 'I2I-rank': # I2I ranking
        # mined_split = i2i_ranked_sampler(logger, 
        #                                  num_samples=args.num_samples, 
        #                                 threshold=0.0, # note we set threshold to 0 here
        #                                 pre_extracted_feats=pre_extracted_feats,
        #                                 duplicates_dict=duplicate_images_dict)

        mined_split, num_imgs_sampled_dict = i2i_ranked_sampler_p2p(args, logger, 
                                                                    prompt_tensors=prompt_tensors_dict[args.prompt_name], 
                                                                    num_samples=args.num_samples, 
                                                                    threshold=0.0, # note we set threshold to 0 here
                                                                    pre_extracted_feats=pre_extracted_feats,
                                                                    filtered_images_dict=filtered_images_dict,
                                                                    duplicates_dict=duplicate_images_dict)            

    elif args.sampling_method == 'I2T-rank': # I2T ranking
        mined_split, num_imgs_sampled_dict = i2t_rank_sampler(args, logger, 
                                                                prompt_tensors=prompt_tensors_dict[args.prompt_name], 
                                                                num_samples=args.num_samples, 
                                                                threshold=0.0, # note we set threshold to 0 here
                                                                pre_extracted_feats=pre_extracted_feats,
                                                                filtered_images_dict=filtered_images_dict,
                                                                duplicates_dict=duplicate_images_dict)   

    # elif args.sampling_method == 'I2T-tshd': # I2T thrshold filtering
    #     mined_split, num_imgs_sampled_dict = i2t_tshd_sampler(args, logger, 
    #                                                             prompt_tensors=prompt_tensors_dict['most_common_name'], 
    #                                                             num_samples=args.num_samples, 
    #                                                             threshold=0.25, # note we set threshold to 0.25 here
    #                                                             pre_extracted_feats=pre_extracted_feats,
    #                                                             filtered_images_dict=filtered_images_dict,
    #                                                             duplicates_dict=duplicate_images_dict)  

    elif args.sampling_method == 'T2T-rank': # T2T ranking
        mined_split, num_imgs_sampled_dict = t2t_ranked_sampler(args, logger,
                                                                prompt_tensors=prompt_tensors_dict[args.prompt_name], 
                                                                # prompt_tensors=prompt_tensors_dict['most_common_name'], 
                                                                num_samples=args.num_samples, 
                                                                threshold=0.0, # note we set threshold to 0 here
                                                                pre_extracted_feats=pre_extracted_feats,
                                                                filtered_images_dict=filtered_images_dict,
                                                                duplicates_dict=duplicate_images_dict) 

    elif args.sampling_method == 'T2T-rank-T2I-tshd': # T2T ranking with T2I thresholding
        mined_split, num_imgs_sampled_dict = t2t_ranked_t2i_tshd_sampler(args, logger, 
                                            # prompt_tensors=prompt_tensors_dict['most_common_name'], # note here we used most_common_name
                                            prompt_tensors=prompt_tensors_dict[args.prompt_name], # note here we used alternates                                            
                                            num_samples=args.num_samples, 
                                            threshold=0.0, # note we set threshold to 0 here
                                            pre_extracted_feats=pre_extracted_feats,
                                            filtered_images_dict=filtered_images_dict,
                                            duplicates_dict=duplicate_images_dict)

    elif args.sampling_method == 'T2T-rank-I2T-tshd': # T2T ranking with I2T thresholding
        mined_split, num_imgs_sampled_dict = t2t_rank_i2t_tshd_sampler(args, logger, 
                                            prompt_tensors=prompt_tensors_dict[args.prompt_name], 
                                            num_samples=args.num_samples, 
                                            threshold=0.0, # note we set threshold to 0 here
                                            pre_extracted_feats=pre_extracted_feats,
                                            filtered_images_dict=filtered_images_dict,
                                            duplicates_dict=duplicate_images_dict)

    elif args.sampling_method == 'T2T-rank-I2I-tshd': # T2T ranking with I2I thresholding
        mined_split, num_imgs_sampled_dict = t2t_rank_i2i_tshd_sampler(args, logger, 
                                            prompt_tensors=prompt_tensors_dict[args.prompt_name], 
                                            num_samples=args.num_samples, 
                                            threshold=0.0, # note we set threshold to 0 here
                                            pre_extracted_feats=pre_extracted_feats,
                                            filtered_images_dict=filtered_images_dict,
                                            duplicates_dict=duplicate_images_dict)
        

    elif args.sampling_method == 'T2I-rank': # T2I ranking
        mined_split, num_imgs_sampled_dict = t2i_ranked_sampler(args, logger, 
                                        prompt_tensors=prompt_tensors_dict[args.prompt_name], 
                                        num_samples=args.num_samples, 
                                        threshold=0.0, 
                                        pre_extracted_feats = pre_extracted_feats,
                                        filtered_images_dict=filtered_images_dict,
                                        duplicates_dict=duplicate_images_dict)

    # elif args.sampling_method == 'crossentropy':
    #     mined_split = crossentropy_sampler(root_folder=root_folder, metrics=metrics, num_samples=args.num_samples, 
    #                                 model=model, preprocess=preprocess, threshold=args.sampling_threshold, 
    #                                 name_type=args.prompt_name, pre_extracted_feats = pre_extracted_feats,
    #                                 duplicates_dict=duplicate_images_dict) 

    # elif args.sampling_method == 'totalentropy':
    #     zeroshot_weights = features.prompt_sampler(prompt_tensors_dict[args.prompt_name], logger, sample_by='mean')            
    #     head = MyLinear(weights=zeroshot_weights, bias=False)
    #     head.to(device=device)
    #     head.eval()

    #     mined_split = totalentropy_sampler(root_folder=root_folder, metrics=metrics, num_samples=args.num_samples, 
    #                                 model=model, preprocess=preprocess, threshold=args.sampling_threshold, 
    #                                 name_type=args.prompt_name, pre_extracted_feats = pre_extracted_feats,
    #                                 duplicates_dict=duplicate_images_dict, head=head)

    #----------------- Feature preparation  
    feature_list = mined_split['feature_list']
    label_list = mined_split['label_list']
    file_list = mined_split['file_list']
    
    logger.info(f'len(file_list): {len(file_list)}')
    # concatenate
    final_file_list = []
    for file_path in file_list:
        final_file_list.extend(file_path)
    logger.info(f'len(final_file_list): {len(final_file_list)}')
    sample_ct = len(final_file_list)

    # concatenate the feature_list and label_list
    feature_tensor = torch.cat(feature_list, dim=0)
    labels_tensor = torch.cat(label_list, dim=0)
    logger.info(f'feature_tensor.shape: {feature_tensor.shape}')
    logger.info(f'labels_tensor.shape: {labels_tensor.shape}')

    feature_dict = {'image_features': feature_tensor, 'labels': labels_tensor}

    # # save the feature_dict to torch pth file
    # with open(pre_extracted_path, 'wb') as f:
    #     torch.save(feature_dict, f)
    # logger.info(f'Saved feature_dict to: {pre_extracted_path}')

    # save the file_list to text file, following the finetuning format
    file_list_path = save_sample_file_list(args, final_file_list, labels_tensor)

    # save num_imgs_sampled_dict to json
    num_imgs_sampled_fn = f'{args.output_folder}/{args.prefix}_num_imgs_sampled.json'
    with open(num_imgs_sampled_fn, 'w') as f:
        json.dump(num_imgs_sampled_dict, f, indent=4)

    return file_list_path, sample_ct


if __name__ == '__main__':

    time_start = time()    
    parser = argparse.ArgumentParser(description='Arguments for script.')
    parser.add_argument('--prefix', type=str, default=None, help='prefix.txt and foldername')
    parser.add_argument('--dataset', type=str, default='semi-aves', help='Dataset name.')
    parser.add_argument('--root', type=str, default=ROOT, help='Root directory for storing mined data.')
    parser.add_argument('--model_cfg', type=str, default='vitb32_openclip_laion400m', 
                        choices=['vitb32_openclip_laion400m', 'vitb32_openclip_laion2b', 
                                 'vitb32_clip', 'vitb16_clip'],
                        help='ViT Transformer arch.')
    parser.add_argument('--database', type=str, default='LAION400M', help='Database from which images are mined.')
    parser.add_argument('--prompt_name', type=str, default='alternates', # note here we use alternates to use average text embedding of all synonyms
                        choices=['most_common_name', 'alternates', 'name'], 
                        help='What label to use for making text prompts.')    

    parser.add_argument('--sampling_method', type=str, default='T2T-rank', 
                        choices=['Random', 'Random-I2I', 'T2T-rank', 'T2T-rank-T2I-tshd', 'I2I-rank',
                                'T2I-rank', 'crossentropy', 'totalentropy',
                                'I2T-rank', 'I2T-tshd', 'T2T-rank-I2T-tshd', 'T2T-rank-I2I-tshd',
                                ], 
                        help='Sampling images randomly or ranked by CLIP(img, prompt) similarity.')
    
    parser.add_argument('--sampling_threshold', type=float, default=0.0, help='random sampling images over a certain CLIP T2I or I2I threshold.')
    parser.add_argument('--num_samples', type=int, default=500, help='Number of images that are to be sampled.')
    parser.add_argument('--zeroshot_img_filter', action='store_true', default=False, help='zeroshot CLIP image filtering.')
    parser.add_argument('--image_dedup', action='store_true', default=False, help='CLIP image deduplication by filtering duplicates with img sim > 0.9.')
    parser.add_argument('--recal_prompt', action='store_true', default=False, help='Recalculate the prompt tensors.')
    parser.add_argument('--seed', type=int, default=0, help='Seed for random number generator.')
    parser.add_argument('--log_mode', type=str, default='both', choices=['console', 'file', 'both'], help='where to log.')

    args = parser.parse_args()

    if not os.path.exists('output'):
        os.mkdir('output')
    
    # set the seed of random number generator
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    case_name = f'{args.dataset}_{args.model_cfg}_{args.prefix}'
    args.case_name = case_name
    args.output_folder = f'output/{case_name}'
    if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)

    ## setup logger
    logger = get_logger(args.output_folder, 'sampling', 'both')
    logger.info(f'case_name: {case_name}')

    # print args
    for arg in vars(args):
        logger.info(f'{arg} = {getattr(args, arg)}')

    dataset_root = f'{args.root}/{args.dataset}'
    # metric_fn = f'{dataset_root}/{args.dataset}_metrics-{args.database.upper()}.json' 
    metric_fn = f'../data/{args.dataset}/{args.dataset}_metrics-{args.database.upper()}.json' 

    with open(metric_fn, 'r') as f:
        metrics = json.load(f)
    logger.info(f'Loaded metrics from: {metric_fn}')
    logger.info(f'len(metrics): {len(metrics)}')

    # load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, preprocess, tokenizer = get_engine(model_cfg=args.model_cfg, device=device)
    model.eval()

    # calculate the prompt embeddings for all classes using different label types
    prompt_tensors_dict, text_prompts_dict = cal_prompt_tensors(args, logger, model, tokenizer, metrics, dataset_root)

    #----------------- sampling -----------------#
    file_list_path, sample_ct = sampling(args, logger, model, preprocess, metrics, dataset_root)
    logger.info(f'sample_ct: {sample_ct}')
    logger.info(f'file_list_path: {file_list_path}')
    logger.info(f'Done, time: {round(time()-time_start)} seconds.')