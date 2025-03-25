import os
import clip
import json
import torch
import random
import argparse
import time
import copy
import pickle
from tqdm import tqdm

def prompt_sampler(prompt_tensors, sample_by='mean'):

    sampled_prompts = []
    for i in prompt_tensors.keys():
        if sample_by == 'mean':
            sampled_prompts.append(prompt_tensors[i]['mean'])
        elif sample_by == 'random':
            sampled_prompts.append(random.choice(prompt_tensors[i]['all']))
        else:
            raise NotImplementedError

    return torch.stack(sampled_prompts, dim=0)



def operate_on_prompt(model, text, operation, tokenize):

    if operation=='encode':
        features = model.encode_text(tokenize(text).cuda())
        features /= features.norm(dim=-1, keepdim=True) # Normalization. +++++
        return features # this is text embedding

    elif operation == 'tokenize':
        tokens = tokenize(text)
        return tokens # this is tokenized text

# Pre-extract all features and pass to data-loader.
def get_text_features(model, prompt_dict, tokenize, operation='encode'):

    tensor_dict = {}
    model.eval()
    with torch.no_grad():
        for key, info in prompt_dict.items():
            # key is the class_id
            source = {}
            prompts = []
            for prompt in info['corpus']:
                prompts.append(prompt)

            stacked_tensor = operate_on_prompt(model, prompts, operation, tokenize)
            stacked_tensor.cpu()

            source['all'] = stacked_tensor

            # also compute the mean tensor if operation is encode
            if operation == 'encode':
                mean_tensor = torch.mean(stacked_tensor, dim=0)
                mean_tensor /= mean_tensor.norm(dim=-1, keepdim=True)
                source['mean'] = mean_tensor

            tensor_dict[key] = source

    return tensor_dict

# Pre-extract all features and pass to data-loader.
"""
def get_text_features(model, prompt_dict, mode, tokenize, prompt_mode = 'All', operation='encode'):

    tensor_list = []
    with torch.no_grad():

        for i, key in enumerate(prompt_dict):
            obj = prompt_dict[key]
            class_id = obj['class_id']
            source = {}

            # no prompts means just using the s-name or c-name depending on the mode
            if prompt_mode == 'no_prompts' or (prompt_mode == 'ChatGPT' and obj['data_source'] != prompt_mode): # Helpful for validation.

                # use the scientific name or common name depending on the mode
                if mode == 'c-name':
                    name = obj['common_name']
                elif mode == 's-name':
                    name = obj['species']
                else:
                    raise ValueError(f'Invalid mode: {mode}')

                text = f'Here is a photo of the: {name}.'  # single prompt

                features = operate_on_prompt(model, text, operation, tokenize) # encode or tokenize

                # print('features.shape: ', features.shape) #[1, 77]
                source['all'] = features
                source['mean'] = features

                tensor_list.append(source) # list of dict
                continue

            # else, for case when prompt_mode == 'All'
            # or other cases, including when prompt_mode == 'ChatGPT' and obj['data_source'] == prompt_mode i.e. obj['data_source']=='ChatGPT'
            prompts = []
            for prompt in obj['corpus']:
                features = operate_on_prompt(model, prompt, operation, tokenize) # encode or tokenize
                # print('features.shape: ', features.shape) #[1, 77]
                prompts.append(features)

            stacked_tensor = torch.stack(prompts, dim=0)
            source['all'] = stacked_tensor # already stacked tensor/tokens

            if operation == 'encode':
                mean_tensor = torch.mean(stacked_tensor, dim=0) # take the mean of all prompts embeddings
                # normalize the mean tensor
                mean_tensor /= mean_tensor.norm(dim=-1, keepdim=True) # torch.linalg.vector_norm(mean_tensor, dim=-1, keepdim=True)
                source['mean'] = mean_tensor

            tensor_list.append(source)

        return tensor_list
"""

def extract_test_feats(model, dataloader):

    img_feats_lst, labels_lst = [], []

    for data in tqdm(dataloader):
    # for data in dataloader:
        imgs, labels, text, source = data
        imgs = imgs.cuda()
        labels = labels.long()

        model.eval()
        with torch.no_grad():
            img_feats = model.encode_image(imgs)
            img_feats /= img_feats.norm(dim=-1, keepdim=True) # Normalization.

        img_feats_lst.append(img_feats.cpu())
        labels_lst.append(labels.cpu())

    img_feats_store = torch.cat(img_feats_lst, dim=0)
    labels_store = torch.cat(labels_lst, dim=0)
    # print('img_feats_store.shape:', img_feats_store.shape)
    # print('labels_store.shape:', labels_store.shape)

    result = {'image_features': img_feats_store,
                'labels': labels_store}

    return result


# if __name__=='__main__':
#     parser = argparse.ArgumentParser(description='Arguments for script.')
#     parser.add_argument('--model_path', type=str, default='', help='learning rate for optimizer')
#     parser.add_argument('--prompt_source', type=str, default='', help='Prompt Source for feature extraction.')

#     args = parser.parse_args()

#     torch.cuda.empty_cache()
#     prompt_source = json.load(open(f'/scratch/user/shubhamprshr/research/data/semi-inat-2021/{args.prompt_source}'))
#     model, _ = clip.load('ViT-B/32', 'cpu')
#     model = model.float()
#     if args.model_path:
#         ckpt = torch.load(f'/scratch/user/shubhamprshr/research/clip_models/finetuned/{args.model_path}')
#         model.load_state_dict(ckpt['clip'])
#     prompt_tensors = get_text_features(copy.deepcopy(model).cpu(), prompt_source, prompt_mode='All')

#     # model_prompts = json.dumps(prompt_tensors, indent=4)
#     with open(f'{args.model_path}-{args.prompt_source}.pkl', 'wb') as outfile:
#         pickle.dump(prompt_tensors, outfile)
#         # outfile.write(model_prompts)





