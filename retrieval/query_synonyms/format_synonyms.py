# for semi-aves and cub200, add sname after text filtering

# import openai
import json
# import re
# import time
# from tqdm import tqdm
# import os
# from imagenet_labels import clip_imagenet_classes
from prompts import class_map, cub200_classes
from query_synonyms import get_aves_class_lst, get_cub200_class_lst, clean_text


def add_names(dataset, class_lst):

    with open(f'output/{dataset}_synonyms_filtered.json', 'r') as f:
        data = json.load(f)
    
    for idx, info in data.items():
        alternatives_set  = set(info['synonyms_filtered'].keys())

        if dataset == 'semi-aves':
            sname = class_lst[int(idx)][0]
            sname_clean = clean_text(sname)
            cname = class_lst[int(idx)][1]
            cname_clean = clean_text(cname)

            if sname_clean not in alternatives_set:
                alternatives_set.add(sname_clean)
            
            if cname_clean not in alternatives_set:
                alternatives_set.add(cname_clean)
            info['common_name'] = cname
        else:
            class_name = class_lst[int(idx)]
            class_name_clean = clean_text(class_name)
            if class_name_clean not in alternatives_set:
                alternatives_set.add(class_name_clean)
        
        info['synonyms_final'] = {}
        for alt in alternatives_set:
            info['synonyms_final'][alt] = 0
    
    # save the updated dict
    with open(f'output/{dataset}_synonyms_filtered_final.json', 'w') as f:
        json.dump(data, f, indent=4)
        

if __name__ == '__main__':

    # build the target dataset dict by picking items from the class_map dict
    target_dataset_dict = {}
    targets = [
            # 'caltech-101', 
            # 'dtd', 'eurosat_clip', 'fgvc-aircraft-2013b-variants102',
            # 'oxford-flower-102', 'oxford-iiit-pets', 'sun397', 
            # 'food-101', 
            # 'stanford-cars'
            'semi-aves',
            # 'cub200'
            ]
    
    for key in targets:
        if key == 'semi-aves':
            target_dataset_dict[key] = get_aves_class_lst()
        elif key == 'cub200':
            target_dataset_dict[key] = get_cub200_class_lst()
        else: # for other datasets 
            target_dataset_dict[key] = class_map[key]

    print('len(target_dataset_dict): ', len(target_dataset_dict))
    
    # loop through each target dataset and run the procedure
    for dataset, class_lst in target_dataset_dict.items():
        print(f'\n{dataset}: {len(class_lst)}')

        add_names(dataset, class_lst)

    print('Done!')