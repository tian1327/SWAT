import torch
import torch.utils.data as data
import numpy as np
import os
from torchvision.datasets import folder as dataset_parser
import json


def make_dataset(dataset_root, split, task='All', pl_list=None):

    split_file_path = os.path.join(dataset_root, split+'.txt')
    print('split_file_path: ', split_file_path)

    with open(split_file_path, 'r') as f:
        img = f.readlines()

    if task == 'semi_fungi':
        img = [x.strip('\n').rsplit('.JPG ') for x in img]
        print(img)

    # elif task[:9] == 'semi_aves':
    else:
        img = [x.strip('\n').rsplit() for x in img]

    ## Use PL + l_train - Pseudo-Labels.
    if pl_list is not None:
        if task == 'semi_fungi':
            pl_list = [x.strip('\n').rsplit('.JPG ') for x in pl_list]
        else:
            pl_list = [x.strip('\n').rsplit() for x in pl_list]
        
        img += pl_list

    for idx, x in enumerate(img):
        if task == 'semi_fungi':
            img[idx][0] = os.path.join(dataset_root, x[0] + '.JPG')
        else:
            img[idx][0] = os.path.join(dataset_root, x[0])
        img[idx][1] = int(x[1])

    classes = [x[1] for x in img]

    num_classes = len(set(classes)) 
    print('# of images in {}: {}'.format(split, len(img)))

    return img, num_classes


class iNatDataset(data.Dataset):
    def __init__(self, dataset_root, split, task='All', transform=None,
            loader=dataset_parser.default_loader, pl_list=None, return_name=False, 
            return_text=False, prompts=[], num_prompts = 5):
        
        self.loader = loader
        self.dataset_root = dataset_root
        self.task = task

        self.imgs, self.num_classes = make_dataset(self.dataset_root, split, self.task, pl_list)

        self.transform = transform
        self.split = split
        self.return_name = return_name
        self.return_text = return_text
        if self.task == 'semi-inat-2021':
            self.label2taxaid = json.load(open(os.join(dataset_root,'label2taxaid.json')))

        # self.categories = json.load(open('./data/semi-inat-2021/categories.json'))['categories']
        self.num_prompts = num_prompts
        self.prompts = prompts
        # self.prompt_padding = 50
        
        # if self.return_text:
        #    text = []
        #    for c in range(len(self.categories)):
        #        text.append(f"This is a photo of the species: {self.categories[c]['species']}.")
        #    self.text = clip.tokenize(text)

    def __getitem__(self, index):
        path, target = self.imgs[index]
        # target_text = f"This is a photo of the species: {self.categories[target]['species']}."

        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
            
        if self.return_text:
            if self.split == 'l_train' or self.split == 'l_train+val':
                return img, target, self.prompts[str(target)]['all'][:self.num_prompts]  
            
                # return img, target, self.prompts[target]['all'][0] 
                # if self.num_prompts == -1:

                #     print("+++++")
                #     print('img.shape: ', img.shape)
                #     print('target: ', target)
                #     print('len(self.prompts): ', len(self.prompts))
                #     print('self.prompts[target][all].shape: ', self.prompts[target]['all'].shape)

                #     return img, target, self.prompts[target]['all'][0]
                # else:
                #     return img, target, self.prompts[target]['all'][:self.num_prompts]         
                   
                # original_length = self.prompts[target]['all'].shape[0]
                # padded_prompts = torch.nn.functional.pad(self.prompts[target]['all'], pad=(0, 0,0,0, 0, self.prompt_padding - self.prompts[target]['all'].shape[0]))
                # if self.return_all:
                    # return img, target, padded_prompts, original_length
                # return img, target,random.choice(self.prompts[target]['all']) # random.choice([self.prompts[target]['all'], self.prompts[target]['mean']], p=[0.8, 0.2])
                
            elif self.split == 'val' or \
                self.split == 'u_train' or \
                self.split == 'u_train_in' or \
                self.split == 'u_train_in_ST-Hard' or \
                self.split == 'test':
            
                return img, target, self.prompts[str(target)]['all'][0]

        elif self.task == 'semi-inat-2021':
            kingdomId = self.label2taxaid[str(target)]['kingdom']
            phylumId = self.label2taxaid[str(target)]['phylum']
            classId = self.label2taxaid[str(target)]['class']
            orderId = self.label2taxaid[str(target)]['order']
            familyId = self.label2taxaid[str(target)]['family']
            genusId = self.label2taxaid[str(target)]['genus']
            if self.return_name:
                return img, target, kingdomId, phylumId, classId, orderId, familyId, genusId, path
            else:
                return img, target, kingdomId, phylumId, classId, orderId, familyId, genusId
        
        return img, target 
        

    def __len__(self):
        return len(self.imgs)

    def get_num_classes(self):
        return self.num_classes