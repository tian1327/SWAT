import sys
sys.path.append("..")  # Adds higher directory to python modules path.
from utils.datasets.dataset_utils import MinedDataset
import torch
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
import argparse
import pickle
from utils.extras import get_engine
import yaml


# read the retrieved path from the config.yml
with open('../config.yml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    ROOT = config['retrieved_path']  

MINED_DATASET_ROOT_DICT = {
    # 'imagenet_1k': 'data/imagenet_1k/retrieved_1m-laion400m-alternates-random',
    # 'semi-aves': 'data/semi-aves/retrieved_1m-laion400m-alternates-random',
    'semi-aves': f'{ROOT}/semi-aves/semi-aves_retrieved_LAION400M-all_synonyms-all',
    'fgvc-aircraft': f'{ROOT}/fgvc-aircraft/fgvc-aircraft_retrieved_LAION400M-all_synonyms-all',
    'eurosat': f'{ROOT}/eurosat/eurosat_retrieved_LAION400M-all_synonyms-all',
    'dtd': f'{ROOT}/dtd/dtd_retrieved_LAION400M-all_synonyms-random',
    # 'dtd': f'{ROOT}/dtd/dtd_retrieved_LAION400M-all_synonyms-all', # this is for retrieval with texture, no better
    'flowers102': f'{ROOT}/flowers102/flowers102_retrieved_LAION400M-all_synonyms-random',
    'oxford_pets': f'{ROOT}/oxford_pets/oxford_pets_retrieved_LAION400M-all_synonyms-random',
    'food101': f'{ROOT}/food101/food101_retrieved_LAION400M-all_synonyms-random',
    'stanford_cars': f'{ROOT}/stanford_cars/stanford_cars_retrieved_LAION400M-all_synonyms-random',
    'imagenet': f'{ROOT}/imagenet/imagenet_retrieved_LAION400M-all_synonyms-random',
    }

CAPTION_MAP_DICT = {
    'semi-aves': f'{ROOT}/semi-aves/semi-aves_metadata-all-0.0-LAION400M.map',
    'fgvc-aircraft': f'{ROOT}/fgvc-aircraft/fgvc-aircraft_metadata-all-0.0-LAION400M.map',
    'eurosat': f'{ROOT}/eurosat/eurosat_metadata-all-0.0-LAION400M.map',
    'dtd': f'{ROOT}/dtd/dtd_metadata-random-0.0-LAION400M.map', 
    # 'dtd': f'{ROOT}/dtd/dtd_metadata-all-0.0-LAION400M.map', # this is for retrieval with texture, no better
    'flowers102': f'{ROOT}/flowers102/flowers102_metadata-random-0.0-LAION400M.map',
    'oxford_pets': f'{ROOT}/oxford_pets/oxford_pets_metadata-random-0.0-LAION400M.map',
    'food101': f'{ROOT}/food101/food101_metadata-random-0.0-LAION400M.map',
    'stanford_cars': f'{ROOT}/stanford_cars/stanford_cars_metadata-random-0.0-LAION400M.map',
    'imagenet': f'{ROOT}/imagenet/imagenet_metadata-random-0.0-LAION400M.map',
}


def extract_mined_feats(model, dataloader, tokenizer):

    img_feats_lst, labels_lst, captions_lst = [], [], []
    filepath_lst = []
    model.eval()

    # +++++ calculate the captions_feats_store
    for data in tqdm(dataloader, desc='Extract captions features'):
        imgs, labels, file_path, captions = data
        imgs = imgs.cuda()
        labels = labels.long()

        model.eval()
        with torch.no_grad():
            captions_tokens = tokenizer(captions)
            captions_feats = model.encode_text(captions_tokens.cuda())
            captions_feats /= captions_feats.norm(dim=-1, keepdim=True) # Normalization.

        labels_lst.append(labels.cpu())
        captions_lst.append(captions_feats.cpu())
        filepath_lst.extend(file_path)
        # break

    captions_feats_store = torch.cat(captions_lst, dim=0)
    labels_store = torch.cat(labels_lst, dim=0)
    print('captions_feats_store.shape:', captions_feats_store.shape)
    print('labels_store.shape:', labels_store.shape)
    print('len(filepath_lst):', len(filepath_lst))

    result = {'caption_features': captions_feats_store,
                'labels': labels_store,
                'filepath': filepath_lst}

    # +++++ calculate the img_feats_store
    for data in tqdm(dataloader, desc='Extract image features'):
        imgs, labels, file_path, captions = data
        imgs = imgs.cuda()
        labels = labels.long()

        model.eval()
        with torch.no_grad():
            img_feats = model.encode_image(imgs)
            img_feats /= img_feats.norm(dim=-1, keepdim=True) # Normalization.

        img_feats_lst.append(img_feats.cpu())

    img_feats_store = torch.cat(img_feats_lst, dim=0)
    print('img_feats_store.shape:', img_feats_store.shape)

    result['image_features']=img_feats_store
 
    return result


def extract_mined_feats_batch(model, dataloader, tokenizer, save_dir, batch_size=100):

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    img_feats_lst, labels_lst, captions_lst, filepath_lst = [], [], [], []
    model.eval()
    
    # +++++ calculate the captions_feats_store
    batch_idx = 0
    for data in tqdm(dataloader, desc='Extract captions features'):
        imgs, labels, file_path, captions = data
        imgs = imgs.cuda()
        labels = labels.long()

        with torch.no_grad():
            captions_tokens = tokenizer(captions)
            captions_feats = model.encode_text(captions_tokens.cuda())
            captions_feats /= captions_feats.norm(dim=-1, keepdim=True) # Normalization.

        labels_lst.append(labels.cpu())
        captions_lst.append(captions_feats.cpu())
        filepath_lst.extend(file_path)

        batch_idx += 1 
        if batch_idx % batch_size == 0:
            captions_feats_store = torch.cat(captions_lst, dim=0)
            labels_store = torch.cat(labels_lst, dim=0)
            
            # Save to disk
            torch.save({'caption_features': captions_feats_store, 
                        'labels': labels_store, 
                        'filepath': filepath_lst},
                       os.path.join(save_dir, f"captions_batch_{batch_idx}.pth"))
            print(f'Saved captions_batch_{batch_idx}.pth')

            # Clear lists to free memory
            captions_lst, labels_lst, filepath_lst = [], [], []   

    # Save the last batch
    if len(captions_lst) > 0:
        captions_feats_store = torch.cat(captions_lst, dim=0)
        labels_store = torch.cat(labels_lst, dim=0)
        torch.save({'caption_features': captions_feats_store, 
                    'labels': labels_store, 
                    'filepath': filepath_lst},
                   os.path.join(save_dir, f"captions_batch_{batch_idx}.pth"))
    
    # combine the saved batches
    captions_lst, labels_lst, filepath_lst = [], [], []
    for fn in os.listdir(save_dir):
        if fn.startswith('captions_batch_'):
            data = torch.load(os.path.join(save_dir, fn))
            captions_lst.append(data['caption_features'])
            labels_lst.append(data['labels'])
            filepath_lst.extend(data['filepath'])         

    captions_feats_store = torch.cat(captions_lst, dim=0)
    labels_store = torch.cat(labels_lst, dim=0)
    print('captions_feats_store.shape:', captions_feats_store.shape)
    print('labels_store.shape:', labels_store.shape)
    print('len(filepath_lst):', len(filepath_lst))

    result = {'caption_features': captions_feats_store,
                'labels': labels_store,
                'filepath': filepath_lst}


    # +++++ calculate the img_feats_store
    batch_idx = 0
    for data in tqdm(dataloader, desc='Extract image features'):
        imgs, labels, file_path, captions = data
        imgs = imgs.cuda()
        labels = labels.long()

        model.eval()
        with torch.no_grad():
            img_feats = model.encode_image(imgs)
            img_feats /= img_feats.norm(dim=-1, keepdim=True) # Normalization.
        
        img_feats_lst.append(img_feats.cpu())

        batch_idx += 1
        if batch_idx % batch_size == 0:
            img_feats_store = torch.cat(img_feats_lst, dim=0)
            torch.save(img_feats_store, os.path.join(save_dir, f"images_batch_{batch_idx}.pth"))
            print(f'Saved images_batch_{batch_idx}.pth')

            img_feats_lst = []
    
    # Save the last batch
    if len(img_feats_lst) > 0:
        img_feats_store = torch.cat(img_feats_lst, dim=0)
        torch.save(img_feats_store, os.path.join(save_dir, f"images_batch_{batch_idx}.pth"))

    # combine the saved batches
    img_feats_lst = []
    for fn in os.listdir(save_dir):
        if fn.startswith('images_batch_'):
            data = torch.load(os.path.join(save_dir, fn))
            img_feats_lst.append(data)
        
    img_feats_store = torch.cat(img_feats_lst, dim=0)
    print('img_feats_store.shape:', img_feats_store.shape)

    result['image_features']=img_feats_store

    # remove the saved dir
    os.system(f'rm -rf {save_dir}')
 
    return result


def extract_test_feats(model, dataloader):

    img_feats_lst, labels_lst = [], []
    filepath_lst = []

    for data in tqdm(dataloader):
        # imgs, labels, path = data
        imgs, labels = data
        path = ''

        imgs = imgs.cuda()
        labels = labels.long()

        model.eval()
        with torch.no_grad():
            img_feats = model.encode_image(imgs)
            img_feats /= img_feats.norm(dim=-1, keepdim=True) # Normalization.

        img_feats_lst.append(img_feats.cpu())
        labels_lst.append(labels.cpu())
        filepath_lst.extend(path)

    img_feats_store = torch.cat(img_feats_lst, dim=0)
    labels_store = torch.cat(labels_lst, dim=0)

    print('img_feats_store.shape:', img_feats_store.shape)
    print('labels_store.shape:', labels_store.shape)
    print('len(filepath_lst):', len(filepath_lst))

    result = {'image_features': img_feats_store, 
                'labels': labels_store,
                'filepath': filepath_lst}
    
    return result


def pre_extract_mined_fea(args, dataset_name, dataset_root, caption_path, 
                        model_cfg, model, preprocess, tokenizer):    
    
    print('\nextracting features for', dataset_name, model_cfg)
    print('dataset_root:', dataset_root)

    # build dataset
    with open(caption_path, 'rb') as f:
        caption_map = pickle.load(f)            

    dataset = MinedDataset(dataset_root=dataset_root, transform=preprocess, caption_map=caption_map)

    # build dataloader
    val_loader = DataLoader(dataset, batch_size=args.bsz, shuffle=False, 
                            num_workers=args.num_workers, drop_last=False, pin_memory=True)

    # +++++ extract the image + caption features
    # dataset_dict = extract_mined_feats(model, val_loader, tokenizer)
    dataset_dict = extract_mined_feats_batch(model, val_loader, tokenizer, save_dir=f'{args.root}/{dataset_name}/temp')      

    # +++++ save the dataset_dict
    save_root = f'{args.root}/{dataset_name}' # here since we mined with the alternates
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    output_fn = f'{save_root}/{dataset_name}_{model_cfg}_mined.pth'

    torch.save(dataset_dict, output_fn)
    print(f'\nSaved to extracted feature to: {output_fn}')



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Arguments for script.')
    parser.add_argument('--dataset', type=str, default='semi-aves', help='Dataset name.')
    parser.add_argument('--root', type=str, default=ROOT, help='Root directory for storing mined data.')
    parser.add_argument('--model_cfg', type=str, default='vitb32_openclip_laion400m', 
                        choices=['vitb32_openclip_laion400m', 'vitb32_openclip_laion2b', 
                                 'vitb32_clip', 'vitb16_clip'],
                        help='ViT Transformer arch.')
    parser.add_argument('--bsz', type=int, default=1024, help='Batch size.')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of workers for dataloader.')
    
    args = parser.parse_args()

    # print out the arguments
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')

    # load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, preprocess, tokenizer = get_engine(model_cfg=args.model_cfg, device=device)
    model.eval()

    # extract the mined set features
    dataset_name = args.dataset
    pre_extract_mined_fea(args=args,
                        dataset_name=dataset_name, 
                        dataset_root=MINED_DATASET_ROOT_DICT[dataset_name],
                        caption_path=CAPTION_MAP_DICT[dataset_name],
                        model_cfg=args.model_cfg,
                        model=model, preprocess=preprocess, tokenizer=tokenizer
                        )