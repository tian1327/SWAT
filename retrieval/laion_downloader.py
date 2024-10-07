import os
import requests
import pickle
from laion_parser import LaionParser
from multiprocessing import Pool
from img2dataset import download
from concurrent.futures import ThreadPoolExecutor #, ProcessPoolExecutor
import shutil
import pandas as pd
from PIL import Image #, UnidentifiedImageError
from io import BytesIO
import argparse
import time
import random
import json
import yaml


def validate_and_save_image(base_path,response_content, img_count):
    valid_image = False
    try:
        image = Image.open(BytesIO(response_content))

        width, height = image.size
        if max(width,height) > 80: # Avoid very small images, they tend to be empty.
            img_path = os.path.join(base_path,f'{img_count}.JPEG')
            image.convert('RGB').save(img_path, 'JPEG')
            img_count+=1
            valid_image = True
    except Exception as e:
        s = "error downloading"
    return img_count, valid_image


def worker(args):
    laion_parser = LaionParser('LAION400M.db', data_source='./')
    item, download_folder = args
    key, metadata =  item
    folder_path = os.path.join(download_folder, str(key))
    os.makedirs(folder_path, exist_ok=True)
    downloads_dict = {key: []}

    chunk_size = 1000
    target_img_count = 1000
    img_count = 0
    start = time.time()
    if RANDOM:
        random.shuffle(metadata)
    for sample in metadata:
        shard, rowid, text, relevance_score = sample
        if relevance_score == 'yes' or isinstance(relevance_score, float): # LLAMA based or similarity based.
            result = laion_parser.find_by_id(rowid=rowid, shard=shard, column='URL')
            url,nsfw = result
            if nsfw == 'NSFW':
                continue
            try:
                response = requests.get(url, timeout=15)
                if response.status_code == 200: # Successfully mined images.
                    img_count, valid_image = validate_and_save_image(base_path=folder_path, 
                                                        response_content=response.content,
                                                        img_count=img_count)
                    # Only valid images are recorded.
                    if valid_image:
                        downloads_dict[key].append((shard, rowid, text, url, relevance_score))
            except Exception as e:
                # HTTP Exception
                continue
                # print('Exception occured, continuing forward.', key, url)
        if img_count == 1000: # Max 1000 images. However will need to handle the tail.
            break
    print(f'Downloaded for {key} - {img_count} -', time.time()-start)
    laion_parser.conn.close()
    return downloads_dict


def download_and_save_imgs(retrieved_captions: dict, download_folder: str, dataset: str, database:str ,name_type:str, ):
    # num_processes = min(os.cpu_count(), 64)
    os.makedirs(download_folder, exist_ok=True)
    params = [((key,value), download_folder) for (key,value) in retrieved_captions.items()]
    
    # params = [((key,value), download_folder) for (key,value) in retrieved_captions.items()]
    # with ProcessPoolExecutor(max_workers=num_processes) as executor:
    #     downloads = list(executor.map(worker, params))
    with Pool(32) as pool:
        downloads = pool.map(worker, params)

    all_downloads = {}
    for item in downloads:
        all_downloads.update(item)  
    
    meta_data_dir = os.path.join(dataset, database, name_type)
    if not os.path.exists(meta_data_dir):
        os.makedirs(meta_data_dir)
    with open(f'./{meta_data_dir}/download-meta-data.pkl', 'wb') as f:
        pickle.dump(all_downloads, f)


def move_file(root_folder, child_folder, filename, file_count, clas):

    file_path = os.path.join(child_folder, filename)
    file_id, ext = filename.split('.')
    dest_path = os.path.join(root_folder, clas, f'{str(file_count)}.{ext}')

    if os.path.isfile(file_path) and not filename.endswith('.json'):
        shutil.move(file_path, dest_path)


def img2dataset_download(parquet_path: str, download_dir):
    
    # if not os.path.exists(dir_name):
    print('Downloading', parquet_path)
    print('download_dir:', download_dir)
    if os.path.exists(download_dir):
        print('Already Downloaded images')
    else:
        os.makedirs(download_dir, exist_ok=True)
        print(f'making download_dir: {download_dir}')
        download(
            processes_count=16,
            thread_count=16,
            url_list=parquet_path,
            resize_mode='no',
            encode_quality=100,
            input_format='parquet',
            output_format='files',
            min_image_size=85,
            number_sample_per_shard=2000000,
            output_folder=download_dir
        )   
        print(f'Downloaded - {parquet_path}')

def create_parquet(root, dataset, retrieved_captions, sampling='ranked', sampling_threshold=0.0, database='LAION400M',):

    metadata_path = os.path.join(f'{root}/{dataset}', f'{dataset}_metadata-{sampling}-{sampling_threshold}-{database}.meta')
    urls_path = f'{root}/{dataset}/{dataset}-urls-{sampling}-{sampling_threshold}-{database}.parquet'

    if os.path.exists(urls_path):
        print("URL parquet exists.")
        return urls_path, metadata_path
    
    dataset_urls = []
    download_metadata = {}
    for i, key in enumerate(retrieved_captions.keys()):
        # Get 10000 URLs, should be fine since we only need 1000 downloads.
        # print(retrieved_captions[key][:10])
        start = time.time()
        if sampling == 'random':
            retrieved_captions[key] = list(retrieved_captions[key])
            random.shuffle(retrieved_captions[key])
            metadata = retrieved_captions[key][:2000]
        elif sampling == 'all':
            metadata = list(retrieved_captions[key])
        elif sampling == 'few':
            metadata = list(retrieved_captions[key])[:5]
        elif sampling == 'ranked': # this is for T2T ranking before downloading
            metadata = list(sorted(retrieved_captions[key][:2000], reverse=True, key= lambda x: x[-1]))
        else:
            raise ValueError('Sampling method not defined.')

        with ThreadPoolExecutor(16) as executor:
            metadata = list(executor.map(process_sample, metadata))
        
        urls = [{'class': key, 'url': metadata_i[-2]} for metadata_i in metadata]

        download_metadata[key] = metadata
        # print(download_metadata[key])
        dataset_urls.extend(urls)
        print(f'{i}, key: {key}, {round(time.time() - start,0)} seconds')

    df = pd.DataFrame(dataset_urls)
    df.to_parquet(urls_path, index=False)
    
    with open(metadata_path, 'wb') as f:
        pickle.dump(download_metadata, f)

    return urls_path, metadata_path


def process_sample(sample):

    laion_parser = LaionParser('LAION400M.db', data_source='database/')

    if len(sample) == 3: # this is just a hotfix from shubham for previous structure change
        shard = sample[0]
        rowid = sample[1]
    else:
        shard = sample[1]
        rowid = sample[2]

    # print(sample)
    # print(len(sample),shard, rowid)
    url, nsfw = laion_parser.find_by_id(rowid=rowid, shard=shard, column='URL')
    laion_parser.conn.close()
    del laion_parser

    return (*sample, url, nsfw)

def restructure_download(parquet_path, download_dir, metadata_path=''):
    
    df = pd.read_parquet(parquet_path)
    # 000000000000
    classes = df['class'].unique()
    print('len(classes):', len(classes))
    child_folder = os.path.join(download_dir, '00000')
    print('Restructuring download folder:', child_folder)

    metadata = pickle.load(open(metadata_path, 'rb'))
    downloaded_ct = {}

    for clas in classes:
        class_path = os.path.join(download_dir, str(clas))
        os.makedirs(class_path, exist_ok=True)
        row_ids = df[df['class'] == clas].index.tolist()
        file_count = 0
        for i, row_id in enumerate(row_ids):
            fomatted_rowid = "{:012}".format(row_id)
            # print(i,row_id)
            # if os.path.exists(os.path.join(download_dir, child_folder, f'{fomatted_rowid}.jpg')):
            if os.path.exists(os.path.join(child_folder, f'{fomatted_rowid}.jpg')):
            
                metadata[str(clas)][i]=(*metadata[str(clas)][i], file_count)
                move_file(download_dir, child_folder, f'{fomatted_rowid}.jpg', file_count, clas)
                file_count +=1
        print(f'Class {clas} - {file_count} images.')
        downloaded_ct[clas] = file_count 

    # shutil.rmtree(child_folder)

    with open(metadata_path,'wb') as f:
        pickle.dump(metadata, f)
        
    print('Restructuring completed.')

    return downloaded_ct
        

if __name__ == '__main__':

    # retrieved_captions = pickle.load(open('./relevant_captions_similarity','rb'))
    parser = argparse.ArgumentParser(description='Arguments for script.')
    # parser.add_argument('--arch', type=str, default='ViT-B/32', help='ViT Transformer arch.')
    parser.add_argument('--database', type=str, default='LAION400M')
    parser.add_argument('--dataset', type=str, default='imagenet_1k', help='Dataset name.')
    parser.add_argument('--root', type=str, default='', help='Root directory for storing mined data.')
    parser.add_argument('--name_type', type=str, default='all_synonyms', choices=['name', 'most_freq_synonym', 'all_synonyms'])
    parser.add_argument('--sampling', type=str, default='all', choices=['ranked', 'random', 'all', 'few'])
    parser.add_argument('--sampling_threshold', type=float, default=0.0)
    parser.add_argument('--tag', type=str, default=None)

    args = parser.parse_args()
    random.seed(0)

    # read the retrieved path from the config.yml
    with open('../config.yml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        args.root = config['retrieved_path']         

    print('Starting loading.')

    if args.dataset == 'eurosat':
        caption_file = f'{args.root}/{args.dataset}/{args.dataset}_mined_captions-{args.database}-satellite.pkl'
    # elif args.dataset == 'dtd':
    #     caption_file = f'{args.root}/{args.dataset}/{args.dataset}_mined_captions-{args.database}-texture.pkl'
    else:
        caption_file = f'{args.root}/{args.dataset}/{args.dataset}_mined_captions-{args.database}.pkl'

    retrieved_captions = pickle.load(open(caption_file,'rb'))
    print('Loaded relevant captions:', caption_file)
    
    # print(retrieved_captions['4'])
    # quit()
    
    download_dir = f'{args.root}/{args.dataset}/{args.dataset}_retrieved_{args.database}-{args.name_type}-{args.sampling}{f"-{args.tag}" if args.tag is not None else ""}'

    start = time.time()
    parquet_path, metadata_path = create_parquet(args.root,
                                                 args.dataset, 
                                                 retrieved_captions=retrieved_captions, 
                                                 sampling=args.sampling, 
                                                 sampling_threshold=args.sampling_threshold, 
                                                 database=args.database)
    
    img2dataset_download(parquet_path=parquet_path, download_dir=download_dir)

    downloaded_ct = restructure_download(parquet_path=parquet_path, download_dir=download_dir, metadata_path=metadata_path)

    # save the downloaded count
    fp = f'{args.root}/{args.dataset}/{args.dataset}_downloaded_ct-{args.database}-{args.name_type}-{args.sampling}.json'
    with open(fp, 'w') as f:
        json.dump(downloaded_ct, f, indent=4)
    print(f'saved downloaded count to {fp}')

    # download_and_save_imgs(retrieved_captions=retrieved_captions, 
    #                        download_folder=download_dir,
    #                        dataset=args.dataset, 
    #                        database=args.database,
    #                        name_type=args.name_type,
    #                        )

    print(f'Finished in {round((time.time()-start)/60,0)} minutes.')
