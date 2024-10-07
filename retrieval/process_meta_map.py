import pickle
import yaml
import sys

def build_map(fn, fn_out, caption_idx):
    result_dict = {}
    with open(fn, 'rb') as f:
        meta = pickle.load(f)
    
    print('len(meta):', len(meta))

    no_image_ct = 0
    no_caption_ct = 0
    valid_img_ct = 0
    for cls_idx, tuple_lst in meta.items():
        # print(cls_idx, len(tuple_lst), tuple_lst[:3])
        # exit()
        
        for idx, tuple in enumerate(tuple_lst):
            img_idx = tuple[-1]
            img_caption = tuple[caption_idx]
            
            if type(img_idx) != int:
                # print(f'img_idx is not int, {cls_idx}, {img_idx}, {idx}, | {meta[cls_idx][idx]}') # no image index, no image
                no_image_ct += 1          
                # exit()      
                continue

            if type(img_caption) != str:
                # raise ValueError(f'img_caption is not string, {cls_idx}, {img_idx}, {idx}, | {meta[cls_idx][idx]}')
                no_caption_ct += 1
                continue

            if cls_idx not in result_dict:
                result_dict[cls_idx] = {}

            # add to result_dict 
            result_dict[cls_idx][str(img_idx)] = img_caption
            valid_img_ct += 1
    
    print(f'no image count: {no_image_ct}')
    print(f'no caption count: {no_caption_ct}')
    print(f'valid image count: {valid_img_ct}')

    with open(fn_out, 'wb') as f:
        pickle.dump(result_dict, f)
    print(f'write to {fn_out}')


if __name__ == '__main__':

    """
    fn = 'data/imagenet_1k/metadata-random-0.0-laion400m.meta'
    fn_out = 'data/imagenet_1k/metadata-random-0.0-laion400m.map'
    build_map(fn, fn_out, caption_idx=-4)
    """

    """
    fn = 'data/imagenet_1k/metadata-random-0.0-laion2b.meta'
    fn_out = 'data/imagenet_1k/metadata-random-0.0-laion2b.map'
    build_map(fn, fn_out, caption_idx=-5)
    """


    with open('../config.yml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader) 
        root = config['retrieved_path'] 
    
    dataset = sys.argv[1]
    print(f'dataset: {dataset}')

    # semi-aves
    if dataset == 'semi-aves':
        fn = '{root}/semi-aves/semi-aves_metadata-all-0.0-LAION400M.meta' # i lost this one accidentally ;(

    # fgvc-aircraft
    elif dataset == 'fgvc-aircraft':
        fn = f'{root}/fgvc-aircraft/fgvc-aircraft_metadata-all-0.0-LAION400M.meta'

    # eurosat
    elif dataset == 'eurosat':
        fn = f'{root}/eurosat/eurosat_metadata-all-0.0-LAION400M.meta'

    # flowers
    elif dataset == 'flowers':
        fn = f'{root}/flowers102/flowers102_metadata-random-0.0-LAION400M.meta'
    
    # dtd
    elif dataset == 'dtd':
        fn = f'{root}/dtd/dtd_metadata-random-0.0-LAION400M.meta'
        # fn = f'{root}/dtd/dtd_metadata-all-0.0-LAION400M.meta'
    
    # oxfordpets
    elif dataset == 'oxfordpets':
        fn = f'{root}/oxfordpets/oxfordpets_metadata-random-0.0-LAION400M.meta'
    
    # food101
    elif dataset == 'food101':
        fn = f'{root}/food101/food101_metadata-random-0.0-LAION400M.meta'
    
    # stanfordcars
    elif dataset == 'stanfordcars':
        fn = f'{root}/stanfordcars/stanfordcars_metadata-random-0.0-LAION400M.meta'
    
    # imagenet
    elif dataset == 'imagenet':
        fn = f'{root}/imagenet/imagenet_metadata-random-0.0-LAION400M.meta'
    
    else:
        raise ValueError(f'unknown dataset: {dataset}')


    fn_out = fn.replace('.meta', '.map')
    print(f'fn: {fn}')
    print(f'fn_out: {fn_out}')

    assert fn != fn_out

    build_map(fn, fn_out, caption_idx=-4)