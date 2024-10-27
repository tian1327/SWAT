import json 
import yaml 

def format_txt(split_list, prefix, output_file):
    txt_list = []
    for entry in split_list:
        path = entry[0]
        label = entry[1]
        txt_list.append(f'{prefix}{path} {label} 1') # 1 means downstream data
    
    # sort txt_list by {label}
    txt_list.sort(key=lambda x: int(x.split(' ')[1]))
    
    with open(output_file, 'w') as f:
        f.write('\n'.join(txt_list))
    print(f'Created {output_file}, {len(txt_list)} lines') 

def format_imagenet_txt(split_list, prefix, output_file, split):
    txt_list = []
    for entry in split_list:
        path = entry[0]
        label = entry[1]

        path_segs = path.split('/')
        folder = path_segs[0]
        cls_code = path_segs[1]
        image_id = path_segs[-1].split('.')[0].split('_')[-1]

        if split == 'train':
            path_new = folder+'/'+cls_code+'_'+image_id+'_'+cls_code+'.JPEG'            
        
        elif split == 'val' or split == 'test':
            path_new = folder+'/ILSVRC2012_val_'+image_id+'_'+cls_code+'.JPEG'
        
        else:
            raise ValueError(f'Invalid split: {split}')

        txt_list.append(f'{prefix}{path_new} {label} 1') # 1 means downstream data
    
    # sort txt_list by {label}
    txt_list.sort(key=lambda x: int(x.split(' ')[1]))
    
    with open(output_file, 'w') as f:
        f.write('\n'.join(txt_list))
    print(f'Created {output_file}, {len(txt_list)} lines') 
    

def create_labels_oxfordpets():
    print('\nCreating labels for OxfordPets')

    file = f'{ROOT}/oxford_pets/split_zhou_OxfordPets.json'
    with open(file, 'r') as f:
        data = json.load(f)
    train_split = data['train']
    val_split = data['val']
    test_split = data['test']

    # create the train.txt, val.txt, test.txt
    prefix=f'images/'
    format_txt(train_split, prefix, 'data/oxford_pets/train.txt')
    format_txt(val_split, prefix, 'data/oxford_pets/val.txt')
    format_txt(test_split, prefix, 'data/oxford_pets/test.txt')


def create_labels_food101():
    print('\nCreating labels for food101')

    file = f'{ROOT}/food101/split_zhou_Food101.json'
    with open(file, 'r') as f:
        data = json.load(f)
    train_split = data['train']
    val_split = data['val']
    test_split = data['test']

    # create the train.txt, val.txt, test.txt
    prefix=f'images/'
    format_txt(train_split, prefix, 'data/food101/train.txt')
    format_txt(val_split, prefix, 'data/food101/val.txt')
    format_txt(test_split, prefix, 'data/food101/test.txt')

def create_labels_stanfordcars():
    print('\nCreating labels for stanfordcars')

    file = f'{ROOT}/stanford_cars/split_zhou_StanfordCars.json'
    with open(file, 'r') as f:
        data = json.load(f)
    train_split = data['train']
    val_split = data['val']
    test_split = data['test']

    # create the train.txt, val.txt, test.txt
    prefix=f''
    format_txt(train_split, prefix, 'data/stanford_cars/train.txt')
    format_txt(val_split, prefix, 'data/stanford_cars/val.txt')
    format_txt(test_split, prefix, 'data/stanford_cars/test.txt')


def create_labels_imagenet():
    print('\nCreating labels for ImageNet')

    file = f'{ROOT}/imagenet/split_ImageNet.json' # this file from CMLP repo, use val for testing
    with open(file, 'r') as f:
        data = json.load(f)
    train_split = data['train']
    val_split = data['val']
    test_split = data['test']

    # create the train.txt, val.txt, test.txt
    prefix=f'images/'

    # here the true train split should be the combination of train and val
    train_split.extend(val_split)

    format_imagenet_txt(train_split, prefix, 'data/imagenet/train.txt', 'train')
    format_imagenet_txt(test_split, prefix, 'data/imagenet/val.txt', 'val') # note here we pass in the test split
    format_imagenet_txt(test_split, prefix, 'data/imagenet/test.txt', 'test')

# def create_labels_imagenet():
#     print('\nCreating labels for ImageNet')

#     file = f'{ROOT}/imagenet/split_ImageNet.json' # this file from CMLP repo, use val for testing
#     with open(file, 'r') as f:
#         data = json.load(f)
#     train_split = data['train']
#     val_split = data['val']
#     test_split = data['test']

#     # create the train.txt, val.txt, test.txt
#     prefix=f'images/'
#     format_txt(train_split, prefix, 'data/imagenet/train.txt')
#     format_txt(val_split, prefix, 'data/imagenet/val.txt')
#     format_txt(test_split, prefix, 'data/imagenet/test.txt')


def create_labels_dtd():
    print('\nCreating labels for DTD')

    file = f'{ROOT}/dtd/dtd/split_zhou_DescribableTextures.json'
    with open(file, 'r') as f:
        data = json.load(f)
    train_split = data['train']
    val_split = data['val']
    test_split = data['test']

    # create the train.txt, val.txt, test.txt
    prefix=f'dtd/dtd/images/'
    format_txt(train_split, prefix, 'data/dtd/train.txt')
    format_txt(val_split, prefix, 'data/dtd/val.txt')
    format_txt(test_split, prefix, 'data/dtd/test.txt')


def create_labels_eurosat():
    print('\nCreating labels for EuroSAT')

    file = f'{ROOT}/eurosat/split_zhou_EuroSAT.json'
    with open(file, 'r') as f:
        data = json.load(f)
    train_split = data['train']
    val_split = data['val']
    test_split = data['test']

    # create the train.txt, val.txt, test.txt
    prefix=f'eurosat/EuroSAT_RGB/'
    format_txt(train_split, prefix, 'data/eurosat/train.txt')
    format_txt(val_split, prefix, 'data/eurosat/val.txt')
    format_txt(test_split, prefix, 'data/eurosat/test.txt')

def create_labels_flowers():
    print('\nCreating labels for Flowers')

    file = f'{ROOT}/flowers102/split_zhou_OxfordFlowers.json'
    with open(file, 'r') as f:
        data = json.load(f)
    train_split = data['train']
    val_split = data['val']
    test_split = data['test']

    # create the train.txt, val.txt, test.txt
    prefix=f'flowers102/jpg/'
    format_txt(train_split, prefix, 'data/flowers102/train.txt')
    format_txt(val_split, prefix, 'data/flowers102/val.txt')
    format_txt(test_split, prefix, 'data/flowers102/test.txt')

def format_aircraft_txt(split_list, prefix, output_file, label_dict):
    txt_list = []
    for line in split_list:
        entry = line.strip().split(' ')
        path = entry[0]
        label_name = ' '.join(entry[1:])
        # print(line)
        # print(entry)
        label = label_dict[label_name]
        txt_list.append(f'{prefix}{path}.jpg {label} 1') # 1 means downstream data
    
    # sort txt_list by {label}
    txt_list.sort(key=lambda x: int(x.split(' ')[1]))
    
    with open(output_file, 'w') as f:
        f.write('\n'.join(txt_list))
    print(f'Created {output_file}, {len(txt_list)} lines') 

def create_labels_aircraft():
    print('\nCreating labels for Aircraft')

    file = f'{ROOT}/fgvc-aircraft/fgvc-aircraft-2013b/data/variants.txt'
    with open(file, 'r') as f:
        labels = f.readlines()
    # build a dictionary of the labels
    labels_dict = {}
    for i, label in enumerate(labels):
        labels_dict[label.strip()] = i
    print('len(labels_dict):', len(labels_dict))
     
    train_fn = f'{ROOT}/fgvc-aircraft/fgvc-aircraft-2013b/data/images_variant_train.txt'
    val_fn = f'{ROOT}/fgvc-aircraft/fgvc-aircraft-2013b/data/images_variant_val.txt'
    test_fn = f'{ROOT}/fgvc-aircraft/fgvc-aircraft-2013b/data/images_variant_test.txt'

    with open(train_fn, 'r') as f:
        train_split = f.readlines()
    with open(val_fn, 'r') as f:
        val_split = f.readlines()
    with open(test_fn, 'r') as f:
        test_split = f.readlines()
    
    # create the train.txt, val.txt, test.txt
    prefix=f'fgvc-aircraft/fgvc-aircraft-2013b/data/images/'

    format_aircraft_txt(train_split, prefix, 'data/fgvc-aircraft/train.txt', labels_dict)
    format_aircraft_txt(val_split, prefix, 'data/fgvc-aircraft/val.txt', labels_dict)
    format_aircraft_txt(test_split, prefix, 'data/fgvc-aircraft/test.txt', labels_dict)



def format_aves_txt(split_fn, prefix, output_file):
    with open(split_fn, 'r') as f:
        split_list = f.readlines()
          
    txt_list = []
    for line in split_list:
        entry = line.strip().split(' ')
        path = entry[0]
        label = entry[1]
        txt_list.append(f'{prefix}{path} {label} 1') # 1 means downstream data
    
    # sort txt_list by {label}
    txt_list.sort(key=lambda x: int(x.split(' ')[1]))
    
    with open(output_file, 'w') as f:
        f.write('\n'.join(txt_list))
    print(f'Created {output_file}, {len(txt_list)} lines') 


def create_labels_aves():
    print('\nCreating labels for Aves')

    train_fn = f'{ROOT}/dataset/semi-aves/l_train.txt'
    val_fn = f'{ROOT}/dataset/semi-aves/val.txt'
    test_fn = f'{ROOT}/dataset/semi-aves/test.txt'


    prefix=f'semi-aves/'
    format_aves_txt(train_fn, prefix, 'data/semi-aves/ltrain.txt')
    format_aves_txt(val_fn, prefix, 'data/semi-aves/val.txt')
    format_aves_txt(test_fn, prefix, 'data/semi-aves/test.txt')


if __name__ == '__main__':
    
    # read the retrieved path from the config.yml
    with open('config.yml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        ROOT = config['dataset_path']  

    # create_labels_dtd()
    # create_labels_eurosat()
    # create_labels_flowers()
    # create_labels_aircraft()
    # create_labels_aves()
        
    # create_labels_oxfordpets()
    # create_labels_food101()
    # create_labels_stanfordcars()
    create_labels_imagenet()

    print('Done')