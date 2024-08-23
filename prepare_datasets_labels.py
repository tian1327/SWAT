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



def create_labels_dtd():
    print('\nCreating labels for DTD')

    file = f'{ROOT}/dataset/dtd/dtd/split_zhou_DescribableTextures.json'
    with open(file, 'r') as f:
        data = json.load(f)
    train_split = data['train']
    val_split = data['val']
    test_split = data['test']

    # create the train.txt, val.txt, test.txt
    prefix=f'{ROOT}/dataset/dtd/dtd/images/'
    format_txt(train_split, prefix, 'data/dtd/train.txt')
    format_txt(val_split, prefix, 'data/dtd/val.txt')
    format_txt(test_split, prefix, 'data/dtd/test.txt')


def create_labels_eurosat():
    print('\nCreating labels for EuroSAT')

    file = f'{ROOT}/dataset/eurosat/split_zhou_EuroSAT.json'
    with open(file, 'r') as f:
        data = json.load(f)
    train_split = data['train']
    val_split = data['val']
    test_split = data['test']

    # create the train.txt, val.txt, test.txt
    prefix=f'{ROOT}/dataset/eurosat/EuroSAT_RGB/'
    format_txt(train_split, prefix, 'data/eurosat/train.txt')
    format_txt(val_split, prefix, 'data/eurosat/val.txt')
    format_txt(test_split, prefix, 'data/eurosat/test.txt')

def create_labels_flowers():
    print('\nCreating labels for Flowers')

    file = f'{ROOT}/dataset/flowers102/split_zhou_OxfordFlowers.json'
    with open(file, 'r') as f:
        data = json.load(f)
    train_split = data['train']
    val_split = data['val']
    test_split = data['test']

    # create the train.txt, val.txt, test.txt
    prefix=f'{ROOT}/dataset/flowers102/jpg/'
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

    file = f'{ROOT}/dataset/fgvc-aircraft/fgvc-aircraft-2013b/data/variants.txt'
    with open(file, 'r') as f:
        labels = f.readlines()
    # build a dictionary of the labels
    labels_dict = {}
    for i, label in enumerate(labels):
        labels_dict[label.strip()] = i
    print('len(labels_dict):', len(labels_dict))
     
    train_fn = f'{ROOT}/dataset/fgvc-aircraft/fgvc-aircraft-2013b/data/images_variant_train.txt'
    val_fn = f'{ROOT}/dataset/fgvc-aircraft/fgvc-aircraft-2013b/data/images_variant_val.txt'
    test_fn = f'{ROOT}/dataset/fgvc-aircraft/fgvc-aircraft-2013b/data/images_variant_test.txt'

    with open(train_fn, 'r') as f:
        train_split = f.readlines()
    with open(val_fn, 'r') as f:
        val_split = f.readlines()
    with open(test_fn, 'r') as f:
        test_split = f.readlines()
    
    # create the train.txt, val.txt, test.txt
    prefix=f'{ROOT}/dataset/fgvc-aircraft/fgvc-aircraft-2013b/data/images/'

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


    prefix=f'{ROOT}/dataset/semi-aves/'
    format_aves_txt(train_fn, prefix, 'data/semi-aves/ltrain.txt')
    format_aves_txt(val_fn, prefix, 'data/semi-aves/val.txt')
    format_aves_txt(test_fn, prefix, 'data/semi-aves/test.txt')


if __name__ == '__main__':
    
    # read the retrieved path from the config.yml
    with open('../config.yml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        ROOT = config['retrieved_path']  

    create_labels_dtd()
    create_labels_eurosat()
    create_labels_flowers()
    create_labels_aircraft()
    create_labels_aves()