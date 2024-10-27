import random

# take the 10 img from each class as the fewshot_train, the other 5 as the fewshot_val
"""
fewshot_train = dict()
fewshot_val = dict()
for class_id, paths in fewshot.items():
    fewshot_train[class_id] = paths[:10]
    fewshot_val[class_id] = paths[10:]

# write out to fewshot_train.txt
with open('data/semi-aves/fewshot_train10.txt', 'w') as f:
    for class_id, paths in fewshot_train.items():
        for path in paths:
            f.write(f'{path} {class_id}\n')

# write out to fewshot_val.txt
with open('data/semi-aves/fewshot_val5.txt', 'w') as f:
    for class_id, paths in fewshot_val.items():
        for path in paths:
            f.write(f'{path} {class_id}\n')
"""

def random_sample_fewshot(dataset, ct, seed):
    print(f'\nRandomly sampling, seed {seed}, {ct} shots, {dataset}')
    if dataset == 'semi-aves':
        train_fn = 'ltrain+val.txt'
    else:
        train_fn = 'train.txt'
    
    with open(f'data/{dataset}/{train_fn}', 'r') as f:
        lines = f.readlines()
    
    # collect line by class
    train = dict()
    fewshot = dict()
    for line in lines:
        path, class_id, source = line.strip('\n').split(' ')
        if class_id in train:
            train[class_id].append(path)
        else:
            train[class_id] = [path]
    
    # set the random seed value
    random.seed(seed)

    # randomly sample ct images from each class
    for class_id, paths in train.items():
        if len(paths) < ct:
            fewshot[class_id] = paths
            print(f'class_id: {class_id}, len(paths): {len(paths)}')
        else:
            fewshot[class_id] = random.sample(paths, ct)
    
    # write out to fewshot{ct}_seed{seed}.txt
    fn = f'data/{dataset}/fewshot{ct}_seed{seed}.txt'
    fewshot_lines = []
    with open(fn, 'w') as f:
        for class_id, paths in fewshot.items():
            for path in paths:
                line = f'{path} {class_id} 1\n'
                fewshot_lines.append(line)
                f.write(line)
    print(f'saved to {fn}')

    """
    # combine with the retrieved data
    retrieve_method = 'T2T500+T2I0.25'
    retrieved_fn = f'data/{dataset}/{retrieve_method}.txt'
    with open(retrieved_fn, 'r') as f:
        retrieved_lines = f.readlines()
    
    # combine the fewshot with the retrieved data
    combined_lines = fewshot_lines + retrieved_lines
    print('len(fewshot_lines):', len(fewshot_lines))
    print('len(retrieved_lines):', len(retrieved_lines))
    print('len(combined_lines):', len(combined_lines))

    # write out to fewshot{ct}_seed{seed}+{retrieve_method}.txt
    fn = f'data/{dataset}/fewshot{ct}_seed{seed}+{retrieve_method}.txt'
    with open(fn, 'w') as f:
        for line in combined_lines:
            f.write(line)
    print(f'saved to {fn}')
    """

if __name__ == '__main__':

    datasets = [
        # 'semi-aves', 
        # 'dtd', 
        # 'fgvc-aircraft', 
        # 'eurosat', 
        # 'flowers102'
        # 'oxford_pets',
        # 'food101',
        # 'stanford_cars'.
        'imagenet'
        ]
    # fewshot_ct = [1, 2, 4, 8, 16]
    fewshot_ct = [4, 8, 16]
    seed_list = [1, 2, 3]
    for seed in seed_list:
        for ct in fewshot_ct:
            for dataset in datasets:
                random_sample_fewshot(dataset, ct, seed)