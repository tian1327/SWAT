# import open_clip
# from analysis.laion.imagenet_1k.labels import clip_imagenet_classes
# from sentence_transformers import SentenceTransformer
import json
import torch
import os
import sys
# sys.path.append("../../") # Adds higher directory to python modules path.

# Get the absolute path of the utils directory
utils_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Add the utils directory to sys.path
sys.path.append(utils_dir)
from utils.extras import get_engine


#---------- load model
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f'Number of GPUs available: {torch.cuda.device_count()}')

model, preprocess, tokenizer = get_engine(model_cfg='vitb32_openclip_laion400m', device=device)
print(f'Loaded model.')


def text_prompt_classifier(class_names_lst):

    prompts = []
    for i, class_name in enumerate(class_names_lst):
        s = f'a photo of a {class_name}'
        # s_tokens = s
        s_tokens = tokenizer(s)
        s_tokens = s_tokens.cuda()
        with torch.no_grad():
            text_embedding = model.encode_text(s_tokens)
        text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
        prompts.append(text_embedding.cpu())
    
    prompt_classifier = torch.stack(prompts, dim=0).squeeze()
    prompt_classifier = prompt_classifier.cuda()

    return prompt_classifier


def clip_filter(filename):

    with open(filename, 'r') as f:
        names = json.load(f)
    
    # get the true class label list
    true_class_labels = [names[label_idx]['query_name'] for label_idx in names]
    print('len(true_class_labels): ', len(true_class_labels))
    print('true_class_labels[:5]: ', true_class_labels[:5])

    # get the prompt classifier using the embeddings of the true class labels
    prompt_classifier = text_prompt_classifier(true_class_labels)

    for i, key in enumerate(names): # prompts
        all_synonyms = names[key]['synonyms'].keys()
        filtered_names = set()
        for name in all_synonyms:
            with torch.no_grad():
                prompt = model.encode_text(tokenizer(f'a photo of a {name}').cuda())
            similarity = (prompt.cuda() @ prompt_classifier.t()) * 100
            similarity = similarity.squeeze()

            if similarity.argmax().item() == int(key): # only add the names that are classified as the same class
                filtered_names.add(name)
            else:
                print(f"{key} - {names[key]['query_name']}: unmatch {name} - {similarity.argmax().item()}")

        names[key]['synonyms_filtered'] = {}
        for name in filtered_names:
            names[key]['synonyms_filtered'][name] = 0
    
    return names


if __name__ == "__main__":
    
    targets = [
        # 'caltech-101', 
        # 'dtd', 'eurosat_clip', 'fgvc-aircraft-2013b-variants102',
        # 'oxford-flower-102', 'oxford-iiit-pets', 'sun397', 
        # 'food-101', 'stanford-cars',
        'semi-aves',
        # 'cub200'
        ]

    for target in targets:
        print('filtering: ', target)
        filename = f'output/{target}_synonyms.json'
        
        filtered_dict = clip_filter(filename)

        outfn = f'output/{target}_synonyms_filtered.json'
        with open(outfn, 'w') as f:
            f.write(json.dumps(filtered_dict, indent=4))
    
    print('Done.')