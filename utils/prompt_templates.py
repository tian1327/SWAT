import json
caltech101_templates = [
    'a photo of a {}.',
    'a painting of a {}.',
    'a plastic {}.',
    'a sculpture of a {}.',
    'a sketch of a {}.',
    'a tattoo of a {}.',
    'a toy {}.',
    'a rendition of a {}.',
    'a embroidered {}.',
    'a cartoon {}.',
    'a {} in a video game.',
    'a plushie {}.',
    'a origami {}.',
    'art of a {}.',
    'graffiti of a {}.',
    'a drawing of a {}.',
    'a doodle of a {}.',
    'a photo of the {}.',
    'a painting of the {}.',
    'the plastic {}.',
    'a sculpture of the {}.',
    'a sketch of the {}.',
    'a tattoo of the {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'the embroidered {}.',
    'the cartoon {}.',
    'the {} in a video game.',
    'the plushie {}.',
    'the origami {}.',
    'art of the {}.',
    'graffiti of the {}.',
    'a drawing of the {}.',
    'a doodle of the {}.',
]

describabletextures_templates = [
    'a photo of a {} texture.',
    'a photo of a {} pattern.',
    'a photo of a {} thing.',
    'a photo of a {} object.',
    'a photo of the {} texture.',
    'a photo of the {} pattern.',
    'a photo of the {} thing.',
    'a photo of the {} object.',
]

eurosat_templates = [
    'a centered satellite photo of {}.',
    'a centered satellite photo of a {}.',
    'a centered satellite photo of the {}.',
]

fgvcaircraft_templates = [
    'a photo of a {}, a type of aircraft.',
    'a photo of the {}, a type of aircraft.',
]

flowers102_templates = [
    'a photo of a {}, a type of flower.',
]

food101_templates = [
    'a photo of {}, a type of food.',
]

oxfordpets_templates = [
    'a photo of a {}, a type of pet.',
]

sun397_templates = [
    'a photo of a {}.',
    'a photo of the {}.',
]

stanfordcars_templates = [
    'a photo of a {}.',
    'a photo of the {}.',
    'a photo of my {}.',
    'i love my {}!',
    'a photo of my dirty {}.',
    'a photo of my clean {}.',
    'a photo of my new {}.',
    'a photo of my old {}.',
]

imagenet_templates = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]

caltech101_templates = [
    'a photo of a {}.',
    'a painting of a {}.',
    'a plastic {}.',
    'a sculpture of a {}.',
    'a sketch of a {}.',
    'a tattoo of a {}.',
    'a toy {}.',
    'a rendition of a {}.',
    'a embroidered {}.',
    'a cartoon {}.',
    'a {} in a video game.',
    'a plushie {}.',
    'a origami {}.',
    'art of a {}.',
    'graffiti of a {}.',
    'a drawing of a {}.',
    'a doodle of a {}.',
    'a photo of the {}.',
    'a painting of the {}.',
    'the plastic {}.',
    'a sculpture of the {}.',
    'a sketch of the {}.',
    'a tattoo of the {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'the embroidered {}.',
    'the cartoon {}.',
    'the {} in a video game.',
    'the plushie {}.',
    'the origami {}.',
    'art of the {}.',
    'graffiti of the {}.',
    'a drawing of the {}.',
    'a doodle of the {}.',
]

semi_aves_templates = {
    's-name': ['a photo of a {}, a type of bird.'],
    'c-name': ['a photo of a {}, a type of bird.'],
    't-name': ['a photo of a {}, a type of bird, commonally known as {}.'],
    'f-name': ['a photo of a {}, a type of bird.'],
    'most_common_name': ['a photo of a {}, a type of bird.'],
    'most_common_name_REAL': ['a photo of a {}, a type of bird.'],
    'name': ['a photo of a {}, a type of bird.'],
    # 'name': imagenet_templates, # openai templates
    'c-name-80prompts': imagenet_templates,
}

TEMPLATES_DIC = {
    'imagenet_1k': imagenet_templates,
    'imagenet_1k_mined': imagenet_templates,
    'flowers102': flowers102_templates,
    'food101': food101_templates,
    'stanford_cars': stanfordcars_templates,
    'fgvc-aircraft': fgvcaircraft_templates,
    'oxford_pets': oxfordpets_templates,
    'imagenet_v2': imagenet_templates,
    'dtd': describabletextures_templates,
    'semi-aves': semi_aves_templates,
    'caltech101': caltech101_templates,
    'eurosat': eurosat_templates,
    'sun397': sun397_templates
}

def prompt_maker(metrics: dict, dataset_name: str, name_type='most_common_name'):
    prompts = {}
    if dataset_name == 'semi-aves':
        prompt_templates = TEMPLATES_DIC[dataset_name][name_type]
    else:
        prompt_templates = TEMPLATES_DIC[dataset_name]
    # print('prompt_templates:', prompt_templates)

    for i, key in enumerate(metrics.keys()):
        label = metrics[key][name_type]

        if name_type == 'alternates': 
            prompt_lst = []
            for alt_name, ct in label.items():                
                pt = [template.format(alt_name) for template in prompt_templates]
                prompt_lst.extend(pt)
            prompts[key] = {'corpus': prompt_lst}
        else:
            prompts[key] = {'corpus': [template.format(label) for template in prompt_templates]}            

    prompts = dict(sorted(prompts.items(), key=lambda x: int(x[0])))

    return prompts


def prompt_maker_aves(metrics: dict, dataset_name: str, name_type='s-name'):
    prompts = {}
    prompt_templates = TEMPLATES_DIC[dataset_name][name_type.split('_')[0]]
    print('prompt_templates:', prompt_templates)

    for i, key in enumerate(metrics.keys()):
        class_id = str(metrics[key]['class_id'])
        s_name = metrics[key]['species']
        c_name = metrics[key]['common_name']
        
        if name_type == 's-name':
            prompt_lst = [template.format(s_name) for template in prompt_templates]
        
        elif name_type == 'c-name':
            prompt_lst = [template.format(c_name) for template in prompt_templates]
        
        elif name_type == 't-name':
            prompt_lst = [template.format(s_name, c_name) for template in prompt_templates]
        
        elif name_type == 'f-name':
            freq_name = metrics[key]['most_freq_synonym']
            prompt_lst = [template.format(freq_name) for template in prompt_templates]
        
        elif name_type == 'c-name-80prompts':
            prompt_lst = [template.format(c_name) for template in prompt_templates]

        elif name_type == 'c-name_attribute':
            prompt_lst = [template.format(c_name) for template in prompt_templates]
            attributes = json.load(open('data/semi-aves/prompts/visual-attrs-semi-aves.json', 'r'))
            attributes_lst = attributes[key]["corpus"]
            # print('attributes_lst:', attributes_lst)
            attributes_prompt_lst = [template.format(c_name)+f' {c_name} {attr}'.replace('Has', 'has') for template in prompt_templates for attr in attributes_lst]
            # attributes_prompt_lst = [f'{c_name} {attr}'.replace('Has', 'has') for attr in attributes_lst]
            prompt_lst.extend(attributes_prompt_lst)
            # print('prompt_lst:', prompt_lst)

        prompts[class_id] = {'corpus': prompt_lst}

    prompts = dict(sorted(prompts.items(), key= lambda x: int(x[0])))
    # print(prompts['0'])

    return prompts