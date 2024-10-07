import json

metric = json.load(open('../data/imagenet/imagenet_metrics-LAION400M.json'))

# loop through each key-value pair in the dictionary
for key, value in metric.items():
    name = value['name']
    # convert to lower case
    name = name.lower()
    alternates = value['alternates']
    alternates_set = set(alternates)

    # check if the name is in the alternates
    if name not in alternates_set:
        print(f'{key}: {name} not in alternates')        