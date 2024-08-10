from query_synonyms import clean_text
import json

fn = 'output/backup/semi-aves_alternatives_gpt4.json'
with open(fn, 'r') as f:
    data = json.load(f)

for k, info in data.items():
    alt_names = info['alternatives']
    names_set = set()
    for name in alt_names.keys():
        names_set.add(clean_text(name))
    info['alternatives'] = {}
    for name in names_set:
        info['alternatives'][name] = 0

# save the data
fn = 'output/semi-aves_alternatives.json'
with open(fn, 'w') as f:
    json.dump(data, f, indent=4)