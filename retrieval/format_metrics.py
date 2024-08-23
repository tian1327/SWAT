import json
import sys

fn = sys.argv[1]

# open the json fn
f = open(fn)
metrics = json.load(f)
# save the metrics to a json file with 4 indentations
with open(fn, 'w') as f:
    json.dump(metrics, f, indent=4)
print(f'Done reformatting {fn}!')