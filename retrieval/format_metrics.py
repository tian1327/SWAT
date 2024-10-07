import json
import sys

fn = sys.argv[1]

# open the json fn
f = open(fn)
metrics = json.load(f)

# sort the metrics by the int(key)
metrics = {int(k): v for k, v in metrics.items()}
metrics = dict(sorted(metrics.items()))

# save the metrics to a json file with 4 indentations
with open(fn, 'w') as f:
    json.dump(metrics, f, indent=4)
print(f'Done reformatting {fn}!')