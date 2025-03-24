import matplotlib.pyplot as plt
import json
# import sys

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.style'] = 'normal'
plt.rcParams['font.variant'] = 'normal'

NUM_CLASSES_DICT = {
    'semi-aves': 200,
    'flowers102': 102,
    'fgvc-aircraft': 100,
    'eurosat': 10,
    'dtd': 47,
    'food101': 101,
    'stanford_cars': 196,
    "oxford_pets": 37,
    'imagenet': 1000,
    'semi-inat-2021': 810,
}

NAMES_DICT = {
    'semi-aves': 'Semi-Aves',
    'flowers102': 'Flowers',
    'fgvc-aircraft': 'Aircraft',
    'eurosat': 'EuroSAT',
    'dtd': 'DTD',
    'food101': 'Food',
    'stanford_cars': 'Cars',
    'oxford_pets': 'Pets',
    'imagenet': 'ImageNet',
}

MARKERSIZE = 12
LINEWIDTH = 4
LABELSIZE = 25
LEGENDSIZE = 20
TICKSIZE = 22

X = []
Y = []
Names = []
Num_cls = []
for dataset in NAMES_DICT.keys():

    # fn = f'retrieval/output/{dataset}_vitb32_openclip_laion400m_T2T500+T2I0.25/num_imgs_sampled.json'
    fn = f'../../retrieval/output/{dataset}_vitb32_openclip_laion400m_T2T500/T2T500_num_imgs_sampled.json'

    with open(fn, 'r') as f:
        num_img = json.load(f)

    # sort num_img based on value in descending order
    num_img_sorted = dict(sorted(num_img.items(), key=lambda x: x[1], reverse=True))

    print(f'len(num_img_sorted): {len(num_img)}')
    num_class = NUM_CLASSES_DICT[dataset]

    x = list(range(num_class))
    y = [v for k, v in num_img_sorted.items()]

    while len(y)<num_class:
        y.append(0)

    x = [(v+1)/num_class*100 for v in x]
    X.append(x)
    Y.append(y)
    Names.append(NAMES_DICT[dataset])
    Num_cls.append(num_class)


# plot a line plot to show the number of images sampled per class
plt.figure(figsize=(10, 5))
for x, y, name in zip(X, Y, Names):
    # plt.plot(x, y, linestyle='solid', linewidth=LINEWIDTH, label=name, color='tab:blue')
    plt.plot(x, y, linestyle='solid', linewidth=LINEWIDTH, label=name)

# plt.title(f"{NAMES_DICT[dataset]}", fontsize=20)
plt.ylabel('# of imags / class', fontsize=LABELSIZE)
# plt.xlabel("Class sorted w/ decreasing frequency", fontsize=19)
plt.xlabel("% of class number", fontsize=LABELSIZE)

plt.legend(handles=plt.gca().get_legend_handles_labels()[0],
           loc='lower left', prop={'size': 19},
           frameon=True, facecolor='white', framealpha=0.3)

plt.yticks(fontsize=TICKSIZE)
plt.xticks(fontsize=TICKSIZE)

plt.tight_layout()
plt.savefig(f'imbalanced_all.png', dpi=300)
plt.clf()  # clear the current figure
print('done')
