import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.style'] = 'normal'
plt.rcParams['font.variant'] = 'normal'

x = [0, 3, 10, 30, 50, 70, 90, 100]
aves = [57.7, 63.1, 63.2, 62.3, 60.7, 60.1, 59.6, 56.8]
aircraft = [55.1, 62.4, 61.9, 62.3, 60.7, 61.7, 60.4, 47.2]
pets = [90.5, 91.6, 91.8, 91.0, 90.6, 91.1, 90.5, 89.7]

# plot 1 row with two subplots
fig, axs = plt.subplots(figsize=(4, 2))

axs.plot(x, aves, label='Aves', marker='o', alpha=0.85)
axs.plot(x, aircraft, label='Aircraft', marker='o', alpha=0.85)
axs.plot(x, pets, label='Pets', marker='o', alpha=0.85)
axs.set_xlabel('Few-shot ratio in a batch (%)')
axs.set_ylabel('Accuracy (%)')
# plot a red vertical line at x = 3
axs.axvline(x=3, color='red', linestyle='--')
# set x ticks
x_new = [0, 10, 30, 50, 70, 90, 100]
axs.set_xticks(x_new)
# add a legend, add the vertical line to the legend
# use coordinates for specifying the location of the legend
legend = axs.legend(['Aves', 'Aircraft', 'Pets', 'natural (3%)'],
        loc='center right', bbox_to_anchor=(1.0, 0.65), fontsize='small')
legend.get_frame().set_alpha(0.3)

plt.tight_layout()
# plt.show()
plt.savefig('ablate_fs-ratio.pdf')

