import matplotlib.pyplot as plt
import numpy as np

x = [0, 3, 10, 30, 50, 70, 90, 100]
aves = [57.7, 63.1, 63.2, 62.3, 60.7, 60.1, 59.6, 56.8]
aircraft = [55.1, 62.4, 61.9, 62.3, 60.7, 61.7, 60.4, 47.2]
pets = [90.5, 91.6, 91.8, 91.0, 90.6, 91.1, 90.5, 89.7]

x2 = [10, 30, 50, 70, 90]
aves2 = [61.9, 63.1, 63.1, 63.4, 63.4]
aircraft2 = [57.1, 61.7, 60.7, 63.5, 65.0]
pets2 = [91.4, 91.2, 90.6, 91.7, 91.5]

# plot 1 row with two subplots
fig, axs = plt.subplots(1, 2, figsize=(6, 2))

axs[0].plot(x, aves, label='Aves', marker='o', alpha=0.85)
axs[0].plot(x, aircraft, label='Aircraft', marker='o', alpha=0.85)
axs[0].plot(x, pets, label='Pets', marker='o', alpha=0.85)
axs[0].set_xlabel('Few-shot ratio in a batch (%)')
axs[0].set_ylabel('Accuracy (%)')
# plot a red vertical line at x = 3
axs[0].axvline(x=3, color='red', linestyle='--')
# set x ticks
x_new = [0, 10, 30, 50, 70, 90, 100]
axs[0].set_xticks(x_new)
# add a legend, add the vertical line to the legend
# use coordinates for specifying the location of the legend
legend = axs[0].legend(['Aves', 'Aircraft', 'Pets', 'natural (3%)'],
        loc='center right', bbox_to_anchor=(1.0, 0.65), fontsize='small')
legend.get_frame().set_alpha(0.3)

axs[1].plot(x2, aves2, label='Aves', marker='o', alpha=0.85)
axs[1].plot(x2, aircraft2, label='Aircraft', marker='o', alpha=0.85)
axs[1].plot(x2, pets2, label='Pets', marker='o', alpha=0.85)
axs[1].axvline(x=50, color='red', linestyle='--')
# set x ticks
axs[1].set_xticks(x2)
axs[1].set_xlabel('Stage 1 training epochs')
axs[1].set_ylabel('Accuracy (%)')

axs[1].legend(['Aves', 'Aircraft', 'Pets', 'SWAT'],
        loc='center right', bbox_to_anchor=(1.0, 0.6), fontsize='small')

plt.tight_layout()
# plt.show()
plt.savefig('fs-ratio_s1-epochs.pdf')

