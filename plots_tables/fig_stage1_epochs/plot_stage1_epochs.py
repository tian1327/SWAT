import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.style'] = 'normal'
plt.rcParams['font.variant'] = 'normal'

x2 = [10, 30, 50, 70, 90]
aves2 = [61.9, 63.1, 63.1, 63.4, 63.4]
aircraft2 = [57.1, 61.7, 60.7, 63.5, 65.0]
pets2 = [91.4, 91.2, 90.6, 91.7, 91.5]

# plot 1 row with two subplots
fig, axs = plt.subplots(figsize=(4, 2))

axs.plot(x2, aves2, label='Aves', marker='o', alpha=0.85)
axs.plot(x2, aircraft2, label='Aircraft', marker='o', alpha=0.85)
axs.plot(x2, pets2, label='Pets', marker='o', alpha=0.85)
axs.axvline(x=50, color='red', linestyle='--')
# set x ticks
axs.set_xticks(x2)
axs.set_xlabel('Stage 1 training epochs')
axs.set_ylabel('Accuracy (%)')

axs.legend(['Aves', 'Aircraft', 'Pets', 'SWAT'],
        loc='center right', bbox_to_anchor=(1.0, 0.6), fontsize='small')

plt.tight_layout()
# plt.show()
plt.savefig('ablate_s1-epochs.pdf')

