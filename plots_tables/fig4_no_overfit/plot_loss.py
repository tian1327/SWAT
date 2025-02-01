import matplotlib.pyplot as plt
import pandas as pd

eps = 80

fn = f'output_stage3_loss/output_semi-aves/stage3_loss_{eps}eps_semi-aves_probing_fewshot_REAL-Prompt_16shots_seed1_{eps}eps/loss.csv'
# read csv with header
df = pd.read_csv(fn, header=0)
print(df.shape)

# plot a line plot to show the train loss on the left axis, and test loss on the right axis
fig, ax1 = plt.subplots()

color = 'tab:blue'

ax1.set_xlabel('Epoch')
ax1.set_ylabel('Train Loss', color=color)
ax1.plot(df['Train_loss'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Test acc', color=color)
ax2.plot(df['Test_acc'], color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.savefig(f'stage3_loss_{eps}eps.png', dpi=300)
print('done')

