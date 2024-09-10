import matplotlib.pyplot as plt
import pandas as pd
from args import get_eval_args
import os
from matplotlib.ticker import MaxNLocator

args = get_eval_args()


Link_types = ['Type 1', 'Type 2', 'Type 3', 'Type 4']

proj1 = './result_flask/test-project'
proj2 = './result_flask/components'
proj3 = './result_flask/wasmiot-supervisor'
proj4 = './result_flask/postgres'

df1 = pd.read_csv(os.path.join(proj1, "positives.csv"))
df2 = pd.read_csv(os.path.join(proj2, "positives.csv"))
df3 = pd.read_csv(os.path.join(proj3, "positives.csv"))
df4 = pd.read_csv(os.path.join(proj4, "positives.csv"))

df1['project'] = '1'
df2['project'] = '2'
df3['project'] = '3'
df4['project'] = '4'

df = pd.concat([df1, df2, df3, df4])
print(df)

counts = df.groupby('project')['link_type'].value_counts().unstack().fillna(0)
print(counts)
plt.rcParams.update({'font.size': 15})

fig, ax = plt.subplots(figsize=(11, 3.5))

counts.plot(kind='barh', stacked=True, ax=ax, alpha=0.8)

ax.set_ylabel('Project No.')
ax.set_xlabel('Number of links')
ax.set_xlim(left=1)
for index, row in counts.iterrows():
    print(index)
    total = int(row.sum())
    # ax.text(i, total + 0.5, round(total),
    #       ha = 'center', weight = 'bold', color = 'black')
    ax.text(total + 0.5, int(index[-1]) - 1, f'{int(total)}', va='center')
ax.legend(title='Link Types')
ax.xaxis.set_major_locator(MaxNLocator(nbins=20))



fig.tight_layout()

# fig.tight_layout()
plt.savefig('./charts/stacked_barchart')
plt.close()
