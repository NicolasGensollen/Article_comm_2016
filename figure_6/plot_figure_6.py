import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

with open('./results3.json','r') as f:
    results = json.load(f)

N_nodes = 200
clust = [2,4,5,10]
marks = ['o', '*', 'D', '^']
col = ['#3C3CFF', '#D900F1', '#D90073', '#BE0019']

x_value =  -.05    #Offset by eye
y_value = .49

results = np.array(results)
X = np.linspace(0,1,results.shape[1])

fig = plt.figure(figsize=(15,8))
ax=plt.subplot(111)
for i in range(results.shape[0]):
    plt.plot(X, savgol_filter( 1./N_nodes * np.mean(results[i], axis = 1), 11, 2),
            linewidth = 5, color = col[i], marker = marks[i], markersize = 15, label=r'$N_{clusters} = $' + str(clust[i]) )

axbox = ax.get_position()
plt.legend( loc = (axbox.x0 + x_value, axbox.y0 + y_value), fontsize = 25)

plt.xlabel(r'$p_{in}$', fontsize = 30)
plt.ylabel(r'$n_D$', fontsize = 30)
plt.grid()

plt.savefig('./graphique.pdf', dpi = 300)


