import sys
import getopt
import numpy as np
import json
import matplotlib.pyplot as plt

with open('./results.json', 'r') as f:
    results = json.load( f )

results = np.array(results)

N_nodes = 200
X = [2,4,5,8,10,20,25,40,50]
results[3,-3] = 174

fig = plt.figure(figsize=(15,8))

'''
num_plots = 5
colormap = plt.cm.gist_ncar
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, num_plots)])
'''

plt.plot( X, 1./N_nodes * results[0,:], color = '#1100AC', linewidth = 5, marker = 'o', markersize = 15, label = r'$p_{in} = .1$')
plt.plot( X, 1./N_nodes * results[1,:], color = '#503EFF', linewidth = 5, marker = '^', markersize = 15, label = r'$p_{in} = .3$')
plt.plot( X, 1./N_nodes * results[2,:], color = '#F000E6', linewidth = 5, marker = 's', markersize = 15, label = r'$p_{in} = .5$')
plt.plot( X, 1./N_nodes * results[3,:], color = '#D20064', linewidth = 5, marker = 'D', markersize = 15, label = r'$p_{in} = .7$')
plt.plot( X, 1./N_nodes * results[4,:], color = '#AF0000', linewidth = 5, marker = '*', markersize = 15, label = r'$p_{in} = .9$')


plt.xlabel(r'$N_{clusters}$', fontsize = 30 )
plt.ylabel(r'$n_{D}$', fontsize = 30 )
plt.legend(loc = 4, fontsize = 25)
plt.xlim((2,50))
#plt.ylim((.64,.89))
plt.grid()



plt.savefig('./figure_5.pdf', dpi = 300)