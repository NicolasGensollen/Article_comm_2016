import json
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import json
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from scipy.signal import savgol_filter


with open('./test_OPT.json', 'r') as f:
    OPT = json.load(f)
with open('./test_RAND.json', 'r') as f:
    RAND = json.load(f)

OPT = np.nan_to_num(np.array(OPT) )
RAND = np.nan_to_num( np.array(RAND) )

with open('./test_OPT.json', 'r') as f:
    OPT = json.load(f)
OPT = np.nan_to_num(np.array(OPT) )

N_nodes = 50

MEANS = [ np.mean( RAND[RAND[:,0]==i][:,1] ) for i in range(1,N_nodes) ]

MEANS2 = [ np.mean( OPT[OPT[:,0]==i][:,1] ) for i in range(1,N_nodes) ]


MEANS = savgol_filter(MEANS, 5, 3)
MEANS2 = savgol_filter(MEANS2, 5, 3)

fig = plt.figure(figsize=(10,10))
plt.scatter( 1./N_nodes * RAND[:,0], RAND[:,1], s= 60, color = '#FD6AF2', alpha = .3 )
plt.scatter( 1./N_nodes * OPT[:,0], OPT[:,1], s= 60, color = '#FDAC86', alpha = .2 )
plt.plot( 1./N_nodes * np.arange(1,N_nodes), MEANS, linewidth = 4, color = '#C703B8', label = 'random' )
plt.plot( 1./N_nodes * np.arange(1,N_nodes), MEANS2, linewidth = 4, color = 'r', label = 'greedy optimization' )
plt.xlim((0.45,1))
plt.ylim((5,27))
plt.xlabel(r'$n_D$', fontsize = 20)
plt.ylabel(r'$log(\mathcal{E})$', fontsize = 20)
plt.legend(loc=3, fontsize = 15)
plt.grid()
plt.savefig('./figure_3.pdf', dpi = 300)
plt.savefig('./figure_3.png')
