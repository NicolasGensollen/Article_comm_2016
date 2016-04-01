import sys
import getopt
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def plot_graphique_3(N_nodes, N_points):

    SIG_RANGE_1 = np.linspace(.01,12,5)
    SIG_RANGE_2 = np.linspace(12,35,30)
    SIG_RANGE_3 = np.linspace(35,50,5)
    SIG_RANGE = np.append( SIG_RANGE_1, np.append( SIG_RANGE_2, SIG_RANGE_3 ) )
    SIG_RANGE = np.unique( SIG_RANGE )
    N_points  = len(SIG_RANGE)
   
    with open('./' + str(N_nodes) + '_nodes/graphique_3_TRACE_CONS.json', 'r') as f:
	    TRACE_CONS = json.load( f )
    with open('./' + str(N_nodes) + '_nodes/graphique_3_TRACE_INV_CONS.json', 'r') as f:
	    TRACE_INV_CONS = json.load( f )
    with open('./' + str(N_nodes) + '_nodes/graphique_3_RANK_CONS.json', 'r') as f:
	    RANK_CONS = json.load( f )
		
    TRACE_CONS     = np.array( TRACE_CONS )
    TRACE_INV_CONS = np.array( TRACE_INV_CONS )
    RANK_CONS      = np.array( RANK_CONS )
	
    YY_TRACE_CONS     = savgol_filter( 1./N_nodes * np.mean( TRACE_CONS, axis = 1 ), 11, 3)
    YY_TRACE_INV_CONS = savgol_filter( 1./N_nodes * np.mean( TRACE_INV_CONS, axis = 1 ), 11, 3)
    YY_RANK_CONS      = savgol_filter( 1./N_nodes * np.mean( RANK_CONS, axis = 1 ), 11, 3)	
	
    print TRACE_CONS.shape[1]
    fig = plt.figure( figsize = (10,6) )
 
    plt.scatter( 1./100 * SIG_RANGE, 1./N_nodes * np.mean( TRACE_CONS, axis = 1 ),     s=100, color = 'b', alpha = .5)
    plt.scatter( 1./100 * SIG_RANGE, 1./N_nodes * np.mean( TRACE_INV_CONS, axis = 1 ), s=100, color = 'r', alpha = .5)
    plt.scatter( 1./100 * SIG_RANGE, 1./N_nodes * np.mean( RANK_CONS, axis = 1 ),      s=100, color = 'm', alpha = .5)
    plt.plot( 1./100 * SIG_RANGE, YY_TRACE_CONS,     color = 'b', linewidth = 3, label = 'trace constrained' )
    plt.plot( 1./100 * SIG_RANGE, YY_TRACE_INV_CONS, color = 'r', linewidth = 3, label = 'trace inverse constrained' )
    plt.plot( 1./100 * SIG_RANGE, YY_RANK_CONS,      color = 'm', linewidth = 3, label = 'rank constrained' )
	
    plt.xlabel(r'$ \sigma_{\lambda} / \mu_{\lambda} $', fontsize = 15 )
    plt.ylabel(r'$n_D$', fontsize = 20 )
    plt.legend( loc = 4, fontsize = 15 )
    plt.xlim((0,.5))
    plt.ylim((.55,1.05))
    plt.grid()
    plt.savefig('./' + str(N_nodes) + '_nodes/graphique_3_3.pdf', dpi = 300 )
	
	
def main():
	N_nodes  = 100
	N_points = 20
	plot_graphique_3( N_nodes, N_points )
	
if __name__ == "__main__":
	main()