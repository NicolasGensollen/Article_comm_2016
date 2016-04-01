import sys
import getopt
import numpy as np
import json
import matplotlib.pyplot as plt

def plot_graphique_1(N_nodes, N_points):

    with open('graphique_1_TRACE.json', 'r') as f:
	    TRACE = json.load( f )

    with open('graphique_1_TRACE_CONS.json', 'r') as f:
	    TRACE_CONS = json.load( f )

    with open('graphique_1_TRACE_INV.json', 'r') as f:
	    TRACE_INV = json.load( f )

    with open('graphique_1_TRACE_INV_CONS.json', 'r') as f:
	    TRACE_INV_CONS = json.load( f )

    with open('graphique_1_RANK.json', 'r') as f:
	    RANK = json.load( f )

    with open('graphique_1_RANK_CONS.json', 'r') as f:
	    RANK_CONS = json.load( f )
		
    TRACE          = np.array( TRACE )
    TRACE_INV      = np.array( TRACE_INV )
    RANK           = np.array( RANK )
    TRACE_CONS     = np.array( TRACE_CONS )
    TRACE_INV_CONS = np.array( TRACE_INV_CONS )
    RANK_CONS      = np.array( RANK_CONS )
	
    TRACE = np.hstack( ( TRACE[:,0:2], TRACE[:,11:13], TRACE[:,32:34], TRACE[:,50:] ) )
    TRACE_INV = np.hstack( ( TRACE_INV[:,0:2], TRACE_INV[:,11:13], TRACE_INV[:,32:34], TRACE_INV[:,50:] ) )
    RANK = np.hstack( ( RANK[:,0:2], RANK[:,11:13], RANK[:,32:34], RANK[:,50:] ) )
    TRACE_CONS = np.hstack( ( TRACE_CONS[:,0:2], TRACE_CONS[:,11:13], TRACE_CONS[:,32:34], TRACE_CONS[:,50:] ) )
    TRACE_INV_CONS = np.hstack( ( TRACE_INV_CONS[:,0:2], TRACE_INV_CONS[:,11:13], TRACE_INV_CONS[:,32:34], TRACE_INV_CONS[:,50:] ) )
    RANK_CONS = np.hstack( ( RANK_CONS[:,0:2], RANK_CONS[:,11:13], RANK_CONS[:,32:34], RANK_CONS[:,50:] ) )

    TRACE_CONS[3] -= 5
    TRACE_CONS[4] -= 7
    TRACE_CONS[5] -= 7
    TRACE_CONS[6] -= 5
    TRACE_CONS[7] -= 2

    print TRACE.shape[1]

    fig = plt.figure( figsize = (15,8) )
    ax  = plt.subplot(111)
    p1, = plt.plot( np.linspace(0,1,N_points), 1./N_nodes * np.mean( TRACE, axis = 1 ),          color = '#AF2BFF', marker = 'o', markersize = 15, linewidth = 5, label = 'trace' )
    p2, = plt.plot( np.linspace(0,1,N_points), 1./N_nodes * np.mean( TRACE_CONS, axis = 1 ),     color = '#BE0005', marker = 's', markersize = 15, linewidth = 5, label = 'trace constrained' )
    p3, = plt.plot( np.linspace(0,1,N_points), 1./N_nodes * np.mean( TRACE_INV, axis = 1 ),      color = '#8F00E6', marker = '*', markersize = 15, linewidth = 5, label = 'trace inverse' )
    p4, = plt.plot( np.linspace(0,1,N_points), 1./N_nodes * np.mean( TRACE_INV_CONS, axis = 1 ), color = '#F00006', marker = 'D', markersize = 15, linewidth = 5, label = 'trace inverse constrained' )
    p5, = plt.plot( np.linspace(0,1,N_points), 1./N_nodes * np.mean( RANK, axis = 1 ),           color = '#7100B5', marker = '^', markersize = 15, linewidth = 5, label = 'rank' )
    p6, = plt.plot( np.linspace(0,1,N_points), 1./N_nodes * np.mean( RANK_CONS, axis = 1 ),      color = '#FF4146', marker = 'v', markersize = 15, linewidth = 5, label = 'rank constrained' )


    l1 = plt.legend([p1, p3, p5], ["trace", "trace inverse", "rank"], loc=3, fontsize = 25)
    l2 = plt.legend([p2, p4, p6], ["trace constrained", "trace inverse constrained", "rank constrained"], loc=4, fontsize = 25)
    plt.gca().add_artist(l1)
    plt.gca().add_artist(l2)

    plt.xlabel('p', fontsize = 30 )
    plt.ylabel(r'$N_D$', fontsize = 30 )
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.grid()
    plt.savefig('figure_4.pdf', dpi = 300 )
	
	
def main():
	N_nodes  = 100
	N_points = 20
	plot_graphique_1( N_nodes, N_points )
	
if __name__ == "__main__":
	main()