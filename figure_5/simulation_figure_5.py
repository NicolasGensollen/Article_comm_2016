##########################################################################################
#				PACKAGES
##########################################################################################

try:
	import numpy as np
	import math
	import matplotlib.pyplot as plt
	import networkx as nx
	import random
	import operator
	import itertools
	import copy
	import operator
	import json
except:
	raise ValueError('Import problem')
	
#Import all the functions written in control module
import control2
from clint.textui import progress

##########################################################################################
#				PARAMETERS
##########################################################################################


Omega = 50. / ( 2 * math.pi )                        # Main frequency is 50Hz
I = .1                                               # Inertia
KD = .1                                              # Dissipation constant
Tmax = 50                                            # Maximum time
Dt = .01                                             # Discretization step
time = np.arange(0,Tmax,Dt)                          # Time  
alpha = float( 2 * KD ) / float( I )                 # Damping
control_time = 10                                    # control time T
r = 5                                                # maximum charge / discharge rate
max_capacity = 150                                   # maximum capacity for the batteries


##########################################################################################
#				FUNCTIONS FOR GENERATING RANDOM INSTANCES
##########################################################################################

def generate_random_topology_ER( N_nodes, p ):
    '''Generate a Erdos-Renyi random graph with N_nodes nodes, and link probability p
	   Return the triplet (Graph, adjacancy matrix, Laplacian matrix)
    '''
    G    = nx.erdos_renyi_graph( N_nodes, p )
    adja = np.array( nx.to_numpy_matrix(G) )
    L    = control2.get_Laplacian( adja )
    return (G, adja, L)

	
def generate_random_topology_BA( N_nodes, n ):
    '''Generate a Barabasi-Albert random graph with N_nodes nodes, and new link parameter n
	   Return the triplet (Graph, adjacancy matrix, Laplacian matrix)
    '''
    G    = nx.barabasi_albert_graph( N_nodes, n )
    adja = np.array( nx.to_numpy_matrix(G) )
    L = control2.get_Laplacian( adja )
    return (G, adja, L)

	
def generate_random_power_distribution( N_nodes, order_of_magnitude = 10**1 ):
    '''Generate a random distribution for production and consumption
	   The order_of_magnitude parameter controls how large the productions and consumptions are (used for units)
	   The sample should sum to zero (i.e production matches demand)
    '''
    P = np.random.random((1,N_nodes))[0]
    P /= np.sum( P )
    assert np.allclose( np.sum( P ), 1 )
    P -= np.mean( P )
    assert np.allclose( np.sum( P ), 0 )
    P *= order_of_magnitude
    assert np.allclose( np.sum( P ), 0 )
    return P

def generate_power_distribution( G, rho, P ):
    N_nodes = G.number_of_nodes()
    search = True
    while search:
        res = np.zeros((1,N_nodes))[0]
        for node in range(0,N_nodes/2):
            if random.random() <= rho:
                res[node] = P
            else:
                res[node] = -P
        for node in range(N_nodes/2,N_nodes):
            if random.random() <= rho:
                res[node] = -P
            else:
                res[node] = P
        if np.sum(res) == 0:
            search = False
    return res
	
def generate_random_capacity_distribution( N_nodes, adja, order_of_magnitude = 2*10**1 ):
    '''Generate a random distribution for the line capacities
	   The order_of_magnitude parameter should be consistent with the one of the power distribution, otherwise there will be flow problems
    '''
    assert N_nodes == adja.shape[0]
    assert N_nodes == adja.shape[1]
    Pmax = np.random.normal( order_of_magnitude, .1 * order_of_magnitude, size = ( 1, N_nodes**2 ) )[0]
    Pmax = Pmax * adja.flatten()
    Pmax = Pmax.reshape((N_nodes, N_nodes))
    for i in range(N_nodes):
        if np.sum( adja[i] ) != 0:
            Pmax[i][i] = np.sum( Pmax[i] ) / float( np.sum( adja[i] ) )
        else:
            Pmax[i][i] = 0
    return Pmax
	

	
def main():

	N_nodes = 200
	p_out = .05
	monte_carlo = 100
	results = []

	p_values = [.1,.3,.5,.7,.9]
	c_values = [2,4,5,8,10,20,25,40,50]
	val = 0

	with progress.Bar(label="Progress: ", expected_size=len(p_values)*len(c_values)*monte_carlo) as bar:
		for p_in in [.1,.3,.5,.7,.9]:
			results.append([])
	
			for N_clusters in [2,4,5,8,10,20,25,40,50]:
				AVG = []
		
				for mc in range( monte_carlo ):

					#Random Initial state
					theta0  = np.random.normal( loc = 0, scale = .01, size = (1,N_nodes) )[0] #Random small phase angles
					dtheta0 = np.random.normal( loc = 0, scale = .01, size = (1,N_nodes) )[0] #Random small frequencies
					Y0 = np.append( theta0, dtheta0 )
					Y0 = np.append( Y0, 1 )

					#Random Initial state
					thetaf  = np.random.normal( loc = 0, scale = .01, size = (1,N_nodes) )[0] #Random small phase angles
					dthetaf = np.random.normal( loc = 0, scale = .01, size = (1,N_nodes) )[0] #Random small frequencies
					Yf = np.append( thetaf, dthetaf )
					Yf = np.append( Yf, 1 )
            
					G = nx.planted_partition_graph(N_clusters, N_nodes/N_clusters, p_in, p_out)
					adja = np.array( nx.to_numpy_matrix( G ) )
					L = control2.get_Laplacian( adja )
					Psource       = generate_random_power_distribution( N_nodes )
					Pmax          = generate_random_capacity_distribution( N_nodes, adja )
					K             =  Pmax  / float( I * Omega )
					A             = control2.build_transition_matrix( Psource, L, Omega, I, K, alpha, Dt )
					A_powers = [ np.identity( A.shape[0] ), A ]
					for k in range( control_time + 1 )[2:]:
						A_powers.append( np.dot( A, A_powers[-1] ) )
					storage_level = 100 * np.ones((1,N_nodes))[0]
            
					drivers = control2.rank_submodular_greedy_lazy( A_powers, adja, range( N_nodes ), control_time, storage_level, max_capacity, r, I, Omega, Y0, 
Yf, True )
					AVG.append( len(drivers) )
					val += 1
					bar.show(val)
				
				results[-1].append( np.mean( AVG ) )
			
	with open('./results.json', 'w') as f:
		json.dump(results,f)
		
if __name__ == "__main__":
	main()
