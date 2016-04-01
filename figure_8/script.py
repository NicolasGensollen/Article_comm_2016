import numpy as np
import random
import math
import json
import networkx as nx
from networkx.readwrite.json_graph.node_link import node_link_graph
from scipy.sparse import issparse
from scipy.sparse import csr_matrix
from scipy.sparse import identity
from scipy.sparse.linalg import svds
import control2
import matplotlib.pyplot as plt

def get_control_matrix_2(driver_nodes_vector):
    N_osc = len(driver_nodes_vector)
    N_D = len(driver_nodes_vector[driver_nodes_vector == 1])
    if N_D == 0:
        return 0
    Q = np.diagflat(driver_nodes_vector)
    if driver_nodes_vector[0] == 1:
        Q2 = Q[:, 0]
        Q2 = np.hstack((Q2[:, np.newaxis], Q[:, np.arange(0, N_osc)[driver_nodes_vector * np.arange(0, N_osc) != 0]]))
    else:
        Q2 = Q[:, np.arange(0, N_osc)[driver_nodes_vector * np.arange(0, N_osc) != 0]]
    assert Q2.shape[0] == N_osc
    assert Q2.shape[1] == N_D
    return np.array(np.bmat([[np.zeros((N_osc, N_D))],
                             [Q2],
                             [np.zeros((1, N_D))]]))

def get_gramian(A_sparse_powers, B, control_time):
    BT = B.T
    BBT = B.dot(BT)
    W = A_sparse_powers[0].dot(BBT).dot( A_sparse_powers[0].T )
    for k in range(1, control_time + 1):
        ABBT = A_sparse_powers[k].dot( BBT )
        W = W + ABBT.dot( A_sparse_powers[k].T )
    return W

def F_trace(A_sparse_powers, S, control_time):
    N_nodes = (A_sparse_powers[0].shape[0] - 1) / 2
    drivers = np.zeros((1, N_nodes))[0]
    drivers[S] = 1
    BS = csr_matrix( get_control_matrix_2(drivers) )
    WS = get_gramian(A_sparse_powers, BS, control_time)
    return WS.diagonal().sum()

def F_rank(A_sparse_powers, S, control_time):
    N_nodes = (A_sparse_powers[0].shape[0] - 1) / 2
    drivers = np.zeros((1, N_nodes))[0]
    drivers[S] = 1
    BS = csr_matrix( get_control_matrix_2(drivers) )
    WS = get_gramian(A_sparse_powers, BS, control_time)
    return np.linalg.matrix_rank(WS.todense())


def F_rank2(A_sparse_powers, S, control_time):
    N_nodes = (A_sparse_powers[0].shape[0] - 1) / 2
    drivers = np.zeros((1, N_nodes))[0]
    drivers[S] = 1
    BS = csr_matrix( get_control_matrix_2(drivers) )
    WS = get_gramian(A_sparse_powers, BS, control_time)
    try:
        sigs = svds(WS, k = WS.shape[0] - 1)[1]
        return len([x for x in sigs if abs(x) > 10**-12])
    except:
        return 0


def F_rank3(A, S, control_time):
    N_nodes = (A.shape[0] - 1) / 2
    drivers = np.zeros((1, N_nodes))[0]
    drivers[S] = 1
    BS = csr_matrix( get_control_matrix_2(drivers) )
    WS = get_gramian(A, BS, control_time)
    k=3
    while k<WS.shape[0]-1:
        sigs = svds(WS, k = k)[1]
        if len([x for x in sigs if abs(x) < 10**-12])>0:
            return len([x for x in sigs if abs(x) > 10**-12])
        k+=1
    return WS.shape[0]



def trace_modular_greedy(A_sparse_powers, pool, control_time):
    S = []
    np.random.shuffle(pool)
    N_nodes = (A_sparse_powers[0].shape[0] - 1) / 2

    Delta = {k: v for k, v in zip(pool, [F_trace(A_sparse_powers, [i], control_time) for i in pool])}
    sorted_Delta = control2.sort_dico_by_values(Delta)
    return sorted_Delta


def rank_submodular_greedy_lazy_size_limit(A_sparse_powers, pool, control_time, size):

    S = []  # Optimal set of drivers
    np.random.shuffle(pool)  # The pool of nodes is shuffled
    N_nodes = (A_sparse_powers[0].shape[0] - 1) / 2

    # Dictionary of initial marginal gains
    # F is computed for each node...
    Delta = {k: v for k, v in zip(pool, [F_rank(A_sparse_powers, [i], control_time) for i in pool])}

    # Marginal gains are sorted
    sorted_Delta = control2.sort_dico_by_values(Delta)

    # Optimal set is initialized with the best node
    S.append(sorted_Delta[0][0])
    pool.remove(sorted_Delta[0][0])
    del Delta[sorted_Delta[0][0]]
    FS = sorted_Delta[0][1]

    # While not full control and if there are still nodes in the pool
    while len(S) < size:

        i = 0
        print i
        search = True
        sorted_Delta = control2.sort_dico_by_values(Delta)
        FS = F_rank(A_sparse_powers, S, control_time)

        while search and i < len(sorted_Delta) - 1:
            sorted_Delta = control2.sort_dico_by_values(Delta)
            marginal = F_rank(A_sparse_powers, [sorted_Delta[0][0]] + S, control_time) - FS

            if marginal >= sorted_Delta[1][1]:
                S.append(sorted_Delta[0][0])
                pool.remove(sorted_Delta[0][0])
                del Delta[sorted_Delta[0][0]]
                FS = marginal + FS
                search = False

            else:
                Delta[sorted_Delta[0][0]] = marginal
                i += 1

        if search:
            try:
                sorted_Delta = control2.sort_dico_by_values(Delta)
                S.append(sorted_Delta[0][0])
                pool.remove(sorted_Delta[0][0])
                del Delta[sorted_Delta[0][0]]
                FS = FS + sorted_Delta[0][1]
            except:
                pass

    return S


def get_driv(sorted_Delta, desired_size):
    S=[]
    i = -1

    while len(S) < desired_size:
        i += 1
        S.append(sorted_Delta[i][0])

    return S

def compute_control_energy(A_sparse_powers, W, control_time, initial_state, final_state):

    vf = final_state - A_sparse_powers[control_time].dot(initial_state)
    Wd = W.todense()
    Wpinv = np.linalg.pinv(Wd)
    return np.transpose(vf).dot(Wpinv).dot(vf)


#Parameters
Omega = 50. / (2 * math.pi)  # Main frequency is 50Hz
I = .1  # Inertia
KD = .1  # Dissipation constant
Tmax = 50  # Maximum time
Dt = .01  # Discretization step
time = np.arange(0, Tmax, Dt)  # Time
alpha = float(2 * KD) / float(I)  # Damping
control_time = 10  # control time T
r = 5  # maximum charge / discharge rate
max_capacity = 150  # maximum capacity for the batteries


#Load data
with open('./A.json', 'r') as f:
    A = np.array(json.load(f))
with open('./K.json', 'r') as f:
    K = np.array(json.load(f))
'''
with open('./G2.json', 'r') as f:
    G = node_link_graph(json.load(f))
'''
G=nx.read_gml('./G.gml')

A = np.array(A)
assert not np.isnan(A).any()
assert not np.isinf(A).any()

A_sparse = csr_matrix(A)
A_sparse_powers = [ identity(A_sparse.shape[0] ), A_sparse ]
for i in range(2,control_time+1):
    A_sparse_powers.append( A_sparse.dot( A_sparse_powers[-1] ) )

N_nodes = G.number_of_nodes()

def get_random_states():
    global N_nodes
    # Random Initial state
    theta0 = np.random.normal(loc=0, scale=.01, size=(1, N_nodes))[0]  # Random small phase angles
    dtheta0 = np.random.normal(loc=0, scale=.01, size=(1, N_nodes))[0]  # Random small frequencies
    Y0 = np.append(theta0, dtheta0)
    Y0 = np.append(Y0, 1)

    # Random Initial state
    thetaf = np.random.normal(loc=0, scale=.01, size=(1, N_nodes))[0]  # Random small phase angles
    dthetaf = np.random.normal(loc=0, scale=.01, size=(1, N_nodes))[0]  # Random small frequencies
    Yf = np.append(thetaf, dthetaf)
    Yf = np.append(Yf, 1)
    return Y0,Yf

corres = {k:v for k,v in zip(range(G.number_of_nodes()), G.nodes())}
margs  = trace_modular_greedy(A_sparse_powers, corres.keys(), control_time)

with open('./margs.json','w') as f:
    json.dump(margs,f)

RANKS = []
ENERGIES = []
ENERGIES_RAND = []

print margs

for size in range(1,N_nodes-1):
    S = get_driv(margs,size)
    print size
    drivers = np.zeros((1, N_nodes))[0]
    drivers[S] = 1
    B = csr_matrix( get_control_matrix_2(drivers) )
    W = get_gramian(A_sparse_powers, B, control_time)
    rrr = np.linalg.matrix_rank(W.todense())
    RANKS.append(rrr)
    if rrr >= 2 * N_nodes:
        AVG = []
        for mm in range(20):
            Y0,Yf = get_random_states()
            ene = compute_control_energy(A_sparse_powers, W, control_time, Y0, Yf)
            AVG.append(ene)
        ENERGIES.append( np.mean(AVG) )
        print ENERGIES[-1]
        AVG = []
        for mm in range(20):
            S2 = random.sample(range(N_nodes), size)
            Y0,Yf = get_random_states()
            drivers = np.zeros((1, N_nodes))[0]
            drivers[S2] = 1
            B = csr_matrix( get_control_matrix_2(drivers) )
            W = get_gramian(A_sparse_powers, B, control_time)
            ene = compute_control_energy(A_sparse_powers, W, control_time, Y0, Yf)
            AVG.append(ene)
        ENERGIES_RAND.append( np.mean( AVG ) )
        print ENERGIES_RAND[-1]
    print '-'*60

with open('./ENERGIES.json', 'w') as f:
    json.dump(ENERGIES, f)
with open('./ENERGIES_RAND.json', 'w') as f:
    json.dump(ENERGIES_RAND, f)
with open('./RANKS.json', 'w') as f:
    json.dump(RANKS, f)

'''
print F_trace(A_sparse_powers, [10,20,30,40,50], control_time)
print F_trace(A_sparse_powers, [5,100,130,150,190], control_time)
print '-'*60
print control2.F_trace(A_powers, [10,20,30,40,50], control_time)
print control2.F_trace(A_powers, [5,100,130,150,190], control_time)
print '-'*60


S1 = random.sample(range(N_nodes),100)
S2 =  random.sample(range(N_nodes),100)
print S1
print S2
drivers1 = np.zeros((1, N_nodes))[0]
drivers1[S1] = 1
#B1 = control2.get_control_matrix_2(drivers1)

B1 = csr_matrix( get_control_matrix_2(drivers1) )

drivers2 = np.zeros((1, N_nodes))[0]
drivers2[S2] = 1
#B2 = control2.get_control_matrix_2(drivers2)

B2 = csr_matrix( get_control_matrix_2(drivers2) )
W1 = get_gramian(A_sparse_powers, B1, control_time)
W2 = get_gramian(A_sparse_powers, B2, control_time)

#W1 = control2.get_gramian(A_powers, B1, control_time)
#W2 = control2.get_gramian(A_powers, B2, control_time)

print np.trace( np.linalg.pinv( W1.todense() ) )
print np.trace( np.linalg.pinv( W2.todense() ) )

#print np.trace( np.linalg.pinv( W1 ) )
#print np.trace( np.linalg.pinv( W2 ) )
#print W1!=W2
print '-'*60

print np.linalg.matrix_rank(W1.todense())
print np.linalg.matrix_rank(W2.todense())

#print np.linalg.matrix_rank(W1)
#print np.linalg.matrix_rank(W2)
print '-'*60

print compute_control_energy(A_sparse_powers, W1, control_time, Y0,Yf)
print compute_control_energy(A_sparse_powers, W2, control_time, Y0,Yf)

#print control2.compute_control_energy(A, W1, control_time, Y0,Yf)
#print control2.compute_control_energy(A, W2, control_time, Y0,Yf)

'''
'''
S = rank_submodular_greedy_lazy_size_limit(A_sparse, corres.keys(), control_time, size)
print S
drivers = np.zeros((1,.5*(A.shape[0]-1)))[0]
drivers[S] = 1
B = csr_matrix( get_control_matrix_2(drivers) )
W = get_gramian(A_sparse,B,control_time)
#sigs = svds(W,W.shape[0]-1 )[1]
#print 'rank = ' + str( len([x for x in sigs if abs(x) > 10**-12]) )
RR.append( np.linalg.matrix_rank(W.todense()) )
print RR[-1]
EE.append( compute_control_energy(A_sparse, W, control_time, Y0, Yf) )
print EE[-1]


fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(121)
plt.plot( range(N_nodes), RR )
ax2 = fig.add_subplot(122)
plt.plot( range(N_nodes), EE)
plt.savefig('./resu.pdf', dpi = 300)
'''
