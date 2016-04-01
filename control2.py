import numpy as np
import math
import matplotlib.pyplot as plt
import networkx as nx
import random
import operator
import itertools
import copy
import operator


def sort_dico_by_values(dico):
    '''Sort dico in reverse order
    '''
    return sorted(dico.items(), key=operator.itemgetter(1), reverse=True)


def is_invertible(a):
    '''Check if matrix a is invertible
    '''
    return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]


def get_Laplacian(adjacency):
    '''Returns the Laplacian matrix from the given adjacency matrix
    '''
    return np.diag(np.sum(adjacency, axis=1)) - adjacency


def get_adja_from_laplacian(laplacian):
    '''Returns the adjacency matrix from the given Laplacian matrix
    '''
    return np.diag(np.diagonal(laplacian)) - laplacian


def max_multiplicity(spectrum):
    '''Given a list, returns a tuple with the element that appears the maximum and the number of time it appears
	   The equality is tested at epsilon
    '''
    epsilon = 10 ** -3
    maxi = 0;
    j = 1;
    ll = None
    sorted_spectrum = sorted(spectrum)
    for i, l in enumerate(sorted_spectrum):
        while i + j < len(sorted_spectrum) and abs(sorted_spectrum[i + j] - l) <= epsilon:
            j += 1
        if j > maxi:
            maxi = j
            ll = l
    return (ll, maxi)


def build_transition_matrix(Psource, L, Omega, I, K, alpha, Dt):
    '''From power distribution Psource, topology laplacian L, main frequency Omega,
	   inertia I, coupling K, and discretization step Dt,
	   returns the transition matrix A
    '''
    N_osc = len(Psource)
    KD = float(I * alpha) / float(2)
    w = (float(1) / float(I * Omega) * Psource)
    w2 = np.append(np.zeros((1, N_osc)), w)
    M2 = np.array(np.bmat([[np.identity(N_osc), np.identity(N_osc) * Dt],
                           [-1 * K * L * Dt, (1 - alpha * Dt) * np.identity(N_osc)]]))
    A = np.array(np.bmat([[M2, w2[:, np.newaxis] * Dt]]))
    A = np.vstack((A, np.append(np.zeros((1, M2.shape[1])), np.array([1]))))
    return A


def get_control_matrix_2(driver_nodes_vector):
    '''Returns the control matrix B corresponding to a given driver node set
       The control is assumed to be on the frequencies of the oscillators
    '''
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


def get_gramian(A_powers, B, control_time):
    '''Returns the Gramian matrix for transition matrix A, control matrix B, and control_time
    '''
    BT = np.transpose(B)
    BBT = np.dot(B, BT)
    W = np.dot(A_powers[0], np.dot(BBT, np.transpose(A_powers[0])))
    for k in range(1, control_time + 1):
        ABBT = np.dot(A_powers[k], BBT)
        W = W + np.dot(ABBT, np.transpose(A_powers[k]))
    return W


'''
def get_opt_control( A, B, W, initial_state, final_state, control_time ):
    Returns the control inputs u_star that minimize control energy for
	   transition matrix A, control matrix B, Gramian matrix W, control_time
	   and initial state and final state

    N_nodes = (A.shape[0] -1 ) / 2
    BT = np.transpose( B )
    AT = np.transpose( A )
    ATk_1 = np.identity(len(A))
    W_pinv =  np.linalg.pinv( W )
    vf = final_state - np.dot( np.linalg.matrix_power(A, control_time), initial_state)
    u_star = np.zeros( ( control_time, B.shape[1] ) )
    u_star[-1] = np.dot( np.dot( BT, W_pinv ), vf )
    for t in np.array( range( control_time )[::-2] ):
        ATk = np.dot(AT,ATk_1)
        u_star[t] = np.dot( np.dot( np.dot( BT, ATk ), W_pinv ), vf )
    return u_star
'''


def full_control(A_powers, S, control_time):
    N_nodes = (A_powers[0].shape[0] - 1) / 2
    drivers = np.zeros((1, N_nodes))[0]
    drivers[S] = 1
    B = get_control_matrix_2(drivers)
    W = get_gramian(A_powers,B,control_time)
    rank = np.linalg.matrix_rank(W)
    return rank >= 2 * N_nodes

def get_opt_control(A, B, W, initial_state, final_state, control_time):
    N_nodes = (A.shape[0] - 1) / 2
    BT = np.transpose(B)
    AT = np.transpose(A)
    W_pinv = np.linalg.pinv(W)
    vf = final_state - np.dot(np.linalg.matrix_power(A, control_time), initial_state)
    u_star = np.zeros((control_time, B.shape[1]))
    for t in np.array(range(control_time)):
        u_star[t] = np.dot(np.dot(np.dot(BT, np.linalg.matrix_power(AT, control_time - t - 1)), W_pinv), vf)
    return u_star


def compute_control_energy(A, W, control_time, initial_state, final_state):
    '''Returns the control energy for
	   transition matrix A, Gramian matrix W, control_time
	   and initial state and final state
	'''
    vf = final_state - np.dot(np.linalg.matrix_power(A, control_time), initial_state)
    return np.dot(np.dot(np.transpose(vf), np.linalg.pinv(W)), vf)


def check_batteries(storage_level, u_star, drivers, max_capacity, r, I, Omega):
    '''Check if a given control sequence u_star is valid for a given storage level
	   vector (situation at the begining of the control phase)
       max_capacity = maximum capacity of the batteries (change this into a vector if you want different capacities)
       r = maximum charge / discharge rate for the batteries	   
    '''

    storage_level2 = copy.deepcopy(storage_level)
    control_time = u_star.shape[0]

    for t in range(control_time):

        if np.any(I * Omega * u_star[t] > r):
            return False

        storage_level2[drivers != 0] -= I * Omega * u_star[t]

        if np.any(storage_level2 > max_capacity):
            return False

        if np.any(storage_level2 < 0):
            return False

    return True


def check_flows(A, adja, B, u_star, initial, final, control_time):
    '''Check if no electrical line is overloaded during the control phase with control signals u_star
	   The flows are calculated from the phase angle differences
    '''
    N_nodes = (A.shape[0] - 1) / 2
    time = np.array(range(control_time))
    Y = np.zeros((control_time, 2 * N_nodes + 1))
    Y[0] = initial

    for t, T in enumerate(time[:-1]):
        Y[t + 1] = np.dot(A, Y[t]) + np.dot(B, u_star[t])

    for i in range(N_nodes):
        for j in range(N_nodes)[i:]:
            if adja[i][j] != 0:
                if np.any(np.abs(Y[:, j] - Y[:, i]) > 1):
                    return False
    return True


def F_trace(A_powers, S, control_time):
    '''Computes the trace of the Gramian WS induced by driver set S
    '''
    N_nodes = (A_powers[0].shape[0] - 1) / 2
    drivers = np.zeros((1, N_nodes))[0]
    drivers[S] = 1
    BS = get_control_matrix_2(drivers)
    WS = get_gramian(A_powers, BS, control_time)
    return np.trace(WS)


def F_trace_inv(A_powers, S, control_time):
    '''Computes the trace of the pseudo inverse of the Gramian WS induced by driver set S
    '''
    N_nodes = (A_powers[0].shape[0] - 1) / 2
    drivers = np.zeros((1, N_nodes))[0]
    drivers[S] = 1
    BS = get_control_matrix_2(drivers)
    WS = get_gramian(A_powers, BS, control_time)
    return -1 * np.trace(np.linalg.pinv(WS))


def F_rank(A_powers, S, control_time):
    '''Computes the rank of the Gramian WS induced by driver set S
    '''
    N_nodes = (A_powers[0].shape[0] - 1) / 2
    drivers = np.zeros((1, N_nodes))[0]
    drivers[S] = 1
    BS = get_control_matrix_2(drivers)
    WS = get_gramian(A_powers, BS, control_time)
    return np.linalg.matrix_rank(WS)


def rank_submodular_greedy_lazy(A_powers, adja, pool, control_time, storage_level, max_capacity, r, I, Omega,
                                initial_state, final_state, constrained):
    '''Performs a greedy lazy optimization using the Gramian rank based metric
	   The algorithm stops as soon as the Gramian is full rank
	   At each step, the node yielding the best improvement in the rank metric is added 
    '''

    S = []  # Optimal set of drivers
    np.random.shuffle(pool)  # The pool of nodes is shuffled
    N_nodes = (A_powers[0].shape[0] - 1) / 2

    # Dictionary of initial marginal gains
    # F is computed for each node...
    Delta = {k: v for k, v in zip(pool, [F_rank(A_powers, [i], control_time) for i in pool])}

    # Marginal gains are sorted
    sorted_Delta = sort_dico_by_values(Delta)

    # Optimal set is initialized with the best node
    S.append(sorted_Delta[0][0])
    pool.remove(sorted_Delta[0][0])
    del Delta[sorted_Delta[0][0]]
    FS = sorted_Delta[0][1]

    constrained_control = False

    # While not full control and if there are still nodes in the pool
    while not constrained_control and len(S) < N_nodes:

        i = 0;
        search = True
        sorted_Delta = sort_dico_by_values(Delta)
        FS = F_rank(A_powers, S, control_time)

        while search and i < len(sorted_Delta) - 1:
            sorted_Delta = sort_dico_by_values(Delta)
            marginal = F_rank(A_powers, [sorted_Delta[0][0]] + S, control_time) - FS

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
                sorted_Delta = sort_dico_by_values(Delta)
                S.append(sorted_Delta[0][0])
                pool.remove(sorted_Delta[0][0])
                del Delta[sorted_Delta[0][0]]
                FS = FS + sorted_Delta[0][1]
            except:
                pass

        drivers = np.zeros((1, N_nodes))[0]
        drivers[S] = 1
        B = get_control_matrix_2(drivers)
        W = get_gramian(A_powers, B, control_time)

        if np.linalg.matrix_rank(W) >= 2 * N_nodes:
            if not constrained:
                constrained_control = True
            else:
                u_star = get_opt_control(A_powers[1], B, W, initial_state, final_state, control_time)

                if (check_batteries(storage_level, u_star, drivers, max_capacity, r, I, Omega) and
                        check_flows(A_powers[1], adja, B, u_star, initial_state, final_state, control_time)):
                    constrained_control = True
    return S


def rank_submodular_greedy_lazy_size_limit(A_powers, pool, control_time, size):
    '''Performs a greedy lazy optimization using the Gramian rank based metric
	   The algorithm stops as soon as the driver set is of a given size
	   At each step, the node yielding the best improvement in the rank metric is added 
    '''

    S = []  # Optimal set of drivers
    np.random.shuffle(pool)  # The pool of nodes is shuffled
    N_nodes = (A_powers[0].shape[0] - 1) / 2

    # Dictionary of initial marginal gains
    # F is computed for each node...
    Delta = {k: v for k, v in zip(pool, [F_rank(A_powers, [i], control_time) for i in pool])}

    # Marginal gains are sorted
    sorted_Delta = sort_dico_by_values(Delta)

    # Optimal set is initialized with the best node
    S.append(sorted_Delta[0][0])
    pool.remove(sorted_Delta[0][0])
    del Delta[sorted_Delta[0][0]]
    FS = sorted_Delta[0][1]

    # While not full control and if there are still nodes in the pool
    while len(S) < size:

        i = 0;
        search = True
        sorted_Delta = sort_dico_by_values(Delta)
        FS = F_rank(A_powers, S, control_time)

        while search and i < len(sorted_Delta) - 1:
            sorted_Delta = sort_dico_by_values(Delta)
            marginal = F_rank(A_powers, [sorted_Delta[0][0]] + S, control_time) - FS

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
                sorted_Delta = sort_dico_by_values(Delta)
                S.append(sorted_Delta[0][0])
                pool.remove(sorted_Delta[0][0])
                del Delta[sorted_Delta[0][0]]
                FS = FS + sorted_Delta[0][1]
            except:
                pass

    return S


def trace_inv_submodular_greedy_lazy_size_limit(A_powers, pool, control_time, size_limit):
    '''Performs a greedy lazy optimization using the pseudo inverse Gramian trace based metric
	   The algorithm stops as soon as a driver set satisfying all constraints is found
	   At each step, the node yielding the best improvement in the metric is added
    '''

    S = []
    np.random.shuffle(pool)
    N_nodes = (A_powers[0].shape[0] - 1) / 2

    Delta = {k: v for k, v in zip(pool, [F_trace_inv(A_powers, [i], control_time) for i in pool])}
    sorted_Delta = sort_dico_by_values(Delta)

    S.append(sorted_Delta[0][0])
    pool.remove(sorted_Delta[0][0])
    del Delta[sorted_Delta[0][0]]
    FS = sorted_Delta[0][1]


    while len(S) < size_limit:

        i = 0
        search = True
        sorted_Delta = sort_dico_by_values(Delta)
        FS = F_trace_inv(A_powers, S, control_time)

        while search and i < len(sorted_Delta) - 1:
            sorted_Delta = sort_dico_by_values(Delta)
            marginal = F_trace_inv(A_powers, [sorted_Delta[0][0]] + S, control_time) - FS

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
                sorted_Delta = sort_dico_by_values(Delta)
                S.append(sorted_Delta[0][0])
                pool.remove(sorted_Delta[0][0])
                del Delta[sorted_Delta[0][0]]
                FS = FS + sorted_Delta[0][1]
            except:
                pass
    return S



def trace_inv_submodular_greedy_lazy(A_powers, adja, pool, control_time, storage_level, max_capacity, r, I, Omega,
                                     initial_state, final_state, constrained):
    '''Performs a greedy lazy optimization using the pseudo inverse Gramian trace based metric
	   The algorithm stops as soon as a driver set satisfying all constraints is found
	   At each step, the node yielding the best improvement in the metric is added 
    '''

    S = []
    np.random.shuffle(pool)
    N_nodes = (A_powers[0].shape[0] - 1) / 2

    Delta = {k: v for k, v in zip(pool, [F_trace_inv(A_powers, [i], control_time) for i in pool])}
    sorted_Delta = sort_dico_by_values(Delta)

    S.append(sorted_Delta[0][0])
    pool.remove(sorted_Delta[0][0])
    del Delta[sorted_Delta[0][0]]
    FS = sorted_Delta[0][1]

    constrained_control = False

    while not constrained_control and len(S) < N_nodes:

        i = 0;
        search = True
        sorted_Delta = sort_dico_by_values(Delta)
        FS = F_trace_inv(A_powers, S, control_time)

        while search and i < len(sorted_Delta) - 1:
            sorted_Delta = sort_dico_by_values(Delta)
            marginal = F_trace_inv(A_powers, [sorted_Delta[0][0]] + S, control_time) - FS

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
                sorted_Delta = sort_dico_by_values(Delta)
                S.append(sorted_Delta[0][0])
                pool.remove(sorted_Delta[0][0])
                del Delta[sorted_Delta[0][0]]
                FS = FS + sorted_Delta[0][1]
            except:
                pass

        drivers = np.zeros((1, N_nodes))[0]
        drivers[S] = 1
        B = get_control_matrix_2(drivers)
        W = get_gramian(A_powers, B, control_time)

        if np.linalg.matrix_rank(W) >= 2 * N_nodes:
            if not constrained:
                constrained_control = True
            else:
                u_star = get_opt_control(A_powers[1], B, W, initial_state, final_state, control_time)

                if (check_batteries(storage_level, u_star, drivers, max_capacity, r, I, Omega) and
                        check_flows(A_powers[1], adja, B, u_star, initial_state, final_state, control_time)):
                    constrained_control = True

    return S


def trace_modular_greedy(A_powers, adja, pool, control_time, storage_level, max_capacity, r, I, Omega, initial_state,
                         final_state, constrained):
    '''Performs a greedy modular optimization using the trace of the Gramian
	   The algorithm stops as soon as a driver set satisfying all constraints is found
    '''

    S = []
    np.random.shuffle(pool)
    N_nodes = (A_powers[0].shape[0] - 1) / 2

    Delta = {k: v for k, v in zip(pool, [F_trace(A_powers, [i], control_time) for i in pool])}
    sorted_Delta = sort_dico_by_values(Delta)
    constrained_control = False
    i = -1

    while not constrained_control and len(S) < N_nodes:
        i += 1
        S.append(sorted_Delta[i][0])

        drivers = np.zeros((1, N_nodes))[0]
        drivers[S] = 1
        B = get_control_matrix_2(drivers)
        W = get_gramian(A_powers, B, control_time)

        if np.linalg.matrix_rank(W) >= 2 * N_nodes:
            if not constrained:
                constrained_control = True
            else:
                u_star = get_opt_control(A_powers[1], B, W, initial_state, final_state, control_time)

                if (check_batteries(storage_level, u_star, drivers, max_capacity, r, I, Omega) and
                        check_flows(A_powers[1], adja, B, u_star, initial_state, final_state, control_time)):
                    constrained_control = True

    return S
