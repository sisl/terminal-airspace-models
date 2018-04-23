"""
smooth.py
Author: Shane Barratt

Smooth trajectories in trajs-raw.pkl and save in trajs-smooth.pkl, e.g.

python smooth.py --data data/kfrg --num_workers 6 --L_landings 200 --L_takeoffs 200 --num_extend 20
python smooth.py --data data/jfk --num_workers 6 --L_landings 125 --L_takeoffs 70 --num_extend 15
python smooth.py --data data/28j --num_workers 6 --L_landings 125 --L_takeoffs 70 --num_extend 15
"""

import argparse
import multiprocessing as mp
import pickle
import os
import gc

import numpy as np
from sklearn.utils import shuffle
import IPython as ipy
from scipy.linalg import solve_banded
import bandmat as bm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils import train_test_split

def smooth(a, Phat, N, lambda_1, lambda_2):
    """
    a: (N,) number of measurements at that timestep
    Phat: (N, 3) sum of measurements at that timestep
    N: num time steps
    lambda_1, lambda_2: regularization parameters

    solves the optimization problem (over P \in R^{Tx3}):
    minimize ||diag(a)*P-Phat||^2 + lambda_1/N*||D_2*P||^2 + lambda_2/N*||D_3*P||^2

    returns:
        - P: (N, 3) matrix with full trajectory
    """
    # A in Banded Matrix form
    A = bm.diag(1.*a)

    # D_2 and D_3 in Banded Matrix form transposed
    D_2_bm_T = bm.BandMat(
        1, 1,
        np.hstack([np.zeros((3, 1)), np.repeat([[1.], [-2.], [1.]], N-2, axis=1), np.zeros((3, 1))])
    )
    D_3_bm_T = bm.BandMat(
        2, 2,
        np.hstack([np.zeros((5, 2)), np.repeat([[-1.], [2.], [0.], [-2.], [1.]], N-4, axis=1), np.zeros((5, 2))])
    )

    # XP=B normal equations
    X = bm.dot_mm(A, A) + lambda_1/N * bm.dot_mm(D_2_bm_T, D_2_bm_T.T) + lambda_2/N * bm.dot_mm(D_3_bm_T, D_3_bm_T.T)
    l_and_u = (X.l, X.u) # lower and upper band bounds
    B = np.hstack([
            np.expand_dims(bm.dot_mv(A, Phat[:, 0]), -1),
            np.expand_dims(bm.dot_mv(A, Phat[:, 1]), -1),
            np.expand_dims(bm.dot_mv(A, Phat[:, 2]), -1)
    ])

    # solve normal equations
    P = solve_banded(l_and_u, X.data, B)

    return P

def val_loss(a, P, Phat):
    return np.linalg.norm(np.diag(a).dot(P-Phat), 2)

def convert_dataset(data, L):
    Phat = np.zeros((L, 3))
    a = np.zeros(L)
    for i in range(data.shape[0]):
        t, p = int(data[i, 0]), data[i, 1:]
        if t < L:
            Phat[t, :] += p
            a[t] += 1

    return Phat, a

def smooth_cross_validation(traj, fraction_val=.3, L=225):
    N = max(L, int(np.max(traj[:, 0])))

    train, test = train_test_split(traj, fraction_val)
    Phat_train, a_train = convert_dataset(train, N)
    Phat_test, a_test = convert_dataset(test, N)

    best_loss, best_lambda_1, best_lambda_2 = float("inf"), 1., 1.
    train, test = train_test_split(traj, fraction_val)
    Phat_train, a_train = convert_dataset(train, N)
    Phat_test, a_test = convert_dataset(test, N)
    for log_lambda_1 in np.arange(0, 8):
        for log_lambda_2 in np.arange(0, 8):
            P = smooth(a_train, Phat_train, N, 10**log_lambda_1, 10**log_lambda_2)
            loss = val_loss(a_test, P, Phat_test) / 50.
            if loss < best_loss:
                best_loss = loss
                best_lambda_1, best_lambda_2 = 10**log_lambda_1, 10**log_lambda_2

    Phat, a = convert_dataset(traj, N)
    return smooth(a, Phat, N, best_lambda_1, best_lambda_2)

def f(x):
    x, L, i, data, fname = x
    traj = np.copy(x['data'][:, :4])

    traj_smooth = smooth_cross_validation(traj, fraction_val=.4, L=L)
    traj_smooth = traj_smooth[:L, :]

    plt.figure(figsize=(14, 10))
    plt.xlim([-12000, 12000])
    plt.ylim([-12000, 12000])
    plt.scatter(traj[:, 1], traj[:, 2], s=6)
    plt.plot(traj_smooth[:, 0], traj_smooth[:, 1], alpha=.6)
    plt.savefig(os.path.join(data, 'smoothing-figures', '%s_%06d.pdf' % (fname, i)))
    plt.close()

    x['data'] = traj_smooth
    gc.collect()

    return x

if __name__ == '__main__':
    N = 300
    traj = lambda t: np.array([1.,1.,10.]) + t*np.array([1.,-1.,1.]) + np.random.normal(size=3) # linear trajectory
    a = np.random.choice([0, 1, 2], size=N, p=[.6,.3,.1]) # probability of 0, 1 or 2 measurements
    Phat = np.zeros((N, 3))
    # Fill in measurement matrix
    for i in range(N):
        for _ in range(a[i]):
            Phat[i, :] += traj(i)
    P = smooth(a, Phat, N, 1e1, 1e1)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='location of data folder')
    parser.add_argument('--num_workers', type=int, default=1, help='number of multiprocessing CPU workers')
    parser.add_argument('--L_landings', type=int, default=None, help='length of final trajectories for landings')
    parser.add_argument('--L_takeoffs', type=int, default=None, help='length of final trajectories for takeoffs')
    parser.add_argument('--num_extend', type=int, default=15, help='max num. seconds to extend')
    opt = parser.parse_args()
    opt.num_workers = min(mp.cpu_count(), opt.num_workers)
    print (opt)

    os.makedirs(os.path.join(opt.data, 'smoothing-figures'), exist_ok=True)

    for fname in ['landings', 'takeoffs']:
        trajs = pickle.load(open(os.path.join(opt.data, '%s.pkl' % fname), 'rb'))

        traj_lengths = [np.max(t['data'][:, 0])-np.min(t['data'][:, 0]) for t in trajs]
        if fname == 'landings':
            if opt.L_landings is None:
                L = int(np.median(traj_lengths))
            else:
                L = opt.L_landings
        else:
            if opt.L_takeoffs is None:
                L = int(np.median(traj_lengths))
            else:
                L = opt.L_takeoffs
        print (fname)
        print ("Using L=%d, num_extend=%d" % (L, opt.num_extend))

        trajs_new = []
        for traj in trajs:
            if fname == 'landings':
                traj['start_time'] = traj['data'][-1, 0]
                traj['data'][:, 0] = -traj['data'][:, 0]
            elif fname == 'takeoffs':
                traj['start_time'] = traj['data'][0, 0]
            traj['data'][:, 0] -= np.min(traj['data'][:, 0])

            if np.max(traj['data'][:, 0]) < L-opt.num_extend: # ignore trajs less than L-50 length
                continue
            trajs_new.append(traj)

        pool = mp.Pool(opt.num_workers)
        trajs_out = pool.map_async(f, [(t, L, i, opt.data, fname) for i, t in enumerate(trajs_new)])
        trajs_out = trajs_out.get()

        pool.close()
        pool.join()

        pickle.dump(trajs_out, open(os.path.join(opt.data, '%s-smooth.pkl' % fname), 'wb'))
