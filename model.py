"""
model.py
Author: Shane Barratt

This script fits models to landing and takeoff data in folder specified by the argument --data and saves the trained models in the folder specified by the argument --experiment.

$ python model.py --data data/jfk --experiment run0/
"""

import argparse
import pickle
import os

import numpy as np

from sklearn.cluster import KMeans

from utils import train_test_split

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='location of data folder')
    parser.add_argument('--experiment', required=True, help='name folder to save models in')
    opt = parser.parse_args()
    print (opt)

    os.makedirs(os.path.join(opt.data, opt.experiment), exist_ok=True)
    names = ['landings', 'takeoffs']

    for name in names:
        print (name)
        traj_fname = os.path.join(opt.data, '%s-smooth.pkl' % name)
        trajs = pickle.load(open(traj_fname, 'rb'))

        L = trajs[0]['data'].shape[0]
        N = len(trajs)

        X = np.zeros((N, L*3))
        for i, traj in enumerate(trajs):
            traj_data = traj['data']
            traj_data = traj_data.reshape(traj_data.shape[0]*traj_data.shape[1])
            X[i, :] = traj_data

        to_delete = np.where(np.linalg.norm(X, axis=1) > 800000)[0]
        print("Deleting", len(to_delete), "for being too far")
        X = np.delete(X, to_delete, axis=0)

        to_delete = []
        for i, traj in enumerate(X):
            traj = traj.reshape(traj.shape[0]//3, 3)
            if np.max(np.linalg.norm(np.diff(traj, axis=0), axis=1)) >250:
                to_delete.append(i)
        print("Deleting", len(to_delete), "for being too fast")
        X = np.delete(X, to_delete, axis=0)

        to_delete = np.where(np.max(X[:, 2::3], axis=1) > 12500.)[0]
        print ("Deleting", len(to_delete), "for being too high")
        X = np.delete(X, to_delete, axis=0)

        to_delete = np.where(np.min(X[:, 2::3], axis=1) < -1000.)[0]
        print ("Deleting", len(to_delete), "for being too low")
        X = np.delete(X, to_delete, axis=0)

        X_trainp, X_test = train_test_split(X, fraction_val=.2)
        X_train, X_val = train_test_split(X_trainp, fraction_val=.25)

        best_prob, best_k = -float("inf"), 25
        for k in [1]+list(range(5,200,5)):
            import IPython as ipy; ipy.embed()
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(X)
            labels, centers = kmeans.labels_, kmeans.cluster_centers_
            means, covs, cluster_probs = [], [], []
            for j in range(k):
                cluster_data = X[labels == j]
                mean = centers[j]
                if cluster_data.shape[0] == 1:
                    cov = np.zeros((L*3, L*3))
                else:
                    cov = np.cov((cluster_data - mean).T)
                print (cluster_data.shape[0], L*3)
                means.append(mean)
                covs.append(cov)
                cluster_probs.append(np.sum(labels == j)/X.shape[0])

            experiment_dir = os.path.join(opt.data, opt.experiment)
            with open(os.path.join(experiment_dir, '%s-%d.model' % (name, k)), 'wb') as f:
                pickle.dump({
                    'means': means,
                    'covs': covs,
                    'cluster_probs': cluster_probs,
                    'X_test': 'X_test_%s.npy' % name,
                    'k': k
                }, f)
                np.save(os.path.join(experiment_dir, 'X_test_%s' % name), X_test)
            print (k)
        print ("\n")
