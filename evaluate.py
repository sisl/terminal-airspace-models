"""
evaluate.py
Author: Shane Barratt

This script evaluates trained models on validation data.

$ python evaluate.py --data data/kfrg --experiment run0/
"""

import argparse
import pickle
import os
import copy
from glob import glob
from math import floor, ceil
import gc

import numpy as np
import pandas as pd
import IPython as ipy

from scipy.stats import entropy

from utils import log_probability, generate

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='location of data folder')
    parser.add_argument('--experiment', required=True, help='location of experiment folder')
    opt = parser.parse_args()
    print (opt)

    experiment_dir = os.path.join(opt.data, opt.experiment)
    figures_dir = os.path.join(experiment_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    for traj_type in ['takeoffs', 'landings']:
        for k in [1]+list(range(5,200,5)):
            results = {}

            model_name = '%s-%d.model' % (traj_type, k)
            model_name = os.path.join(experiment_dir, model_name)
            model = pickle.load(open(model_name, 'rb'))
            model_fname = os.path.basename(model_name)
            model_fname = model_fname[:model_fname.index('.')]
            means = model['means']
            covs = model['covs']
            cluster_probs = model['cluster_probs']
            k = model['k']
            X_test_fname = model['X_test']
            X_test = np.load(os.path.join(experiment_dir, X_test_fname))

            # Reorder covariance matrix
            j = np.argmax(cluster_probs)
            cov = covs[j]
            N = cov.shape[0]
            cov_prime = cov.reshape(int(N/3), 3, int(N/3), 3)
            cov = cov_prime.reshape(N, N, order='F')

            results['cov'] = cov
            results['means'] = means

            # SVD covs
            for i in range(k):
                u,s,vt = np.linalg.svd(covs[i])
                covs[i] = (u[:, :5], s[:5], vt[:5, :])

            # Plot most probable
            j = np.argmax(cluster_probs)
            X_samples = []
            for _ in range(25):
                X_samples.append(generate(means[j], covs[j]))
            X_samples_25 = np.array(X_samples)

            results['X_samples'] = X_samples_25
            results['most_probable'] = j

            results['log_probs'] = []
            results['X_test'] = X_test

            results['position_kls'] = []
            results['position_sample'] = []

            results['velocity_longitudinal_kls'] = []
            results['velocity_longitudinal_test'] = []
            results['velocity_longitudinal_sample'] = []

            results['velocity_vertical_kls'] = []
            results['velocity_vertical_test'] = []
            results['velocity_vertical_sample'] = []

            results['turnrate_kls'] = []
            results['turnrate_test'] = []
            results['turnrate_sample'] = []

            for _ in range(10):
                # Take X_test.shape[0] samples
                X_samples = []
                for _ in range(X_test.shape[0]):
                    j = np.random.choice(np.arange(0, len(means)), p=cluster_probs)
                    t = generate(means[j], covs[j])
                    X_samples.append(t)
                X_samples = np.array(X_samples)
                results['samples'] = X_samples

                # log_prob = log_probability(means, covs, cluster_probs, X_test)
                # results['log_probs'].append(log_prob)
            
                # KL-divergence of position histogram
                positions_test = X_test.reshape(X_test.shape[0]*X_test.shape[1]//3, 3)
                H_test, _ = np.histogramdd(positions_test, bins=100, range=[(-9000, 9000), (-9000, 9000), (0, 10000)])
                positions_samples = X_samples.reshape(X_samples.shape[0]*X_samples.shape[1]//3, 3)
                H_samples, _ = np.histogramdd(positions_samples, bins=100, range=[(-9000, 9000), (-9000, 9000), (0, 10000)])

                H_test_dithered = H_test.flatten() + 10.
                H_test_dithered /= np.sum(H_test_dithered)

                H_samples_dithered = H_samples.flatten() + 10.
                H_samples_dithered /= np.sum(H_samples_dithered)

                a = entropy(H_samples_dithered, H_test_dithered)
                results['position_kls'].append(a)
                results['position_test'] = positions_test
                results['position_sample'] = positions_samples

                # Velocity norm Longitudinal histogram
                velocities_test = np.diff(positions_test[:, :2], axis=0)
                velocity_norms_test = np.linalg.norm(velocities_test,axis=1)

                velocities_samples = np.diff(positions_samples[:, :2], axis=0)
                velocity_norms_samples = np.linalg.norm(velocities_samples,axis=1)

                l = min(np.min(velocity_norms_test), np.min(velocity_norms_samples))
                h = 150
                H_test, _ = np.histogram(velocity_norms_test, bins=100, range=(l,h))
                H_samples, _ = np.histogram(velocity_norms_samples, bins=100, range=(l,h))

                H_test_dithered = H_test + 10.
                H_test_dithered /= np.sum(H_test_dithered)

                H_samples_dithered = H_samples + 10.
                H_samples_dithered /= np.sum(H_samples_dithered)

                b = entropy(H_samples_dithered, H_test_dithered)
                results['velocity_longitudinal_kls'].append(b)
                results['velocity_longitudinal_test'] = velocity_norms_test
                results['velocity_longitudinal_sample'] = velocity_norms_samples

                # Velocity norm Vertical histogram
                # print ("Velocity Vertical norm")
                velocities_test = np.diff(positions_test[:, -1], axis=0)
                velocity_norms_test = velocities_test

                velocities_samples = np.diff(positions_samples[:, -1], axis=0)
                velocity_norms_samples = velocities_samples
                l = -20
                h = 200

                H_test, _ = np.histogram(velocity_norms_test, bins=100, range=(l,h))
                H_samples, _ = np.histogram(velocity_norms_samples, bins=100, range=(l,h))

                H_test_dithered = H_test + 10.
                H_test_dithered /= np.sum(H_test_dithered)

                H_samples_dithered = H_samples + 10.
                H_samples_dithered /= np.sum(H_samples_dithered)

                c = entropy(H_samples_dithered, H_test_dithered)
                results['velocity_vertical_kls'].append(c)
                results['velocity_vertical_test'] = velocity_norms_test
                results['velocity_vertical_sample'] = velocity_norms_samples

                # Turn rate histogram
                # print ("Turn rate")
                X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1]//3, 3)
                theta_test = np.empty(0)
                for traj in X_test_reshaped:
                    v_test = np.diff(traj[:, :2], axis=0)
                    dx, dy = v_test[:, 0], v_test[:, 1]
                    theta = np.arctan2(dy, dx)
                    theta_test = np.r_[theta_test, np.diff(theta)]

                X_samples_reshaped = X_samples.reshape(X_samples.shape[0], X_samples.shape[1]//3, 3)
                theta_samples = np.empty(0)
                for traj in X_samples_reshaped:
                    v_samples = np.diff(traj[:, :2], axis=0)
                    dx, dy = v_samples[:, 0], v_samples[:, 1]
                    theta = np.arctan2(dy, dx)
                    theta_samples = np.r_[theta_samples, np.diff(theta)]

                H_test, _ = np.histogram(theta_test, bins=100, range=(-.1, .1))
                H_samples, _ = np.histogram(theta_samples, bins=100, range=(-.1, .1))

                H_test_dithered = H_test + 10.
                H_test_dithered /= np.sum(H_test_dithered)

                H_samples_dithered = H_samples + 10.
                H_samples_dithered /= np.sum(H_samples_dithered)

                d = entropy(H_samples_dithered, H_test_dithered)

                results['turnrate_kls'].append(d)
                results['turnrate_test'] = theta_test
                results['turnrate_sample'] = theta_samples
            print ('Finished %s - %d' % (traj_type, k))
            np.save(os.path.join('/scratch/jfk-results/', '%s-%d-results' % (traj_type, k)), results)
            gc.collect()