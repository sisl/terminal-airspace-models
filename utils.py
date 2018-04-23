"""
utils.py
Author: Shane Barratt

This script provides coordinate transformations from Geodetic -> ECEF, ECEF -> ENU
and Geodetic -> ENU (the composition of the two previous functions). Running the script
by itself runs tests.
based on https://gist.github.com/govert/1b373696c9a27ff4c72a.
It also provides some other useful functions.
"""
import math

from sklearn.utils import shuffle
import numpy as np

def train_test_split(data, fraction_val):
    data_shuffled = shuffle(data)
    n_val = int(data.shape[0]*fraction_val)

    return data_shuffled[n_val:], data_shuffled[:n_val]

a = 6378137
b = 6356752.3142
f = (a - b) / a
e_sq = f * (2-f)

def geodetic_to_ecef(lat, lon, h):
    # (lat, lon) in WSG-84 degrees
    # h in meters
    lamb = math.radians(lat)
    phi = math.radians(lon)
    s = math.sin(lamb)
    N = a / math.sqrt(1 - e_sq * s * s)

    sin_lambda = math.sin(lamb)
    cos_lambda = math.cos(lamb)
    sin_phi = math.sin(phi)
    cos_phi = math.cos(phi)

    x = (h + N) * cos_lambda * cos_phi
    y = (h + N) * cos_lambda * sin_phi
    z = (h + (1 - e_sq) * N) * sin_lambda

    return x, y, z

def ecef_to_enu(x, y, z, lat0, lon0, h0):
    lamb = math.radians(lat0)
    phi = math.radians(lon0)
    s = math.sin(lamb)
    N = a / math.sqrt(1 - e_sq * s * s)

    sin_lambda = math.sin(lamb)
    cos_lambda = math.cos(lamb)
    sin_phi = math.sin(phi)
    cos_phi = math.cos(phi)

    x0 = (h0 + N) * cos_lambda * cos_phi
    y0 = (h0 + N) * cos_lambda * sin_phi
    z0 = (h0 + (1 - e_sq) * N) * sin_lambda

    xd = x - x0
    yd = y - y0
    zd = z - z0

    xEast = -sin_phi * xd + cos_phi * yd
    yNorth = -cos_phi * sin_lambda * xd - sin_lambda * sin_phi * yd + cos_lambda * zd
    zUp = cos_lambda * cos_phi * xd + cos_lambda * sin_phi * yd + sin_lambda * zd

    return xEast, yNorth, zUp

def geodetic_to_enu(lat, lon, h, lat_ref, lon_ref, h_ref):
    x, y, z = geodetic_to_ecef(lat, lon, h)
    
    return ecef_to_enu(x, y, z, lat_ref, lon_ref, h_ref)

def log_probability(means, covs, cluster_probs, X_val):
    # Iterate over validation data and calculate probability of each one under model
    n_clusters = len(means)
    cov_pinvs = []
    for j in range(n_clusters):
        u, s, v = covs[j]
        pinv = np.matmul(v.T, np.matmul(np.diag(1./s), u.T))
        cov_pinvs.append(pinv)

    x = 0.0
    for traj in X_val:
        possibilities = []
        for j in range(n_clusters):
            u, s, v = covs[j]
            mean = means[j]
            cov_pinv = cov_pinvs[j]

            dev = (traj - mean)[:, np.newaxis]
            log_2pi_pseudo_det = np.sum(np.log(2*np.pi*s))
            maha = np.dot(traj.T, np.dot(cov_pinv, traj))
            prob = -.5 * (log_2pi_pseudo_det + maha) + np.log(cluster_probs[j])

            # prob = cluster_probs[j] * np.exp(log_2pi_pseudo_det)**(-.5)*np.exp(-.5*maha)
            possibilities.append(prob)
        x += np.max(possibilities)
    return x

def generate(mean, cov, r=1.):
    u, s, vt = cov
    z = np.random.normal(size=vt.shape[1])
    traj = mean + np.dot(vt.T, np.dot(np.diag(r*s**.5), np.dot(vt, z)))
    return traj

if __name__ == '__main__':
    def are_close(a, b):
        return abs(a-b) < 1e-4

    latLA = 34.00000048
    lonLA = -117.3335693
    hLA = 251.702

    x0, y0, z0 = geodetic_to_ecef(latLA, lonLA, hLA)
    x = x0 + 1
    y = y0
    z = z0
    xEast, yNorth, zUp = ecef_to_enu(x, y, z, latLA, lonLA, hLA)
    assert are_close(0.88834836, xEast)
    assert are_close(0.25676467, yNorth)
    assert are_close(-0.38066927, zUp)

    x = x0
    y = y0 + 1
    z = z0
    xEast, yNorth, zUp = ecef_to_enu(x, y, z, latLA, lonLA, hLA)
    assert are_close(-0.45917011, xEast)
    assert are_close(0.49675810, yNorth)
    assert are_close(-0.73647416, zUp)

    x = x0
    y = y0
    z = z0 + 1
    xEast, yNorth, zUp = ecef_to_enu(x, y, z, latLA, lonLA, hLA)
    assert are_close(0.00000000, xEast)
    assert are_close(0.82903757, yNorth)
    assert are_close(0.55919291, zUp)