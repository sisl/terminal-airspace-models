"""
extract_types.py
Author: Shane Barratt

This script extracts landings/takeoffs/touch-and-gos/other out of a given trajectory file, e.g.,

$ python extract_types.py --data data/kfrg --close 2000
$ python extract_types.py --data data/jfk --close 2000
$ python extract_types.py --data data/28j --close 1500

Results in four pkl files, 'landings.pkl', 'takeoffs.pkl', 'touchandgos.pkl', 'other.pkl' in --data.
"""

import copy
import os
import pickle
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument('--data', required=True, help='location of data folder')
parser.add_argument('--close', type=int, required=True, help='what we define as close to the airport (in meters)')

opt = parser.parse_args()
print (opt)

# Load Trajectories
trajs_path = os.path.join(opt.data, 'trajs-raw.pkl')
trajs = pickle.load(open(trajs_path, 'rb'))

landings, takeoffs, touchandgos, other = [], [], [], []

for traj in trajs:
    t = copy.deepcopy(traj['data'][:, 0])
    p = copy.deepcopy(traj['data'][:, 1:4])

    # distance of first and last points to origin
    first_dist = np.linalg.norm(p[0, :2], 2)
    last_dist = np.linalg.norm(p[-1, :2], 2)

    # index of closest point to origin
    closest_ind = np.argmin(np.linalg.norm(p, axis=1))
    closest_time = t[closest_ind]
    frac_time_closest = (closest_time-t[0])/(t[-1]-t[0])

    # average dz/dt
    dz  = (p[-1, -1] - p[0, -1])/(t[-1] - t[0])

    # throw away extra measurements
    traj['data'] = traj['data'][:, :4]

    # scale z by 10
    traj['data'][:, -1] *= 10
    print (last_dist, first_dist, dz, frac_time_closest)
    if last_dist < opt.close and first_dist > opt.close and dz < -.5 and p[-1, -1] < 300. and frac_time_closest > .9: # landings
        traj['data'] = traj['data'][:closest_ind, :]
        landings.append(traj)
    elif first_dist < opt.close and last_dist > opt.close and dz > .5 and p[0, -1] < 300. and frac_time_closest < .1: # takeoffs
        traj['data'] = traj['data'][closest_ind:, :]
        takeoffs.append(traj)
    elif first_dist < opt.close and last_dist < opt.close and p[-1, -1] < 300. and p[0, -1] < 300.: # touch and go
        touchandgos.append(traj)
    else: # other
        other.append(traj)

plt.figure()
plt.xlim(-9000, 9000)
plt.ylim(-9000, 9000)
for traj in touchandgos[:250]:
    p = traj['data'][:, 1:4]
    plt.plot(p[:, 0], p[:, 1])
plt.savefig(os.path.join(opt.data, 'touchandgos.png'))
plt.close()

plt.figure()
plt.xlim(-9000, 9000)
plt.ylim(-9000, 9000)
for traj in landings[:250]:
    p = traj['data'][:, 1:4]
    plt.plot(p[:, 0], p[:, 1])
plt.savefig(os.path.join(opt.data, 'landings.png'))
plt.close()

plt.figure()
plt.xlim(-9000, 9000)
plt.ylim(-9000, 9000)
for traj in takeoffs[:250]:
    p = traj['data'][:, 1:4]
    plt.plot(p[:, 0], p[:, 1])
plt.savefig(os.path.join(opt.data, 'takeoffs.png'))
plt.close()

plt.figure()
plt.xlim(-9000, 9000)
plt.ylim(-9000, 9000)
for traj in other[:250]:
    p = traj['data'][:, 1:4]
    plt.plot(p[:, 0], p[:, 1])
plt.savefig(os.path.join(opt.data, 'other.png'))
plt.close()

print ("Dumping %d landings/takeoffs/touchandgos/other to %s." % (len(trajs), opt.data))
print ("Landings: %d" % len(landings))
print ("Takeoffs: %d" % len(takeoffs))
print ("Touch and gos: %d" % len(touchandgos))
print ("other: %d" % len(other))

pickle.dump(landings, open(os.path.join(opt.data, 'landings.pkl'), 'wb'))
pickle.dump(takeoffs, open(os.path.join(opt.data, 'takeoffs.pkl'), 'wb'))
pickle.dump(touchandgos, open(os.path.join(opt.data, 'touchandgos.pkl'), 'wb'))
pickle.dump(other, open(os.path.join(opt.data, 'other.pkl'), 'wb'))
