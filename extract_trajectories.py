"""
extract_trajectories.py
Author: Shane Barratt

This script extracts trajectories according to a list of arguments, e.g.,

$ python extract_trajectories.py --data /scratch/tsaa_data/tsaa_raw_data --output data/jfk --place new_york --airport_altitude 12 \
--airport_lat 40.642591 --airport_lon -73.776100 --max_altitude 3000 --d_nm 5 --min_t 20 --max_t 500 --max_time_btwn_samples 40

$ python extract_trajectories.py --data /scratch/tsaa_data/tsaa_raw_data --output data/kfrg --place new_york --airport_altitude 24 \
--airport_lat 40.729631 --airport_lon -73.414530 --max_altitude 3000 --d_nm 5 --min_t 20 --max_t 500 --max_time_btwn_samples 40

$ python extract_trajectories.py --data /scratch/tsaa_data/tsaa_raw_data --output data/28j --place 'florida' --airport_altitude 14.6 \
--airport_lat 29.658153 --airport_lon -81.689668 --max_altitude 3000 --d_nm 5 --min_t 20 --max_t 500 --max_time_btwn_samples 40

Results in a serialized list of dictionaries in the specified output folder, where each dictionary has two fields:
traj['fname'] = csv file trajectory came from
traj['data'] = N x 4 numpy array where columns are [time, xEast, yNorth, zUp]
"""

import os
import time
import pickle
import gc
import argparse
import glob

import pandas as pd
import numpy as np

from utils import geodetic_to_enu
geodetic_to_enu = np.vectorize(geodetic_to_enu)

# Constants
M_TO_NM = 0.000539957
FT_TO_M = .3048

# Argument Parser
parser = argparse.ArgumentParser()

parser.add_argument('--data', required=True, help='location of raw data folder')
parser.add_argument('--output', required=True, help='folder to put trajectories in')
parser.add_argument('--place', required=True, help='new_york | florida | los_angeles')

parser.add_argument('--airport_altitude', required=True, type=float, help='altitude of airport (in meters)')
parser.add_argument('--airport_lat', required=True, type=float, help='altitude of airport (in meters)')
parser.add_argument('--airport_lon', required=True, type=float, help='altitude of airport (in meters)')

parser.add_argument('--max_altitude', required=True, type=float, help='height of bounding box surrounding airport (in feet)')
parser.add_argument('--d_nm', required=True, type=float, help='1/2 length of bounding box surrounding airport (in nautical miles)')
parser.add_argument('--min_t', required=True, type=int, help='minimum length of trajectory (in seconds)')
parser.add_argument('--max_t', required=True, type=int, help='maximum length of trajectory (in seconds)')
parser.add_argument('--max_time_btwn_samples', required=True, type=int, help='amount of time in which absence of sensor measurement implies a new trajectory (in seconds)')

opt = parser.parse_args()
print (opt)

if opt.place == 'new_york':
    opt.place_folder = 'NewYork_vcs0'
elif opt.place == 'florida':
    opt.place_folder = 'Florida_vcs2'
elif opt.place == 'los_angeles':
    opt.place_folder = 'LosAngeles_vcs11'
else:
    raise Exception('Place %s not supported.' % opt.place)

d_m = opt.d_nm/M_TO_NM
max_altitude = opt.max_altitude*FT_TO_M

columns = ['t', 'aircraft_id', 'address_qualifier', 'target_address', 'lat', 'lon', \
 'pressure_altitude_available', 'pressure_altitude', 'geometric_altitude_available', \
  'geometric_altitude', 'NACp_available', 'NACp', 'NACV_available', 'NACv', \
  'north_south_available', 'east_west_available', 'vertical_available', \
  'north_south_velocity', 'east_west_velocity', 'vertical_velocity']

location_folder = os.path.join(opt.data, opt.place_folder)

csvs = glob.glob(os.path.join(location_folder, '*.csv'))
trajs = []

for i, f in enumerate(csvs):
    start = time.time()
    num_trajs_processed = 0

    # Read csv
    df = pd.read_csv(f, names=columns)

    # Remove entries which don't have pressure altitude
    df = df[(df.pressure_altitude_available == 1)]

    # convert all (lat, lon, altitude) pairs to ENU coordinates centered around airpot
    xEast, yNorth, zUp = geodetic_to_enu(
        df.lat,
        df.lon,
        df.pressure_altitude * FT_TO_M,
        opt.airport_lat,
        opt.airport_lon,
        opt.airport_altitude
    )

    # select rows which are in 2d_m x 2d_m x max_altitude cuboid centered at airport
    ind = (zUp < max_altitude) & \
          (xEast > -d_m) & \
          (xEast < d_m) & \
          (yNorth > -d_m) & \
          (yNorth < d_m)
    df_around_airport = df[ind]

    # find the unique target addresses
    targ_addresses = df_around_airport.target_address.unique()

    for targ_address in targ_addresses:
        # extract all rows associated with targ_address
        df_targ = df_around_airport[df_around_airport.target_address == targ_address]
        df_targ = df_targ.sort_values('t')

        # if there aren't measurements for a long enough time, assume new trajectory
        transition_points = np.where(np.diff(df_targ.t) > opt.max_time_btwn_samples)[0]

        # split the dataframe into these trajectories
        split_trajs_dataframes = []
        ind_start = 0
        for k in transition_points:
            split_trajs_dataframes.append(df_targ[ind_start:k+1])
            ind_start = k+1
        split_trajs_dataframes.append(df_targ[ind_start:])

        # Now extract the relevant columns and turn it into a numpy array
        for df_traj in split_trajs_dataframes:
            trajectory_length_in_seconds = (df_traj.t.max() - df_traj.t.min())
            if trajectory_length_in_seconds < opt.min_t or trajectory_length_in_seconds > opt.max_t:
                continue

            xEast, yNorth, zUp = geodetic_to_enu(df_traj.lat, df_traj.lon, df_traj.pressure_altitude * .3048, opt.airport_lat, opt.airport_lon, opt.airport_altitude)

            traj = dict()
            traj['fname'] = os.path.basename(f)
            traj['data'] = np.zeros((df_traj.t.shape[-1], 4))
            traj['data'][:, 0] = df_traj.t
            traj['data'][:, 1] = xEast
            traj['data'][:, 2] = yNorth
            traj['data'][:, 3] = zUp

            trajs.append(traj)
            num_trajs_processed += 1

    gc.collect()
    print ("Processed %d trajectories from %s" % (num_trajs_processed, f))
    print ("time elapsed: %d s" % (time.time()-start))

print ("Dumping trajs to %s." % opt.output)
os.makedirs(opt.output, exist_ok=True)
pickle.dump(trajs, open(os.path.join(opt.output, 'trajs-raw.pkl'), 'wb'))