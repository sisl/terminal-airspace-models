# Learning Probabilistic Trajectory Models of Aircraft in Terminal Airspace from Position Data

We unfortunately cannot share the raw dataset because of confidentiality reasons. However, we do provide a pre-trained landing and takeoff model for the KJFK airport.

### Installation

```
$ git clone git@github.com:sisl/terminal-airspace-models.git
$ cd terminal-airspace-models
$ pip install -r requirements.txt
```

### Experiment with pre-trained models

Follow the instructions in ```PreTrained.ipynb```.
```
$ jupyter notebook PreTrained.ipynb
```

### Experiment with our Trajectory Reconstruction procedure

Run the cells in ```TrajectoryReconstruction.ipynb```.
```
$ jupyter notebook TrajectoryReconstruction.ipynb
```

### Pilot Turing Test

The PDF files used to conduct the Pilot Turing Test (PTT) are in the folder ```turing_test```.

### Training the model

For example, for the KJFK airport:

Extract trajectories:
```
$ python extract_trajectories.py --data /scratch/tsaa_data/tsaa_raw_data --output data/jfk --place new_york --airport_altitude 12 \
--airport_lat 40.642591 --airport_lon -73.776100 --max_altitude 3000 --d_nm 5 --min_t 20 --max_t 500 --max_time_btwn_samples 40
```

Extract landing/takeoffs:
```
$ python extract_types.py --data data/jfk --close 2000
```

Smooth landing/takeoffs:
```
$ python smooth.py --data data/jfk --num_workers 6 --L_landings 125 --L_takeoffs 70 --num_extend 15
```

Train models with varying values of k:
```
$ python model.py --data data/jfk --experiment run0/
```

Evaluate models:
```
$ python evaluate.py --data data/kfrg --experiment run0/
```