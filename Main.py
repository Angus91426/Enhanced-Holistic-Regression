"""
main.py
=======
Entry point for reproducing the Enhanced Holistic Regression (EHR)
multicollinearity-detection experiments. Set BASE_DIR (results are written
there and the folder is created automatically), then choose which simulations
and which data scales to run.

Data scales follow Table 2 of the manuscript. The MR list is ordered
[MR2, MR3, MR4, MR8, MR10, MR15, MR20]; MR2 = 0 as the table has no MR(2).
"""

import os

import Simulation as Sim

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# Root for all results. Change this to wherever you want the output to go.
BASE_DIR = os.path.join( os.path.dirname( os.path.abspath( __file__ ) ), 'Results' )

# Folder holding the pre-processed real-world datasets (the *_Processed.csv files).
DATA_DIR = os.path.join( os.path.dirname( os.path.abspath( __file__ ) ), 'Data' )

# --------------------------------------------------------------------------- #
# Data settings (Table 2): [MR2, MR3, MR4, MR8, MR10, MR15, MR20]
# --------------------------------------------------------------------------- #
SCALES = {
    'small': { 'n': 2000,  'p': 1000,  'MR': [0, 5, 3, 1, 1, 0, 0] },
    'large': { 'n': 20000, 'p': 10000, 'MR': [0, 3, 3, 1, 1, 1, 1] },
}

# --------------------------------------------------------------------------- #
# What to run
# --------------------------------------------------------------------------- #
RUN_SCALES     = ['small']   # which data scales to run, 'large'
RUN_DETECTION  = False
RUN_REDUCTION  = True
RUN_REALWORLD  = False

# Coefficient-sweep experiment: Enhanced Corr vs Enhanced Eigvec across
# coef_min = 0.0..1.0 on the large-scale dataset.
RUN_COEF_SWEEP   = False
COEF_SWEEP_SCALE = 'large'
COEF_SWEEP_MINS  = [round( 0.1 * i, 1 ) for i in range( 11 )]   # 0.0, 0.1, ..., 1.0

# Shared simulation settings.
NOISE_SCALE = 0.01
COEF_MINS   = [0.0]
SIMULATIONS = 10
SEED        = 911122


if __name__ == '__main__':
    os.makedirs( BASE_DIR, exist_ok = True )

    for scale in RUN_SCALES:
        cfg = SCALES[scale]
        n, p, MR = cfg['n'], cfg['p'], cfg['MR']
        print( f'\n########## {scale}-scale | n={n} p={p} MR={MR} ##########' )

        # ---- 1. Synthetic detection ablation study + significance test ---- #
        if RUN_DETECTION:
            Sim.run_detection_simulation(
                BASE_DIR = BASE_DIR,
                n = n, p = p, MR = MR,
                noise_scale = NOISE_SCALE,
                coef_mins = COEF_MINS,
                simulations = SIMULATIONS,
                seed = SEED,
                outlier_threshold = 5,
                run_A = True, run_B = True, run_C = True, run_D = True, run_E = True,
                run_Enhanced_Corr = True, run_Enhanced_Eigen = True, run_Bertsimas = True,
                run_significance = True,
                significance_baseline = 'Original',
            )

        # ---- 2. Dimensionality-reduction screen comparison ---- #
        if RUN_REDUCTION:
            Sim.run_reduction_simulation(
                BASE_DIR = BASE_DIR,
                n = n, p = p, MR = MR,
                noise_scale = NOISE_SCALE,
                coef_mins = COEF_MINS,
                simulations = SIMULATIONS,
                seed = SEED,
            )

    # ---- Coefficient sweep: Enhanced Corr vs Enhanced Eigvec, coef_min 0.0..1.0 ---- #
    if RUN_COEF_SWEEP:
        cfg = SCALES[COEF_SWEEP_SCALE]
        print( f'\n########## coef-sweep | {COEF_SWEEP_SCALE}-scale | coef_min={COEF_SWEEP_MINS} ##########' )
        Sim.run_detection_simulation(
            BASE_DIR = BASE_DIR,
            n = cfg['n'], p = cfg['p'], MR = cfg['MR'],
            noise_scale = NOISE_SCALE,
            coef_mins = COEF_SWEEP_MINS,
            simulations = SIMULATIONS,
            seed = SEED,
            outlier_threshold = 5,
            run_A = False, run_B = False, run_C = False, run_D = False, run_E = False,
            run_Enhanced_Corr = True, run_Enhanced_Eigen = True, run_Bertsimas = False,
            run_significance = False,
            csv_id = 'CoefSweep',
        )

    # ---- 3. Real-world detection (pre-processed datasets) ---- #
    if RUN_REALWORLD:
        Sim.run_realworld_detection(
            BASE_DIR = BASE_DIR,
            data_dir = DATA_DIR,
            datasets = None,              # None => every *_Processed.csv in DATA_DIR
            run_enhanced = True,
            run_bertsimas = True,
            reduction = False,
            total_time_limit = 86400,
            per_solve_time_limit = 100,
        )
