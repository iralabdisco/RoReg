import os, shutil
from multiprocessing import Pool
from pathlib import Path
import numpy as np

PY3="python3"

GPU = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU

PREPROCESSING_DIR="/benchmark/experiments/RoReg/preprocessing/"
VOXEL_DIR = "/benchmark/experiments/RoReg/voxel/"

base_command = (f'{PY3}' + ' testset.py ')

# datasets = ['urban05', 'apartment', 'gazebo_summer', 'gazebo_winter', 'hauptgebaude', 'plain', 'stairs',
#             'wood_autumn', 'wood_summer', 'long_office_household', 'pioneer_slam', 'pioneer_slam3',
#             'box_met', 'p2at_met', 'planetary_map']

datasets = ['pioneer_slam3', 'box_met', 'p2at_met', 'planetary_map']


commands = []

for dataset in datasets:
    problem_name = dataset + "_global"
    f = os.path.join(VOXEL_DIR, problem_name+".npy")
    voxel = np.load(f)
    print(f'{problem_name} voxel size: {voxel}')
    full_command = (base_command +
                    f' --voxel_size {voxel}'
                    f' --dataset {dataset}')

    time_command = (f'command time --verbose -o {PREPROCESSING_DIR}/preprocessing_stats/{dataset}_time.txt '
                    + full_command)
    nvidia_command = (f'nvidia-smi --query-gpu=timestamp,memory.used -i 0 --format=csv -lms 1 > '
                      f'{PREPROCESSING_DIR}/preprocessing_stats/{dataset}_memory.txt')

    full_command_stats = f'parallel -j2 -u --halt now,success=1 ::: \'{time_command}\' \'{nvidia_command}\''
    commands.append(full_command_stats)

if not os.path.exists(PREPROCESSING_DIR):
    os.makedirs(PREPROCESSING_DIR)
    os.makedirs(PREPROCESSING_DIR+"/preprocessing_stats/")

# save config in result directory
txt_commands = os.path.join(PREPROCESSING_DIR, "readme.md")
with open(txt_commands, 'w') as f:
    for item in commands:
        f.write("%s\n" % item)

pool = Pool(1)
pool.map(os.system, commands)
