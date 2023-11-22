import os, shutil
from multiprocessing import Pool
from pathlib import Path

PY3="python3"

GPU = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU

BENCHMARK_DIR="/benchmark/point_clouds_registration_benchmark/"
OUTPUT_DIR = "./data/origin_data/BENCHMARK/"

base_command = (f'{PY3}' + ' prepare_benchmark_dataset.py ')

problem_txts = ['kaist/urban05_global.txt',
                'eth/apartment_global.txt',
                'eth/gazebo_summer_global.txt',
                'eth/gazebo_winter_global.txt',
                'eth/hauptgebaude_global.txt',
                'eth/plain_global.txt',
                'eth/stairs_global.txt',
                'eth/wood_autumn_global.txt',
                'eth/wood_summer_global.txt',
                'tum/long_office_household_global.txt',
                'tum/pioneer_slam_global.txt',
                'tum/pioneer_slam3_global.txt',
                'planetary/box_met_global.txt',
                'planetary/p2at_met_global.txt',
                'planetary/planetary_map_global.txt']

pcd_dirs = ['kaist/urban05/',
            'eth/apartment/',
            'eth/gazebo_summer/',
            'eth/gazebo_winter/',
            'eth/hauptgebaude/',
            'eth/plain/',
            'eth/stairs/',
            'eth/wood_autumn/',
            'eth/wood_summer/',
            'tum/long_office_household/',
            'tum/pioneer_slam/',
            'tum/pioneer_slam3/',
            'planetary/box_met/',
            'planetary/p2at_met/',
            'planetary/p2at_met/']

out_dirs = ['urban05/',
            'apartment/',
            'gazebo_summer/',
            'gazebo_winter/',
            'hauptgebaude/',
            'plain/',
            'stairs/',
            'wood_autumn/',
            'wood_summer/',
            'long_office_household/',
            'pioneer_slam/',
            'pioneer_slam3/',
            'box_met/',
            'p2at_met/',
            'planetary_map/']

commands = []

for problem_txt, pcd_dir, out_dir in zip(problem_txts, pcd_dirs, out_dirs):
    full_command = (base_command +
                    f' --input_txt={BENCHMARK_DIR}/{problem_txt}' +
                    f' --pcd_dir={BENCHMARK_DIR}/{pcd_dir}' +
                    f' --output_dir={OUTPUT_DIR}/{out_dir}')

    problem_name = Path(problem_txt).stem
    commands.append(full_command)

pool = Pool(1)
pool.map(os.system, commands)
