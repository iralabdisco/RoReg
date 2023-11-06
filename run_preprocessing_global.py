import os, shutil
from multiprocessing import Pool
from pathlib import Path
PY3="python3"

GPU = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU

BENCHMARK_DIR="/benchmark/point_clouds_registration_benchmark/"
PREPROCESSING_DIR="/benchmark/experiments/RoReg/preprocessing"
VOXEL_SIZE = 0.1

base_command = ( f'{PY3}' + ' testset_benchmark.py ' +
                 f'--voxel_size={VOXEL_SIZE}')


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

commands = []

for problem_txt, pcd_dir in zip(problem_txts, pcd_dirs):
    full_command = (base_command +
                    f' --input_txt={BENCHMARK_DIR}/{problem_txt}' +
                    f' --pcd_dir={BENCHMARK_DIR}/{pcd_dir}' +
                    f' --out_dir={PREPROCESSING_DIR}')

    problem_name = Path(problem_txt).stem
    time_command = f'command time --verbose -o {PREPROCESSING_DIR}/{problem_name}_preprocessing_time.txt ' + full_command
    nvidia_command = (f'nvidia-smi --query-gpu=timestamp,memory.used -i 0 --format=csv -lms 1 > {PREPROCESSING_DIR}/{problem_name}_preprocessing_memory.txt')

    full_command_stats = f'parallel -j2 -u --halt now,success=1 ::: \'{time_command}\' \'{nvidia_command}\''
    commands.append(full_command_stats)

# delete and recreate result directory
answer = input(f"Delete previous {PREPROCESSING_DIR} experiments? [Y/N] ")
if answer != "Y":
    print("Quitting...")
    exit()
shutil.rmtree(PREPROCESSING_DIR, ignore_errors=True)
os.makedirs(PREPROCESSING_DIR)

# save config in result directory
txt_commands = os.path.join(PREPROCESSING_DIR, "readme.md")
with open(txt_commands, 'w') as f:
    for item in commands:
        f.write("%s\n" % item)

pool = Pool(1)
pool.map(os.system, commands)