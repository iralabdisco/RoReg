import open3d as o3d
import argparse
import pandas as pd
import benchmark_helpers
from tqdm import tqdm
import numpy as np
import os

def main(args):
    # Load problems txt file
    df = pd.read_csv(args.input_txt, sep=' ', comment='#')
    df = df.reset_index()
    problem_name = os.path.splitext(os.path.basename(args.input_txt))[0]

    extensions_benchmark = []
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        problem_id, source_pcd, target_pcd, source_transform, target_pcd_filename = \
            benchmark_helpers.load_problem(row, args.pcd_dir)

        obb = source_pcd.get_oriented_bounding_box()
        max_extension = obb.extent.max()
        extensions_benchmark.append(max_extension)

        obb = target_pcd.get_oriented_bounding_box()
        max_extension = obb.extent.max()
        extensions_benchmark.append(max_extension)

    avg_extention = sum(extensions_benchmark) / len(extensions_benchmark)
    print(avg_extention)

    match3d_voxel_size = 0.025
    match3d_extention = 3.8
    voxel_size = avg_extention/match3d_extention*match3d_voxel_size
    print(voxel_size)

    np.save(f'{args.out_dir}/{problem_name}', voxel_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input_txt',
        type=str,
        default="/benchmark/point_clouds_registration_benchmark/kaist/urban05_global.txt",
        help='path to problems txt')
    parser.add_argument(
        '--pcd_dir',
        type=str,
        default="/benchmark/point_clouds_registration_benchmark/kaist/urban05/",
        help='path to pcd dir')
    parser.add_argument(
        '--out_dir',
        type=str,
        default="/benchmark/ROREG_TEST/TUM_long/",
        help='path to output dir')
    args = parser.parse_args()

    main(args)