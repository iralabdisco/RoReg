import pandas as pd
import open3d as o3d
import numpy as np
import os
from tqdm import tqdm
import benchmark_helpers
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_txt',
        type=str,
        default="/benchmark/point_clouds_registration_benchmark/devel/registration_pairs/gazebo_summer_global.txt",
        help='path to problems txt')
    parser.add_argument(
        '--pcd_dir',
        type=str,
        default="/benchmark/point_clouds_registration_benchmark/eth/gazebo_summer/",
        help='path to pcd dir')
    parser.add_argument(
        '--output_dir',
        type=str,
        default="./data/origin_data/benchmark/gazebo_summer/problems/",
        help='path to output dir')

    args = parser.parse_args()

    # Load problems txt file
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "PointCloud"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "Keypoints"), exist_ok=True)

    #create gt.log
    gt_file = os.path.join(args.output_dir, "PointCloud", "gt.log")
    with open(gt_file, 'w') as fp:
        pass

    df = pd.read_csv(args.input_txt, sep=' ', comment='#')
    df = df.reset_index()
    problem_name = os.path.splitext(os.path.basename(args.input_txt))[0]

    index = 0
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        problem_id, source_pcd, target_pcd, source_transform, target_pcd_filename = \
            benchmark_helpers.load_problem(row, args.pcd_dir)

        moved_source_pcd = source_pcd.transform(source_transform)

        target_path = os.path.join(args.output_dir, "PointCloud", "cloud_bin_"+str(index)+".ply")
        o3d.io.write_point_cloud(target_path, target_pcd)

        if len(np.asarray(target_pcd.points)) >= 5000:
            target_kps = np.random.choice(len(np.asarray(target_pcd.points)), 5000, replace=False)
        else:
            print(target_pcd_filename)
            target_kps = np.arange(len(np.asarray(target_pcd.points)))

        target_kps_path = os.path.join(args.output_dir, "Keypoints", "cloud_bin_"+str(index)+"Keypoints.txt")
        np.savetxt(target_kps_path, target_kps, newline=" ")

        source_path = os.path.join(args.output_dir, "PointCloud", "cloud_bin_"+str(index+ 1)+".ply")
        o3d.io.write_point_cloud(source_path, moved_source_pcd)

        if len(np.asarray(moved_source_pcd.points)) >= 5000:
            source_kps = np.random.choice(len(np.asarray(moved_source_pcd.points)), 5000, replace=False)
        else:
            print(target_pcd_filename)
            source_kps = np.arange(len(np.asarray(moved_source_pcd.points)))

        source_kps_path = os.path.join(args.output_dir, "Keypoints", "cloud_bin_"+str(index + 1)+"Keypoints.txt")
        np.savetxt(source_kps_path, source_kps, newline=" ")

        gt_transform = np.linalg.inv(source_transform)
        with open(gt_file, 'a') as f:
            f.write(str(index) + "\t" + str(index+1) + "\t" + str(0) + "\n")
            f.write(str(gt_transform[0,0]) + "\t" + str(gt_transform[0, 1]) + "\t" + str(gt_transform[0, 2]) + "\t" + str(
                gt_transform[0, 3]) + "\n")
            f.write(str(gt_transform[1, 0]) + "\t" + str(gt_transform[1, 1]) + "\t" + str(gt_transform[1, 2]) + "\t" + str(
                gt_transform[1, 3]) + "\n")
            f.write(str(gt_transform[2, 0]) + "\t" + str(gt_transform[2, 1]) + "\t" + str(gt_transform[2, 2]) + "\t" + str(
                gt_transform[2, 3]) + "\n")
            f.write(str(gt_transform[3, 0]) + "\t" + str(gt_transform[3, 1]) + "\t" + str(gt_transform[3, 2]) + "\t" + str(
                gt_transform[3, 3]) + "\n")
        index = index+2
