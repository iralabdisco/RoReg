import benchmark_helpers
import pandas as pd
import argparse
import numpy as np
import os
from tqdm import tqdm
import csv
import copy
import sys


def main(args):
    # Load problems txt file
    df = pd.read_csv(args.input_txt, sep=' ', comment='#')
    df = df.reset_index()
    problem_name = os.path.splitext(os.path.basename(args.input_txt))[0]

    # initialize result file
    os.makedirs(args.results_dir, exist_ok=True)
    header_comment = "# " + " ".join(sys.argv[:]) + "\n"
    header = ['id', 'initial_error', 'final_error', 'transformation', 'status_code']
    result_name = problem_name + "_result.txt"
    result_filename = os.path.join(args.results_dir, result_name)

    # RoReg stuff
    match_dir = f'{args.roreg_dir}match_5000'
    Save_dir = f'{match_dir}/yohoo/1000iters'

    with open(result_filename, mode='w') as f:
        f.write(header_comment)
        csv_writer = csv.writer(f, delimiter=';')
        csv_writer.writerow(header)

    initial_errors = []
    final_errors = []
    overlaps = []
    n_identities = 0
    index = 0
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):

        problem_id, source_pcd, target_pcd, source_transform, target_pcd_filename = \
            benchmark_helpers.load_problem(row, args.input_pcd_dir)

        target_pcd_filename = os.path.splitext(target_pcd_filename)[0]

        # calculate initial error
        moved_source_pcd = copy.deepcopy(source_pcd)
        moved_source_pcd = moved_source_pcd.transform(source_transform)
        initial_error = benchmark_helpers.calculate_error(source_pcd, moved_source_pcd)
        initial_errors.append(initial_error)

        roreg_path = os.path.join(Save_dir, f'{index}-{index+1}.npz')
        roreg_transform = np.load(roreg_path)['trans']

        # calculate final error
        moved_source_pcd = moved_source_pcd.transform(roreg_transform)
        final_error = benchmark_helpers.calculate_error(source_pcd, moved_source_pcd)
        final_errors.append(final_error)

        overlap = benchmark_helpers.overlap(target_pcd, source_pcd, 0.1)
        overlaps.append(overlap)

        if np.allclose(roreg_transform, np.eye(4)):
            n_identities += 1

        # write results to file
        str_solution = ' '.join(map(str, roreg_transform.ravel()))
        results = [problem_id, initial_error, final_error, str_solution, 'ok']
        with open(result_filename, mode='a') as f:
            csv_writer = csv.writer(f, delimiter=';', quoting=csv.QUOTE_NONE, escapechar=' ')
            csv_writer.writerow(results)

        index = index + 2

    print(f'Dataset: {problem_name}')
    print(f'Initial error: {np.median(initial_errors): .2f}')
    print(f'Final error: {np.median(final_errors): .2f}')
    print(f'Overlap: {np.median(overlaps): .2f}')
    print(f'N identity transforms: {n_identities}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluate RoReg results')

    # I/O files and dirs
    parser.add_argument('--input_txt', type=str,
                        default="/benchmark/point_clouds_registration_benchmark/eth/wood_autumn_global.txt",
                        help='Path to the problem .txt')
    parser.add_argument('--input_pcd_dir', type=str,
                        default="/benchmark/point_clouds_registration_benchmark/eth/wood_autumn/",
                        help='Directory which contains the pcd files')
    parser.add_argument('--roreg_dir', type=str,
                        default="/benchmark/ROREG_TEST/ETH_WOOD_AUTUMN/",
                        help='Directory to take results from')
    parser.add_argument('--results_dir', type=str,
                        default="/benchmark/ROREG_TEST/ETH_WOOD_AUTUMN/",
                        help='Directory to save the results to')

    args = parser.parse_args()

    main(args)