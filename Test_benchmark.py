import sys
import argparse
import parses.parses_test as parses_test
from test.evaluator import yoho_evaluator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_iter',default=1000,type=int,help='ransac iterations')
    parser.add_argument('--testset',default='3dmatch',type=str,help='dataset name')
    parser.add_argument('--ransac_ird',default=0.1,type=float,help='inliner threshold of ransac')
    parser.add_argument('--keynum',default=5000,type=int,help='inliner threshold of ransac')
    parser.add_argument('--match_n',default=0.5,type=float,help = 'use how many predicted correspondences, 0.99 for all, >=1 for top-h')
    parser.add_argument('--tau_1',default=0.05,type=float,help='tau 1 for FMR')
    parser.add_argument('--tau_2',default=0.1,type=float,help='tau 2 for FMR')
    parser.add_argument('--tau_3',default=0.2,type=float,help='tau 3 for RR')
    parser.add_argument('--RD',action='store_true')
    parser.add_argument('--RM',action='store_true')
    parser.add_argument('--ET',default='yohoo',type=str,help = 'yohoc/yohoo')
    parser.add_argument(
        '--input_txt',
        type=str,
        default="/benchmark/point_clouds_registration_benchmark/eth/wood_autumn_global.txt",
        help='path to problems txt')
    parser.add_argument(
        '--pcd_dir',
        type=str,
        default="/benchmark/point_clouds_registration_benchmark/eth/wood_autumn/",
        help='path to pcd dir')
    parser.add_argument(
        '--features_dir',
        type=str,
        default="/benchmark/ROREG_TEST/ETH_WOOD_AUTUMN/",
        help='path to features dir')
    args = parser.parse_args()

    # execuate
    config, nouse = parses_test.get_config()
    evalor=yoho_evaluator(config)

    evalor.run_benchmark(args.input_txt, args.pcd_dir, args.features_dir)


if __name__ == '__main__':
    main()
