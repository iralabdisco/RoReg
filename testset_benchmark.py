"""
Generate YOHO input for Testset.
PC*60 rotations->FCGF backbone-> FCGF Group feature for PC keypoints.
"""

import time
import numpy as np
import os
import pandas as pd
import argparse
import open3d as o3d
import torch
from tqdm import tqdm
import MinkowskiEngine as ME
from dataops.dataset import get_dataset_name
from utils.utils import make_non_exists_dir
from backbone.fcgf import load_model
from utils.knn_search import knn_module
from utils.utils_o3d import make_open3d_point_cloud
import benchmark_helpers


class FCGFDataset():
    def __init__(self,datasets,config):
        self.cfg = config
        self.voxel_size = config.voxel_size
        self.Rgroup=np.load(f'{self.cfg.groupdir}/Rotation.npy')

class testset_create():
    def __init__(self,config):
        self.config=config
        self.voxel_size = config.voxel_size
        self.Rgroup=np.load(f'{self.config.groupdir}/Rotation.npy')
        self.knn=knn_module.KNN(1)

    def get_kps(self, pcd, keypoints):
        source_xyz = np.array(pcd.points)
        xyz_len = source_xyz.shape[0]
        keypoints = max(xyz_len, keypoints)
        indexes = np.random.choice(xyz_len, keypoints, replace=False)
        kps = source_xyz[indexes, :]
        return kps

    def get_item_from_pcd(self, pcd: o3d.geometry.PointCloud, g_id):
        xyz0 = np.asarray(pcd.points)
        xyz0 = xyz0 @ self.Rgroup[g_id].T
        # Voxelization
        _, sel0 = ME.utils.sparse_quantize(xyz0 / self.voxel_size, return_index=True)
        # Make point clouds using voxelized points
        pcd0 = make_open3d_point_cloud(xyz0)
        # Select features and points using the returned voxelized indices
        pcd0.points = o3d.utility.Vector3dVector(np.array(pcd0.points)[sel0])
        # Get coords
        xyz0 = np.array(pcd0.points)
        feats = np.ones((xyz0.shape[0], 1))
        coords0 = np.floor(xyz0 / self.voxel_size)

        return (xyz0, coords0, feats)

    def get_dict_from_item(self, list_data):
        xyz0, coords0, feats0 = list_data
        xyz0 = [xyz0]
        coords0 = [coords0]
        feats0 = [feats0]
        xyz_batch0 = []
        dsxyz_batch0 = []

        def to_tensor(x):
            if isinstance(x, torch.Tensor):
                return x
            elif isinstance(x, np.ndarray):
                return torch.from_numpy(x)
            else:
                raise ValueError(f'Can not convert to torch tensor, {x}')

        for batch_id, _ in enumerate(coords0):
            xyz_batch0.append(to_tensor(xyz0[batch_id]))
            _, inds = ME.utils.sparse_quantize(coords0[batch_id], return_index=True)
            dsxyz_batch0.append(to_tensor(xyz0[batch_id][inds]))

        coords_batch0, feats_batch0 = ME.utils.sparse_collate(coords0, feats0)

        # Concatenate all lists
        xyz_batch0 = torch.cat(xyz_batch0, 0).float()
        dsxyz_batch0 = torch.cat(dsxyz_batch0, 0).float()
        cuts_node = 0
        cuts = [0]
        for batch_id, _ in enumerate(coords0):
            cuts_node += coords0[batch_id].shape[0]
            cuts.append(cuts_node)

        return {
            'pcd0': xyz_batch0,
            'dspcd0': dsxyz_batch0,
            'cuts': cuts,
            'sinput0_C': coords_batch0,
            'sinput0_F': feats_batch0.float(),
        }

    def get_features_from_pcd(self, pcd, device, model, output_dir, filename_save):
        features = []
        features_gid = []

        make_non_exists_dir(f'{output_dir}/FCGF_Input_Group_feature')

        Keys_i_orig = self.get_kps(pcd, 5000)
        np.save(f'{output_dir}/FCGF_Input_Group_feature/{filename_save}_kpts.npy',
                Keys_i_orig)
        for g_id in range(60):
            input_item = self.get_item_from_pcd(pcd, g_id)
            input_dict = self.get_dict_from_item(input_item)
            sinput0 = ME.SparseTensor(
                input_dict['sinput0_F'].to(device),
                coordinates=input_dict['sinput0_C'].to(device))
            F0 = model(sinput0).F
            F0 = F0.detach()

            cuts = input_dict['cuts']
            feature = F0[cuts[0]:cuts[0 + 1]]
            pts = input_dict['dspcd0'][cuts[0]:cuts[0 + 1]]  # *config.voxel_size

            Keys_i = Keys_i_orig @ self.Rgroup[g_id]

            xyz_down = pts.T[None, :, :].cuda()  # 1,3,n
            Keys_i = torch.from_numpy(Keys_i.T[None, :, :].astype(np.float32)).cuda()
            d, nnindex = self.knn(xyz_down, Keys_i)
            nnindex = nnindex[0, 0]
            one_R_output = feature[nnindex, :].cpu().numpy()  # 5000*32

            features.append(one_R_output[:, :, None])
            features_gid.append(g_id)
            if len(features_gid) == 60:
                sort_args = np.array(features_gid)
                sort_args = np.argsort(sort_args)
                output = np.concatenate(features, axis=-1)[:, :, sort_args]
                np.save(f'{output_dir}/FCGF_Input_Group_feature/{filename_save}.npy',
                        output)
        torch.cuda.empty_cache()

    def Feature_extracting_benchmark(self, input_txt, data_dir, output_dir):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        checkpoint = torch.load(self.config.model)
        config = checkpoint['config']

        num_feats = 1
        Model = load_model(config.model)
        model = Model(
            num_feats,
            config.model_n_out,
            bn_momentum=0.05,
            normalize_feature=config.normalize_feature,
            conv1_kernel_size=config.conv1_kernel_size,
            D=3)

        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device)
        model.eval()

        # Load problems txt file
        df = pd.read_csv(input_txt, sep=' ', comment='#')
        df = df.reset_index()
        problem_name = os.path.splitext(os.path.basename(args.input_txt))[0]

        with torch.no_grad():
            for _, row in tqdm(df.iterrows(), total=df.shape[0]):
                problem_id, source_pcd, target_pcd, source_transform, target_pcd_filename = \
                    benchmark_helpers.load_problem(row, data_dir)

                # Get source features
                source_pcd = source_pcd.transform(source_transform)

                self.get_features_from_pcd(source_pcd, device, model, output_dir, problem_id)
                target_pcd_filename = os.path.splitext(target_pcd_filename)[0]
                if not os.path.exists(f'{output_dir}/FCGF_Input_Group_feature/{target_pcd_filename}.npy'):
                    self.get_features_from_pcd(target_pcd, device, model, output_dir, target_pcd_filename)

if __name__=="__main__":
    basedir = './data'
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        default='./checkpoints/FCGF/backbone/best_val_checkpoint.pth',
        type=str,
        help='path to backbone latest checkpoint (default: None)')
    parser.add_argument(
        '--voxel_size',
        default=0.025,
        type=float,
        help='voxel size to preprocess point cloud')
    parser.add_argument(
        '--groupdir',
        default='./utils/group_related',
        type=str,
        help='group related files')
    parser.add_argument(
        '--input_txt',
        type=str,
        default="/benchmark/point_clouds_registration_benchmark/tum/long_office_household_global.txt",
        help='path to problems txt')
    parser.add_argument(
        '--pcd_dir',
        type=str,
        default="/benchmark/point_clouds_registration_benchmark/tum/long_office_household/",
        help='path to pcd dir')
    parser.add_argument(
        '--out_dir',
        type=str,
        default="/benchmark/ROREG_TEST/TUM_long/",
        help='path to output dir')
    
    args = parser.parse_args()

    testset_creater=testset_create(args)
    testset_creater.Feature_extracting_benchmark(args.input_txt, args.pcd_dir, args.out_dir)
    