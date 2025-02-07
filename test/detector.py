import os
import torch
import abc
import utils.utils as utils
import time
from tqdm import tqdm
import numpy as np
from network import name2network
import benchmark_helpers
import pandas as pd

class yoho_det():
    def __init__(self,cfg):
        self.cfg=cfg
        self.network=name2network['RD_test'](cfg).cuda()
        self.best_model_fn=f'{self.cfg.model_fn}/RD/model_best.pth'
        self.Rgroup=np.load(f'{self.cfg.SO3_related_files}/Rotation.npy').astype(np.float32)
        self._load_model()

    #Model_import
    def _load_model(self):
        if os.path.exists(self.best_model_fn):
            checkpoint=torch.load(self.best_model_fn)
            self.network.load_state_dict(checkpoint['network_state_dict'],strict=True)
        else:
            raise ValueError("No model exists")
    
    def run(self,dataset):
        self.network.eval()
        if dataset.name[0:4]=='3dLo':
            datasetname=f'3d{dataset.name[4:]}'
        else:
            datasetname=dataset.name
        savedir=f'{self.cfg.output_cache_fn}/{datasetname}/det_score'
        utils.make_non_exists_dir(savedir)
        print(f'Evaluating the saliency of points using rotaion guided detector on {dataset.name}')
        for pc_id in tqdm(range(len(dataset.pc_ids))):
            if os.path.exists(f'{savedir}/{pc_id}.npy'):continue
            feats=np.load(f'{self.cfg.output_cache_fn}/{datasetname}/YOHO_Output_Group_feature/{pc_id}.npy')
            batch={
                'feats':torch.from_numpy(feats.astype(np.float32))
            }
            batch=utils.to_cuda(batch)
            with torch.no_grad():
                scores=self.network(batch)['scores'].cpu().numpy()
            # normalization for NMS comparision only
            argscores = np.argsort(scores)
            scores[argscores] = np.arange(scores.shape[0])/scores.shape[0]
            np.save(f'{savedir}/{pc_id}.npy', scores)

    def run_benchmark(self, input_txt, pcd_dir, features_dir):
        self.network.eval()
        savedir = f'{features_dir}/det_score'
        utils.make_non_exists_dir(savedir)
        print(f'Evaluating the saliency of points using rotaion guided detector on {input_txt}')

        # Load problems txt file
        df = pd.read_csv(input_txt, sep=' ', comment='#')
        df = df.reset_index()
        problem_name = os.path.splitext(os.path.basename(input_txt))[0]

        with torch.no_grad():
            for _, row in tqdm(df.iterrows(), total=df.shape[0]):
                problem_id, source_pcd_filename, target_pcd_filename, source_transform = \
                    benchmark_helpers.load_problem_no_pcd(row, pcd_dir)

                # Source
                feats = np.load(f'{features_dir}/YOHO_Output_Group_feature/{problem_id}.npy')
                batch = {
                    'feats': torch.from_numpy(feats.astype(np.float32))
                }
                batch = utils.to_cuda(batch)
                with torch.no_grad():
                    scores = self.network(batch)['scores'].cpu().numpy()
                # normalization for NMS comparision only
                argscores = np.argsort(scores)
                scores[argscores] = np.arange(scores.shape[0]) / scores.shape[0]
                np.save(f'{savedir}/{problem_id}.npy', scores)

                # Target
                target_pcd_filename = os.path.splitext(target_pcd_filename)[0]
                feats = np.load(f'{features_dir}/YOHO_Output_Group_feature/{target_pcd_filename}.npy')
                batch = {
                    'feats': torch.from_numpy(feats.astype(np.float32))
                }
                batch = utils.to_cuda(batch)
                with torch.no_grad():
                    scores = self.network(batch)['scores'].cpu().numpy()
                # normalization for NMS comparision only
                argscores = np.argsort(scores)
                scores[argscores] = np.arange(scores.shape[0]) / scores.shape[0]
                np.save(f'{savedir}/{target_pcd_filename}.npy', scores)

