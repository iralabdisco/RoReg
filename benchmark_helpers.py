import open3d as o3d
import numpy as np
import os
from pykdtree.kdtree import KDTree
import sys


class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds =  [os.open(os.devnull,os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0],1)
        os.dup2(self.null_fds[1],2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0],1)
        os.dup2(self.save_fds[1],2)
        # Close all file descriptors
        for fd in self.null_fds + self.save_fds:
            os.close(fd)


def overlap(cloud1, cloud2, distance):
    cloud1 = np.array(cloud1.points)
    cloud2 = np.array(cloud2.points)
    cloud2_tree = KDTree(cloud2)
    # distances,indexes  = cloud1_tree.query(cloud1, k = 11, n_jobs=-1)
    # means= np.mean(distances[:,1:None],axis=1)
    # std_dev= np.std(distances[:,1:None],axis=1)
    # cloud2_distances, cloud2_indexes = cloud2_tree.query(cloud1, n_jobs=-1)
    # result =  means+std_dev - cloud2_distances
    # neigh_found = len([x for x in result if np.isfinite(x) and x >=0])
    # for index,point in enumerate(cloud1):
    #     distances,indexes  = cloud1_tree.query(point, k= 10, n_jobs=-1)
    #     medians[index] = np.median(distances[1:None])
        # cloud2_distance, cloud2_indexes = cloud2_tree.query(point, distance_upper_bound=median, n_jobs=-1)
        # print(medians[index])
        # if np.isfinite(cloud2_distance):
            # neigh_found = neigh_found + 1
            # neigh_found = neigh_found + len([x for x in cloud2_distance if np.isfinite(x)])
    dist, idx = cloud2_tree.query(cloud1,1, eps=distance/100, distance_upper_bound = distance)
    neigh_found = np.count_nonzero(np.isfinite(dist))
    overlap = neigh_found/len(cloud1)
    return overlap



def load_problem(row, input_dir):

    with suppress_stdout_stderr():

        problem_id = row['id']

        source_pcd_filename = row['source']
        source_pcd_file = os.path.join(input_dir, source_pcd_filename)
        source_pcd_orig = o3d.io.read_point_cloud(source_pcd_file, remove_nan_points=True,
                                                  remove_infinite_points=True)

        target_pcd_filename = row['target']
        target_pcd_file = os.path.join(input_dir, target_pcd_filename)
        target_pcd_orig = o3d.io.read_point_cloud(target_pcd_file, remove_nan_points=True,
                                                  remove_infinite_points=True)

        source_transform = np.eye(4)
        source_transform[0][0] = row['t1']
        source_transform[0][1] = row['t2']
        source_transform[0][2] = row['t3']
        source_transform[0][3] = row['t4']
        source_transform[1][0] = row['t5']
        source_transform[1][1] = row['t6']
        source_transform[1][2] = row['t7']
        source_transform[1][3] = row['t8']
        source_transform[2][0] = row['t9']
        source_transform[2][1] = row['t10']
        source_transform[2][2] = row['t11']
        source_transform[2][3] = row['t12']

    return problem_id, source_pcd_orig, target_pcd_orig, source_transform, target_pcd_filename

def load_problem_no_pcd(row, input_dir):
    problem_id = row['id']

    source_pcd_filename = row['source']

    target_pcd_filename = row['target']

    source_transform = np.eye(4)
    source_transform[0][0] = row['t1']
    source_transform[0][1] = row['t2']
    source_transform[0][2] = row['t3']
    source_transform[0][3] = row['t4']
    source_transform[1][0] = row['t5']
    source_transform[1][1] = row['t6']
    source_transform[1][2] = row['t7']
    source_transform[1][3] = row['t8']
    source_transform[2][0] = row['t9']
    source_transform[2][1] = row['t10']
    source_transform[2][2] = row['t11']
    source_transform[2][3] = row['t12']

    return problem_id, source_pcd_filename, target_pcd_filename, source_transform


def calculate_error(cloud1: o3d.geometry.PointCloud, cloud2: o3d.geometry.PointCloud) -> float:
    assert len(cloud1.points) == len(cloud2.points), "len(cloud1.points) != len(cloud2.points)"

    centroid, _ = cloud1.compute_mean_and_covariance()
    weights = np.linalg.norm(np.asarray(cloud1.points) - centroid, 2, axis=1)
    distances = np.linalg.norm(np.asarray(cloud1.points) - np.asarray(cloud2.points), 2, axis=1) / len(weights)
    return np.sum(distances / weights)