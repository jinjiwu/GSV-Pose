import numpy as np
import pickle
import open3d as o3d
import copy

def dump_pickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def gen_data(ref_points,src_points):
    all_points = o3d.geometry.PointCloud()
    all_points.points = o3d.utility.Vector3dVector(np.concatenate((ref_points,src_points), axis=0))

    # 法线估计
    radius = 0.02  # 搜索半径
    max_nn = 60  # 邻域内用于估算法线的最大点数
    all_points.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius, max_nn))

    all_normals = np.array(all_points.normals)
    
    all_data = []
    all_data.append(dict(points=np.asarray(all_points.points), normals=np.asarray(all_normals), label=27))
    all_data = np.asarray(all_data)
    dump_pickle(np.asarray(all_data), '/home/zhangyuekun/GPV_Pose/network/weight_net/data/test.pkl')
    # dump_pickle(np.asarray(all_data), 'train.pkl')
    # dump_pickle(np.asarray(all_data), 'val.pkl')
    return all_data




# for subset in ['train', 'test']:
#     process(subset)
