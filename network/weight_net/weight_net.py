import sys 
sys.path.append("/home/zhangyuekun/GeoTransformer")
sys.path.append("/home/zhangyuekun/GeoTransformer/experiments")
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed
import numpy as np

from geotransformer.utils.torch import release_cuda, to_cuda
from geotransformer.modules.ops import point_to_node_partition, index_select
from geotransformer.modules.registration import get_node_correspondences
from geotransformer.modules.geotransformer import (
    GeometricTransformer,
    SuperPointMatching,
    SuperPointTargetGenerator,
    LocalGlobalRegistration,
)
from geotransformer.utils.data import (
    registration_collate_fn_stack_mode,
    calibrate_neighbors_stack_mode,
    build_dataloader_stack_mode,
)

from network.weight_net.config import make_cfg
from network.weight_net.model import create_model
from network.weight_net.gen_data import gen_data
from network.weight_net.dataset import test_data_loader
from network.weight_net.test import Tester

def load_template():
    # cat_name2id = {'bottle': 1, 'bowl': 2, 'camera': 3, 'can': 4, 'laptop': 5, 'mug': 6}
    template = []
    bottle = np.load("/home/zhangyuekun/GPV_Pose/network/weight_net/template/bottle.npy")
    bowl = np.load("/home/zhangyuekun/GPV_Pose/network/weight_net/template/bowl.npy")
    camera = np.load("/home/zhangyuekun/GPV_Pose/network/weight_net/template/camera.npy")
    can = np.load("/home/zhangyuekun/GPV_Pose/network/weight_net/template/can.npy")
    laptop = np.load("/home/zhangyuekun/GPV_Pose/network/weight_net/template/laptop.npy")
    mug = np.load("/home/zhangyuekun/GPV_Pose/network/weight_net/template/mug.npy")
    template.append(bottle)
    template.append(bowl)
    template.append(camera)
    template.append(can)
    template.append(laptop)
    template.append(mug)
    return template

class weighter():
    def __init__(self):
        self.cfg = make_cfg()
        self.model = create_model(self.cfg)
        self.template = load_template()

    def get_weight(self,PC,object_id):
        # PC = np.random.rand(6,1028,3)
        # object_id = [1,3,2,4,6,5]
        bs = len(PC)
        weight_all = []
        for i in range(bs):
            search_points = PC[i]
            template_points = self.template[int(object_id[i])]
            # data_dict = to_cuda(data_dict)
            # data_dict feats
            # 生成test.pkl
            gen_data(template_points,search_points.cpu())
            tester =Tester(self.cfg)
            # 计算结果并保存
            ref_points_f,ref_corr_points,corr_scores = tester.run()
            # 将权重存储到i_weight.npy中
            weight_i = compareOverlap(ref_points_f.cpu(),ref_corr_points.cpu(),corr_scores.cpu())
            weight_all.append(weight_i)
        return weight_all

        
        

def compareOverlap(pt1,pt2,weight):
    n = 1028
    # pt1 = np.load('/home/zhangyuekun/GeoTransformer/experiments/gentransformer.NOCS/ref_points_f.npy') # 1028 * 3
    # pt2 = np.load("/home/zhangyuekun/GeoTransformer/experiments/gentransformer.NOCS/ref_corr_points.npy") # n * 3
    # weight = np.load("/home/zhangyuekun/GeoTransformer/experiments/gentransformer.NOCS/corr_scores.npy")
    weight = np.asarray(weight.cpu())

    cnt = 0
    # weight 映射到 (0.3 ~ 1)
    weight = 0.3+(weight-np.min(weight))/(np.max(weight)-np.min(weight))
    weight = 0.3+((1-0.3)/(np.max(weight)-np.min(weight)))*(weight-np.min(weight))

    # print("min",min(weight))
    # print("max",max(weight))
    
    # 需要去重（source 和 ref 中都有重采样的重复点）重复点保留最大权重
    rep_index = []
    for i in range(len(pt2)):
        for j in range(len(pt2)):
            if i==j: continue
            if (pt2[i] == pt2[j]).all():
                idx = min(i,j)
                if idx not in rep_index:
                    rep_index.append(idx)
    pt2 = np.delete(pt2,rep_index,0)
    weight = np.delete(weight,rep_index)

    weight_all = np.ones(1028,dtype=float)
    weight_all = weight_all*0.3
    # print(max(weight_all))
    for j in range(len(pt2)):
        for i in range(len(pt1)):
            if (pt2[j] == pt1[i]).all():
                cnt += 1
                # print(i," : ",j)
                # print(weight_all[i],"+",weight[j]*0.7)
                weight_all[i] += weight[j]*0.7
                break
    # print(cnt)
    # for i in range(len(weight_all)):
        # print(weight_all[i])
    # print(max(weight_all))
    # np.save("/home/zhangyuekun/GPV_Pose/vote_weight/"+str(index)+"_weight.npy",weight_all)
    return weight_all




if __name__ == "__main__":
    wgter = weighter()
    wgter.get_weight()