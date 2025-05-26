import sys

sys.path.append("GeoTransformer")
sys.path.append("GeoTransformer/experiments")
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
from tools.timer import timer_epoch
import torch_scatter


def load_template():
    # cat_name2id = {'bottle': 1, 'bowl': 2, 'camera': 3, 'can': 4, 'laptop': 5, 'mug': 6}
    template = []
    bottle = np.load("network/weight_net/template/bottle.npy")
    bowl = np.load("network/weight_net/template/bowl.npy")
    camera = np.load("network/weight_net/template/camera.npy")
    can = np.load("network/weight_net/template/can.npy")
    laptop = np.load("network/weight_net/template/laptop.npy")
    mug = np.load("network/weight_net/template/mug.npy")
    template.append(bottle)
    template.append(bowl)
    template.append(camera)
    template.append(can)
    template.append(laptop)
    template.append(mug)
    return template


class weighter:
    def __init__(self):
        self.cfg = make_cfg()
        self.model = create_model(self.cfg)
        self.template = load_template()
        self.tester = Tester(self.cfg)

    def get_weight(self, PC, object_id):
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
            # gen_data(template_points, search_points.cpu())

            # 计算结果并保存
            ref_points_f, ref_corr_points, corr_scores = self.tester.run()
            # 将权重存储到i_weight.npy中
            timer_epoch.start("knn")
            weight_i = torch_optimized_compareOverlap_with_torch_scatter(
                ref_points_f, ref_corr_points, corr_scores
            )
            timer_epoch.stop("knn")
            # weight_i = optimized_compareOverlap(
            #     ref_points_f.cpu(), ref_corr_points.cpu(), corr_scores.cpu()
            # )
            weight_all.append(weight_i.cpu().numpy())
        return weight_all


def torch_optimized_compareOverlap_with_torch_scatter(
    pt1: torch.Tensor,
    pt2_original: torch.Tensor,
    weight_original: torch.Tensor,
    fixed_n_pt1: int = 1028,
    initial_weight_value: float = 0.3,
    added_weight_factor: float = 0.7,
) -> torch.Tensor:
    """
    优化的函数，用于比较重叠并计算权重，兼容 PyTorch 1.10.1 (使用 torch_scatter)。

    参数:
        pt1 (torch.Tensor): 参考点云，形状为 (N, Dims)。
                            假设已位于 CUDA 设备上。
        pt2_original (torch.Tensor): 源点云，形状为 (M, Dims)，可能包含重复点。
                                     假设与 pt1位于同一 CUDA 设备上。
        weight_original (torch.Tensor): 对应于 pt2_original 的权重，形状为 (M,)。
                                        假设与 pt1 位于同一 CUDA 设备上。
        fixed_n_pt1 (int): 输出权重张量的固定大小。这决定了从 pt1 开头计算权重的点的数量。
        initial_weight_value (float): 权重的默认值以及归一化权重的下限。
        added_weight_factor (float):匹配点的权重添加到 pt1 权重时的乘数因子。

    返回:
        torch.Tensor: pt1 中前 `fixed_n_pt1` 个点各自计算出的权重，形状为 (fixed_n_pt1,)。
                      张量与输入张量位于同一设备。
    """
    device = pt1.device
    calc_dtype = torch.float32  # 计算时使用的数据类型

    # 1. 处理 pt2_original 或 weight_original 为空的情况
    if pt2_original.shape[0] == 0 or weight_original.shape[0] == 0:
        return torch.full(
            (fixed_n_pt1,), float(initial_weight_value), device=device, dtype=calc_dtype
        )

    # 2. 权重归一化到 [initial_weight_value, 1.0]
    w_orig_casted = weight_original.to(dtype=calc_dtype)
    min_w = torch.min(w_orig_casted)
    max_w = torch.max(w_orig_casted)

    normalized_weight: torch.Tensor
    if max_w > min_w:
        normalized_weight = initial_weight_value + (
            (1.0 - initial_weight_value) * (w_orig_casted - min_w) / (max_w - min_w)
        )
    else:
        normalized_weight = torch.full_like(w_orig_casted, float(initial_weight_value))

    # 3. 对 pt2_original 去重并聚合权重
    unique_pt2_rows, inverse_indices = torch.unique(
        pt2_original, dim=0, return_inverse=True
    )
    num_unique_pt2 = unique_pt2_rows.shape[0]

    if num_unique_pt2 == 0:
        return torch.full(
            (fixed_n_pt1,), float(initial_weight_value), device=device, dtype=calc_dtype
        )

    # 使用 torch_scatter.scatter 替代 scatter_reduce_ (reduce="amax")
    # torch_scatter.scatter(reduce="max") 对于未在 index 中出现的 dim_size 索引，其输出值默认为0。
    # 因为 normalized_weight >= 0.3，这符合我们取最大值的期望。
    final_pt2_weights = torch_scatter.scatter(
        src=normalized_weight,
        index=inverse_indices,
        dim=0,
        dim_size=num_unique_pt2,
        reduce="max",
    )

    # 4. 初始化 pt1 点的 weight_all (这是输出张量)
    weight_all = torch.full(
        (fixed_n_pt1,), float(initial_weight_value), device=device, dtype=calc_dtype
    )

    # 确定用于匹配的 pt1 中的有效点数
    effective_n_pt1 = min(pt1.shape[0], fixed_n_pt1)

    if effective_n_pt1 == 0:  # 如果 pt1 为空或 fixed_n_pt1 为 0
        return weight_all

    pt1_sliced = pt1[:effective_n_pt1]

    # 5. 将 unique_pt2_rows 与 pt1_sliced 匹配并更新 weight_all
    if num_unique_pt2 > 0:  # 仅当 pt2 中有唯一行可供匹配时才继续
        expanded_pt1 = pt1_sliced.unsqueeze(1)
        expanded_unique_pt2 = unique_pt2_rows.unsqueeze(0)
        match_matrix = (expanded_pt1 == expanded_unique_pt2).all(dim=2)
        matched_pt1_indices, matched_unique_pt2_indices = torch.where(match_matrix)

        if matched_pt1_indices.shape[0] > 0:  # 如果存在任何匹配
            # 初始化一个张量来存储每个 unique_pt2 点在 pt1_sliced 中的首次匹配索引
            # 使用 effective_n_pt1 作为哨兵值（表示未匹配或大于任何有效索引）
            min_idx_in_pt1_for_unique_pt2_initial = torch.full(
                (num_unique_pt2,),
                effective_n_pt1,  # 哨兵值
                device=device,
                dtype=torch.long,
            )

            # 使用 torch_scatter.scatter 替代 scatter_reduce_ (reduce="amin")
            # 通过提供 out 参数，torch_scatter.scatter 会使用 out 中的值作为初始值进行归约
            min_idx_in_pt1_for_unique_pt2 = torch_scatter.scatter(
                src=matched_pt1_indices.long(),
                index=matched_unique_pt2_indices,
                dim=0,
                out=min_idx_in_pt1_for_unique_pt2_initial.clone(),  # 传入初始值
                dim_size=num_unique_pt2,
                reduce="min",
            )

            valid_matches_mask = min_idx_in_pt1_for_unique_pt2 < effective_n_pt1
            target_pt1_indices_to_update = min_idx_in_pt1_for_unique_pt2[
                valid_matches_mask
            ]
            weights_values_to_add = final_pt2_weights[valid_matches_mask]

            # 使用内置的 scatter_add_ (这个方法在 PyTorch 1.10.1 中是存在的)
            weight_all.scatter_add_(
                dim=0,
                index=target_pt1_indices_to_update,
                src=weights_values_to_add * added_weight_factor,
            )
    return weight_all


def optimized_compareOverlap(pt1, pt2_original, weight_original):
    """
    优化后的函数，用于比较重叠并计算权重。

    参数:
        pt1 (np.ndarray): 参考点云，期望形状为 (N, 3)，其中 N 通常为 1028。
        pt2_original (np.ndarray): 源点云，形状为 (M, 3)，可能包含重复点。
        weight_original (torch.Tensor or np.ndarray): 对应于 pt2_original 的权重，形状为 (M,)。
                                                    如果是 PyTorch 张量，则假定其在 CPU 上。
    返回:
        np.ndarray: pt1 中每个点的计算权重，形状为 (1028,)。
    """

    # 确保 weight 是一个 NumPy 数组
    if hasattr(weight_original, "cpu"):  # 检查是否为 PyTorch 张量
        weight = weight_original.cpu().numpy()
    else:
        weight = np.asarray(weight_original)  # 确保其为 numpy 数组

    # pt1 中点的预期数量，基于原始代码的使用情况
    fixed_n_pt1 = 1028

    # 尽早处理 pt2/weight 为空的情况
    if pt2_original.shape[0] == 0 or weight.shape[0] == 0:
        return np.full(fixed_n_pt1, 0.3, dtype=float)

    # 1. 权重归一化到 [0.3, 1.0]
    min_w, max_w = np.min(weight), np.max(weight)
    if max_w > min_w:
        normalized_weight = 0.3 + (0.7 * (weight - min_w) / (max_w - min_w))
    else:  # 所有权重相同（或只有一个权重）
        normalized_weight = np.full_like(weight, 0.3, dtype=float)

    # 2. 对 pt2_original 去重并聚合权重（保留重复点的最大 normalized_weight）
    # np.unique 返回唯一行及其反向映射索引。
    # 唯一行默认是排序的，这对于此用例没有问题。
    unique_pt2_rows, inverse_indices = np.unique(
        pt2_original, axis=0, return_inverse=True
    )

    num_unique_rows = unique_pt2_rows.shape[0]

    # 初始化 unique_pt2_rows 的权重。使用 0.0 是安全的，因为 normalized_weight >= 0.3
    final_pt2_weights = np.zeros(num_unique_rows, dtype=float)
    # 对于每个唯一行，找到其对应的最大 normalized_weight
    np.maximum.at(final_pt2_weights, inverse_indices, normalized_weight)

    # 3. 初始化 pt1 点的 weight_all
    # 根据原始代码，weight_all 大小固定为 1028，并初始化为 0.3
    weight_all = np.full(fixed_n_pt1, 0.3, dtype=float)

    # 4. 将 unique_pt2_rows 与 pt1 匹配并更新 weight_all
    # 为 pt1 创建一个查找字典: {point_tuple: first_occurrence_index}
    # 这确保了如果 pt1 有重复点，其行为与原始代码中的 `break` 一致。
    pt1_lookup = {}
    # 遍历 pt1 中至多 fixed_n_pt1 个点。
    # 原始代码暗示 pt1 有 1028 个点。
    for i, p_1_coord in enumerate(pt1[:fixed_n_pt1]):
        p_1_tuple = tuple(p_1_coord)
        if p_1_tuple not in pt1_lookup:
            pt1_lookup[p_1_tuple] = i

    # 遍历 pt2 中的唯一（不重复）点
    for j in range(num_unique_rows):
        pt_j_coords = unique_pt2_rows[j]
        pt_j_tuple = tuple(pt_j_coords)

        if pt_j_tuple in pt1_lookup:
            idx_in_pt1 = pt1_lookup[pt_j_tuple]
            # 由于 pt1_lookup 的构建方式，idx_in_pt1 保证小于 fixed_n_pt1
            weight_all[idx_in_pt1] += final_pt2_weights[j] * 0.7

    return weight_all


def compareOverlap(pt1, pt2, weight):
    n = 1028
    # pt1 = np.load('GeoTransformer/experiments/gentransformer.NOCS/ref_points_f.npy') # 1028 * 3
    # pt2 = np.load("GeoTransformer/experiments/gentransformer.NOCS/ref_corr_points.npy") # n * 3
    # weight = np.load("GeoTransformer/experiments/gentransformer.NOCS/corr_scores.npy")
    weight = np.asarray(weight.cpu())

    cnt = 0
    # weight 映射到 (0.3 ~ 1)
    weight = 0.3 + (weight - np.min(weight)) / (np.max(weight) - np.min(weight))
    weight = 0.3 + ((1 - 0.3) / (np.max(weight) - np.min(weight))) * (
        weight - np.min(weight)
    )

    # print("min",min(weight))
    # print("max",max(weight))

    # 需要去重（source 和 ref 中都有重采样的重复点）重复点保留最大权重
    rep_index = []
    for i in range(len(pt2)):
        for j in range(len(pt2)):
            if i == j:
                continue
            if (pt2[i] == pt2[j]).all():
                idx = min(i, j)
                if idx not in rep_index:
                    rep_index.append(idx)
    pt2 = np.delete(pt2, rep_index, 0)
    weight = np.delete(weight, rep_index)

    weight_all = np.ones(1028, dtype=float)
    weight_all = weight_all * 0.3
    # print(max(weight_all))
    for j in range(len(pt2)):
        for i in range(len(pt1)):
            if (pt2[j] == pt1[i]).all():
                cnt += 1
                # print(i," : ",j)
                # print(weight_all[i],"+",weight[j]*0.7)
                weight_all[i] += weight[j] * 0.7
                break
    # print(cnt)
    # for i in range(len(weight_all)):
    # print(weight_all[i])
    # print(max(weight_all))
    # np.save("vote_weight/"+str(index)+"_weight.npy",weight_all)
    return weight_all


if __name__ == "__main__":
    wgter = weighter()
    wgter.get_weight()
