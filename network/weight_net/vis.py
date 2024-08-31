import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from sklearn.manifold import TSNE
from tqdm import tqdm
import pickle

from geotransformer.utils.open3d import (
    make_open3d_point_cloud,
    make_open3d_axes,
    make_open3d_corr_lines,
)

def dump_pickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def draw_point_to_node(points, nodes, point_to_node, node_colors=None):
    if node_colors is None:
        node_colors = np.random.rand(*nodes.shape)
    # point_colors = node_colors[point_to_node] * make_scaling_along_axis(points, alpha=0.3).reshape(-1, 1)
    point_colors = node_colors[point_to_node]
    node_colors = np.ones_like(nodes) * np.array([[1, 0, 0]])

    ncd = make_open3d_point_cloud(nodes, colors=node_colors)
    pcd = make_open3d_point_cloud(points, colors=point_colors)
    axes = make_open3d_axes()

    o3d.visualization.draw([pcd, ncd, axes])

def save_final_cor(ref_corr_points,src_corr_points,corr_scores,estimated_transform):
    out_put = []
    out_put.append(dict(ref_points=ref_corr_points, src_points=src_corr_points, scores=corr_scores, transform=estimated_transform))
    dump_pickle(out_put,"cor_result.pkl")

    ref_corr_points = np.asarray(ref_corr_points)
    src_corr_points = np.asarray(src_corr_points)

    # select_index = 
    # ref_corr_points = [x for x in ref_corr_points if ]
    ref_points = o3d.geometry.PointCloud()
    src_points = o3d.geometry.PointCloud()

    ref_points.points = o3d.utility.Vector3dVector(ref_corr_points)
    src_points.points = o3d.utility.Vector3dVector(src_corr_points)
    o3d.io.write_point_cloud("./ref_corr_points.ply", ref_points)
    np.save("./ref_corr_points.npy",ref_corr_points)
    np.save('./corr_scores.npy',corr_scores)
    np.save('./estimated_transform.npy',estimated_transform)


    color_r = 0
    color_g = 0
    color_b = 0
    colors = np.zeros([ref_corr_points.shape[0], 3])
    for i in range(len(ref_corr_points)):
        if i%128==0:
            color_r = np.random.rand()
            color_g = np.random.rand()
            color_b = np.random.rand()
        colors[i,:] = [color_r,color_g,color_b]
    ref_points.colors = o3d.utility.Vector3dVector(colors)
    src_points.colors = o3d.utility.Vector3dVector(colors)
    src_points.translate((1,1,1),relative=False)

    select_points = o3d.geometry.PointCloud()
    select_points = ref_points+src_points
    
    o3d.io.write_point_cloud("./final_match_points.ply", select_points)




def draw_node_correspondences2(
    ref_node_corr_knn_points, # （node_num，points_num，3）
    src_node_corr_knn_points, # （node_num，points_num，3）
    ref_node_corr_knn_masks, # (node_num,points_num)
    src_node_corr_knn_masks,
    ref_node_corr_indices, # 128
    src_node_corr_indices, # 128
    ref_node_colors=None,
    offsets=(0, 2, 0)
):
    src_node_corr_knn_points = np.asarray(src_node_corr_knn_points)
    ref_node_corr_knn_points = np.asarray(ref_node_corr_knn_points)
    # 将node中被掩码的点，赋值为0
    for node in range(len(ref_node_corr_knn_points)):
        for points in range(len(ref_node_corr_knn_points[1])):
            if ref_node_corr_knn_masks[node][points] is False:
                ref_node_corr_knn_points[node][points] = [0,0,0]
    for node in range(len(src_node_corr_knn_points)):
        for points in range(len(src_node_corr_knn_points[1])):
            if src_node_corr_knn_masks[node][points] is False:
                src_node_corr_knn_points[node][points] = [0,0,0]
            
    # 我过滤了一部分点，但是没有消除 ref_node_corr_indices ？
    ref_points_selected = []
    src_points_selected = []
    for node in range(len(ref_node_corr_knn_points)):
        after_select = [x for x in ref_node_corr_knn_points[node] if x[0]!= 0 or x[1]!=0 or x[2]!=0]
        for point in after_select:
            point += offsets
        ref_points_selected.append(after_select)
    for node in range(len(src_node_corr_knn_points)):
        after_select = [x for x in src_node_corr_knn_points[node] if x[0]!= 0 or x[1]!=0 or x[2]!=0]
        src_points_selected.append(after_select)
    # print("go check")
    # print(np.asarray(ref_node_corr_indices).shape)
    # print(np.asarray(src_node_corr_indices).shape)
    # print(np.asarray(ref_node_corr_knn_points).shape) 
    # print(np.asarray(src_node_corr_knn_points).shape)
    # 共有 128 个Node
    # 用一个列表，把两个物体的点云，按照node合并，
    all_points = o3d.geometry.PointCloud()
    ref_points = o3d.geometry.PointCloud()
    src_points = o3d.geometry.PointCloud()


    color_r = 0
    color_g = 0
    color_d = 0
    # print(len(ref_node_corr_indices))
    # print(ref_node_corr_indices)
    # print(np.asarray(ref_points_selected).shape)
    # print(np.asarray(ref_node_corr_knn_points).shape)

    
    for i in range(len(ref_node_corr_indices)):
        # tmp_points = ref_node_corr_knn_points[ref_node_corr_indices[i]]+src_node_corr_knn_points[src_node_corr_indices[i]]
        tmp_points_ref = ref_points_selected[ref_node_corr_indices[i]]
        tmp_points_src = src_points_selected[src_node_corr_indices[i]]
        pcd_ref = o3d.geometry.PointCloud()
        pcd_ref.points = o3d.utility.Vector3dVector(tmp_points_ref)
        pcd_src = o3d.geometry.PointCloud()
        pcd_src.points = o3d.utility.Vector3dVector(tmp_points_src)
        # color_r = color_r + 1/128
        # color_g = color_g + 1/128
        # color_d = color_d + 1/128
        color_r = np.random.rand()
        color_g = np.random.rand()
        color_d = np.random.rand()
        pcd_ref.paint_uniform_color([color_r,color_g,color_d])
        pcd_src.paint_uniform_color([color_r,color_g,color_d])
        ref_points += pcd_ref
        src_points += pcd_src
    all_points = ref_points+src_points
    print("pointsall: ",np.asarray(all_points.points))
    o3d.io.write_point_cloud("./colored_result.ply", all_points)
    # o3d.io.write_point_cloud("./colored_result_src.ply", src_points)





def draw_node_correspondences(
    ref_points,
    ref_nodes,
    ref_point_to_node,
    src_points,
    src_nodes,
    src_point_to_node,
    node_correspondences=None,
    ref_node_colors=None,
    src_node_colors=None,
    offsets=(0, 2, 0),
):
    src_nodes = np.asarray(src_nodes.cpu()) + offsets
    src_points = np.asarray(src_points.cpu()) + offsets

    if ref_node_colors is None:
        ref_node_colors = np.random.rand(*ref_nodes.shape)
    # src_point_colors = src_node_colors[src_point_to_node] * make_scaling_along_axis(src_points).reshape(-1, 1)
    ref_point_colors = ref_node_colors[ref_point_to_node.cpu()]
    ref_node_colors = np.ones_like(ref_nodes) * np.array([[1, 0, 0]])

    if src_node_colors is None:
        src_node_colors = np.random.rand(*src_nodes.shape)
    # tgt_point_colors = tgt_node_colors[tgt_point_to_node] * make_scaling_along_axis(tgt_points).reshape(-1, 1)
    src_point_colors = src_node_colors[src_point_to_node.cpu()]
    src_node_colors = np.ones_like(src_nodes) * np.array([[1, 0, 0]])

    ref_ncd = make_open3d_point_cloud(ref_nodes, colors=ref_node_colors)
    ref_pcd = make_open3d_point_cloud(ref_points, colors=ref_point_colors)
    src_ncd = make_open3d_point_cloud(src_nodes, colors=src_node_colors)
    src_pcd = make_open3d_point_cloud(src_points, colors=src_point_colors)
    corr_lines = make_open3d_corr_lines(ref_nodes, src_nodes, node_correspondences)
    axes = make_open3d_axes(scale=0.1)

    # o3d.visualization.draw([ref_pcd, ref_ncd, src_pcd, src_ncd, corr_lines, axes])
    o3d.visualization.draw([ref_pcd, ref_ncd, src_pcd, src_ncd, axes])