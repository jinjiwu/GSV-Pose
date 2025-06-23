import torch
import torch.nn.functional as F
import numpy as np
import absl.flags as flags

FLAGS = flags.FLAGS


class PcCut:
    def __init__(self, seed=42):
        self.K = np.array(
            [[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]],
            dtype=np.float32,
        )
        self.rng = np.random.default_rng(seed)  # 使用随机数生成器

    def cut_use_one_plane(self, points, cut_ratio=0.1):
        """
        使用一个平面切割点云,切割比例为 cut_ratio
        """
        if points is None or points.shape[0] == 0:
            print("Warning: 点云数据缺失")
            return np.array([]), np.array([])

        N_total = points.shape[0]

        # Step 1: 随机生成平面法向量
        # n = np.random.randn(3)
        n = self.rng.standard_normal(3)
        n /= np.linalg.norm(n)
        print(f"Cut plane normal vector: {n}")

        # Step 2: 随机选一个点为平面上的点
        p0 = points[self.rng.integers(0, N_total)]

        # Step 3: 计算所有点到平面的有符号距离
        distances = (points - p0) @ n  # 点乘计算每个点到平面的距离

        # Step 4: 按距离排序，裁剪前 cut_ratio 百分比的点
        sorted_indices = np.argsort(distances)
        N_cut = int(N_total * cut_ratio)

        mask = np.ones(N_total, dtype=bool)
        mask[sorted_indices[:N_cut]] = False  # 标记为被遮挡

        visible_points = points[mask]
        occluded_points = points[~mask]

        return visible_points

    def cut_use_two_plane(self, points, cut_ratio=0.1):
        """
        使用两个平面切割点云, 切割比例为 cut_ratio
        """
        if points is None or points.shape[0] == 0:
            print("Warning: 点云数据缺失")
            return np.array([]), np.array([])

        # 分为两个的单次切割
        visible_points1 = self.cut_use_one_plane(points, cut_ratio / 2.0)
        visible_points2 = self.cut_use_one_plane(
            visible_points1, cut_ratio / 2.0 / (1 - cut_ratio / 2.0)
        )

        return visible_points2


def PC_sample(obj_mask, Depth, camK, coor2d):
    """
    :param Depth: bs x 1 x h x w
    :param camK:
    :param coor2d:
    :return:
    """
    # handle obj_mask
    if obj_mask.shape[1] == 2:  # predicted mask
        obj_mask = F.softmax(obj_mask, dim=1)
        _, obj_mask = torch.max(obj_mask, dim=1)
    """
    import matplotlib.pyplot as plt
    plt.imshow(obj_mask[0, ...].detach().cpu().numpy())
    plt.show()
    """
    bs, H, W = Depth.shape[0], Depth.shape[2], Depth.shape[3]
    x_label = coor2d[:, 0, :, :]
    y_label = coor2d[:, 1, :, :]

    rand_num = FLAGS.random_points
    samplenum = rand_num

    PC = torch.zeros([bs, samplenum, 3], dtype=torch.float32, device=Depth.device)
    pc_cut = PcCut(FLAGS.cut_seed)

    for i in range(bs):
        dp_now = Depth[i, ...].squeeze()  # 256 x 256
        x_now = x_label[i, ...]  # 256 x 256
        y_now = y_label[i, ...]
        obj_mask_now = obj_mask[i, ...].squeeze()  # 256 x 256
        dp_mask = dp_now > 0.0
        fuse_mask = obj_mask_now.float() * dp_mask.float()

        camK_now = camK[i, ...]

        # analyze camK
        fx = camK_now[0, 0]
        fy = camK_now[1, 1]
        ux = camK_now[0, 2]
        uy = camK_now[1, 2]

        x_now = (x_now - ux) * dp_now / fx
        y_now = (y_now - uy) * dp_now / fy

        p_n_now = torch.cat(
            [
                x_now[fuse_mask > 0].view(-1, 1),
                y_now[fuse_mask > 0].view(-1, 1),
                dp_now[fuse_mask > 0].view(-1, 1),
            ],
            dim=1,
        )

        # cut point cloud use plane
        if hasattr(FLAGS, "cut_method") and not FLAGS.train:
            if FLAGS.cut_method == "one_plane":
                p_n_now = pc_cut.cut_use_one_plane(
                    p_n_now.cpu().numpy(), cut_ratio=FLAGS.cut_ratio
                )
            elif FLAGS.cut_method == "two_plane":
                p_n_now = pc_cut.cut_use_two_plane(
                    p_n_now.cpu().numpy(), cut_ratio=FLAGS.cut_ratio
                )
            p_n_now = torch.from_numpy(p_n_now).to(Depth.device)

        # basic sampling
        if FLAGS.sample_method == "basic":
            l_all = p_n_now.shape[0]
            if l_all <= 1.0:
                return None, None
            if l_all >= samplenum:
                replace_rnd = False
            else:
                replace_rnd = True

            choose = np.random.choice(
                l_all, samplenum, replace=replace_rnd
            )  # can selected more than one times
            p_select = p_n_now[choose, :]
        else:
            p_select = None
            raise NotImplementedError

        # reprojection
        if p_select.shape[0] > samplenum:
            p_select = p_select[p_select.shape[0] - samplenum : p_select.shape[0], :]

        PC[i, ...] = p_select[:, :3]

    return PC / 1000.0
