import torch
import numpy as np
import os
import cv2
import pickle
import shutil
from pathlib import Path
import glob
import random
import torch.backends
import json

from network.GPVPose import GPVPose
from tools.geom_utils import generate_RT, generate_sRT
from config.config import *
from absl import app
import time

import absl.flags as flags

FLAGS = flags.FLAGS
from evaluation.load_data_eval import PoseDataset
import torch.nn as nn
import time

# from creating log
import tensorflow as tf
import evaluation
from evaluation.eval_utils import setup_logger, compute_mAP, draw_detections, draw_img
from evaluation.eval_utils_v2 import compute_degree_cm_mAP
from tqdm import tqdm

device = "cuda"


def evaluate(argv):
    if not os.path.exists(FLAGS.model_save):
        os.makedirs(FLAGS.model_save)
    tf.compat.v1.disable_eager_execution()
    logger = setup_logger("eval_log", os.path.join(FLAGS.model_save, "log_eval.txt"))
    Train_stage = "PoseNet_only"
    FLAGS.train = False

    model_name = os.path.basename(FLAGS.resume_model).split(".")[0]
    # build dataset annd dataloader

    val_dataset = PoseDataset(source=FLAGS.dataset, mode="test")
    output_path = os.path.join(FLAGS.model_save, f"eval_result_{model_name}")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    import pickle

    pred_result_save_path = os.path.join(output_path, "pred_result.pkl")
    if os.path.exists(pred_result_save_path) and False:
        with open(pred_result_save_path, "rb") as file:
            pred_results = pickle.load(file)
    else:
        network = GPVPose(Train_stage)
        network = network.to(device)

        if FLAGS.resume:
            state_dict = torch.load(FLAGS.resume_model)
            network.load_state_dict(state_dict)
        else:
            raise NotImplementedError
        # start to test
        network = network.eval()
        pred_results = []

        for i, data in tqdm(enumerate(val_dataset, 1)):
            if data is None:
                print("data is None")
                continue
            data, detection_dict, gts = data
            mean_shape = data["mean_shape"].to(device)
            sym = data["sym_info"].to(device)
            if len(data["cat_id_0base"]) == 0:
                detection_dict["pred_RTs"] = np.zeros((0, 4, 4))
                detection_dict["pred_scales"] = np.zeros((0, 4, 4))
                pred_results.append(detection_dict)
                continue
            start_time = time.time()
            # draw_img(
            #     f"eval_logs/roi_eval_{i}.png",
            #     data["roi_img"][0].permute(1, 2, 0).cpu().numpy(),
            # )
            output_dict = network(
                rgb=data["roi_img"].to(device),
                depth=data["roi_depth"].to(device),
                depth_normalize=data["depth_normalize"].to(device),
                obj_id=data["cat_id_0base"].to(device),
                camK=data["cam_K"].to(device),
                gt_mask=data["roi_mask"].to(device),
                gt_R=None,
                gt_t=None,
                gt_s=None,
                mean_shape=mean_shape,
                gt_2D=data["roi_coord_2d"].to(device),
                sym=sym,
                def_mask=data["roi_mask"].to(device),
            )
            end_time = time.time()
            logger.info("inference total time: {}".format(end_time - start_time))
            p_green_R_vec = output_dict["p_green_R"].detach()
            p_red_R_vec = output_dict["p_red_R"].detach()
            p_T = output_dict["Pred_T"].detach()
            p_s = output_dict["Pred_s"].detach()
            f_green_R = output_dict["f_green_R"].detach()
            f_red_R = output_dict["f_red_R"].detach()
            from tools.training_utils import get_gt_v

            pred_s = p_s + mean_shape
            pred_RT = generate_RT(
                [p_green_R_vec, p_red_R_vec],
                [f_green_R, f_red_R],
                p_T,
                mode="vec",
                sym=sym,
            )

            if pred_RT is not None:
                pred_RT = pred_RT.detach().cpu().numpy()
                pred_s = pred_s.detach().cpu().numpy()
                detection_dict["pred_RTs"] = pred_RT
                detection_dict["pred_scales"] = pred_s
            else:
                assert NotImplementedError
            pred_results.append(detection_dict)

            # -------save result-----------
            autoSave = False
            if autoSave:
                if FLAGS.our_camK:
                    intrinsics = np.array(
                        [[386.49, 0, 320.494], [0, 386.008, 236.679], [0, 0, 1]],
                        dtype=np.float,
                    )
                    # intrinsics = np.array(
                    #     [[385.964, 0, 320.494], [0, 385.484, 236.679], [0, 0, 1]],
                    #     dtype=np.float,
                    # )
                else:
                    intrinsics = np.array(
                        [[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]],
                        dtype=np.float,
                    )
                img = detection_dict["image_path"]
                img_list = img.split("/")
                img_scene = img_list[3]
                img_num = img_list[4]
                img_path = "data/Real/test/" + img_scene + "/" + img_num + "_color.png"
                # print(img_path)
                import cv2

                img_oral = cv2.imread(img_path)
                print("img_path", img_path)
                print("gt_RTs", detection_dict["gt_RTs"])
                draw_detections(
                    img_oral,
                    "eval_logs/result/pic_vote_box",
                    "real_test",
                    img_scene,
                    img_num,
                    intrinsics,
                    detection_dict["pred_RTs"],
                    detection_dict["pred_scales"],
                    detection_dict["pred_class_ids"],
                    detection_dict["gt_RTs"],
                    detection_dict["gt_scales"],
                    detection_dict["gt_class_ids"],
                    draw_gt=not FLAGS.our_camK,
                )
                pc = output_dict["recon"].detach()
                print(pc.shape)
                # 保存重建结果点云
                # folder = os.path.exists("/home/zhangyuekun/GPV_Pose/result/pc_recon/real_test/"+img_scene +"/"+ img_num)
                # if not folder:
                #     os.makedirs("/home/zhangyuekun/GPV_Pose/result/pc_recon/real_test/"+img_scene +"/"+ img_num)

                # save_pointscloud(pc,"/home/zhangyuekun/GPV_Pose/result/pc_recon/real_test/"+img_scene +"/"+ img_num)

        with open(pred_result_save_path, "wb") as file:
            pickle.dump(pred_results, file)

    if FLAGS.eval_inference_only:
        import sys

        sys.exit()

    degree_thres_list = list(range(0, 61, 1))
    shift_thres_list = [i / 2 for i in range(21)]
    iou_thres_list = [i / 100 for i in range(101)]

    # iou_aps, pose_aps, iou_acc, pose_acc = compute_mAP(pred_results, output_path, degree_thres_list, shift_thres_list,
    #                                                  iou_thres_list, iou_pose_thres=0.1, use_matches_for_pose=True,)
    synset_names = ["BG"] + ["bottle", "bowl", "camera", "can", "laptop", "mug"]
    if FLAGS.per_obj in synset_names:
        idx = synset_names.index(FLAGS.per_obj)
    else:
        idx = -1
    iou_aps, pose_aps, eval_dict = compute_degree_cm_mAP(
        pred_results,
        synset_names,
        output_path,
        degree_thres_list,
        shift_thres_list,
        iou_thres_list,
        iou_pose_thres=0.1,
        use_matches_for_pose=True,
    )

    with open(
        os.path.join(
            output_path,
            f"eval_dict_ratio{FLAGS.cut_ratio}_seed{FLAGS.cut_seed}_{FLAGS.cut_method}.pkl",
        ),
        "wb",
    ) as f:
        pickle.dump(eval_dict, f)

    # # fw = open('{0}/eval_logs.txt'.format(result_dir), 'a')
    iou_25_idx = iou_thres_list.index(0.25)
    iou_50_idx = iou_thres_list.index(0.5)
    iou_75_idx = iou_thres_list.index(0.75)
    degree_05_idx = degree_thres_list.index(5)
    degree_10_idx = degree_thres_list.index(10)
    shift_02_idx = shift_thres_list.index(2)
    shift_05_idx = shift_thres_list.index(5)
    shift_10_idx = shift_thres_list.index(10)

    messages = []

    if FLAGS.per_obj in synset_names:
        messages.append("mAP:")
        messages.append("3D IoU at 25: {:.1f}".format(iou_aps[idx, iou_25_idx] * 100))
        messages.append("3D IoU at 50: {:.1f}".format(iou_aps[idx, iou_50_idx] * 100))
        messages.append("3D IoU at 75: {:.1f}".format(iou_aps[idx, iou_75_idx] * 100))
        messages.append(
            "5 degree, 2cm: {:.1f}".format(
                pose_aps[idx, degree_05_idx, shift_02_idx] * 100
            )
        )
        messages.append(
            "5 degree, 5cm: {:.1f}".format(
                pose_aps[idx, degree_05_idx, shift_05_idx] * 100
            )
        )
        messages.append(
            "10 degree, 2cm: {:.1f}".format(
                pose_aps[idx, degree_10_idx, shift_02_idx] * 100
            )
        )
        messages.append(
            "10 degree, 5cm: {:.1f}".format(
                pose_aps[idx, degree_10_idx, shift_05_idx] * 100
            )
        )
        messages.append(
            "10 degree, 10cm: {:.1f}".format(
                pose_aps[idx, degree_10_idx, shift_10_idx] * 100
            )
        )
    else:
        messages.append("average mAP:")
        messages.append("3D IoU at 25: {:.1f}".format(iou_aps[idx, iou_25_idx] * 100))
        messages.append("3D IoU at 50: {:.1f}".format(iou_aps[idx, iou_50_idx] * 100))
        messages.append("3D IoU at 75: {:.1f}".format(iou_aps[idx, iou_75_idx] * 100))
        messages.append(
            "5 degree, 2cm: {:.1f}".format(
                pose_aps[idx, degree_05_idx, shift_02_idx] * 100
            )
        )
        messages.append(
            "5 degree, 5cm: {:.1f}".format(
                pose_aps[idx, degree_05_idx, shift_05_idx] * 100
            )
        )
        messages.append(
            "10 degree, 2cm: {:.1f}".format(
                pose_aps[idx, degree_10_idx, shift_02_idx] * 100
            )
        )
        messages.append(
            "10 degree, 5cm: {:.1f}".format(
                pose_aps[idx, degree_10_idx, shift_05_idx] * 100
            )
        )
        messages.append(
            "10 degree, 10cm: {:.1f}".format(
                pose_aps[idx, degree_10_idx, shift_10_idx] * 100
            )
        )

        for idx in range(1, len(synset_names)):
            messages.append("category {}".format(synset_names[idx]))
            messages.append("mAP:")
            messages.append(
                "3D IoU at 25: {:.1f}".format(iou_aps[idx, iou_25_idx] * 100)
            )
            messages.append(
                "3D IoU at 50: {:.1f}".format(iou_aps[idx, iou_50_idx] * 100)
            )
            messages.append(
                "3D IoU at 75: {:.1f}".format(iou_aps[idx, iou_75_idx] * 100)
            )
            messages.append(
                "5 degree, 2cm: {:.1f}".format(
                    pose_aps[idx, degree_05_idx, shift_02_idx] * 100
                )
            )
            messages.append(
                "5 degree, 5cm: {:.1f}".format(
                    pose_aps[idx, degree_05_idx, shift_05_idx] * 100
                )
            )
            messages.append(
                "10 degree, 2cm: {:.1f}".format(
                    pose_aps[idx, degree_10_idx, shift_02_idx] * 100
                )
            )
            messages.append(
                "10 degree, 5cm: {:.1f}".format(
                    pose_aps[idx, degree_10_idx, shift_05_idx] * 100
                )
            )
            messages.append(
                "10 degree, 10cm: {:.1f}".format(
                    pose_aps[idx, degree_10_idx, shift_10_idx] * 100
                )
            )

    for msg in messages:
        logger.info(msg)


class NoisyEvaluator:
    def __init__(
        self,
        cut_ratio,
        seed,
        cut_method="one_plane",
        iou_thres=0.5,
        degree_thres=10,
        shift_thres=5,
    ):
        """"""
        # stat
        self.alpha = 0.05  # significance level

        self.cut_ratio = [0.1, 0.2]
        self.cut_ratio = cut_ratio
        self.seed = seed
        self.cut_method = cut_method
        self.our_method_result = [
            f"eval_logs/eval_result_gpv_pose_update/eval_dict_ratio{self.cut_ratio}_seed{self.seed}_{self.cut_method}.pkl"
        ]
        self.gpv_method_result = [
            f"external/GPV_Pose/eval_logs/eval_result_gpv_pose_update/eval_dict_ratio{self.cut_ratio}_seed{self.seed}_{self.cut_method}.pkl"
        ]
        self.iou_thres = iou_thres
        self.degree_thres = degree_thres
        self.shift_thres = shift_thres

    def calc_p_value(self, our_result_p, gpv_result_p):
        with open(our_result_p, "rb") as f:
            our_result = pickle.load(f)
        with open(gpv_result_p, "rb") as f:
            gpv_result = pickle.load(f)

        iou_stat, iou_p_value = self.stat_iou(our_result, gpv_result, self.iou_thres)
        print(f"IOU stat: {iou_stat}, p-value: {iou_p_value}")

        degree_cm_stat, degree_cm_p_value = self.stat_degree_cm(
            our_result, gpv_result, self.degree_thres, self.shift_thres
        )
        print(f"Degree CM stat: {degree_cm_stat}, p-value: {degree_cm_p_value}")

    def eval(self):
        """"""
        n_results = len(self.our_method_result)
        for i in range(n_results):
            our_result_p = self.our_method_result[i]
            gpv_result_p = self.gpv_method_result[i]

            print(f"Evaluating {our_result_p} vs {gpv_result_p}")
            self.calc_p_value(our_result_p, gpv_result_p)

    def stat_iou(self, our_result, gpv_result, iou_thres=0.5):
        our_iou_t = []
        gpv_iou_t = []
        for k, v in our_result.items():
            if k not in gpv_result:
                print(f"Key {k} not found in GPV result")
                continue
            our_iou = v["iou"]["iou_3d"].reshape(-1)
            gpv_iou = gpv_result[k]["iou"]["iou_3d"].reshape(-1)

            our_t = our_iou >= iou_thres
            gpv_t = gpv_iou >= iou_thres

            our_iou_t.append(our_t)
            gpv_iou_t.append(gpv_t)
        our_iou_t = np.concatenate(our_iou_t)
        gpv_iou_t = np.concatenate(gpv_iou_t)
        ret = self.calc_McNemar(our_iou_t, gpv_iou_t)
        return ret

    def stat_degree_cm(self, our_result, gpv_result, degree_thres=10, shift_thres=5):
        our_degree_cm_t = []
        gpv_degree_cm_t = []
        for k, v in our_result.items():
            if k not in gpv_result:
                print(f"Key {k} not found in GPV result")
                continue
            our_degree_cm = v["RT"]["degree_cm"]
            gpv_degree_cm = gpv_result[k]["RT"]["degree_cm"]

            our_t = (
                (our_degree_cm[:, 0] <= degree_thres)
                & (our_degree_cm[:, 1] <= shift_thres)
                & (np.all(our_degree_cm > 0, axis=1))
            )
            gpv_t = (
                (gpv_degree_cm[:, 0] <= degree_thres)
                & (gpv_degree_cm[:, 1] <= shift_thres)
                & (np.all(gpv_degree_cm > 0, axis=1))
            )

            if len(our_t) != len(gpv_t):
                continue

            our_degree_cm_t.append(our_t)
            gpv_degree_cm_t.append(gpv_t)
        our_degree_cm_t = np.concatenate(our_degree_cm_t)
        gpv_degree_cm_t = np.concatenate(gpv_degree_cm_t)

        ret = self.calc_McNemar(our_degree_cm_t, gpv_degree_cm_t)
        return ret

    def calc_McNemar(self, our_bool_result, gpv_bool_result):
        """
        McNemar's test for paired nominal data
        """
        from statsmodels.stats.contingency_tables import mcnemar

        # Create a contingency table
        contingency_table = np.zeros((2, 2), dtype=int)

        contingency_table[0, 0] = np.sum(our_bool_result & gpv_bool_result)
        contingency_table[0, 1] = np.sum(our_bool_result & ~gpv_bool_result)
        contingency_table[1, 0] = np.sum(~our_bool_result & gpv_bool_result)
        contingency_table[1, 1] = np.sum(~our_bool_result & ~gpv_bool_result)
        print(contingency_table)

        # Perform McNemar's test
        result = mcnemar(contingency_table, exact=True)
        # 提取不一致的配对数 b 和 c
        b = contingency_table[0, 1]
        c = contingency_table[1, 0]

        p_value_onesided = None
        conclusion = ""

        # 我们的备择假设是 "our" 方法更好，所以我们期望 b > c
        # 1. 首先检查数据趋势是否支持我们的假设
        if b > c:
            # 如果趋势正确，单尾 p-value 是双尾 p-value 的一半
            p_value_onesided = result.pvalue / 2

            # 2. 将单尾 p-value 与显著性水平 alpha 进行比较
            if p_value_onesided < self.alpha:
                conclusion = f"结果显著 (p={p_value_onesided:.4f} < {self.alpha})。证据表明 'our' 方法显著优于 'gpv' 方法。"
            else:
                conclusion = f"结果不显著 (p={p_value_onesided:.4f} >= {self.alpha})。没有足够的证据表明 'our' 方法更好。"
        else:
            # 如果 b <= c，数据本身就不支持 "our" 方法更好这个假设。
            # 在这种情况下，单尾p-value会很大(>=0.5)，所以我们直接得出结论。
            # (严谨的计算是 1 - p_value_twosided / 2, 但结果肯定是 > alpha)
            p_value_onesided = 1 - (result.pvalue / 2)
            conclusion = f"数据趋势不支持假设 (our优于gpv的次数 {b} <= gpv优于our的次数 {c})。不能断定 'our' 方法更好。"

        print(conclusion)

        return result.statistic, p_value_onesided


def eval_main():
    """"""
    ne = NoisyEvaluator(
        cut_ratio=0.15,
        seed=42,
        cut_method="two_plane",
        iou_thres=0.9,
        degree_thres=10,
        shift_thres=2,
    )
    ne.eval()


if __name__ == "__main__":
    # app.run(evaluate)
    # # eval_main()

    # app.run(evaluate)
    eval_main()

    # two_plane, cut_ratio=0.15, iou_thres=0.9, degree_thres=10, shift_thres=2
    # cut_ratio=0.4,seed=2025,cut_method="two_plane",iou_thres=0.7,degree_thres=5,shift_thres=2,
