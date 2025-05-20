import pickle
import numpy as np
import cv2


def main():
    p = "data/Real/test/scene_my_copy/0000_label.pkl"
    with open(p, "rb") as f:
        data = pickle.load(f)
    print(data)
    # 'handle_visibility': array([1, 1, 1, 1, 1, 1])
    for k, v in data.items():
        if k == "bboxes":
            data[k] = np.array([[151, 187, 363, 520]], np.int32)
        else:
            data[k] = v[4:5]
    print(data)
    new_p = "data/Real/test/scene_rgbd/0000_label.pkl"
    with open(new_p, "wb") as f:
        pickle.dump(data, f)
    print(data)


def main1():
    p = "data/segmentation_results/REAL275/results_test_scene_my_copy_0000.pkl"
    with open(p, "rb") as f:
        data = pickle.load(f)
    print(data)
    # 'handle_visibility': array([1, 1, 1, 1, 1, 1])
    for k, v in data.items():
        if k == "image_path":
            data[k] = "data/Real/test/scene_rgbd/0000"
        elif k == "pred_bboxes":
            data[k] = np.array(
                [[151, 187, 363, 520]],
                np.int32,
            )
        # elif k == "gt_bboxes":
        #     data[k] = np.array(
        #         [[164, 127, 373, 517]],
        #         np.int32,
        #     )
        elif k == "pred_masks":
            mask_path = "data/Real/test/scene_rgbd/0000_mask.png"
            mask_img = cv2.imread(mask_path)
            mask = mask_img.astype(bool)
            print("dddd", mask.sum())
            data[k] = mask
        elif k == "pred_class_ids":
            data[k] = np.array([5], np.int32)
        else:
            data[k] = v[4:5]
    # print(data)
    new_p = "data/segmentation_results/REAL275/results_test_scene_rgbd_0000.pkl"
    with open(new_p, "wb") as f:
        pickle.dump(data, f)
    # print(data)


main()
main1()
