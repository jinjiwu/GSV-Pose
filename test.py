import matplotlib

# matplotlib.use("TkAgg")  # 在代码开头设置
import matplotlib.pyplot as plt
import cv2
import numpy as np

fig = plt.figure()

img = cv2.imread("img/real_test_scene_rgbd_0000_pred.png")
# img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

pts = np.array([[187, 151], [520, 363]])
rrcc = np.array([[122, 68], [522, 468]])

plt.imshow(img)
plt.plot(pts[..., 0], pts[..., 1], "ro")
plt.plot(rrcc[..., 0], rrcc[..., 1], "bo")
plt.show()

# p = "our_rgbd/original/rgb_2_mask.png"
# img = cv2.imread(p)
# mask = 255 - img
# mask_bool = mask > 0
# cv2.imwrite(p, mask)
