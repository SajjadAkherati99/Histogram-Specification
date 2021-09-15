import cv2
import numpy as np
from matplotlib import pyplot as plt


# def hist_match1(source, target):
#     shape = np.array(source.shape)
#     out = np.zeros(shape)
#     hist = cv2.calcHist([source], [0], None, [256], [0, 256])
#     source_cdf = hist.cumsum()
#     source_cdf = source_cdf / source_cdf.max()
#
#     hist = cv2.calcHist([target], [0], None, [256], [0, 256])
#     target_cdf = hist.cumsum()
#     target_cdf = target_cdf / target_cdf.max()
#
#     for i in range(256):
#         ind = np.where(target_cdf >= source_cdf[i])[0]
#         if len(ind) == 0:
#             break
#         x, y = np.where(source == i)
#         out[x, y] = ind[0]
#
#     return out


def hist_match(source, target):
    old_shape = source.shape
    source = source.ravel()
    target = target.ravel()

    s, ind = np.unique(source, return_inverse='true')

    hist = cv2.calcHist([source], [0], None, [256], [0, 256])
    source_cdf = hist.cumsum()
    source_cdf = source_cdf / source_cdf.max()

    hist = cv2.calcHist([target], [0], None, [256], [0, 256])
    target_cdf = hist.cumsum()
    target_cdf = target_cdf / target_cdf.max()

    interp_t_values = np.interp(source_cdf, target_cdf, range(0, 256))

    return interp_t_values[ind].reshape(old_shape)


img_dark = cv2.imread('Dark.jpg')
img_pink = cv2.imread('Pink.jpg')

img_dark[:, :, 0] = hist_match(img_dark[:, :, 0], img_pink[:, :, 0])
img_dark[:, :, 1] = hist_match(img_dark[:, :, 1], img_pink[:, :, 1])
img_dark[:, :, 2] = hist_match(img_dark[:, :, 2], img_pink[:, :, 2])

hist_b = cv2.calcHist([img_dark], [0], None, [256], [0, 256])
hist_g = cv2.calcHist([img_dark], [1], None, [256], [0, 256])
hist_r = cv2.calcHist([img_dark], [2], None, [256], [0, 256])
fig, axs = plt.subplots(3, 1)
axs[0].plot(hist_b, 'tab:blue')
axs[0].set_title('blue channel histogram')
axs[1].plot(hist_g, 'tab:green')
axs[1].set_title('green channel histogram')
axs[2].plot(hist_r, 'tab:red')
axs[2].set_title('red channel histogram')
plt.savefig("res05.jpg")
plt.show()

cv2.imwrite('res06.jpg', img_dark)
