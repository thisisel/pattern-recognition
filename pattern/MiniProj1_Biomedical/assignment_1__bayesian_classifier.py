# -*- coding: utf-8 -*-
import cv2
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import random
import math
from dataclasses import dataclass
from typing import Tuple, Optional


# %matplotlib inline


"""
1. Use elephant_walking.bmp as an input image and show it.
"""

main_bgr = cv2.imread("elephant_walking.bmp")
main_rgb = cv2.cvtColor(main_bgr, cv2.COLOR_BGR2RGB)

plt.axis("off")
plt.imshow(main_rgb)

"""Auxiliary Structures and Functions"""

def read_samples(sample_paths: list):
    samples_arr = np.empty(len(sample_paths), dtype=object)

    for i, p_path in enumerate(sample_paths):
        pic_bgr = cv2.imread(p_path)

        if pic_bgr is not None:
            pic_rgb = cv2.cvtColor(pic_bgr, cv2.COLOR_BGR2RGB)
            pic_rgb_pixels = pic_rgb.reshape(-1, 3)
            samples_arr[i] = pic_rgb_pixels

    return samples_arr


def stack_all_samples(samples_arr):
    n = len(samples_arr)
    stacked_pixels = np.concatenate([samples_arr[i] for i in range(n)], axis=0)

    return stacked_pixels


@dataclass
class Segment:
    name: str
    grayscale_val: int
    samples: NDArray
    prior: Optional[float] = None
    mean: Optional[NDArray] = None
    cov: Optional[NDArray] = None

    def est_params(self) -> Tuple[NDArray, NDArray]:
        self.mean = np.mean(self.samples, axis=0)
        diff = self.samples - self.mean
        self.cov = np.cov(diff, rowvar=False)


"""2. Use `‘s1.bmp’`, `‘s2.bmp’` and `‘s3.bmp’` to identify sample pixels for class 1
 `w1 = sky`
"""
sky_sample_paths = ["s1.bmp", "s2.bmp", "s3.bmp"]
sky_sample_arr = read_samples(sky_sample_paths)
sky_sample_pixels = stack_all_samples(sky_sample_arr)
sky = Segment(name="sky", grayscale_val=255, samples=sky_sample_pixels)

print(
    f"total number of pixels gathered from {len(sky_sample_paths)} samples: {sky_sample_pixels.shape[0]}"
)

"""3. Use ‘g1.bmp’, ‘g2.bmp’ and ‘g3.bmp’ to identify sample pixels for class 2 (grass). w2 = grass"""

grass_sample_paths = ["g1.bmp", "g2.bmp", "g3.bmp"]
grass_sample_arr = read_samples(grass_sample_paths)
grass_sample_pixels = stack_all_samples(grass_sample_arr)
grass = Segment(name="grass", grayscale_val=125, samples=grass_sample_pixels)

print(
    f"total number of pixels gathered from {len(grass_sample_paths)} samples: {grass_sample_pixels.shape[0]}"
)

"""4. Use `‘e1.bmp’`, `‘e2.bmp’` and `‘e3.bmp’` to identify sample pixels for class 3 (elephant).

> *Hint: Delete white pixels and don’t use these pixels as sample pixels for class 3 w3 = elephant*

"""

el_sample_paths = ["e1.bmp", "e2.bmp", "e3.bmp"]
el_sample_arr = read_samples(el_sample_paths)
el_sample_pixels_unmasked = stack_all_samples(el_sample_arr)
el_sample_pixels = np.delete(
    el_sample_pixels_unmasked, np.where(el_sample_pixels_unmasked == 255), axis=0
)
el = Segment(name="elephant", grayscale_val=0, samples=el_sample_pixels)

print(
    f"total number of pixels gathered from {len(el_sample_paths)} samples before filtering: {el_sample_pixels_unmasked.shape[0]}"
)
dropped_num = el_sample_pixels_unmasked.shape[0] - el_sample_pixels.shape[0]
print(f"{dropped_num} white pixels were dropped")

"""5. Plot in a 3D space (Red-Green-Blue) preceding pixels.

  Do the classes seem separable for these characteristics i.e R, G and B ?
"""

fig_2 = plt.figure(figsize=(8, 8))
ax = fig_2.add_subplot(111, projection="3d")

ax.scatter(
    sky.samples[:, 0],
    sky.samples[:, 1],
    sky.samples[:, 2],
    c="b",
    marker="*",
    label="Sky pixels",
)
ax.scatter(
    grass.samples[:, 0],
    grass.samples[:, 1],
    grass.samples[:, 2],
    c="g",
    marker="+",
    label="Grass pixels",
)
ax.scatter(
    el_sample_pixels[:, 0],
    el_sample_pixels[:, 1],
    el_sample_pixels[:, 2],
    c="r",
    marker="s",
    label="Elephant pixels",
)

ax.set_xlabel("R", labelpad=20)
ax.set_ylabel("G", labelpad=20)
ax.set_zlabel("B", labelpad=20)
plt.subplots_adjust(left=0.1, right=0.8, top=0.8, bottom=0.2)
# ax.set_box_aspect([2, 2, 2])  # Aspect ratio is 1:1:1

ax.legend()
plt.tight_layout()
plt.show()

"""6. Estimate parameters of probabilities 𝑝(𝑥|𝑤𝑖 ) for a **Gaussian model** starting from these samples.

"""

sky.est_params()
grass.est_params()
el.est_params()

"""7. Use ‘gt1.bmp’ as a ground-truth image.
"""

gt_bgr = cv2.imread("gt1.bmp")
gt_rgb = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB)

plt.axis("off")
plt.imshow(gt_rgb)

"""8. Estimate the prior probabilities 𝑝(𝑤𝑖 ) in the following way:
* Generate by chance the coordinates of (x, y) of 1000 pixels of the image
*  Count the number of pixels belonging to each class by using ground-truth image and divide by 1000.
"""

h = gt_rgb.shape[0]
w = gt_rgb.shape[1]
n_points = 1000

x_list = [random.randint(0, w - 1) for _ in range(n_points)]
y_list = [random.randint(0, h - 1) for _ in range(n_points)]

r_count = 0
g_count = 0
b_count = 0

r_vect = np.array([255, 0, 0])
g_vect = np.array([0, 255, 0])

for y, x in zip(y_list, x_list):
    if all(gt_rgb[y, x, :] == r_vect):
        r_count += 1
    elif all(gt_rgb[y, x, :] == g_vect):
        g_count += 1
    else:
        b_count += 1

sky.prior = b_count / n_points
grass.prior = g_count / n_points
el.prior = b_count / n_points

print(
    f"P(El)= {r_count/n_points} \nP(Grass) = {g_count/n_points} \nP(Sky) = {b_count/n_points}"
)
print(f"P(S) = {(r_count + g_count + b_count)/n_points}")

"""9. Use a Bayesian classifier to automatically segment the image in 3 areas (black, gray and white) corresponding to the classes."""


def discriminant_f(  # noqa: F811
    data_point: NDArray,
    segment: Segment,
) -> float:  # noqa:
    segment.mean.shape[0]
    d = segment.mean.shape[0]
    inv_cov = np.linalg.inv(segment.cov)
    det_cov = np.linalg.det(segment.cov)
    diff = data_point - segment.mean

    term_1 = (d / 2) * math.log(2 * np.pi, 2.7183)
    term_2 = 0.5 * math.log(det_cov, 2.7183)
    term_3 = 0.5 * np.dot(diff.T, np.dot(inv_cov, diff))
    term_4 = math.log(segment.prior, 2.7183)

    g_x = -(term_1 + term_2 + term_3) + term_4

    return g_x


h = main_rgb.shape[0]
w = main_rgb.shape[1]

x_list = [x for x in range(w)]
y_list = [y for y in range(h)]

sky_c = 0
el_c = 0
g_c = 0
grayscale_img = np.zeros((h, w), dtype=np.uint8)

for y in y_list:
    for x in x_list:
        p = main_rgb[y, x, :]

        sky_chance = discriminant_f(data_point=p, segment=sky)
        grass_chance = discriminant_f(data_point=p, segment=grass)
        el_chance = discriminant_f(data_point=p, segment=el)

        grayscale_map = {
            el_chance: el.grayscale_val,
            grass_chance: grass.grayscale_val,
            sky_chance: sky.grayscale_val,
        }

        winner = grayscale_map.get(max(grayscale_map.keys()))
        grayscale_img[y, x] = winner

        if winner == 0:
            el_c += 1
        elif winner == 125:
            g_c += 1

        else:
            sky_c += 1


print(f"sky {sky_c}")
print(f"grass {g_c}")
print(f"el {el_c}")

plt.imshow(grayscale_img)
plt.axis("off")  # Hide axes
plt.show()

"""10. What is the total error of classification for all pixels in the image? Using ground-truth image to check whether pixels are well classified or not."""
