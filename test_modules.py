# =======================================
# Uncomment and run to tests superresolution model.

import cv2
import numpy as np
from modules.reconstruction import superresolution
import os
image_path = 'test_images/noha1.PNG'
save_dir = "check_folder"
model_path = 'utils/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth'
scale = 4
window_size = 8

# read image
(imgname, imgext) = os.path.splitext(os.path.basename(image_path))
img_lq = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.

img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
output = superresolution(img_lq)
cv2.imwrite(f'{save_dir}/{imgname}_SwinIR.png', output)
# ======================================
