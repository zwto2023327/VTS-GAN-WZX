import os
import glob
import argparse
import cv2
import dlib
from tqdm import tqdm
from imutils import face_utils
from PIL import Image
import imageio
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--Data_dir', type=str, default='/data/wzx/GRID/VTS-GAN-OUTPUT')
    parser.add_argument("--Output_dir", type=str, default='/data/wzx/GRID/VTS-GAN-OUTPUT-MARK')
    parser.add_argument("--Predictor", type=str, default='/data/wzx/VTS-GAN-WZX/preprocess/shape_predictor_68_face_landmarks.dat')
    args = parser.parse_args()
    return args

args = parse_args()
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args.Predictor)
files = sorted(glob.glob(os.path.join(args.Data_dir, '*', '*', '*', '*.png')))
for lm in tqdm(files):
    img = cv2.imread(lm)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        p = shape[0]
        x_min = p[0]
        x_max = p[0]
        y_min = p[1]
        y_max = p[1]
        for (x, y) in shape:
            img[y, x] = (255, 255, 255)
            x_min = min(x, x_min)
            x_max = max(x, x_max)
            y_min = min(y, y_min)
            y_max = max(y, y_max)

        height, width = img.shape[:2]
        for iy in range(height):
            for ix in range(width):
                b, g, r = img[iy, ix]
                if (iy >= y_min and iy <= y_max) and (ix >= x_min and ix <= x_max):
                    continue
                img[iy, ix] = (0, 0, 0)
    t, f_name = os.path.split(lm)
    t, m_name = os.path.split(t)
    t, s_name = os.path.split(t)
    _, u_name = os.path.split(t)
    save_path = os.path.join(args.Output_dir, u_name, s_name, m_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    new_im = Image.fromarray(img)
    imageio.imsave(os.path.join(save_path, f_name), new_im)