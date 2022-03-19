"""
Adapted from code: https://github.com/WillBrennan/SemanticSegmentation/blob/master/evaluate_images.py
https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.violinplot.html
"""
import shutil
import os
import argparse
import logging
import pathlib
import functools
import pandas as pd
import cv2
import math
import numpy as np
from skimage import color
import torch
from torchvision import transforms

from semantic_segmentation import models
from semantic_segmentation import load_model
from semantic_segmentation import draw_results

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)

    parser.add_argument('--model-type', type=str, choices=models, required=True)

    parser.add_argument('--threshold', type=float, default=0.5)

    parser.add_argument('--save', action='store_true')
    parser.add_argument('--display', action='store_true')

    return parser.parse_args()

def process_clustering(path_to_cluster_tsv, path_to_reference_tsv=None):
	clusters_pd = pd.read_csv(path_to_cluster_tsv, sep="\t", header=None)
	if path_to_reference_tsv:
		reference_pd = pd.read_csv(path_to_reference_tsv, sep="\t", header=None)
		reference_pd[0] = reference_pd[0].apply(lambda x: x.split("/")[-1].replace(".jpg", ""))
		image_to_reference_caption = reference_pd[[0,2]].set_index(0).T.to_dict()
	else:
		image_to_reference_caption = None
	total_clusters = clusters_pd[1].unique()
	cluster_to_images = {}
	for cluster in total_clusters:
		images = clusters_pd[clusters_pd[1] == cluster]
		image_urls = images[0].apply(lambda x: x.split("/")[-1]).unique()
		cluster_to_images[cluster] = image_urls
	return image_to_reference_caption, cluster_to_images

image_to_reference_caption, cluster_to_images = process_clustering("/Users/yadapruksachatkun/Downloads/yada/mscoco2014_val/clustering_mscoco2014_val_k50_out.tsv")


def find_files(dir_path: pathlib.Path, file_exts):
    assert dir_path.exists()
    assert dir_path.is_dir()

    for file_ext in file_exts:
        yield from dir_path.rglob(f'*{file_ext}')


def _load_image(image_path: pathlib.Path):
    image = cv2.imread(str(image_path))
    assert image is not None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_width = (image.shape[1] // 32) * 32
    image_height = (image.shape[0] // 32) * 32

    image = image[:image_height, :image_width]
    return image


def get_images(image_path, model, model_type, threshold=0.5):
    logging.basicConfig(level=logging.INFO)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f'running inference on {device}')


    model = torch.load(model, map_location=device)
    model = load_model(models[model_type], model)
    model.to(device).eval()

    logging.info(f'evaluating images from {image_path}')
    image_dir = pathlib.Path(image_path)

    fn_image_transform = transforms.Compose(
        [
            transforms.Lambda(lambda image_path: _load_image(image_path)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    ita = []
    for image_file in find_files(image_dir, ['.png', '.jpg', '.jpeg']):
        logging.info(f'segmenting {image_file} with threshold of {threshold}')
        image = fn_image_transform(image_file)

        with torch.no_grad():
            image = image.to(device).unsqueeze(0)
            results = model(image)['out']
            results = torch.sigmoid(results)

            results = results > threshold

        # get only the rgb2lta
        for category, category_image, mask_image in draw_results(image[0], results[0], categories=model.categories):
            mask_image_rgb = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)
            skin_rgb_vals = mask_image_rgb.nonzero()
            skin_vals = list(set(zip(skin_rgb_vals[0], skin_rgb_vals[1])))
            if len(skin_vals) == 0:
                print("continue")
                continue
            lab_vals = color.rgb2lab([mask_image_rgb[x][y] for x,y in skin_vals])
            ita_vals = [ np.arctan((l_val - 50) / float(b_val)) * (180/ math.pi) for l_val, _, b_val in lab_vals]
            print("For image %s" % image_file)
            print("Mean ITA value %s" % np.mean(ita_vals))
            print("Mean ITA value %s" % np.median(ita_vals))
            print("saving images for skin detection to view")
            ita.append(np.mean(ita_vals))
    return ita
cohort_ita_vals = []
breakpoint()
for cluster, images in cluster_to_images.items():
	os.makedirs(f"cluster_{cluster}", exist_ok=True)
	for image in images:
		shutil.copy(f"mscoco2014_val/faces/{image}", f"cluster_{cluster}")
	ita = get_images(f"cluster_{cluster}", "pretrained/model_segmentation_skin_30.pth", "FCNResNet101")
	cohort_ita_vals.append({"cohort": cluster,  "ita": ita})
cohort_ita_vals_pd = pd.DataFrame(cohort_ita_vals)
breakpoint()
