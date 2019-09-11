import numpy as np
import os
import argparse
import pix2face_estimation.geometry_utils as geometry_utils
import face3d
import vxl
import pix2face
from PIL import Image

def run_pipeline():
	parser = argparse.ArgumentParser()
	arser.add_argument('--pipeline', help='sets pipeline')
	args = parser.parse_args()

	json_file = open('./.pipeline')
	data = json.load(json_file)
	pipeline_config = data["pipelines"][args.pipeline]
	json_file.close()

	cuda_device = 0
	pix2face_data_dir = pipeline_config["inputDir"]
	model = pix2face.test.load_pretrained_model(cuda_device=cuda_device)

	


if __name__ == "__main__":
	print("hello")