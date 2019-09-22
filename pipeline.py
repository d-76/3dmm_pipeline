import numpy as np
import os
import argparse
import pix2face_estimation.geometry_utils as geometry_utils
import face3d
import vxl
import pix2face
import json
from PIL import Image

def run_pipeline():
	# Read pipeline type from command line
	parser = argparse.ArgumentParser()
	parser.add_argument('--pipeline', help='sets pipeline')
	args = parser.parse_args()

	# Read json config
	json_file = open('./.pipeline')
	data = json.load(json_file)
	pipeline_config = data["pipelines"][args.pipeline]
	json_file.close()

	# Load pretrained model
	cuda_device = 0
	model = pix2face.test.load_pretrained_model(cuda_device=cuda_device)

	this_dir = os.path.dirname(__file__)
	pvr_data_dir = os.path.join(this_dir, 'data_3DMM/')
	debug_dir = ''
	debug_mode = False
	num_subject_coeffs = 199  # max 199
	num_expression_coeffs = 29  # max 29

	# load needed data files
	head_mesh = face3d.head_mesh(pvr_data_dir)
	subject_components = np.load(os.path.join(pvr_data_dir, 'pca_components_subject.npy'))
	expression_components = np.load(os.path.join(pvr_data_dir, 'pca_components_expression.npy'))
	subject_ranges = np.load(os.path.join(pvr_data_dir,'pca_coeff_ranges_subject.npy'))
	expression_ranges = np.load(os.path.join(pvr_data_dir,'pca_coeff_ranges_expression.npy'))

	# keep only the PCA components that we will be estimating
	subject_components = vxl.vnl.matrix(subject_components[0:num_subject_coeffs,:])
	expression_components = vxl.vnl.matrix(expression_components[0:num_expression_coeffs,:])
	subject_ranges = vxl.vnl.matrix(subject_ranges[0:num_subject_coeffs,:])
	expression_ranges = vxl.vnl.matrix(expression_ranges[0:num_expression_coeffs,:])

	# create rendering object (encapsulates OpenGL context)
	renderer = face3d.mesh_renderer()
	# create coefficient estimator
	coeff_estimator = face3d.media_coefficient_from_PNCC_and_offset_estimator(head_mesh, subject_components, expression_components, subject_ranges, expression_ranges, debug_mode, debug_dir)


	data_dir = pipeline_config["inputDir"]
	output_dir = pipeline_config["outputDir"]
	directories = [x[0] for x in os.walk(data_dir)]
	for directory in directories:
		files = os.listdir(os.path.join(data_dir, directory))
		for file in files:
			file_path, ext_path = os.path.splitext(file)
			img_fname = os.path.join(path, directory, file)
			img = np.array(Image.open(img_fname))
			outputs = pix2face.test.test(model, [img,])
			pncc = outputs[0][0]
			offsets = outputs[0][1]
			pncc_rgb = pncc / 300.0 + 0.5
			offsets_rgb = offsets / 60.0 + 0.5
			pix2face_data = pix2face_estimation.coefficient_estimation.load_pix2face_data()

			# create rendering object (encapsulates OpenGL context)
			renderer = face3d.mesh_renderer()
			
			# create coefficient estimator
			coeff_estimator = face3d.media_coefficient_from_PNCC_and_offset_estimator(head_mesh, subject_components, expression_components, subject_ranges, expression_ranges, debug_mode, debug_dir)
			
			# Estimate Coefficients from PNCC and Offsets
			print('Estimating Coefficients..')
			img_ids = ['img0',]
			coeffs, result = coeff_estimator.estimate_coefficients_perspective(img_ids, [pncc,], [offsets,])

			# Print Yaw, Pitch, Roll of Head
			R_cam = np.array(coeffs.camera(0).rotation.as_matrix())  # rotation matrix of estimated camera
			R0 = np.diag((1,-1,-1))  # R0 is the rotation matrix of a frontal camera
			R_head = np.dot(R0,R_cam)
			yaw, pitch, roll = geometry_utils.matrix_to_Euler_angles(R_head, order='YXZ')
			print('yaw, pitch, roll = %0.1f, %0.1f, %0.1f (degrees)' % (np.rad2deg(yaw), np.rad2deg(pitch), np.rad2deg(roll)))

			# Render 3D-Jittered Images
			print('Rendering Jittered Images..')
			jitterer = face3d.media_jitterer_perspective([img,], coeffs, head_mesh, subject_components, expression_components, renderer, "")

			for entity in pipeline_config["entities"]:
				if entity["type"] == "emote":
					# manually alter expression
					new_expression_coeffs = np.zeros_like(coeffs.expression_coeffs(0))
					parameters = entity["parameters"]
					new_expression_coeffs[0] = parameters["anger"]
					new_expression_coeffs[1] = parameters["disgust"]
					new_expression_coeffs[2] = parameters["fear"]
					new_expression_coeffs[3] = parameters["happiness"]
					new_expression_coeffs[4] = parameters["sadness"]
					new_expression_coeffs[5] = parameters["surprise"]
					render_img = jitterer.render(coeffs.camera(0), coeffs.subject_coeffs(), new_expression_coeffs, subject_components, expression_components)
				elif entity["type"] == "pose":
					# manually alter pose
					delta_R = vxl.vgl.rotation_3d(geometry_utils.Euler_angles_to_quaternion(np.pi/3, 0, 0, order='YXZ'))
					cam = coeffs.camera(0)
					new_R = cam.rotation * delta_R
					new_cam = face3d.perspective_camera_parameters(cam.focal_len, cam.principal_point, new_R, cam.translation, cam.nx, cam.ny)
					render_img = jitterer.render(new_cam, coeffs.subject_coeffs(), coeffs.expression_coeffs(0), subject_components, expression_components)

				Image.fromarray(render_expr[:,:,0:3]).save(os.path.join(os.path.join(output_dir, directory), file_path + "_" + entity["name"] + "." + ext_path))
			Image.fromarray(img).save(os.path.join(os.path.join(output_dir, directory), file_path + "_original." + ext_path))

if __name__ == "__main__":
	run_pipeline()
