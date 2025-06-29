import cv2
import os
import numpy as np
from MonoVO import MonoVO

DEBUG = True
PER_FRAME_ERROR = True

from vo.utils.visualization import Visualizer

def offline_vo(cap, save_path):
	"""Run D3VO on offline video"""
	intrinsic = np.array([[1.79214807e+03, 0.00000000e+00, 5.31603784e+02],
                                [0.00000000e+00, 1.79119403e+03, 9.72880606e+02],
                                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
	monoVO = MonoVO(intrinsic)

	# Run D3VO offline with prerecorded video
	i = 0
	while cap.isOpened():
		ret, frame = cap.read()
		if ret == True:				
			frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
			frame = cv2.resize(frame, (W, H))
			print("\n*** frame %d/%d ***" % (i, CNT))
			monoVO.process_frame(frame)

			if i >= 1:
				prev_rel_pose = monoVO.mp.frames[-2].pose
				current_rel_pose = np.linalg.inv(monoVO.mp.frames[-1].pose)
				current_global_pose = np.dot(prev_rel_pose, current_rel_pose)
				print("Current global pose: ", current_global_pose)

				poses = monoVO.mp.relative_to_global()

				visualizer.world_pose = poses[-1]

				visualizer.draw_trajectory(visualizer.world_pose, color="red", line_width=2)

				visualizer.draw_camera_model(visualizer.world_pose, scale=0.5, name_prefix="camera")			
				
				visualizer.render()


				
		else:
			break
		i += 1

		if DEBUG:
			cv2.imshow('d3vo', frame)
			if cv2.waitKey(1) == 27:     # Stop if ESC is pressed
				break	
	

	# Store pose predictions to a file (do not save identity pose of first frame)
	save_path = os.path.join(save_path)
	np.save(save_path, monoVO.mp.relative_to_global())
	print("-> Predictions saved to", save_path)


if __name__ == "__main__":
	video_path = '/home/tspxr-test/Deep-Visual-SLAM-main/test_dataset/sequence_01.MOV'
	cap = cv2.VideoCapture(video_path)
	W = 640
	H = 480
	CNT = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

	visualizer = Visualizer(draw_plane=True, is_record=True, video_fps=24, video_name="visualization_opti.mp4")

	# run offline visual odometry on provided video
	offline_vo(cap, './output_pose.npy')
	visualizer.close()