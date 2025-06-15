import numpy as np
import matplotlib.pyplot as plt

# trajectory 불러오기
trajectory = np.load('./camera_trajectory.npy')  # shape: (N, 3)

# # x: 왼/오, z: 전/후 → 지도처럼 표현
x = trajectory[:, 0]
y = trajectory[:, 1]

print("X 변화량:", np.max(trajectory[:, 0]) - np.min(trajectory[:, 0]))
print("Y 변화량:", np.max(trajectory[:, 1]) - np.min(trajectory[:, 1])) 
print("Z 변화량:", np.max(trajectory[:, 2]) - np.min(trajectory[:, 2]))

# 첫 번째와 마지막 포즈 비교
print("시작:", trajectory[0])
print("끝:", trajectory[-1])
print("변화:", trajectory[-1] - trajectory[0])

# 프레임 간 거리 누적
dists = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
model_distance = np.sum(dists)
print("모델 전체 거리:", model_distance)

# # 스케일 보정
# scale = 12.7/ model_distance
# trajectory_scaled = trajectory * scale



# plt.figure(figsize=(8, 6))
# plt.plot(trajectory_scaled[:, 0], trajectory_scaled[:, 2], marker='o', color='red', linewidth=2, markersize=3)
# plt.xlabel('X (left/right)')
# plt.ylabel('Z (forward)')
# plt.title('Camera Trajectory (Top-down View)')
# plt.grid(True)
# plt.axis('equal')

# plt.savefig('camera_real_scale.png', dpi=300)

plt.figure(figsize=(8, 6))
plt.plot(y, x, marker='o', color='red', linewidth=2, markersize=3)
plt.ylabel('X (left/right')
plt.xlabel('Y')
plt.title('Camera Trajectory (Top-down View)')
plt.grid(True)
plt.axis('equal')  # 비율 유지
plt.savefig('camera_trajectory_Y.png', dpi=300)

