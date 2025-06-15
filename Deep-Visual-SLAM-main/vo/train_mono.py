import os
import tensorflow as tf
import yaml
import numpy as np
import cv2
import datetime
from tqdm import tqdm
import sys


current_dir = os.path.dirname(os.path.abspath(__file__))


if os.path.basename(current_dir) == 'vo':

    project_root = os.path.dirname(current_dir)
else:
    project_root = current_dir

# 프로젝트 루트를 sys.path에 추가
sys.path.append(project_root)


# 모델 및 데이터 로더 임포트
from dataset.mars_logger import MarsLoggerHandler
from monodepth_learner import Learner
from model.depth_net import DispNet
from model.pose_net import PoseNet

def main():

    with open('vo/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # GPU 설정
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        visible_gpus = config.get('Experiment', {}).get('gpus', [0])
        selected_gpus = [gpus[i] for i in visible_gpus if i < len(gpus)]
        tf.config.set_visible_devices(selected_gpus, 'GPU')
        print(f"using GPU: {visible_gpus}")
    
    # 3. 모델 초기화
    batch_size = config['Train']['batch_size']
    image_shape = (config['Train']['img_h'], config['Train']['img_w'])
    
    # DispNet
    depth_net = DispNet(image_shape=image_shape, batch_size=batch_size, prefix='disp_resnet')
    depth_net.build((batch_size, *image_shape, 3))
    # 더미 추론
    _ = depth_net(tf.random.normal((batch_size, *image_shape, 3)))
    
    # PoseNet 
    pose_net = PoseNet(image_shape=image_shape, batch_size=batch_size, prefix='mono_posenet')
    # 모델 빌드
    pose_net.build((batch_size, *image_shape, 6))
    
    # 4. 데이터 로더 초기화
    data_loader = MarsLoggerHandler(config)
    
    # 5. Learner 초기화
    learner = Learner(depth_model=depth_net, pose_model=pose_net, config=config)
    
    # 6. 옵티마이저 설정
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config['Train']['init_lr'],
        beta_1=config['Train']['beta1'],
        weight_decay=config['Train']['weight_decay']
    )
    
    # 7. 메트릭 초기화
    train_total_loss = tf.keras.metrics.Mean(name='train_total_loss')
    train_pixel_loss = tf.keras.metrics.Mean(name='train_pixel_loss')
    train_smooth_loss = tf.keras.metrics.Mean(name='train_smooth_loss')
    val_total_loss = tf.keras.metrics.Mean(name='val_total_loss')
    val_pixel_loss = tf.keras.metrics.Mean(name='val_pixel_loss')
    val_smooth_loss = tf.keras.metrics.Mean(name='val_smooth_loss')
    
    # 8. 디렉토리 설정
    # 모델 저장 디렉토리
    model_dir = os.path.join(config['Directory']['weights'], config['Directory']['exp_name'])
    os.makedirs(model_dir, exist_ok=True)
    
    # 시각화 저장 디렉토리
    vis_dir = os.path.join(config['Directory']['results'], config['Directory']['exp_name'])
    os.makedirs(vis_dir, exist_ok=True)
    
    #(TensorBoard)
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(config['Directory']['log_dir'], config['Directory']['exp_name'], current_time)
    train_log_dir = os.path.join(log_dir, 'train')
    val_log_dir = os.path.join(log_dir, 'validation')
    os.makedirs(train_log_dir, exist_ok=True)
    os.makedirs(val_log_dir, exist_ok=True)
    
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    val_summary_writer = tf.summary.create_file_writer(val_log_dir)
    
    # 9. 학습 루프
    total_train_steps = len(data_loader.train_data) // batch_size
    total_val_steps = len(data_loader.valid_data) // batch_size
    
    print(f"total train step: {total_train_steps}, totel val step: {total_val_steps}")
    print(f"traning... TensorBoard: tensorboard --logdir={log_dir}")
    
    for epoch in range(config['Train']['epoch']):
        print(f"\n epoch {epoch+1}/{config['Train']['epoch']} start")
        
        # 학습률 조정 (선택적)
        if epoch == config['Train']['epoch'] - 1:
            optimizer.learning_rate.assign(config['Train']['final_lr'])
        
        print(f"learning_rate: {optimizer.learning_rate.numpy():.7f}")
        
        # 메트릭 초기화
        train_total_loss.reset_state()
        train_pixel_loss.reset_state()
        train_smooth_loss.reset_state()
        
        # 훈련 루프
        train_progbar = tqdm(range(0, len(data_loader.train_data), batch_size), desc=f"train epoch: {epoch+1}")
        
        for step in train_progbar:
            # 배치 데이터 준비
            batch_end = min(step + batch_size, len(data_loader.train_data))
            batch_size_actual = batch_end - step
            
            if batch_size_actual <= 0:
                continue
            
            # 배치 데이터 로드
            left_imgs, tgt_imgs, right_imgs, intrinsics = [], [], [], []
            
            for i in range(step, batch_end):
                sample = data_loader.train_data[i]
                
                # 이미지 로드 및 전처리
                left_img = cv2.imread(sample['source_left'])
                left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB) / 255.0
                
                tgt_img = cv2.imread(sample['target_image'])
                tgt_img = cv2.cvtColor(tgt_img, cv2.COLOR_BGR2RGB) / 255.0
                
                right_img = cv2.imread(sample['source_right'])
                right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB) / 255.0
                
                # 크기 조정 (필요한 경우)
                target_size = (config['Train']['img_w'], config['Train']['img_h'])
                
                if left_img.shape[:2] != target_size[::-1]:
                    left_img = cv2.resize(left_img, target_size)
                
                if tgt_img.shape[:2] != target_size[::-1]:
                    tgt_img = cv2.resize(tgt_img, target_size)
                    
                if right_img.shape[:2] != target_size[::-1]:
                    right_img = cv2.resize(right_img, target_size)
                
                left_imgs.append(left_img)
                tgt_imgs.append(tgt_img)
                right_imgs.append(right_img)
                intrinsics.append(sample['intrinsic'])

            # 텐서로 변환
            left_imgs = tf.convert_to_tensor(np.array(left_imgs), dtype=tf.float32)
            tgt_imgs = tf.convert_to_tensor(np.array(tgt_imgs), dtype=tf.float32)
            right_imgs = tf.convert_to_tensor(np.array(right_imgs), dtype=tf.float32)
            intrinsics = tf.convert_to_tensor(np.array(intrinsics), dtype=tf.float32)
            
            # 훈련 스텝
            with tf.GradientTape() as tape:
                total_loss, pixel_loss, smooth_loss, pred_depths = learner.forward_step(
                    ref_images=[left_imgs, right_imgs],
                    tgt_image=tgt_imgs,
                    intrinsic=intrinsics,
                    training=True
                )
            
            # 그래디언트 계산 및 적용
            grads = tape.gradient(total_loss, depth_net.trainable_variables + pose_net.trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, clip_norm=5.0)

            optimizer.apply_gradients(zip(grads, depth_net.trainable_variables + pose_net.trainable_variables))
            
            # 메트릭 업데이트
            train_total_loss(total_loss)
            train_pixel_loss(pixel_loss)
            train_smooth_loss(smooth_loss)
            
            # 진행 표시 업데이트
            train_progbar.set_postfix({
                'total_loss': f'{train_total_loss.result():.4f}',
                'pixel_loss': f'{train_pixel_loss.result():.4f}',
                'smooth_loss': f'{train_smooth_loss.result():.4f}'
            })
            
            # 시각화 및 TensorBoard 로깅 (지정된 간격으로)
            if step % config['Train']['train_plot_interval'] == 0:
                global_step = epoch * total_train_steps + (step // batch_size)
                
                # 깊이 맵 시각화
                depth_vis = visualize_depth(tgt_imgs[0].numpy(), pred_depths[0][0].numpy())
                depth_vis_path = os.path.join(vis_dir, f'train_depth_e{epoch+1}_s{step//batch_size}.png')
                cv2.imwrite(depth_vis_path, cv2.cvtColor(depth_vis, cv2.COLOR_RGB2BGR))

                            
                # disparity 값의 분포 통계 계산
                curr_disp = pred_depths[0][0].numpy()
                
                # 통계값 계산
                disp_min = np.min(curr_disp)
                disp_max = np.max(curr_disp)
                disp_mean = np.mean(curr_disp)
                disp_median = np.median(curr_disp)
                disp_std = np.std(curr_disp)
                
                # 백분위수 계산
                percentiles = [1, 5, 25, 50, 75, 95, 99]
                percentile_values = np.percentile(curr_disp, percentiles)
                
                # 통계값 출력
                print(f"\nDisparity result (step {step//batch_size}):")
                print(f"   min: {disp_min:.6f}, max: {disp_max:.6f}")
                print(f"  avg: {disp_mean:.6f}, median: {disp_median:.6f}, std : {disp_std:.6f}")
                print(f"  peprcentile (1, 5, 25, 50, 75, 95, 99): {percentile_values}")
                
                # 히스토그램 계산
                hist, bins = np.histogram(curr_disp.flatten(), bins=50, range=(disp_min, disp_max))
                    
                # TensorBoard에 로깅
                with train_summary_writer.as_default():
                    # 손실 로깅
                    tf.summary.scalar('total_loss', train_total_loss.result(), step=global_step)
                    tf.summary.scalar('pixel_loss', train_pixel_loss.result(), step=global_step)
                    tf.summary.scalar('smooth_loss', train_smooth_loss.result(), step=global_step)
                    
                    # 학습률 로깅
                    tf.summary.scalar('learning_rate', optimizer.learning_rate, step=global_step)
                    
                    # 이미지 , 깊이 결과 
                    depth_vis_tensor = tf.convert_to_tensor(depth_vis[np.newaxis, ...])
                    tf.summary.image('depth_prediction', depth_vis_tensor, step=global_step)
                    tf.summary.scalar('disp_min', disp_min, step=global_step)
                    tf.summary.scalar('disp_max', disp_max, step=global_step)
                    tf.summary.scalar('disp_mean', disp_mean, step=global_step)
                    tf.summary.scalar('disp_median', disp_median, step=global_step)
                    tf.summary.scalar('disp_std', disp_std, step=global_step)
                      # 백분위수 로깅
                    for p, v in zip(percentiles, percentile_values):
                        tf.summary.scalar(f'disp_percentile_{p}', v, step=global_step)
                    
                    # 히스토그램 로깅
                    tf.summary.histogram('disparity_distribution', curr_disp, step=global_step)
                    
                    # 이미지 로깅 (기존 코드)
                    depth_vis_tensor = tf.convert_to_tensor(depth_vis[np.newaxis, ...])
                    tf.summary.image('depth_prediction', depth_vis_tensor, step=global_step)
                    
        
        # TensorBoard에 최종 손실 로깅
        with train_summary_writer.as_default():
            tf.summary.scalar('epoch_total_loss', train_total_loss.result(), step=epoch)
            tf.summary.scalar('epoch_pixel_loss', train_pixel_loss.result(), step=epoch)
            tf.summary.scalar('epoch_smooth_loss', train_smooth_loss.result(), step=epoch)
        
        print(f"epoch {epoch+1} train complete: avg loss = {train_total_loss.result():.4f}")
        
        # 검증 루프
        if len(data_loader.valid_data) > 0:
            # 메트릭 초기화
            val_total_loss.reset_state()
            val_pixel_loss.reset_state()
            val_smooth_loss.reset_state()
            
            val_progbar = tqdm(range(0, len(data_loader.valid_data), batch_size), desc="검증")
            
            for step in val_progbar:
                # 배치 데이터 준비 (훈련과 유사)
                batch_end = min(step + batch_size, len(data_loader.valid_data))
                batch_size_actual = batch_end - step
                
                if batch_size_actual <= 0:
                    continue
                
                # 배치 데이터 로드
                left_imgs, tgt_imgs, right_imgs, intrinsics = [], [], [], []
                
                for i in range(step, batch_end):
                    sample = data_loader.valid_data[i]
                    
                    # 이미지 로드 및 전처리
                    left_img = cv2.imread(sample['source_left'])
                    if left_img is None:
                        continue
                    left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB) / 255.0
                    
                    tgt_img = cv2.imread(sample['target_image'])
                    if tgt_img is None:
                        continue
                    tgt_img = cv2.cvtColor(tgt_img, cv2.COLOR_BGR2RGB) / 255.0
                    
                    right_img = cv2.imread(sample['source_right'])
                    if right_img is None:
                        continue
                    right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB) / 255.0
                    

                    target_size = (config['Train']['img_w'], config['Train']['img_h'])
                    
                    if left_img.shape[:2] != target_size[::-1]:
                        left_img = cv2.resize(left_img, target_size)
                    
                    if tgt_img.shape[:2] != target_size[::-1]:
                        tgt_img = cv2.resize(tgt_img, target_size)
                        
                    if right_img.shape[:2] != target_size[::-1]:
                        right_img = cv2.resize(right_img, target_size)
                    
                    left_imgs.append(left_img)
                    tgt_imgs.append(tgt_img)
                    right_imgs.append(right_img)
                    intrinsics.append(sample['intrinsic'])
                
                if not left_imgs:  # 이미지 로드 실패
                    continue
                
                # 텐서로 변환
                left_imgs = tf.convert_to_tensor(np.array(left_imgs), dtype=tf.float32)
                tgt_imgs = tf.convert_to_tensor(np.array(tgt_imgs), dtype=tf.float32)
                right_imgs = tf.convert_to_tensor(np.array(right_imgs), dtype=tf.float32)
                intrinsics = tf.convert_to_tensor(np.array(intrinsics), dtype=tf.float32)
                
                # 검증 단계 (그래디언트 없음)
                total_loss, pixel_loss, smooth_loss, pred_depths = learner.forward_step(
                    ref_images=[left_imgs, right_imgs],
                    tgt_image=tgt_imgs,
                    intrinsic=intrinsics,
                    training=False
                )
                
                val_total_loss(total_loss)
                val_pixel_loss(pixel_loss)
                val_smooth_loss(smooth_loss)
                
                # 진행 표시 업데이트
                val_progbar.set_postfix({
                    'total_loss': f'{val_total_loss.result():.4f}',
                    'pixel_loss': f'{val_pixel_loss.result():.4f}',
                    'smooth_loss': f'{val_smooth_loss.result():.4f}'
                })
                
                # step마다 로깅깅
                if step % config['Train']['valid_plot_interval'] == 0:
                    global_step = epoch * total_val_steps + (step // batch_size)
                    
                    # 깊이 맵 시각화
                    depth_vis = visualize_depth(tgt_imgs[0].numpy(), pred_depths[0][0].numpy())
                    depth_vis_path = os.path.join(vis_dir, f'val_depth_e{epoch+1}_s{step//batch_size}.png')
                    cv2.imwrite(depth_vis_path, cv2.cvtColor(depth_vis, cv2.COLOR_RGB2BGR))
                    
                    # TensorBoard에 로깅
                    with val_summary_writer.as_default():
                        depth_vis_tensor = tf.convert_to_tensor(depth_vis[np.newaxis, ...])
                        tf.summary.image('depth_prediction', depth_vis_tensor, step=global_step)
            
            #최종 검증 손실 로깅
            with val_summary_writer.as_default():
                tf.summary.scalar('epoch_total_loss', val_total_loss.result(), step=epoch)
                tf.summary.scalar('epoch_pixel_loss', val_pixel_loss.result(), step=epoch)
                tf.summary.scalar('epoch_smooth_loss', val_smooth_loss.result(), step=epoch)
            
            print(f"epoch {epoch+1} val complate: avg loss = {val_total_loss.result():.4f}")
        
        # save
        if (epoch + 1) % config['Train']['save_freq'] == 0 or epoch == config['Train']['epoch'] - 1:
            
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            save_log_dir = os.path.join(model_dir, f"run_{current_time}")
            os.makedirs(save_log_dir, exist_ok=True)

            depth_weights_path = os.path.join(save_log_dir, f'depth_net_epoch_{epoch+1}.weights.h5')
            pose_weights_path = os.path.join(save_log_dir, f'pose_net_epoch_{epoch+1}.weights.h5')

            print(f"Saving model...: {depth_weights_path}, {pose_weights_path}")


            # 가중치만 저장
            depth_net.save_weights(depth_weights_path)
            pose_net.save_weights(pose_weights_path)

            # (마지막 체크포인트)
            latest_depth_weights = os.path.join(save_log_dir, 'depth_net_latest.weights.h5')
            latest_pose_weights = os.path.join(save_log_dir, 'pose_net_latest.weights.h5')
            depth_net.save_weights(latest_depth_weights)
            pose_net.save_weights(latest_pose_weights)

            # (구조 + 가중치)
            depth_model_path = os.path.join(save_log_dir, f'depth_model_epoch_{epoch+1}.h5')
            pose_model_path = os.path.join(save_log_dir, f'pose_model_epoch_{epoch+1}.h5')
            
            try:
                tf.keras.models.save_model(depth_net, depth_model_path, save_format='h5')
                tf.keras.models.save_model(pose_net, pose_model_path, save_format='h5')
                print(f"All models saved successfully in {save_log_dir}")
            except Exception as e:
                print(f"Failed to save models (weights saved only): {e}")

    
    print("train complete!")
    print(f"model save path: {model_dir}")
    print(f"TensorBoard log path: {log_dir}")
    print(f"TensorBoard call: tensorboard --logdir={log_dir}")

def visualize_depth(image, depth_map):
    # 이미지를 uint8로 변환
    image = (image * 255).astype(np.uint8)
    
    # 깊이 맵 처리 - 이상치 제거 추가
    depth_map = 1.0 / (depth_map + 1e-6)  # 역수 변환
    
    # 이상치 제거 
    vmin = np.percentile(depth_map, 5)
    vmax = np.percentile(depth_map, 95)
    
    # 값 범위 제한 및 정규화
    depth_map = np.clip(depth_map, vmin, vmax)
    depth_map = (depth_map - vmin) / (vmax - vmin + 1e-6)
    depth_map = (depth_map * 255).astype(np.uint8)
    
    # Jet 컬러맵 적용
    depth_color = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
    
    # 이미지와 같이 시각화화
    return np.hstack([image, depth_color])

if __name__ == "__main__":
    main() 