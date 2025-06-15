// const poseModelPath = '../assets/export_pose_with_preprocess_js_fp16/model.json'; // 포즈넷 경로 추가
// const poseModel = await tf.loadGraphModel(poseModelPath);

// const depthNodelPath = '../assets/export_depth_with_preprocess_js_fp16/model.json'; // DepthNet 경로 추가
// const depthModel = await tf.loadGraphModel(depthNodelPath);
// console.log('PoseNet 모델 로드 완료', poseModel);
// console.log('DepthNet 모델 로드 완료', depthModel);
try {
  const poseModel = await tf.loadGraphModel('../assets/export_pose_with_preprocess_js_fp16/model.json');
  console.log('PoseNet 모델 로드 완료', poseModel);
} catch (err) {
  console.error('PoseNet 모델 로드 실패:', err);
}

try {
  const depthModel = await tf.loadGraphModel('../assets/export_depth_with_preprocess_js_fp16/model.json');
  console.log('DepthNet 모델 로드 완료', depthModel);
} catch (err) {
  console.error('DepthNet 모델 로드 실패:', err);
}
// const video = document.getElementById('video');
// const canvas = document.getElementById('canvas');

// const predictCanvas = document.getElementById('predictCanvas');
// const renderCanvas = document.getElementById('renderCanvas');
// const ctx = canvas.getContext('2d');
// const predictCtx = predictCanvas.getContext('2d');
// const renderCtx = renderCanvas.getContext('2d');

// const depthModelPath = '../assets/tfjs/depthnet/model.json';

// let depthModel = null;
// let poseModel = null;

// // 웹캠 스트림 시작
// navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
//   video.srcObject = stream;
//   // video.play();
// });

// async function loadModels() {
//   try {
//     depthModel = await tf.loadLayersModel(depthModelPath);
//     console.log('DepthNet 모델 로드 완료');
//   } catch (e) {
//     console.error('DepthNet 모델 로드 실패:', e);
//   }
//   try {
//     poseModel = await tf.loadGraphModel(poseModelPath);
//     console.log('PoseNet 모델 로드 완료');
//   } catch (e) {
//     console.error('PoseNet 모델 로드 실패:', e);
//   }
// }
// // loadModels();

// // 실시간 추론 루프
// async function processFrame() {
//   if (!depthModel || !poseModel) return requestAnimationFrame(processFrame);

//   // 1. 웹캠 프레임을 canvas에 복사
//   ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

//   // 2. canvas 이미지를 텐서로 변환
//   let inputTensor = tf.browser.fromPixels(canvas).expandDims(0).toFloat();

//   // 3. DepthNet 추론
//   let output = await depthModel.executeAsync(inputTensor);
//   let depthMap = output;
//   if (Array.isArray(output)) depthMap = output[0];
//   depthMap = depthMap.squeeze();
//   await tf.browser.toPixels(depthMap, predictCanvas);

//   // 4. PoseNet 추론
//   let poseOutput = await poseModel.executeAsync(inputTensor);
//   // 예시: poseOutput에서 keypoints 추출 및 시각화 (모델 구조에 따라 다름)
//   // 아래는 일반적인 heatmap 기반 keypoint 예시
//   let keypoints = poseOutput; // 실제 모델 구조에 맞게 후처리 필요
//   if (Array.isArray(poseOutput)) keypoints = poseOutput[0];
//   keypoints = keypoints.squeeze();

//   // renderCanvas에 원본 프레임 그리기
//   renderCtx.drawImage(video, 0, 0, renderCanvas.width, renderCanvas.height);

//   // keypoints 시각화 (예시: keypoints shape이 [num_keypoints, 2]일 때)
//   const keypointsData = await keypoints.array();
//   renderCtx.fillStyle = 'red';
//   keypointsData.forEach(([y, x]) => {
//     renderCtx.beginPath();
//     renderCtx.arc(x, y, 5, 0, 2 * Math.PI);
//     renderCtx.fill();
//   });

//   // 메모리 해제
//   tf.dispose([inputTensor, output, poseOutput, keypoints]);

//   requestAnimationFrame(processFrame);
// }

// // canvas 크기 설정 (웹캠 해상도에 맞게)
// video.addEventListener('loadedmetadata', () => {
//   canvas.width = video.videoWidth;
//   canvas.height = video.videoHeight;
//   predictCanvas.width = video.videoWidth;
//   predictCanvas.height = video.videoHeight;
//   renderCanvas.width = video.videoWidth;
//   renderCanvas.height = video.videoHeight;
//   processFrame();
// });