let model;
let video;
let canvas;
let ctx;

let prevFrameTensor = null;

// three.js 관련 변수
let threeRenderer, threeScene, threeCamera, axesHelper, pointObject;

function setupThreeJS() {
  const container = document.getElementById('three-container');
  threeRenderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  threeRenderer.setSize(400, 400);
  container.appendChild(threeRenderer.domElement);

  threeScene = new THREE.Scene();
  threeCamera = new THREE.PerspectiveCamera(75, 1, 0.1, 1000);
  threeCamera.position.set(0, 0, 5);

  // 3D 좌표축
  axesHelper = new THREE.AxesHelper(2);
  threeScene.add(axesHelper);

  // 포즈를 반영할 점(구체)
  const geometry = new THREE.SphereGeometry(0.1, 32, 32);
  const material = new THREE.MeshBasicMaterial({ color: 0xff0000 });
  pointObject = new THREE.Mesh(geometry, material);
  threeScene.add(pointObject);

  animateThree();
}

function animateThree() {
  requestAnimationFrame(animateThree);
  threeRenderer.render(threeScene, threeCamera);
}

// 포즈의 translation(x, y, z)만을 사용해 점 위치 이동
function setPoseToObject(pose) {
  // pose: [4][4] 배열
  // translation: pose[0][3], pose[1][3], pose[2][3]
  // 스케일이 너무 작거나 크면 곱셈/나눗셈으로 조정 (예: 1~2배)
  const scale = 1; // 필요시 조정
  pointObject.position.set(
    pose[0][3] * scale,
    pose[1][3] * scale,
    pose[2][3] * scale
  );
}

// 모델 로드 함수
async function loadModel() {
  const MODEL_URL = './assets/export_pose_with_preprocess_js_fp16/model.json';
  model = await tf.loadGraphModel(MODEL_URL);
  console.log('모델 로드 완료');
}

// 웹캠(핸드폰 카메라) 설정 함수
async function setupCamera() {
  video = document.getElementById('video');
  canvas = document.getElementById('canvas');
  ctx = canvas.getContext('2d');

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    video.onloadedmetadata = () => {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      detectPose();
    };
  } catch (err) {
    alert('카메라에 접근할 수 없습니다. 허용해주세요.');
    console.error(err);
  }
}

// 포즈 추정 및 결과 처리 함수
async function detectPose() {
  if (!model) return;

  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  let currFrameTensor = tf.browser.fromPixels(canvas)
    .resizeBilinear([384, 640])
    .toFloat()
    .div(tf.scalar(255));

  let inputTensor;
  if (prevFrameTensor) {
    inputTensor = tf.concat([prevFrameTensor, currFrameTensor], -1).expandDims(0);
  } else {
    inputTensor = tf.concat([currFrameTensor, currFrameTensor], -1).expandDims(0);
  }

  const outputs = model.execute(inputTensor);
  let outputTensor = Array.isArray(outputs) ? outputs[0] : outputs;
  const outputData = await outputTensor.array();
  const pose = outputData;

  // three.js 3D 포즈 translation만 반영 (점 이동)
  setPoseToObject(pose);

  // prevFrameTensor 안전하게 dispose 및 갱신
  if (prevFrameTensor) {
    prevFrameTensor.dispose();
  }
  prevFrameTensor = currFrameTensor.clone();

  // currFrameTensor는 clone 후에만 dispose
  tf.dispose([inputTensor, outputs, outputTensor]);
  currFrameTensor.dispose();

  requestAnimationFrame(detectPose);
}

// 초기화 함수
async function init() {
  await loadModel();
  await setupCamera();
  setupThreeJS();
}

init();