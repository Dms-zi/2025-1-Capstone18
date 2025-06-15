const video = document.getElementById('video'); // 웹캠 영상 표시용
const canvas = document.getElementById('resultCanvas'); // 추론 결과 영상 표시용
const ctx = canvas.getContext('2d');

const socket = new WebSocket('ws://localhost:8888');
socket.binaryType = 'arraybuffer';

socket.onmessage = function(event) {
  // 서버에서 받은 JPEG 프레임을 이미지로 변환 후 캔버스에 그림
  const blob = new Blob([event.data], {type: 'image/jpeg'});
  const img = new Image();
  img.onload = function() {
    canvas.width = img.width;
    canvas.height = img.height;
    ctx.drawImage(img, 0, 0, img.width, img.height);
    URL.revokeObjectURL(img.src);
  };
  img.src = URL.createObjectURL(blob);
};

// 주기적으로 웹캠 프레임을 서버로 전송 (예: 30fps)
function sendFrame() {
  const tempCanvas = document.createElement('canvas');
  tempCanvas.width = video.videoWidth;
  tempCanvas.height = video.videoHeight;
  const tempCtx = tempCanvas.getContext('2d');
  tempCtx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);
  tempCanvas.toBlob(blob => {
    if (socket.readyState === WebSocket.OPEN) {
      blob.arrayBuffer().then(buffer => socket.send(buffer));
    }
  }, 'image/jpeg');
}
setInterval(sendFrame, 33); // 약 30fps

// video 태그가 준비되면 자동 시작
video.addEventListener('loadedmetadata', () => {
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
});