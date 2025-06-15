// 실시간 웹캠 depth map (TensorFlow.js 공식 모델 사용)
async function setupWebcam() {
    const video = document.getElementById('video');
    // 웹캠 스트림 요청
    const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 320, height: 240, facingMode: 'environment' }
    });
    video.srcObject = stream;
    return new Promise(resolve => {
        video.onloadedmetadata = () => {
            video.play();
            resolve(video);
        };
    });
}

async function main() {
    await setupWebcam();

    // 공식 depth-estimation 모델 로드 (MobileNetV3, 빠르고 가벼움)
    const model = await depthEstimation.createEstimator('mobilenetv3');

    const video = document.getElementById('video');
    const depthCanvas = document.getElementById('depthCanvas');
    const ctx = depthCanvas.getContext('2d');

    async function renderDepth() {
        if (video.paused || video.ended) {
            requestAnimationFrame(renderDepth);
            return;
        }

        // 추론 (입력: video, 출력: tf.Tensor [H, W, 1])
        const depth = await model.estimateDepth(video);

        // 0~1로 정규화
        const depthData = await depth.data();
        const [h, w] = depth.shape;
        let min = Infinity, max = -Infinity;
        for (let i = 0; i < depthData.length; i++) {
            if (depthData[i] < min) min = depthData[i];
            if (depthData[i] > max) max = depthData[i];
        }
        const range = max - min;

        // Canvas에 그레이스케일로 시각화
        const imageData = ctx.createImageData(w, h);
        for (let i = 0; i < depthData.length; i++) {
            // 가까울수록 밝게(흰색), 멀수록 어둡게(검정)
            const v = Math.floor(((depthData[i] - min) / range) * 255);
            imageData.data[i * 4 + 0] = v;
            imageData.data[i * 4 + 1] = v;
            imageData.data[i * 4 + 2] = v;
            imageData.data[i * 4 + 3] = 255;
        }
        ctx.putImageData(imageData, 0, 0);

        depth.dispose();
        requestAnimationFrame(renderDepth);
    }

    renderDepth();
}

window.addEventListener('DOMContentLoaded', main);