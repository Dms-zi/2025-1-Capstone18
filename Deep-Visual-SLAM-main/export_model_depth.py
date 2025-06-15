import tensorflow as tf
from tensorflow import keras as tf_keras
from model.depth_net import DispNet
import os
import tensorflowjs as tfjs
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--img_h', type=int, default=384, help='Image height')
parser.add_argument('--img_w', type=int, default=640, help='Image width')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for the model')
parser.add_argument('--pretrained_dir', type=str, default='./weights/vo/mode=axisAngle_res=384_640_ep=30_bs=16_initLR=1e-05_endLR=1e-05_prefix=Monodepth2-resnet18-Posenet-mars-depthScale/depth_net_epoch_28_model.weights.h5')
parser.add_argument('--saved_model_dir', type=str, default='./assets/saved_models_depthnet/', help='Output directory for the model')
parser.add_argument('--tfjs_dir', type=str, default='./assets/tfjs_models/depthnet/', help='Output directory for the tfjs model')
parser.add_argument('--output_depth', action='store_true', help='Convert disparity to depth using inverse')
args = parser.parse_args()

'''
2025.06.09
--output_depth True하여 export하였습니다. (Disparity -> Depth)
'''

class DepthExportWrapper(tf_keras.Model):
    def __init__(self, model: tf_keras.Model, image_shape, output_depth=False):
        super(DepthExportWrapper, self).__init__()
        self.model = model
        # rename model 
        self.model._name = 'export_depth_model'

        # rename model layers
        for layer in self.model.layers:
            layer._name = f"export_depth_{layer.name}"

        self.image_shape = image_shape
        self.output_depth = output_depth

    def preprocess(self, image):
        """
        image (tf.Tensor): 입력 이미지 [1, H, W, 3].
        """
        # train.py- normalize_image
        image = tf.cast(image, tf.float32)
        image = image / 255.0
        return image

    def disparity_to_depth(self, disp):

        # monodepth_learner.py- disp_to_depth

        min_depth = 0.1  # config에서 가져온 걸로
        max_depth = 10.0  
        
        min_disp = 1.0 / max_depth
        max_disp = 1.0 / min_depth
        scaled_disp = tf.cast(min_disp, tf.float32) + tf.cast(max_disp - min_disp, tf.float32) * disp
        depth = tf.cast(1.0, tf.float32) / scaled_disp
        
        return depth

    # @tf.function(jit_compile=True)
    def call(self, inputs, training=False):
        # 전처리
        inputs = self.preprocess(inputs)
        
        # 모델 추론 - DispNet은 disp1, disp2, disp3, disp4를 반환
        disp1, disp2, disp3, disp4 = self.model(inputs, training=training)
        
        # 고해상도 모델만 사용용
        output = disp1
        
        # outputdepth 옵션이 켜져 있으면 disparity를 depth로 변환
        if self.output_depth:
            output = self.disparity_to_depth(output)
        
        # 배치 차원 제거 (1, H, W, 1) -> (H, W, 1)
        output = tf.squeeze(output, axis=0)
        
        return output
    
    def get_config(self):
        config = super(DepthExportWrapper, self).get_config()
        config.update({
            'image_shape': self.image_shape,
        })
        return config

if __name__ == '__main__':
    os.makedirs(args.saved_model_dir, exist_ok=True)
    os.makedirs(args.tfjs_dir, exist_ok=True)
    
    image_shape = (args.img_h, args.img_w)
    
    # DispNet 모델 생성 (export용이므로 pretrained=False 해야함함)
    base_model = DispNet(image_shape=image_shape, batch_size=1)
    dispnet_input_shape = (1, *image_shape, 3)
    base_model.build(dispnet_input_shape)
    
    # 가중치 로드
    base_model.load_weights(args.pretrained_dir)
    
    # Export wrapper로 감싸기
    wrapped_model = DepthExportWrapper(model=base_model, image_shape=image_shape, output_depth=args.output_depth)
    wrapped_model.build(input_shape=dispnet_input_shape)
    
    # 테스트 실행
    test_input = tf.random.normal((1, *image_shape, 3)) * 255.0  # 0-255 범위
    outputs = wrapped_model(test_input)
    print(f"Output shape: {outputs.shape}")
    print(f"Output min/max: {tf.reduce_min(outputs):.4f} / {tf.reduce_max(outputs):.4f}")

    # 모든 레이어 이름 변경
    for layer in wrapped_model.layers:
        layer._name = f"a_export_depth_{layer.name}"

    # SavedModel 형식으로 저장
    tf.saved_model.save(wrapped_model, args.saved_model_dir)
    print(f"Model saved to {args.saved_model_dir}")

    # TensorFlow.js 변환 (주석 처리 - 수동으로 실행)
    # tfjs.converters.convert_tf_saved_model(args.saved_model_dir,
    #                                     args.tfjs_dir,
    #                                     quantization_dtype_map=tfjs.quantization.QUANTIZATION_DTYPE_FLOAT16,
    #                                         control_flow_v2=True, 
    #                                         )

# 사용 예시:
# Disparity 출력: python export_depth_model.py --img_h 384 --img_w 640 --pretrained_dir ./weights/vo/your_experiment/depth_net_epoch_28_model.weights.h5
# Depth 출력(250609 현재 사용): python export_depth_model.py --img_h 384 --img_w 640 --output_depth --pretrained_dir ./weights/vo/your_experiment/depth_net_epoch_28_model.weights.h5

# TensorFlow.js 변환 명령어:
# tensorflowjs_converter --input_format tf_saved_model --output_format tfjs_graph_model --quantize_float16 --control_flow_v2 True ./assets/saved_models/depthnet ./assets/tfjs_models/depthnet