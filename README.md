# 2025-1 Capstone: Development and Research on AI-based Lightweight Visual-SLAM Algorithm

## Overview
This project aims to implement a lightweight Visual-SLAM system using only a monocular camera, enabling real-time depth and pose estimation even in indoor environments without expensive sensors. Through self-supervised learning, we eliminate the need for labeled data, and integrate the entire pipeline into a web-based inference system for broader accessibility.

## Key Features
- Monodepth2 + PoseNet: Self-supervised depth and pose estimation from monocular RGB sequences.
- D3VO-based Backend Optimization: Enhances global trajectory accuracy using Bundle Adjustment (g2o).
- Web-based Visualization: Users can upload smartphone videos and see SLAM results on the web.
- Custom Enhancements: Data preprocessing pipeline, custom DataLoader, `predict.py` optimization integration, and an interactive web interface have been implemented beyond the original repository.

## Demonstration Videos
- SLAM Inference Result (`slam/main.py`):  
  [View Result on Google Drive](https://drive.google.com/file/d/19z45ElBhBX0xBy3EpUCI0i3cUZ-zqkKT/view?usp=sharing)

- Web-based SLAM Demo (`web/server.js`):  
  [View Web Inference Demo]

## Project Repository
- Original Repository: [https://github.com/chansoopark98/Deep-Visual-SLAM](https://github.com/chansoopark98/Deep-Visual-SLAM)  
  Author: chansoopark98

- This Fork: We extended the original project by:
  - Creating a full data preprocessing and sequence framing pipeline
  - Implementing a new DataLoader for indoor datasets
  - Modifying `predict.py` to support g2o-based optimization
  - Integrating inference with a Node.js backend and frontend for web visualization

## Use Cases
- Indoor navigation and mapping with minimal hardware
- Real-time SLAM for robotics, AR/VR, and mobile devices
- Educational and research applications where sensor budget is limited

## Contributors
- Chansoo Park: Original author
- Team 18 (Sahmyook University, 2025-1 Capstone)
  - Eunji Lee (Team Lead): Data collection, model training 
  - Hyungjun Lee, Jaewon Lee, Hyukgeun Cho: Data preparation, web service implementation  
  - Prof. Sungwan Kim (Supervisor), Mentor: Chansoo Park
