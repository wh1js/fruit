import os
import cv2
import numpy as np
import argparse
from ultralytics import YOLO, FastSAM

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="FastSAM with YOLOv11 prompts for image or video segmentation")
    parser.add_argument('--input', type=str, default='path/to/input', help='Path to input image, folder, or video')
    parser.add_argument('--yolo_model', type=str, default='yolov11n.pt', help='Path to YOLOv11 model')
    parser.add_argument('--sam_model', type=str, default='FastSAM-s.pt', help='Path to FastSAM model')
    parser.add_argument('--conf', type=float, default=0.5, help='YOLOv11 confidence threshold')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size for inference')
    parser.add_argument('--output', type=str, default='output', help='Output directory for masks and visualized results')
    return parser.parse_args()

def process_image(image_path, yolo_model, sam_model, conf, imgsz, output_dir):
    """处理单张图像的分割"""
    try:
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Failed to load image {image_path}")
            return

        # 步骤 1: 使用 YOLOv11 进行对象检测
        yolo_results = yolo_model.predict(image_path, conf=conf, imgsz=imgsz)
        bboxes = yolo_results[0].boxes.xyxy.cpu().numpy().tolist() if yolo_results else []

        # 步骤 2: 使用 YOLOv11 的边界框作为提示，执行 FastSAM 分割
        sam_results = sam_model.predict(source=image_path, bboxes=bboxes, imgsz=imgsz)

        # 步骤 3: 保存分割掩码并可视化
        mask_dir = os.path.join(output_dir, 'masks')
        vis_dir = os.path.join(output_dir, 'visualized')
        os.makedirs(mask_dir, exist_ok=True)
        os.makedirs(vis_dir, exist_ok=True)

        for i, result in enumerate(sam_results):
            masks = result.masks.data.cpu().numpy()
            for j, mask in enumerate(masks):
                # 保存掩码
                mask_path = os.path.join(mask_dir, f'mask_{os.path.basename(image_path).split(".")[0]}_{i}_{j}.png')
                cv2.imwrite(mask_path, mask * 255)

                # 可视化：将掩码叠加到原图
                vis_img = img.copy()
                mask = (mask > 0).astype(np.uint8) * 255
                colored_mask = np.zeros_like(vis_img)
                colored_mask[:, :, 2] = mask  # 红色掩码
                vis_img = cv2.addWeighted(vis_img, 0.8, colored_mask, 0.2, 0)
                vis_path = os.path.join(vis_dir, f'vis_{os.path.basename(image_path).split(".")[0]}_{i}_{j}.png')
                cv2.imwrite(vis_path, vis_img)

        print(f"Processed image: {image_path}")

    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")

def process_video(video_path, yolo_model, sam_model, conf, imgsz, output_dir):
    """处理视频的分割"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Failed to open video {video_path}")
            return

        # 获取视频属性
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 创建视频输出
        video_out_path = os.path.join(output_dir, f'output_{os.path.basename(video_path)}')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_out_path, fourcc, fps, (width, height))

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 保存临时帧以供预测
            temp_frame_path = os.path.join(output_dir, 'temp_frame.jpg')
            cv2.imwrite(temp_frame_path, frame)

            # 使用 YOLOv11 检测
            yolo_results = yolo_model.predict(temp_frame_path, conf=conf, imgsz=imgsz)
            bboxes = yolo_results[0].boxes.xyxy.cpu().numpy().tolist() if yolo_results else []

            # 使用 FastSAM 分割
            sam_results = sam_model.predict(source=temp_frame_path, bboxes=bboxes, imgsz=imgsz)

            # 可视化帧
            for result in sam_results:
                masks = result.masks.data.cpu().numpy()
                for mask in masks:
                    mask = (mask > 0).astype(np.uint8) * 255
                    colored_mask = np.zeros_like(frame)
                    colored_mask[:, :, 2] = mask  # 红色掩码
                    frame = cv2.addWeighted(frame, 0.8, colored_mask, 0.2, 0)

            out.write(frame)
            frame_count += 1
            print(f"Processed frame {frame_count}")

        cap.release()
        out.release()
        os.remove(temp_frame_path)
        print(f"Processed video: {video_path}")

    except Exception as e:
        print(f"Error processing video {video_path}: {str(e)}")

def main():
    """主函数"""
    args = parse_args()

    # 加载模型
    try:
        yolo_model = YOLO(args.yolo_model)
        sam_model = FastSAM(args.sam_model)
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    # 处理输入
    if os.path.isfile(args.input):
        # 处理单张图像或视频
        if args.input.lower().endswith(('.jpg', '.jpeg', '.png')):
            process_image(args.input, yolo_model, sam_model, args.conf, args.imgsz, args.output)
        elif args.input.lower().endswith(('.mp4', '.avi', '.mov')):
            process_video(args.input, yolo_model, sam_model, args.conf, args.imgsz, args.output)
        else:
            print(f"Unsupported file format: {args.input}")
    elif os.path.isdir(args.input):
        # 处理文件夹中的所有图像
        for file in os.listdir(args.input):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(args.input, file)
                process_image(image_path, yolo_model, sam_model, args.conf, args.imgsz, args.output)
    else:
        print(f"Invalid input path: {args.input}")

if __name__ == "__main__":
    main()