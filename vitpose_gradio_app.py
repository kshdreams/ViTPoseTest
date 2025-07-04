# vitpose_gradio_app.py
import gradio as gr
import torch
import cv2
import numpy as np
import tempfile
import os
import json
import pandas as pd
from PIL import Image
from transformers import (
    AutoProcessor,
    RTDetrForObjectDetection,
    VitPoseForPoseEstimation,
)

# 디바이스 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# 로그 저장용 전역 변수
log_messages = []

def add_log(message):
    """로그 메시지를 추가하고 콘솔에도 출력"""
    import datetime
    timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]  # 밀리초까지 표시 (마이크로초 제거)
    log_entry = f"[{timestamp}] {message}"
    log_messages.append(log_entry)
    print(log_entry)
    
    # 로그가 너무 많아지면 오래된 것부터 삭제 (최대 100개 유지)
    if len(log_messages) > 100:
        log_messages.pop(0)

# 전역 변수로 처리된 데이터 저장
processed_video_data = {
    "thumbnails": [],
    "poses_data": [],
    "fps": 30,
    "frame_interval": 5,
    "video_path": None
}

# 모델 변수들 (전역)
person_image_processor = None
person_model = None
image_processor = None
pose_model = None

def load_model(model_name):
    """지정된 모델을 로드하는 함수"""
    global person_image_processor, person_model, image_processor, pose_model
    
    print(f"모델 로딩 중: {model_name}")
    add_log(f"모델 로딩 중: {model_name}")
    
    try:
        # 사람 검출 모델 (한 번만 로드)
        if person_model is None:
            person_image_processor = AutoProcessor.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
            person_model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd_coco_o365", device_map=device)
        
        # 포즈 추정 모델 (동적 변경)
        image_processor = AutoProcessor.from_pretrained(model_name)
        pose_model = VitPoseForPoseEstimation.from_pretrained(model_name, device_map=device)
        
        print(f"모델 로딩 완료: {model_name}")
        add_log(f"모델 로딩 완료: {model_name}")
        
    except Exception as e:
        print(f"모델 로딩 실패: {e}")
        add_log(f"모델 로딩 실패: {e}")
        raise e

# 초기 모델 로딩
print("모델 로딩 중...")
try:
    load_model("yonigozlan/synthpose-vitpose-huge-hf")
    print("모델 로딩 완료!")
except Exception as e:
    print(f"초기 모델 로딩 실패: {e}")
    person_image_processor = None
    person_model = None
    image_processor = None
    pose_model = None

def detect_persons(image):
    """이미지에서 사람을 검출하는 함수"""
    if person_model is None or person_image_processor is None:
        return []
    
    inputs = person_image_processor(images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = person_model(**inputs)
    
    results = person_image_processor.post_process_object_detection(
        outputs, target_sizes=torch.tensor([(image.height, image.width)]), threshold=0.3
    )
    result = results[0]
    
    # Human label은 COCO 데이터셋에서 0 인덱스
    person_boxes = result["boxes"][result["labels"] == 0]
    person_boxes = person_boxes.cpu().numpy()
    
    # VOC (x1, y1, x2, y2)에서 COCO (x1, y1, w, h) 형식으로 변환
    if len(person_boxes) > 0:
        person_boxes[:, 2] = person_boxes[:, 2] - person_boxes[:, 0]
        person_boxes[:, 3] = person_boxes[:, 3] - person_boxes[:, 1]
        
        # 박스 크기를 10% 확장
        for box in person_boxes:
            x, y, w, h = box
            # 중심점 기준으로 10% 확장
            expand_x = w * 0.1
            expand_y = h * 0.1
            box[0] = max(0, x - expand_x)  # x1
            box[1] = max(0, y - expand_y)  # y1
            box[2] = min(image.width - box[0], w + 2 * expand_x)  # w
            box[3] = min(image.height - box[1], h + 2 * expand_y)  # h
    
    return person_boxes

def get_full_image_box(image):
    """전체 이미지를 하나의 박스로 반환하는 함수"""
    width, height = image.size
    # COCO 형식 (x1, y1, w, h)로 전체 이미지 박스 반환
    return np.array([[0, 0, width, height]])

def extract_pose(image, person_boxes):
    """검출된 사람들의 포즈를 추출하는 함수"""
    if pose_model is None or image_processor is None or len(person_boxes) == 0:
        return []
    
    inputs = image_processor(image, boxes=[person_boxes], return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = pose_model(**inputs)
    
    pose_results = image_processor.post_process_pose_estimation(outputs, boxes=[person_boxes], threshold=0.3)
    return pose_results[0]  # 첫 번째 이미지의 결과

import math

def draw_points(image, keypoints, scores, pose_keypoint_color, keypoint_score_threshold, radius, show_keypoint_weight):
    if pose_keypoint_color is not None:
        # 색상 배열이 키포인트보다 짧으면 반복해서 사용
        if len(pose_keypoint_color) < len(keypoints):
            # 색상 배열을 키포인트 개수만큼 반복
            repeat_times = (len(keypoints) // len(pose_keypoint_color)) + 1
            pose_keypoint_color = np.tile(pose_keypoint_color, (repeat_times, 1))[:len(keypoints)]
    
    for kid, (kpt, kpt_score) in enumerate(zip(keypoints, scores)):
        x_coord, y_coord = int(kpt[0]), int(kpt[1])
        if kpt_score > keypoint_score_threshold:
            if pose_keypoint_color is not None and kid < len(pose_keypoint_color):
                color = tuple(int(c) for c in pose_keypoint_color[kid])
            else:
                # 기본 색상 사용
                color = (0, 255, 0)  # 녹색
            
            if show_keypoint_weight:
                cv2.circle(image, (int(x_coord), int(y_coord)), radius, color, -1)
                transparency = max(0, min(1, kpt_score))
                cv2.addWeighted(image, transparency, image, 1 - transparency, 0, dst=image)
            else:
                cv2.circle(image, (int(x_coord), int(y_coord)), radius, color, -1)

def draw_links(image, keypoints, scores, keypoint_edges, link_colors, keypoint_score_threshold, thickness, show_keypoint_weight, stick_width=2):
    height, width, _ = image.shape
    if keypoint_edges is not None and link_colors is not None:
        # 링크 색상 배열이 엣지보다 짧으면 반복해서 사용
        if len(link_colors) < len(keypoint_edges):
            repeat_times = (len(keypoint_edges) // len(link_colors)) + 1
            link_colors = np.tile(link_colors, (repeat_times, 1))[:len(keypoint_edges)]
        
        for sk_id, sk in enumerate(keypoint_edges):
            # 인덱스 범위 체크
            if sk[0] >= len(keypoints) or sk[1] >= len(keypoints):
                continue
                
            x1, y1, score1 = (int(keypoints[sk[0], 0]), int(keypoints[sk[0], 1]), scores[sk[0]])
            x2, y2, score2 = (int(keypoints[sk[1], 0]), int(keypoints[sk[1], 1]), scores[sk[1]])
            if (
                x1 > 0
                and x1 < width
                and y1 > 0
                and y1 < height
                and x2 > 0
                and x2 < width
                and y2 > 0
                and y2 < height
                and score1 > keypoint_score_threshold
                and score2 > keypoint_score_threshold
            ):
                if sk_id < len(link_colors):
                    color = tuple(int(c) for c in link_colors[sk_id])
                else:
                    color = (255, 0, 0)  # 기본 빨간색
                    
                if show_keypoint_weight:
                    X = (x1, x2)
                    Y = (y1, y2)
                    mean_x = np.mean(X)
                    mean_y = np.mean(Y)
                    length = ((Y[0] - Y[1]) ** 2 + (X[0] - X[1]) ** 2) ** 0.5
                    angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
                    polygon = cv2.ellipse2Poly(
                        (int(mean_x), int(mean_y)), (int(length / 2), int(stick_width)), int(angle), 0, 360, 1
                    )
                    cv2.fillConvexPoly(image, polygon, color)
                    transparency = max(0, min(1, 0.5 * (keypoints[sk[0], 2] + keypoints[sk[1], 2])))
                    cv2.addWeighted(image, transparency, image, 1 - transparency, 0, dst=image)
                else:
                    cv2.line(image, (x1, y1), (x2, y2), color, thickness=thickness)

def draw_keypoints_on_image(image, pose_results, person_boxes=None, input_boxes=None):
    """이미지에 키포인트와 사람 검출 박스를 그리는 함수 (Whole Body - 133개 키포인트)"""
    numpy_image = np.array(image)
        
    keypoint_edges = pose_model.config.edges
    
    palette = np.array(
        [
            [255, 128, 0],
            [255, 153, 51],
            [255, 178, 102],
            [230, 230, 0],
            [255, 153, 255],
            [153, 204, 255],
            [255, 102, 255],
            [255, 51, 255],
            [102, 178, 255],
            [51, 153, 255],
            [255, 153, 153],
            [255, 102, 102],
            [255, 51, 51],
            [153, 255, 153],
            [102, 255, 102],
            [51, 255, 51],
            [0, 255, 0],
            [0, 0, 255],
            [255, 0, 0],
            [255, 255, 255],
        ]
    )

    link_colors = palette[[0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16]]
    keypoint_colors = palette[[16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0]+[4]*(52-17)]

    # 사람 검출 박스 그리기 (빨간색)
    if person_boxes is not None and len(person_boxes) > 0:
        for i, box in enumerate(person_boxes):
            x, y, w, h = box
            # 박스 그리기 (빨간색, 두께 2)
            cv2.rectangle(numpy_image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)
            # 박스 번호 표시
            cv2.putText(numpy_image, f"Person {i+1}", (int(x), int(y)-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # 모델 입력 박스 그리기 (파란색, 점선)
    if input_boxes is not None and len(input_boxes) > 0:
        for i, box in enumerate(input_boxes):
            x, y, w, h = box
            # 점선 박스 그리기 (파란색, 두께 1)
            # 점선 효과를 위해 짧은 선들을 그어서 점선처럼 보이게 함
            for j in range(0, int(w), 10):
                # 위쪽 선
                cv2.line(numpy_image, (int(x + j), int(y)), (int(x + min(j + 5, w)), int(y)), (255, 0, 0), 1)
                # 아래쪽 선
                cv2.line(numpy_image, (int(x + j), int(y + h)), (int(x + min(j + 5, w)), int(y + h)), (255, 0, 0), 1)
            for j in range(0, int(h), 10):
                # 왼쪽 선
                cv2.line(numpy_image, (int(x), int(y + j)), (int(x), int(y + min(j + 5, h))), (255, 0, 0), 1)
                # 오른쪽 선
                cv2.line(numpy_image, (int(x + w), int(y + j)), (int(x + w), int(y + min(j + 5, h))), (255, 0, 0), 1)
            
            # 입력 박스 라벨 표시
            cv2.putText(numpy_image, f"Input {i+1}", (int(x), int(y + h + 20)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)

    for pose_result in pose_results:
        scores = np.array(pose_result["scores"])
        keypoints = np.array(pose_result["keypoints"])
        
        # 키포인트 그리기 (참고 코드와 동일한 방식)
        draw_points(numpy_image, keypoints, scores, keypoint_colors, 
                   keypoint_score_threshold=0.3, radius=2, show_keypoint_weight=False)
        
        # 링크 그리기 (참고 코드와 동일한 방식)
        draw_links(numpy_image, keypoints, scores, keypoint_edges, link_colors, 
                  keypoint_score_threshold=0.3, thickness=1, show_keypoint_weight=False)
    
    return numpy_image

def process_video(video_file, frame_interval=1, progress=None):
    """비디오를 처리하여 포즈 데이터를 추출하는 함수"""
    try:
        add_log(f"비디오 파일 열기 시도: {video_file}")
        cap = cv2.VideoCapture(video_file)
        
        if not cap.isOpened():
            raise ValueError(f"비디오 파일을 열 수 없습니다: {video_file}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            raise ValueError("비디오에 프레임이 없습니다")
        
        poses_data = []
        thumbnails = []
        
        frame_idx = 0
        processed_frames = 0
        cached_person_boxes = None  # 사람 검출 결과 캐싱
        previous_box_areas = []  # 이전 박스들의 면적 추적
        detection_interval = 30  # 10프레임마다 사람 검출
        
        add_log(f"비디오 처리 시작... FPS: {fps}, 총 프레임 수: {total_frames}")
        add_log(f"사람 검출 간격: {detection_interval}프레임마다")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp = frame_idx / fps
            
            # 프로그레스 업데이트
            if progress is not None:
                progress_percent = (frame_idx / total_frames)
                progress(progress_percent, desc=f"프레임 {frame_idx}/{total_frames} 처리 중...")
            
            # 프로그레스를 콘솔에 출력
            if frame_idx % 100 == 0:  # 100프레임마다 출력
                progress_percent = (frame_idx / total_frames) * 100
                add_log(f"프레임 {frame_idx}/{total_frames} 처리 중... ({progress_percent:.1f}%)")
            
            if frame_idx % frame_interval == 0:
                try:
                    add_log(f"프레임 {frame_idx} 처리 시작")
                    
                    # 프로그레스 업데이트 - 프레임 처리 시작
                    if progress is not None:
                        progress(progress_percent, desc=f"프레임 {frame_idx} 처리 중...")
                    
                    # OpenCV BGR을 PIL RGB로 변환
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    
                    # 사람 검출 (10프레임마다 갱신)
                    if cached_person_boxes is None or frame_idx % detection_interval == 0:
                        add_log(f"프레임 {frame_idx} 사람 검출 시작")
                        person_boxes = detect_persons(pil_image)
                        if len(person_boxes) > 0:
                            # 이전 박스 면적과 비교하여 조정
                            adjusted_boxes = []
                            current_box_areas = []
                            
                            for i, box in enumerate(person_boxes):
                                current_area = box[2] * box[3]
                                current_box_areas.append(current_area)
                                
                                # 이전 박스 면적이 있으면 비교
                                if i < len(previous_box_areas):
                                    previous_area = previous_box_areas[i]
                                    # 박스 크기는 줄어들지 않도록 설정 (항상 이전 크기 이상 유지)
                                    if current_area < previous_area:
                                        # 이전 크기를 유지하되, 현재 중심점 사용
                                        ratio = (previous_area / current_area) ** 0.5
                                        new_w = int(box[2] * ratio)
                                        new_h = int(box[3] * ratio)
                                        # 중심점 유지하면서 크기 조정
                                        center_x = box[0] + box[2] // 2
                                        center_y = box[1] + box[3] // 2
                                        new_x = max(0, center_x - new_w // 2)
                                        new_y = max(0, center_y - new_h // 2)
                                        # 이미지 경계 체크
                                        new_x = min(new_x, pil_image.width - new_w)
                                        new_y = min(new_y, pil_image.height - new_h)
                                        adjusted_boxes.append([new_x, new_y, new_w, new_h])
                                        add_log(f"프레임 {frame_idx} 박스 {i+1}: 크기 유지 ({current_area:.0f} → {previous_area:.0f})")
                                    else:
                                        # 현재 박스가 더 크면 새로운 크기로 업데이트
                                        adjusted_boxes.append(box)
                                        add_log(f"프레임 {frame_idx} 박스 {i+1}: 크기 증가 ({previous_area:.0f} → {current_area:.0f})")
                                else:
                                    # 새로운 박스는 그대로 사용
                                    adjusted_boxes.append(box)
                                    add_log(f"프레임 {frame_idx} 박스 {i+1}: 새로운 박스 ({current_area:.0f})")
                            
                            # 현재 박스 면적들을 이전 면적으로 저장
                            previous_box_areas = current_box_areas
                            cached_person_boxes = adjusted_boxes
                            add_log(f"프레임 {frame_idx} 사람 검출 완료: {len(adjusted_boxes)}명 (박스 크기 조정됨)")
                        else:
                            # 사람이 검출되지 않으면 전체 이미지 사용
                            person_boxes = get_full_image_box(pil_image)
                            previous_box_areas = []  # 이전 면적 초기화
                            add_log(f"프레임 {frame_idx} 사람 미검출, 전체 이미지 사용")
                    else:
                        # 캐시된 사람 검출 결과 사용
                        person_boxes = cached_person_boxes
                        add_log(f"프레임 {frame_idx} 캐시된 사람 검출 결과 사용: {len(person_boxes)}명")
                    
                    pose_results = extract_pose(pil_image, adjusted_boxes)
                    
                    # 키포인트가 그려진 이미지 생성 (박스 포함)
                    keypoint_image = draw_keypoints_on_image(pil_image, pose_results, person_boxes, adjusted_boxes)
                    
                    # 썸네일 크기로 리사이즈 (비율 유지)
                    h, w = keypoint_image.shape[:2]
                    target_width = 640
                    target_height = int(h * target_width / w)
                    
                    # 높이가 너무 크면 높이 기준으로 조정
                    if target_height > 480:
                        target_height = 480
                        target_width = int(w * target_height / h)
                    
                    keypoint_image = cv2.resize(keypoint_image, (target_width, target_height))
                    thumbnails.append(keypoint_image)
                    
                    # 포즈 데이터 저장
                    frame_data = {
                        "timestamp": round(timestamp, 3),
                        "frame_idx": frame_idx,
                        "persons": []
                    }
                    
                    for person_idx, person_pose in enumerate(pose_results):
                        person_data = {
                            "person_id": person_idx,
                            "keypoints": []
                        }
                        
                        for i, (keypoint, label, score) in enumerate(zip(
                            person_pose["keypoints"], 
                            person_pose["labels"], 
                            person_pose["scores"]
                        )):
                            keypoint_name = pose_model.config.id2label[label.item()] if pose_model else f"keypoint_{i}"
                            x, y = keypoint
                            person_data["keypoints"].append({
                                "id": i,
                                "name": keypoint_name,
                                "x": float(x),
                                "y": float(y),
                                "confidence": float(score)
                            })
                        
                        frame_data["persons"].append(person_data)
                    
                    poses_data.append(frame_data)
                    processed_frames += 1
                    add_log(f"프레임 {frame_idx} 처리 완료")
                    
                    # 콘솔에도 진행상황 출력
                    if processed_frames % 10 == 0:
                        add_log(f"처리된 프레임: {processed_frames}, 전체 진행률: {frame_idx}/{total_frames} ({frame_idx/total_frames*100:.1f}%)")
                
                except Exception as frame_error:
                    add_log(f"프레임 {frame_idx} 처리 중 오류: {frame_error}")
                    import traceback
                    traceback.print_exc()
                    # 프레임 오류가 발생해도 계속 진행
                    pass
            
            frame_idx += 1
        
        cap.release()
        add_log(f"비디오 처리 완료. 총 {processed_frames}개 프레임 처리됨")
        
        return thumbnails, poses_data
        
    except Exception as e:
        add_log(f"process_video 함수에서 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return [], []



def get_logs():
    """현재까지의 로그를 반환"""
    return "\n".join(log_messages)

def clear_logs():
    """로그 초기화"""
    global log_messages
    log_messages = []
    return "로그가 초기화되었습니다."

def get_keypoint_frame_at_time(timestamp):
    """특정 시간대의 키포인트 프레임을 반환"""
    if not processed_video_data["thumbnails"] or timestamp is None:
        return None
    
    # 시간을 프레임 인덱스로 변환
    frame_idx = int(timestamp * processed_video_data["fps"])
    # frame_interval을 고려한 썸네일 인덱스 계산
    frame_interval = processed_video_data.get("frame_interval", 5)  # 기본값 5
    thumbnail_idx = frame_idx // frame_interval
    
    # 범위 체크
    if 0 <= thumbnail_idx < len(processed_video_data["thumbnails"]):
        return processed_video_data["thumbnails"][thumbnail_idx]
    
    return processed_video_data["thumbnails"][-1] if processed_video_data["thumbnails"] else None

def get_pose_data_at_time(timestamp):
    """특정 시간대의 포즈 데이터를 반환"""
    if not processed_video_data["poses_data"] or timestamp is None:
        return None
    
    # 시간을 프레임 인덱스로 변환
    frame_idx = int(timestamp * processed_video_data["fps"])
    # frame_interval을 고려한 데이터 인덱스 계산
    frame_interval = processed_video_data.get("frame_interval", 5)  # 기본값 5
    data_idx = frame_idx // frame_interval
    
    # 범위 체크
    if 0 <= data_idx < len(processed_video_data["poses_data"]):
        return processed_video_data["poses_data"][data_idx]
    
    return processed_video_data["poses_data"][-1] if processed_video_data["poses_data"] else None

def create_skeleton_video(video_path, poses_data, fps, frame_interval=5, progress=None):
    """스켈레톤이 그려진 비디오를 생성하는 함수"""
    try:
        add_log("스켈레톤 비디오 생성 시작")
        
        # 원본 비디오 열기
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"비디오 파일을 열 수 없습니다: {video_path}")
        
        # 비디오 정보 가져오기
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 출력 비디오 파일 설정
        output_path = video_path.replace(".mp4", "_skeleton.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            raise ValueError("출력 비디오 파일을 생성할 수 없습니다")
        
        add_log(f"출력 비디오 설정 완료: {width}x{height}, {fps}fps")
        
        frame_idx = 0
        processed_frames = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 프로그레스 업데이트
            if progress is not None:
                progress_percent = frame_idx / total_frames
                progress(progress_percent, desc=f"스켈레톤 비디오 생성 중... {frame_idx}/{total_frames}")
            
            # frame_interval에 맞는 프레임만 처리
            if frame_idx % frame_interval == 0:
                try:
                    # OpenCV BGR을 PIL RGB로 변환
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    
                    # 해당 프레임의 포즈 데이터 찾기
                    data_idx = frame_idx // frame_interval
                    if data_idx < len(poses_data):
                        frame_pose_data = poses_data[data_idx]
                        
                        # 포즈 데이터를 draw_keypoints_on_image 형식으로 변환
                        pose_results = []
                        for person in frame_pose_data["persons"]:
                            keypoints = []
                            labels = []
                            scores = []
                            
                            for kp in person["keypoints"]:
                                keypoints.append([kp["x"], kp["y"]])
                                labels.append(kp["id"])
                                scores.append(kp["confidence"])
                            
                            pose_results.append({
                                "keypoints": np.array(keypoints),
                                "labels": np.array(labels),
                                "scores": np.array(scores)
                            })
                        
                        # 스켈레톤이 그려진 이미지 생성
                        skeleton_image = draw_keypoints_on_image(pil_image, pose_results)
                        
                        # PIL을 OpenCV BGR로 변환
                        skeleton_bgr = cv2.cvtColor(np.array(skeleton_image), cv2.COLOR_RGB2BGR)
                        
                        # 원본 해상도로 리사이즈
                        skeleton_bgr = cv2.resize(skeleton_bgr, (width, height))
                        
                        # 비디오에 프레임 추가
                        out.write(skeleton_bgr)
                        processed_frames += 1
                        
                        add_log(f"스켈레톤 프레임 {frame_idx} 처리 완료")
                    else:
                        # 포즈 데이터가 없는 프레임은 원본 그대로 사용
                        out.write(frame)
                
                except Exception as frame_error:
                    add_log(f"스켈레톤 프레임 {frame_idx} 처리 중 오류: {frame_error}")
                    # 오류 발생 시 원본 프레임 사용
                    out.write(frame)
            else:
                # frame_interval에 맞지 않는 프레임은 원본 그대로 사용
                out.write(frame)
            
            frame_idx += 1
        
        # 리소스 해제
        cap.release()
        out.release()
        
        add_log(f"스켈레톤 비디오 생성 완료: {output_path}")
        add_log(f"총 {processed_frames}개 프레임 처리됨")
        
        return output_path
        
    except Exception as e:
        add_log(f"스켈레톤 비디오 생성 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_pose_extraction(video, model_name=None, frame_interval=5, progress=None):
    """Gradio에서 호출되는 메인 함수"""
    if video is None:
        return None, None, None, 0, "비디오를 업로드해주세요."
    
    # Gradio는 업로드된 파일 경로를 직접 반환함
    video_path = video
    
    # 모델이 지정된 경우 로드
    if model_name and model_name != "yonigozlan/synthpose-vitpose-huge-hf":
        try:
            load_model(model_name)
        except Exception as e:
            return None, None, None, 0, f"모델 로드 실패: {str(e)}"
    
    try:
        # 비디오 처리
        thumbnails, poses_data = process_video(video_path, frame_interval=frame_interval, progress=progress)
        
        if not thumbnails:
            return None, None, None, 0, "처리 실패"
        
        # 비디오 FPS 정보 가져오기
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        cap.release()
        
        # 전역 변수에 데이터 저장
        processed_video_data["thumbnails"] = thumbnails
        processed_video_data["poses_data"] = poses_data
        processed_video_data["fps"] = fps
        processed_video_data["frame_interval"] = 5
        processed_video_data["video_path"] = video_path
        
        # 첫 번째 썸네일을 미리보기로 사용
        preview_image = thumbnails[0]
        
        # JSON 파일 저장
        json_path = video_path.replace(".mp4", "_poses.json")
        with open(json_path, "w", encoding='utf-8') as f:
            json.dump(poses_data, f, indent=2, ensure_ascii=False)
        
        # CSV 파일 저장
        csv_path = video_path.replace(".mp4", "_poses.csv")
        csv_rows = []
        
        for frame_data in poses_data:
            for person in frame_data["persons"]:
                for kp in person["keypoints"]:
                    csv_rows.append({
                        "timestamp": frame_data["timestamp"],
                        "frame_idx": frame_data["frame_idx"],
                        "person_id": person["person_id"],
                        "keypoint_id": kp["id"],
                        "keypoint_name": kp["name"],
                        "x": kp["x"],
                        "y": kp["y"],
                        "confidence": kp["confidence"]
                    })
        
        df = pd.DataFrame(csv_rows)
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        # 스켈레톤 비디오 생성
        add_log("스켈레톤 비디오 생성 시작")
        skeleton_video_path = create_skeleton_video(video_path, poses_data, fps, frame_interval, progress)
        
        if skeleton_video_path:
            add_log(f"스켈레톤 비디오 생성 완료: {skeleton_video_path}")
            return preview_image, json_path, csv_path, skeleton_video_path, duration, f"처리 완료! 총 {len(thumbnails)}개 프레임 처리됨 (FPS: {fps:.1f})"
        else:
            add_log("스켈레톤 비디오 생성 실패")
            return preview_image, json_path, csv_path, None, duration, f"처리 완료! 총 {len(thumbnails)}개 프레임 처리됨 (FPS: {fps:.1f})"
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        add_log(f"처리 중 오류 발생: {e}")
        add_log(f"상세 오류 정보:\n{error_details}")
        return None, None, None, 0, f"오류 발생: {str(e)}"
