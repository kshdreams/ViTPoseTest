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

# COCO 17개 키포인트 + 추가 12개 키포인트 정의
COCO_17_KEYPOINTS = [
    "Nose", "L_Eye", "R_Eye", "L_Ear", "R_Ear",
    "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow",
    "L_Wrist", "R_Wrist", "L_Hip", "R_Hip",
    "L_Knee", "R_Knee", "L_Ankle", "R_Ankle"
]

ADDITIONAL_KEYPOINTS = [
    "r_big_toe", "l_big_toe", "l_calc", "r_calc"
]

# 전체 29개 키포인트 목록
SELECTED_KEYPOINTS = COCO_17_KEYPOINTS + ADDITIONAL_KEYPOINTS

# 키포인트 이름별 색상 매핑
KEYPOINT_COLOR_MAP = {
    # 얼굴(분홍)
    "Nose": (255,192,203), "L_Eye": (255,192,203), "R_Eye": (255,192,203), "L_Ear": (255,192,203), "R_Ear": (255,192,203),
    # 왼쪽 상체/하체(파랑)
    "L_Shoulder": (0,0,255), "L_Elbow": (0,0,255), "L_Wrist": (0,0,255), "L_Hip": (0,0,255),
    "L_Knee": (0,0,255), "L_Ankle": (0,0,255), "l_toe": (0,0,255), "l_big_toe": (0,0,255), "l_calc": (0,0,255),
    # 오른쪽 상체/하체(노랑)
    "R_Shoulder": (255,255,0), "R_Elbow": (255,255,0), "R_Wrist": (255,255,0),
    "R_Hip": (255,255,0),
    "R_Knee": (255,255,0), "R_Ankle": (255,255,0), "r_toe": (255,255,0), "r_big_toe": (255,255,0), "r_calc": (255,255,0),
}

# 이름 기반 엣지 정의
KEYPOINT_EDGES = [
    ("Nose", "L_Eye"), ("Nose", "R_Eye"), ("L_Eye", "L_Ear"), ("R_Eye", "R_Ear"),  # 머리
    ("L_Shoulder", "R_Shoulder"), ("L_Shoulder", "L_Elbow"), ("L_Elbow", "L_Wrist"),
    ("R_Shoulder", "R_Elbow"), ("R_Elbow", "R_Wrist"),
    ("L_Shoulder", "L_Hip"), ("R_Shoulder", "R_Hip"), ("L_Hip", "R_Hip"),
    ("L_Hip", "L_Knee"), ("L_Knee", "L_Ankle"),
    ("R_Hip", "R_Knee"), ("R_Knee", "R_Ankle"),
    {"r_big_toe", "R_Ankle"}, {"r_calc", "R_Ankle"},
    {"l_big_toe", "L_Ankle"}, {"l_calc", "L_Ankle"},
    # 필요하면 추가 키포인트 엣지도 여기에 추가
]

def filter_keypoints(pose_results, selected_keypoints):
    if pose_model is None:
        return pose_results
    filtered_results = []
    for pose_result in pose_results:
        filtered_keypoints = []
        for i, (keypoint, label, score) in enumerate(zip(
            pose_result["keypoints"], 
            pose_result["labels"], 
            pose_result["scores"]
        )):
            keypoint_name = pose_model.config.id2label[label.item()] if pose_model else f"keypoint_{i}"
            if keypoint_name in selected_keypoints:
                filtered_keypoints.append({
                    "name": keypoint_name,
                    "coord": keypoint,
                    "score": score
                })
        if filtered_keypoints:
            filtered_results.append({
                "keypoints": filtered_keypoints
            })
    return filtered_results

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
    global person_image_processor, person_model, image_processor, pose_model, SELECTED_KEYPOINTS
    
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
        
        # id2label.values()에서 원하는 키포인트만 뽑아서 SELECTED_KEYPOINTS 동적 할당
        id2label_names = list(pose_model.config.id2label.values())
        # 원하는 이름만 골라서 아래 리스트에 추가 (실제 id2label에 있는 이름만!)
        wanted = [
            "Nose", "L_Eye", "R_Eye", "L_Ear", "R_Ear",
            "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow",
            "L_Wrist", "R_Wrist", "L_Hip", "R_Hip",
            "L_Knee", "R_Knee", "L_Ankle", "R_Ankle",
            "l_big_toe", "r_big_toe", "l_calc", "r_calc"
        ]
        SELECTED_KEYPOINTS = [name for name in id2label_names if name in wanted]
        print("SELECTED_KEYPOINTS:", SELECTED_KEYPOINTS)
        
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
    
    # 선택된 키포인트만 필터링
    filtered_results = filter_keypoints(pose_results[0], SELECTED_KEYPOINTS)
    
    add_log(f"키포인트 필터링: {len(pose_results[0])}개 → {len(filtered_results)}개")
    
    return filtered_results

import math

def draw_points(image, keypoints, keypoint_score_threshold, radius, show_keypoint_weight):
    for kp in keypoints:
        name = kp["name"]
        x_coord, y_coord = int(kp["coord"][0]), int(kp["coord"][1])
        kpt_score = kp["score"]
        if kpt_score > keypoint_score_threshold:
            color = KEYPOINT_COLOR_MAP.get(name, (255,255,255))
            if show_keypoint_weight:
                cv2.circle(image, (x_coord, y_coord), radius, color, -1)
                transparency = max(0, min(1, kpt_score))
                cv2.addWeighted(image, transparency, image, 1 - transparency, 0, dst=image)
            else:
                cv2.circle(image, (x_coord, y_coord), radius, color, -1)

def draw_links(image, keypoints, keypoint_edges, keypoint_score_threshold, thickness, show_keypoint_weight, stick_width=2):
    name_to_idx = {kp["name"]: idx for idx, kp in enumerate(keypoints)}
    for n1, n2 in keypoint_edges:
        if n1 not in name_to_idx or n2 not in name_to_idx:
            print(f"skip edge ({n1}, {n2}) (name not found)")
            continue
        idx1, idx2 = name_to_idx[n1], name_to_idx[n2]
        kp1, kp2 = keypoints[idx1], keypoints[idx2]
        x1, y1, score1 = int(kp1["coord"][0]), int(kp1["coord"][1]), kp1["score"]
        x2, y2, score2 = int(kp2["coord"][0]), int(kp2["coord"][1]), kp2["score"]
        if (
            x1 > 0 and x1 < image.shape[1] and y1 > 0 and y1 < image.shape[0] and
            x2 > 0 and x2 < image.shape[1] and y2 > 0 and y2 < image.shape[0] and
            score1 > keypoint_score_threshold and score2 > keypoint_score_threshold
        ):
            color = KEYPOINT_COLOR_MAP.get(n1, (255,255,255))
            cv2.line(image, (x1, y1), (x2, y2), color, thickness=thickness)

def draw_keypoints_on_image(image, pose_results, person_boxes=None, input_boxes=None):
    numpy_image = np.array(image)
    for pose_result in pose_results:
        keypoints = pose_result["keypoints"]
        # name 기반 엣지
        draw_points(numpy_image, keypoints, keypoint_score_threshold=0.3, radius=2, show_keypoint_weight=False)
        draw_links(numpy_image, keypoints, KEYPOINT_EDGES, keypoint_score_threshold=0.3, thickness=1, show_keypoint_weight=False)
    return numpy_image

def process_video(video_file, frame_interval=1, progress=None):
    """비디오를 처리하여 포즈 데이터를 추출하는 함수"""
    try:
        add_log(f"비디오 파일 열기 시도: {video_file}")
        cap = cv2.VideoCapture(video_file)
        
        # 프로그레스 중복 호출 방지를 위한 변수
        last_progress_update = 0
        
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
            
            # 프로그레스 업데이트 (0.0 ~ 1.0 범위로 제한, 중복 호출 방지)
            if progress is not None and frame_idx - last_progress_update >= 10:  # 10프레임마다만 업데이트
                progress_percent = min(1.0, frame_idx / total_frames)
                progress(progress_percent, desc=f"프레임 {frame_idx}/{total_frames} 처리 중...")
                last_progress_update = frame_idx
            
            # 프로그레스를 콘솔에 출력
            if frame_idx % 100 == 0:  # 100프레임마다 출력
                console_progress = min(100.0, (frame_idx / total_frames) * 100)
                add_log(f"프레임 {frame_idx}/{total_frames} 처리 중... ({console_progress:.1f}%)")
            
            if frame_idx % frame_interval == 0:
                try:
                    add_log(f"프레임 {frame_idx} 처리 시작")
                    
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
                            adjusted_boxes = person_boxes  # adjusted_boxes도 설정
                            previous_box_areas = []  # 이전 면적 초기화
                            add_log(f"프레임 {frame_idx} 사람 미검출, 전체 이미지 사용")
                    else:
                        # 캐시된 사람 검출 결과 사용
                        person_boxes = cached_person_boxes
                        adjusted_boxes = cached_person_boxes  # adjusted_boxes도 설정
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
                        for kp in person_pose["keypoints"]:
                            keypoint_name = kp["name"]
                            x, y = kp["coord"]
                            score = kp["score"]
                            person_data["keypoints"].append({
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
        add_log(f"비디오 처리 완료. 총 {processed_frames}개 프레임 처리됨 (필터링된 키포인트: {len(SELECTED_KEYPOINTS)}개)")
        
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
        
        # 프로그레스 중복 호출 방지를 위한 변수
        last_progress_update = 0
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
            
            # 프로그레스 업데이트 (0.0 ~ 1.0 범위로 제한, 중복 호출 방지)
            if progress is not None and frame_idx - last_progress_update >= 10:  # 10프레임마다만 업데이트
                progress_percent = min(1.0, frame_idx / total_frames)
                progress(progress_percent, desc=f"스켈레톤 비디오 생성 중... {frame_idx}/{total_frames}")
                last_progress_update = frame_idx
            
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
                        
                        # 포즈 데이터를 draw_keypoints_on_image 형식으로 변환 (필터링된 키포인트)
                        pose_results = []
                        for person in frame_pose_data["persons"]:
                            keypoints = []
                            for kp in person["keypoints"]:
                                if kp["name"] in SELECTED_KEYPOINTS:
                                    keypoints.append({
                                        "name": kp["name"],
                                        "coord": [kp["x"], kp["y"]],
                                        "score": kp["confidence"]
                                    })
                            if keypoints:
                                pose_results.append({
                                    "keypoints": keypoints
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
                for idx, kp in enumerate(person["keypoints"]):
                    csv_rows.append({
                        "timestamp": frame_data["timestamp"],
                        "frame_idx": frame_data["frame_idx"],
                        "person_id": person["person_id"],
                        "keypoint_id": idx,
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
            return preview_image, json_path, csv_path, skeleton_video_path, duration, f"처리 완료! 총 {len(thumbnails)}개 프레임 처리됨 (FPS: {fps:.1f}, 키포인트: {len(SELECTED_KEYPOINTS)}개)"
        else:
            add_log("스켈레톤 비디오 생성 실패")
            return preview_image, json_path, csv_path, None, duration, f"처리 완료! 총 {len(thumbnails)}개 프레임 처리됨 (FPS: {fps:.1f}, 키포인트: {len(SELECTED_KEYPOINTS)}개)"
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        add_log(f"처리 중 오류 발생: {e}")
        add_log(f"상세 오류 정보:\n{error_details}")
        return None, None, None, 0, f"오류 발생: {str(e)}"
