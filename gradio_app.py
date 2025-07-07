# gradio_app.py
import gradio as gr
import time
from vitpose_gradio_app import run_pose_extraction, get_keypoint_frame_at_time, get_logs, clear_logs, get_pose_data_at_time, load_model, processed_video_data, add_log



def refresh_logs():
    """로그를 새로고침"""
    return get_logs()

def load_json_data(json_path):
    """JSON 파일을 로드하여 뷰어에 표시"""
    if json_path is None:
        return None
    
    try:
        import json
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        return {"error": f"JSON 로드 실패: {str(e)}"}

def expand_json_data(json_data):
    """JSON 데이터를 전체 확장하여 표시"""
    if json_data is None:
        return None
    return json_data

def collapse_json_data(json_data):
    """JSON 데이터를 축소하여 표시 (첫 3개 항목만)"""
    if json_data is None:
        return None
    
    if isinstance(json_data, list) and len(json_data) > 3:
        return json_data[:3] + [{"...": f"총 {len(json_data)}개 항목 중 3개만 표시"}]
    return json_data



def load_selected_model(model_name):
    """선택된 모델을 로드"""
    try:
        load_model(model_name)
        return f"✅ {model_name} 모델 로드 완료!"
    except Exception as e:
        return f"❌ {model_name} 모델 로드 실패: {str(e)}"



def get_video_info(video_path):
    """비디오 정보를 추출하는 함수"""
    if video_path is None:
        return "비디오를 업로드하면 정보가 여기에 표시됩니다."
    
    try:
        import cv2
        import os
        from datetime import timedelta
        
        # 비디오 캡처 객체 생성
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return "❌ 비디오 파일을 열 수 없습니다."
        
        # 비디오 정보 추출
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 길이 계산
        duration_seconds = total_frames / fps if fps > 0 else 0
        duration = timedelta(seconds=int(duration_seconds))
        
        # 파일 크기
        file_size = os.path.getsize(video_path)
        file_size_mb = file_size / (1024 * 1024)
        
        # 비디오 코덱 정보
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        
        cap.release()
        
        # 정보 포맷팅
        info_text = f"""
### � 비디오 정보

**� 해상도**: {width} × {height} ({width * height:,} 픽셀)

**⏱️ 길이**: {duration} ({duration_seconds:.1f}초)

**� 프레임**: {total_frames:,}개 프레임

**� FPS**: {fps:.1f} fps

**� 파일 크기**: {file_size_mb:.1f} MB

**�️ 코덱**: {codec}

**� 비율**: {width/height:.2f}:1 ({'가로형' if width > height else '세로형' if height > width else '정사각형'})
        """
        
        return info_text
        
    except Exception as e:
        return f"❌ 비디오 정보 추출 실패: {str(e)}"

with gr.Blocks(title="VitPose Video Analysis", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# �‍♂️ VitPose Video Pose Extractor")
    gr.Markdown("### AI 기반 비디오 포즈 분석 및 키포인트 추출")
    
    # 메인 컨텐츠 영역
    with gr.Row():
        # 왼쪽 패널: 입력 및 제어
        with gr.Column(scale=1):
            gr.Markdown("### � 모델 선택")
            model_dropdown = gr.Dropdown(
                choices=[
                    "yonigozlan/synthpose-vitpose-huge-hf (Huge)",
                    "yonigozlan/synthpose-vitpose-base-hf (Base)"
                ],
                value="yonigozlan/synthpose-vitpose-huge-hf (Huge)",
                label="포즈 추정 모델",
                info="Huge: 더 정확하지만 느림, Base: 빠르지만 덜 정확"
            )
            load_model_btn = gr.Button("� 모델 로드", variant="secondary", size="sm")
            model_status = gr.Textbox(
                label="모델 상태", 
                interactive=False, 
                value="모델을 선택하고 로드하세요.",
                lines=2
            )
            
            gr.Markdown("### � 비디오 업로드")
            video_input = gr.Video(label="비디오 파일 선택", height=200)
            
            # 비디오 정보 표시
            video_info = gr.Markdown(
                "비디오를 업로드하면 정보가 여기에 표시됩니다.",
                label="� 비디오 정보"
            )
            
            gr.Markdown("### ⚙️ 처리 설정")
            frame_interval_dropdown = gr.Dropdown(
                choices=[
                    "모든 프레임 (1)",
                    "2프레임마다 (2)",
                    "5프레임마다 (5)",
                    "10프레임마다 (10)",
                    "15프레임마다 (15)",
                    "30프레임마다 (30)"
                ],
                value="5프레임마다 (5)",
                label="프레임 처리 간격",
                info="간격이 클수록 빠르지만 덜 정밀함"
            )
            
            gr.Markdown("### ⚙️ 제어")
            extract_btn = gr.Button("� 포즈 추출 시작", variant="primary", size="lg")
            status_text = gr.Textbox(
                label="상태", 
                interactive=False, 
                value="비디오를 업로드하고 포즈 추출을 시작하세요.",
                lines=2
            )
            
                # 모델 로드 이벤트
    load_model_btn.click(
        fn=lambda x: load_selected_model(x.split(" (")[0]),
        inputs=[model_dropdown],
        outputs=[model_status]
    )
    
    # 비디오 업로드 시 정보 표시
    video_input.change(
        fn=get_video_info,
        inputs=[video_input],
        outputs=[video_info]
    )
    
    gr.Markdown("### � 다운로드")
    with gr.Row():
        json_output = gr.File(label="� JSON", scale=1)
        csv_output = gr.File(label="� CSV", scale=1)
        skeleton_video_output = gr.File(label="� 스켈레톤 비디오", scale=1)
        
        # 오른쪽 패널: 결과 표시
        with gr.Column(scale=1):
            gr.Markdown("### � 처리 결과")
            status_text = gr.Textbox(
                label="상태", 
                interactive=False, 
                value="비디오를 업로드하고 포즈 추출을 시작하세요.",
                lines=3
            )
    
    # 하단 패널: 로그 및 데이터
    with gr.Row():
        # 로그 섹션
        with gr.Column(scale=1):
            gr.Markdown("### � 실시간 로그")
            log_display = gr.Textbox(
                label="처리 로그", 
                interactive=False, 
                lines=8,
                max_lines=12,
                value="로그가 여기에 표시됩니다..."
            )
            with gr.Row():
                refresh_btn = gr.Button("� 새로고침", size="sm", variant="secondary")
                clear_btn = gr.Button("�️ 초기화", size="sm", variant="secondary")
        
        # JSON 뷰어 섹션
        with gr.Column(scale=1):
            gr.Markdown("### � 현재 시점 포즈 데이터")
            json_viewer = gr.JSON(
                label="선택된 시점의 포즈 데이터"
            )
            with gr.Row():
                expand_btn = gr.Button("� 전체보기", size="sm", variant="secondary")
                collapse_btn = gr.Button("� 축소", size="sm", variant="secondary")
    

    
    # 포즈 추출 버튼 클릭 이벤트
    def on_extract_complete(video, model_name, frame_interval_str, progress=gr.Progress()):
        if video is None:
            return None, None, None, None, "비디오를 업로드해주세요."
        
        # 모델 이름에서 실제 모델 ID 추출
        actual_model = model_name.split(" (")[0] if model_name else "yonigozlan/synthpose-vitpose-huge-hf"
        
        # 프레임 간격 파싱
        frame_interval = int(frame_interval_str.split("(")[1].split(")")[0])
        
        result = run_pose_extraction(video, actual_model, frame_interval, progress)
        if len(result) == 6:
            preview_image, json_path, csv_path, skeleton_video_path, duration, status = result
            json_data = load_json_data(json_path)
            return [
                json_path,      # json_output  
                csv_path,       # csv_output
                skeleton_video_path,  # skeleton_video_output
                status,         # status_text
                get_logs(),     # log_display
                json_data       # json_viewer (전체 데이터)
            ]
        else:
            return [None, None, None, "처리 실패", get_logs(), None]
    
    extract_btn.click(
        fn=on_extract_complete,
        inputs=[video_input, model_dropdown, frame_interval_dropdown],
        outputs=[json_output, csv_output, skeleton_video_output, status_text, log_display, json_viewer]
    )
    
    # 로그 자동 새로고침 (페이지 로드 시)
    demo.load(
        fn=get_logs,
        outputs=[log_display]
    )
    

    
    # 로그 관련 버튼들
    refresh_btn.click(
        fn=refresh_logs,
        inputs=[],
        outputs=[log_display]
    )
    
    clear_btn.click(
        fn=clear_logs,
        inputs=[],
        outputs=[log_display]
    )
    

    
    # JSON 뷰어 관련 버튼들
    expand_btn.click(
        fn=expand_json_data,
        inputs=[json_viewer],
        outputs=[json_viewer]
    )
    
    collapse_btn.click(
        fn=collapse_json_data,
        inputs=[json_viewer],
        outputs=[json_viewer]
    )

demo.launch(share=True, server_name="0.0.0.0")
