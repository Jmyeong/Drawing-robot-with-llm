import sys, os
import mujoco
import numpy as np
import json
import glob
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------------------------------------------------------
# 1. 라이브러리 경로 설정 (사용자 환경에 맞게 수정)
# ---------------------------------------------------------
PROJECT_ROOT = '/ssd1/jm_data/project/Drawing_robot/Drawing-with-OMX' # [중요] 실제 경로로 수정하세요
sys.path.append(os.path.join(PROJECT_ROOT, 'package/kinematics_helper/'))
sys.path.append(os.path.join(PROJECT_ROOT, 'package/mujoco_helper/'))
sys.path.append(os.path.join(PROJECT_ROOT, 'package/utility/'))
sys.path.append(os.path.join(PROJECT_ROOT, 'package/openmanipulator/'))

try:
    from ik import *
    from mujoco_parser import *
    from utils import *
    from transforms import *
    from ik_utils import *
except ImportError as e:
    print(f"❌ 라이브러리 로드 실패: {e}")
    sys.exit(1)

# ---------------------------------------------------------
# 2. 설정 (Configuration)
# ---------------------------------------------------------
CONFIG = {
    # [입력] QuickDraw 데이터셋 경로 (jsonl 파일들이 있는 폴더)
    'dataset_root': "/ssd1/jm_data/project/Drawing_robot/dataset/dataset/quickdraw_multi_stroke",
    
    # [출력] 데이터 저장 경로
    'save_dir': './dataset_output_joint_data_2',
    
    # [설정] MuJoCo XML 파일 경로
    'xml_path': os.path.join(PROJECT_ROOT, "asset/omx/scene_omx_f_drawing.xml"),
    
    'joint_names': ["joint1", "joint2", "joint3", "joint4", "joint5"],
    
    # [옵션] 시뮬레이션 시각화 여부 (True: 창 뜸, False: 백그라운드)
    # 대량 생성 시에는 False를 권장합니다.
    'visualize': False,
    
    # [작업 영역] 로봇이 그림을 그릴 실제 물리적 범위 (미터 단위)
    'CANVAS_WIDTH_M': 0.10,   # 가로 8cm
    'CANVAS_HEIGHT_M': 0.10,  # 세로 8cm
    'START_X': 0.15,          # 로봇 앞 15cm (X축)
    'START_Y': 0.0,           # 로봇 중앙 (Y축)
    
    # [높이 설정]
    'z_draw': 0.015, # 펜이 바닥에 닿는 높이 (보정 필요할 수 있음)
    'z_hover': 0.100, # 이동 높이
    
    # [생성 설정]
    'samples_per_category': 50, # 클래스당 생성할 샘플 수
    'ik_max_tick': 2000,         # IK 계산 최대 반복 수
    'ik_err_th': 0.02,           # IK 허용 오차
}

os.makedirs(CONFIG['save_dir'], exist_ok=True)

# ---------------------------------------------------------
# 3. 로봇 데이터 생성기 클래스
# ---------------------------------------------------------
class RobotDataGenerator:
    def __init__(self):
        self.env = MuJoCoParserClass(name='omx', rel_xml_path=CONFIG['xml_path'], verbose=False)
        self.q0 = np.array([0, 0, 0, 0, 0.2240]) # 초기 홈 포즈
        self.reset_env()
        
        # 그림 저장용 데이터
        self.recorded_strokes = [] 
        self.current_stroke = []

        if CONFIG['visualize']:
            self.env.init_viewer(
                title='Dataset Generator',
                transparent=False,
                azimuth=120, distance=0.4, lookat=(0.15, 0.0, 0.05)
            )
            
    def reset_env(self):
        self.env.reset()
        self.env.forward(q=self.q0, joint_names=CONFIG['joint_names'])
        self.recorded_strokes = []
        self.current_stroke = []

    def scale_strokes_to_robot(self, strokes):
        """
        QuickDraw 좌표(픽셀 등)를 로봇의 미터 단위 작업 영역으로 변환
        Multi-Stroke 구조 [[[x,y],...], ...] 유지
        """
        all_points = []
        for stroke in strokes:
            for pt in stroke:
                all_points.append(pt)
        
        if not all_points: return None
        pts = np.array(all_points)

        # 1. 원본 데이터의 바운딩 박스 계산
        min_x, min_y = np.min(pts, axis=0)
        max_x, max_y = np.max(pts, axis=0)
        
        w = max_x - min_x
        h = max_y - min_y
        if w == 0 or h == 0: return None

        # 2. 스케일 계산 (비율 유지하면서 작업 영역에 맞춤)
        scale_x = CONFIG['CANVAS_WIDTH_M'] / w
        scale_y = CONFIG['CANVAS_HEIGHT_M'] / h
        final_scale = min(scale_x, scale_y) * 0.8 # 80% 크기로 여백 확보

        scaled_w = w * final_scale
        scaled_h = h * final_scale
        
        scaled_strokes = []
        for stroke in strokes:
            new_stroke = []
            for pt in stroke:
                x, y = pt[0], pt[1]
                
                # 정규화 (0~1) 및 스케일링
                norm_x = (x - min_x) * final_scale
                norm_y = (y - min_y) * final_scale
                
                # [좌표 변환] Image(x, y) -> Robot(Front, Left)
                # Robot X (앞뒤): 시작점 + 높이 - Y (이미지 Y는 아래로 증가하므로 반전)
                robot_x = CONFIG['START_X'] + (scaled_h - norm_y) 
                
                # Robot Y (좌우): 시작점 + X - (너비 절반) -> 중앙 정렬
                robot_y = CONFIG['START_Y'] + norm_x - (scaled_w / 2)
                
                new_stroke.append([robot_x, robot_y])
            scaled_strokes.append(new_stroke)
            
        return scaled_strokes

    def solve_ik_point(self, target_pos, current_q):
        """Inverse Kinematics 계산"""
        qpos, ik_err_stack, _ = solve_ik(
            env=self.env,
            joint_names_for_ik=CONFIG['joint_names'][0:4],
            body_name_trgt='end_effector_target',
            q_init=current_q,
            p_trgt=target_pos,
            R_trgt=None, # [중요] 자세 제어 끔 (자유롭게 움직임)
            max_ik_tick=CONFIG['ik_max_tick'],
            ik_stepsize=1.0,
            ik_eps=1e-4,
            render=False
        )
        err = np.max(np.abs(ik_err_stack))
        return qpos, err

    def record_step(self, q_target, is_drawing=False):
        """시뮬레이션 업데이트 및 그리기 경로 기록"""
        self.env.forward(q=q_target, joint_names=CONFIG['joint_names'][0:4])
        
        # 펜이 내려가 있으면(drawing) 현재 획에 점 추가
        if is_drawing:
            ee_pos = self.env.get_p_body('end_effector_target')
            self.current_stroke.append((ee_pos[0], ee_pos[1]))
        # 펜이 올라가면 현재 획을 저장하고 초기화
        else:
            if len(self.current_stroke) > 0:
                self.recorded_strokes.append(self.current_stroke)
                self.current_stroke = []

        if CONFIG['visualize']:
            # 시각화 모드일 때만 렌더링
            self.env.render()

    def save_drawing_image(self, save_path):
        """기록된 궤적을 2D 이미지로 저장 (확인용)"""
        # 마지막 획 처리
        if len(self.current_stroke) > 0:
            self.recorded_strokes.append(self.current_stroke)
            self.current_stroke = []

        if not self.recorded_strokes: return

        plt.figure(figsize=(4, 4))
        for stroke in self.recorded_strokes:
            stroke = np.array(stroke)
            # Robot Y(좌우) -> Plot X, Robot X(앞뒤) -> Plot Y
            plt.plot(stroke[:, 1], stroke[:, 0], color='red', linewidth=2)

        # 캔버스 범위 고정
        margin = 0.05
        plt.xlim(CONFIG['START_Y'] - margin, CONFIG['START_Y'] + margin)
        plt.ylim(CONFIG['START_X'], CONFIG['START_X'] + CONFIG['CANVAS_HEIGHT_M'] + margin)
        
        plt.gca().set_aspect('equal')
        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.05, dpi=100)
        plt.close()

    def process_drawing(self, strokes):
        """입력된 스트로크를 로봇 관절 궤적으로 변환"""
        scaled_strokes = self.scale_strokes_to_robot(strokes)
        if scaled_strokes is None: return None

        trajectory_data = [] # [J1, J2, J3, J4, Pen_State]
        self.reset_env()
        current_q = self.q0[0:4]

        # 뷰어가 닫혔으면 중단
        if CONFIG['visualize'] and not self.env.is_viewer_alive():
             return None

        for stroke in scaled_strokes:
            if len(stroke) == 0: continue

            # 1. Hover (이동: Z Hover)
            start_pt = stroke[0]
            target_pos = np.array([start_pt[0], start_pt[1], CONFIG['z_hover']])
            q_hover, err = self.solve_ik_point(target_pos, current_q)
            if err > CONFIG['ik_err_th']: return None # IK 실패시 데이터 버림
            
            trajectory_data.append(np.append(q_hover, 0.0)) # Pen Up (0.0)
            self.record_step(q_hover, is_drawing=False)
            current_q = q_hover
            
            # 2. Down (펜 내림: Z Draw)
            target_pos[2] = CONFIG['z_draw']
            q_down, err = self.solve_ik_point(target_pos, current_q)
            if err > CONFIG['ik_err_th']: return None
            
            trajectory_data.append(np.append(q_down, 1.0)) # Pen Down (1.0)
            self.record_step(q_down, is_drawing=True)
            current_q = q_down
            
            # 3. Draw (획 긋기)
            for i in range(1, len(stroke)):
                pt = stroke[i]
                target_pos = np.array([pt[0], pt[1], CONFIG['z_draw']])
                q_draw, err = self.solve_ik_point(target_pos, current_q)
                
                if err > CONFIG['ik_err_th']: break 
                
                trajectory_data.append(np.append(q_draw, 1.0)) # Pen Down (1.0)
                self.record_step(q_draw, is_drawing=True)
                current_q = q_draw
            
            # 4. Up (펜 듦: Z Hover)
            end_pt = stroke[-1]
            end_pos = np.array([end_pt[0], end_pt[1], CONFIG['z_hover']])
            q_up, err = self.solve_ik_point(end_pos, current_q)
            
            trajectory_data.append(np.append(q_up, 0.0)) # Pen Up (0.0)
            self.record_step(q_up, is_drawing=False)
            current_q = q_up
            
        return np.array(trajectory_data)

# ---------------------------------------------------------
# 4. JSON 파서 (Multi-Stroke 포맷 지원)
# ---------------------------------------------------------
def parse_llm_jsonl_line(line_str):
    try:
        outer_data = json.loads(line_str)
        assistant_content = None
        
        # 메시지에서 assistant의 응답 찾기
        for msg in outer_data.get("messages", []):
            if msg.get("role") == "assistant":
                content = msg.get("content")
                if isinstance(content, list):
                    for item in content:
                        if item.get("type") == "text":
                            assistant_content = item.get("text")
                            break
                elif isinstance(content, str):
                    assistant_content = content
                break
        
        if not assistant_content: return None
        
        # 내부 JSON 파싱 (strokes 키 확인)
        inner_data = json.loads(assistant_content)
        strokes = inner_data.get("strokes", [])
        
        # 데이터가 비어있으면 None
        if not strokes: return None
        
        return strokes # [[[x,y], ...], ...] 반환

    except Exception:
        return None

# ---------------------------------------------------------
# 5. 메인 함수
# ---------------------------------------------------------
def main():
    generator = RobotDataGenerator()
    
    # 데이터셋 파일 목록 가져오기 (.jsonl)
    jsonl_files = glob.glob(os.path.join(CONFIG['dataset_root'], "*.jsonl"))
    print(f"📂 Found {len(jsonl_files)} categories in {CONFIG['dataset_root']}")
    
    index_data = [] 
    
    for file_path in jsonl_files:
        # 파일명에서 카테고리 이름 추출 (예: cat.jsonl -> cat)
        category_name = os.path.splitext(os.path.basename(file_path))[0]
        print(f"Processing: {category_name}")
        
        count = 0
        with open(file_path, 'r') as f:
            # tqdm으로 진행률 표시
            for line_idx, line in enumerate(tqdm(f, total=CONFIG['samples_per_category'], desc=category_name)):
                if count >= CONFIG['samples_per_category']: break
                
                # 뷰어 닫히면 종료
                if CONFIG['visualize'] and not generator.env.is_viewer_alive():
                    print("Viewer closed.")
                    break

                # 1. 파싱
                strokes = parse_llm_jsonl_line(line)
                if strokes is None: continue 

                # 2. 로봇 데이터 생성 (IK)
                traj = generator.process_drawing(strokes)
                
                if traj is not None:
                    # 파일명: category_index.npy
                    file_prefix = f"{category_name}_{line_idx}"
                    
                    # 3. 관절 데이터 저장 (.npy)
                    npy_path = os.path.join(CONFIG['save_dir'], f"{file_prefix}.npy")
                    np.save(npy_path, traj)
                    
                    # 4. 그림 이미지 저장 (.png) - 결과 확인용
                    img_path = os.path.join(CONFIG['save_dir'], f"{file_prefix}.png")
                    generator.save_drawing_image(img_path)
                    
                    # 인덱스 데이터 추가
                    index_data.append({
                        "prompt": f"Draw a {category_name}",
                        "file_path": f"{file_prefix}.npy",
                        "image_path": f"{file_prefix}.png",
                        "length": len(traj),
                    })
                    count += 1
    
    if CONFIG['visualize']:
        generator.env.close_viewer()

    # 메타데이터 저장
    with open(os.path.join(CONFIG['save_dir'], "dataset_index.json"), "w") as f:
        json.dump(index_data, f, indent=4)
        
    print(f"\n✨ All Joint Data Saved to {CONFIG['save_dir']}")

if __name__ == "__main__":
    main()