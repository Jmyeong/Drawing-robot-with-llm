import sys, os
import time
import re
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # 2D 결과 저장용 (MuJoCo viewer에는 영향 없음)
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------
# 1. 라이브러리 경로 설정
# ---------------------------------------------------------
PROJECT_ROOT = '/ssd1/jm_data/project/Drawing_robot/Drawing-with-OMX'

sys.path.append(os.path.join(PROJECT_ROOT, 'package/kinematics_helper/'))
sys.path.append(os.path.join(PROJECT_ROOT, 'package/mujoco_helper/'))
sys.path.append(os.path.join(PROJECT_ROOT, 'package/utility/'))
sys.path.append(os.path.join(PROJECT_ROOT, 'package/openmanipulator/'))

try:
    from mujoco_parser import *
    from ik_utils import *
except ImportError as e:
    print(f"❌ 라이브러리 로드 실패: {e}")
    sys.exit(1)

# ======================================
# 2. 설정 (Configuration)
# ======================================
MODEL_PATH = "/ssd1/jm_data/project/Drawing_robot/output_final/checkpoint-epoch-10"
BASE_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"  # tokenizer 로드 실패시 fallback

XML_PATH = os.path.join(PROJECT_ROOT, "/ssd1/jm_data/project/Drawing_robot/Drawing-with-OMX/asset/omx/scene_omx_f_drawing.xml")
SAVE_IMG_DIR = "./simulation_results"
os.makedirs(SAVE_IMG_DIR, exist_ok=True)

NUM_BINS = 256
MAX_STEPS = 150
JOINT_DIM = 5

device = "cuda" if torch.cuda.is_available() else "cpu"

# ======================================
# 3. 모델 클래스 (학습 코드에 100% 맞춤)
# ======================================
class RobotDrawerRT2:
    def __init__(self):
        print(f"🚀 Loading Model from: {MODEL_PATH}")

        # 1) tokenizer: 학습 체크포인트에서 로드 (가장 중요)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
        except Exception as e:
            print(f"⚠️ 토크나이저 로드 실패 → base로 fallback: {e}")
            self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=True)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 2) action token 보장 (<bin_0> ~ <bin_255>)
        action_tokens = [f"<bin_{i}>" for i in range(NUM_BINS)]
        vocab = self.tokenizer.get_vocab()
        missing = [t for t in action_tokens if t not in vocab]
        if len(missing) > 0:
            self.tokenizer.add_tokens(missing)
            print(f"➕ Added missing action tokens: {len(missing)}")

        # 3) model
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map="auto",
            attn_implementation="eager",
        )
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.eval()

        # 학습 코드와 동일한 범위
        self.min_val = -np.pi
        self.max_val = np.pi

    def decode_trajectory(self, token_string: str):
        """
        학습 코드 ValidationHandler.decode_trajectory()와 동일한 규칙:
        - <bin_#>를 모두 뽑아 5개 단위로 자른다
        - 4 joint: (-pi ~ pi)
        - pen: (0 ~ 1)
        """
        bin_indices = [int(x) for x in re.findall(r"<bin_(\d+)>", token_string)]
        if not bin_indices:
            return None

        valid_len = (len(bin_indices) // JOINT_DIM) * JOINT_DIM
        if valid_len == 0:
            return None
        bin_indices = bin_indices[:valid_len]

        flat = np.array(bin_indices, dtype=np.float32)
        traj = flat.reshape(-1, JOINT_DIM)

        # joints
        for i in range(4):
            traj[:, i] = (traj[:, i] / (NUM_BINS - 1)) * (self.max_val - self.min_val) + self.min_val
        # pen
        traj[:, 4] = traj[:, 4] / (NUM_BINS - 1)

        # 생성이 길어질 수 있으니 MAX_STEPS로 컷
        traj = traj[:MAX_STEPS]
        return traj

    # 학습과 동일한 후처리(필터/스무딩)
    def filter_and_smooth(self, traj, jump_threshold=0.3):
        cleaned = traj.copy()
        T = cleaned.shape[0]

        # spike 제거 (joint 4개만)
        for i in range(1, T):
            for j in range(4):
                diff = cleaned[i, j] - cleaned[i-1, j]
                if abs(diff) > jump_threshold:
                    cleaned[i, j] = cleaned[i-1, j]

        # moving average smoothing
        smoothed = cleaned.copy()
        window = 3
        kernel = np.ones(window) / window
        for j in range(4):
            v = np.convolve(cleaned[:, j], kernel, mode="same")
            v[0] = cleaned[0, j]
            v[-1] = cleaned[-1, j]
            smoothed[:, j] = v
        return smoothed

    @torch.no_grad()
    def predict(self, prompt: str):
        # 학습 데이터의 prompt 스타일과 최대한 동일하게
        messages = [
            {"role": "system", "content": "You are a robot controller."},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(device)

        print(f"🤖 Generating for '{prompt}'...")
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=600,     # 학습 validation과 동일
            min_new_tokens=50,
            do_sample=False,
            repetition_penalty=1.0, # 학습 validation과 동일 (당신 학습코드는 1.0)
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        gen_text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=False
        )

        traj = self.decode_trajectory(gen_text)
        if traj is None:
            print("❌ decode failed (no valid <bin_*> sequence)")
            return None

        traj = self.filter_and_smooth(traj, jump_threshold=0.3)

        # 당신 요청: (2번) offset은 임의로
        # -> “허공에 그리는” 문제는 IK/plane 문제가 더 크지만,
        #    지금은 요청대로 단순 offset만 적용
        traj[:, 1] += 0.05
        traj[:, 2] -= 0.05

        return traj

# ======================================
# 4. 시뮬레이션 시각화 + 2D 저장 클래스
# ======================================
class SimVisualizer:
    def __init__(self):
        if not os.path.exists(XML_PATH):
            sys.exit(f"❌ XML Not Found: {XML_PATH}")

        xml_dir = os.path.dirname(XML_PATH)
        xml_file = os.path.basename(XML_PATH)
        cwd = os.getcwd()
        try:
            os.chdir(xml_dir)
            self.env = MuJoCoParserClass(name='omx', rel_xml_path=xml_file, verbose=False)
        finally:
            os.chdir(cwd)

        self.joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5"]
        self.stroke_list = []
        self.current_stroke = []

        self.env.init_viewer(
            title='RT-2 Simulation',
            transparent=False,
            azimuth=120, distance=0.4, lookat=(0.15, 0.0, 0.05)
        )

    def save_drawing_plot(self, prompt_text):
        if self.current_stroke:
            self.stroke_list.append(self.current_stroke)
            self.current_stroke = []

        if not self.stroke_list:
            print("⚠️ 그려진 내용이 없어 저장하지 않습니다.")
            return

        plt.figure(figsize=(5, 5))
        for i, stroke in enumerate(self.stroke_list):
            if len(stroke) < 2:
                continue
            points = np.array(stroke)
            robot_x = points[:, 0]
            robot_y = points[:, 1]
            lbl = 'Pen Path' if i == 0 else None
            plt.plot(robot_y, robot_x, color='red', linewidth=2, label=lbl)

        plt.xlim(-0.10, 0.10)
        plt.ylim(0.10, 0.30)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title(f"Result: {prompt_text}")
        plt.xlabel("Robot Y (Left/Right) [m]")
        plt.ylabel("Robot X (Front) [m]")
        plt.grid(True, linestyle='--', alpha=0.5)

        safe_name = prompt_text.replace(" ", "_")
        timestamp = int(time.time())
        save_path = os.path.join(SAVE_IMG_DIR, f"{safe_name}_{timestamp}_line.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"💾 2D Line Drawing saved to: {save_path}")

    def play_trajectory(self, trajectory, prompt_text=""):
        self.env.reset()
        self.stroke_list = []
        self.current_stroke = []

        if len(trajectory) > 0:
            start_q = trajectory[0][:4]
            self.env.forward(q=start_q, joint_names=self.joint_names[:4])

        print("▶️ Playing Simulation...")
        all_points_for_viewer = []

        for step_data in trajectory:
            if not self.env.is_viewer_alive():
                break

            target_q = step_data[:4]
            pen_val = float(step_data[4])
            is_drawing = pen_val > 0.5

            self.env.step(ctrl=target_q, ctrl_idxs=[0, 1, 2, 3])
            self.env.forward(q=target_q, joint_names=self.joint_names[:4])

            if is_drawing:
                ee_pos = self.env.get_p_body('end_effector_target')
                self.current_stroke.append(ee_pos.copy())
                all_points_for_viewer.append(ee_pos.copy())
            else:
                if len(self.current_stroke) > 0:
                    self.stroke_list.append(self.current_stroke)
                    self.current_stroke = []

            if len(all_points_for_viewer) > 0:
                self.env.plot_spheres(p_list=all_points_for_viewer[-100:], r=0.002, rgba=(1, 0, 0, 1))

            self.env.render()
            time.sleep(0.02)

        print("✨ Simulation Finished!")
        if len(self.current_stroke) > 0:
            self.stroke_list.append(self.current_stroke)

        self.save_drawing_plot(prompt_text)

# ======================================
# 5. 메인 실행
# ======================================
def main():
    agent = RobotDrawerRT2()
    sim = SimVisualizer()

    print("\n" + "="*50)
    print(" 🎨 RT-2 Drawing Simulation ")
    print("="*50 + "\n")

    while True:
        try:
            prompt = input("Draw >> ").strip()
            if prompt.lower() in ['quit', 'exit', 'q']:
                break
            if not prompt:
                continue

            traj = agent.predict(prompt)
            if traj is None:
                print("❌ 궤적 생성 실패")
                continue

            sim.play_trajectory(traj, prompt_text=prompt)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

    sim.env.close_viewer()
    print("Bye!")

if __name__ == "__main__":
    main()
