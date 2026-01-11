import sys, os
import mujoco
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import cv2
from PIL import Image, ImageDraw
from IPython.display import display
from OpenGL.GL import *
import time

# Get the absolute path to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

sys.path.append(os.path.join(project_root, "package/kinematics_helper/"))
sys.path.append(os.path.join(project_root, "package/mujoco_helper/"))
sys.path.append(os.path.join(project_root, "package/utility/"))
sys.path.append(os.path.join(project_root, "package/openmanipulator/"))

from ik import solve_ik
from mujoco_parser import MuJoCoParserClass
from utils import MultiSliderClass
from transforms import rpy2r
from ik_utils import solve_ik_and_interpolate, interpolate_and_smooth_nd
import datetime
from motor import OMX_Controller
from pos_utils import rads_to_bytes, bytes_to_rads
from tqdm import tqdm

print("MuJoCo:[%s]" % (mujoco.__version__))
"""
requirements:
mujoco
imageio
matplotlib
dynamixel-sdk
"""

"""codes to make glfw use GPU in rendering time"""

os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"
os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
os.environ["__VK_LAYER_NV_optimus"] = "NVIDIA_only"


"""canvas width, canvas height"""

canvas_width = 0.125
canvas_height = 0.179


def rpy_deg2r(r):
    r_rad = np.deg2rad(r)
    return rpy2r(r_rad)


REAL_ROBOT = False

appropriate_z_offset = -0.140
# appropriate_z_offset = -0.22
sim2real_gap = 0.00


def get_scaled_sketch_coordinates(image_path, canvas_width_m, canvas_height_m=None):

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("이미지를 불러올 수 없습니다. 경로를 확인하세요.")

    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("이미지에서 윤곽선을 찾을 수 없습니다.")

    all_points = np.concatenate(contours)
    x, y, w, h = cv2.boundingRect(all_points)

    if canvas_height_m is None:
        canvas_height_m = canvas_width_m * (h / w)
        scale = canvas_width_m / w
    else:

        scale_x = canvas_width_m / w
        scale_y = canvas_height_m / h
        scale = min(scale_x, scale_y)

        canvas_width_m = w * scale
        canvas_height_m = h * scale

    scaled_contours = []
    for cnt in contours:
        cnt_scaled = (cnt - np.array([x, y])) * scale
        scaled_contours.append(cnt_scaled)

    return scaled_contours, (canvas_width_m, canvas_height_m)


# 사용 예시:
image_path = (
    "./notebook/drawing/image/cute_carrot.png"  # 스케치 이미지 경로 (ex: PNG 파일)
)
img = mpimg.imread(image_path)
scale = 0.9

# 이미지 표시
# plt.imshow(img)
# plt.axis("off")
# plt.show()

scaled_contours, canvas_size = get_scaled_sketch_coordinates(
    image_path, canvas_width * scale, canvas_height * scale
)
print(f"canvas_size: {canvas_size}\n scaled_contours: {len(scaled_contours)}")


def interpolate_contour(contour, num_points_between=5):
    """
    Interpolate points between existing contour points for smoother drawing.

    Args:
        contour: numpy array of shape (N, 2) containing contour points
        num_points_between: number of points to insert between each pair

    Returns:
        interpolated contour with more points
    """
    interpolated = []
    for i in range(len(contour)):
        current = contour[i]
        next_point = contour[(i + 1) % len(contour)]

        # Add current point
        interpolated.append(current)

        # Add interpolated points (except for last segment to avoid duplication)
        if i < len(contour) - 1:
            for j in range(1, num_points_between + 1):
                t = j / (num_points_between + 1)
                interp_point = current + t * (next_point - current)
                interpolated.append(interp_point)

    return np.array(interpolated)


converted_contours = [cnt.reshape(-1, 2) for cnt in scaled_contours]
things_to_draw = []
for cnt in converted_contours:
    # Interpolate to add more points for smoother drawing (increased for precision)
    cnt = interpolate_contour(cnt, num_points_between=2)

    cnt[:, 0] = cnt[:, 0] - np.array([canvas_size[0] / 2])
    cnt[:, 1] = (cnt[:, 1] - np.array([canvas_size[1] / 2])) * (
        -1
    )  # image coordinate(*-1)
    z = np.full((cnt.shape[0], 1), appropriate_z_offset)
    result = np.concatenate((cnt, z), axis=1)
    things_to_draw.append(result)

# visualization
# plt.figure(figsize=(6, 6))
# for cnt in things_to_draw:
#     plt.plot(cnt[:, 0], cnt[:, 1], "b-")
# plt.gca().set_aspect("equal", adjustable="box")
# plt.xlabel("X (m)")
# plt.ylabel("Y (m)")
# plt.title("Scaled Sketch Coordinates")
# plt.show()


openmanipulator_path = "asset/omx/scene_omx_f_drawing.xml"
env = MuJoCoParserClass(name="omx", rel_xml_path=openmanipulator_path, verbose=False)
# env.HZ = 200
# joint_names = ["joint1","joint2","joint3","joint4","gripper_crank_joint"]
joint_names = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
]  # , "gripper_joint_1"]

q0 = np.array([0, 0, 0, 0, 0.2240])
p0 = env.get_p_body(body_name="end_effector_target")
p0[0] = env.get_p_body("obj_board")[0]

R0 = rpy_deg2r([0, 0, 0])

env.reset()  # reset
env.forward(q=q0, joint_names=joint_names)  # initial position
end_effector_pos = env.get_p_body("end_effector_target")

# Loop
q_ik_init = q0.copy()
board_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "obj_board_geom")
qpos_to_draw = []
qpos_to_draw.append(q_ik_init[0:4].tolist())
for i in range(len(things_to_draw)):
    cnt = things_to_draw[i]  # np.array for each contour
    cnt = cnt + p0
    j = 0
    if i == 0:
        _, traj_smt = solve_ik_and_interpolate(
            env=env,
            joint_names_for_ik=joint_names[0:4],
            body_name_trgt="end_effector_target",
            p_trgt=cnt[0],
            R_trgt=None,
            max_ik_tick=1000,
            ik_err_th=1e-4,
            restore_state=True,
        )
        qpos_to_draw.extend(traj_smt.tolist())
        env.forward(q=qpos_to_draw[-1], joint_names=joint_names[0:4])
    for j in range(cnt.shape[0]):
        qpos, ik_err_stack, ik_info = solve_ik(
            env=env,
            joint_names_for_ik=joint_names[0:4],
            body_name_trgt="end_effector_target",
            p_trgt=cnt[j, :],
            R_trgt=None,
            max_ik_tick=10000,
            ik_stepsize=1.0,
            ik_eps=1e-5,
            ik_th=np.radians(3.0),
            render=False,
            verbose_warning=False,
            restore_state=False,
        )
        for l in range(3):  # Increased for slower, more precise drawing
            qpos_to_draw.append(qpos.tolist())
        env.forward(q=qpos, joint_names=joint_names[0:4])
        if j == cnt.shape[0] - 1 and i != len(things_to_draw) - 1:
            current_point = cnt[j]
            next_cnt = things_to_draw[i + 1] + p0
            goal_point = next_cnt[0]
            vec = goal_point - current_point
            for interp1 in range(5):
                qpos, ik_err_stack, ik_info = solve_ik(
                    env=env,
                    joint_names_for_ik=joint_names[0:4],
                    body_name_trgt="end_effector_target",
                    p_trgt=current_point + np.array([0, 0, 0.05]) * (interp1) / 5,
                    R_trgt=None,
                    max_ik_tick=1000,
                    ik_stepsize=1.0,
                    ik_eps=1e-6,
                    ik_th=np.radians(1.0),
                    render=False,
                    verbose_warning=False,
                    restore_state=False,
                )
                for l in range(3):
                    qpos_to_draw.append(qpos.tolist())
                env.forward(q=qpos, joint_names=joint_names[0:4])
            for interp2 in range(5):
                qpos, ik_err_stack, ik_info = solve_ik(
                    env=env,
                    joint_names_for_ik=joint_names[0:4],
                    body_name_trgt="end_effector_target",
                    p_trgt=current_point + np.array([0, 0, 0.05]) + vec * interp2 / 10,
                    R_trgt=None,
                    max_ik_tick=1000,
                    ik_stepsize=1.0,
                    ik_eps=1e-6,
                    ik_th=np.radians(1.0),
                    render=False,
                    verbose_warning=False,
                    restore_state=False,
                )
                for l in range(1):
                    qpos_to_draw.append(qpos.tolist())
                env.forward(q=qpos, joint_names=joint_names[0:4])
            for interp3 in range(5):
                qpos, ik_err_stack, ik_info = solve_ik(
                    env=env,
                    joint_names_for_ik=joint_names[0:4],
                    body_name_trgt="end_effector_target",
                    p_trgt=goal_point + np.array([0, 0, 0.05]) * (5 - interp3) / 5,
                    R_trgt=None,
                    max_ik_tick=1000,
                    ik_stepsize=1.0,
                    ik_eps=1e-5,
                    ik_th=np.radians(5.0),
                    render=False,
                    verbose_warning=False,
                    restore_state=False,
                )
                for l in range(3):
                    qpos_to_draw.append(qpos.tolist())
                env.forward(q=qpos, joint_names=joint_names[0:4])
_, traj_smt = solve_ik_and_interpolate(
    env=env,
    joint_names_for_ik=joint_names[0:4],
    body_name_trgt="end_effector_target",
    p_trgt=end_effector_pos,
    R_trgt=None,
    max_ik_tick=1000,
    ik_err_th=1e-4,
    restore_state=True,
)
print(f"qpos_to_draw: {np.shape(qpos_to_draw)}")
print(f"traj_smt: {np.shape(traj_smt)}")
qpos_to_draw = np.vstack((np.array(qpos_to_draw), traj_smt))

# texture options
texture_scale = 5000
texture_width, texture_height = int(canvas_width * texture_scale), int(
    canvas_height * texture_scale
)
print(f"texture width: {texture_width}, height: {texture_height}")
background_color = (255, 255, 255)

img = Image.new("RGB", (texture_width, texture_height), color=background_color)
draw = ImageDraw.Draw(img)

img.save("asset/omx/assets/drawing_board_template.png")


def update_texture(env, img_array, texid, expected_len, tex_adr):

    # image array → flat RGB byte array (bottom-up)
    flat = np.flipud(img_array).astype(np.uint8).flatten()

    # update texture data
    env.model.tex_data[tex_adr : tex_adr + expected_len] = flat
    mujoco.mjr_uploadTexture(env.model, env.viewer.ctx, texid)


def world_to_texture(
    x,
    y,
    canvas_x=0.3,
    canvas_y=0.0,
    canvas_width=canvas_width,
    canvas_height=canvas_height,
    tex_width=texture_width,
    tex_height=texture_height,
):
    """
    Global world (x, y) → texture image (u, v) index
    Corrected for y-axis inversion between world and image space.
    """
    # 기준점
    min_x = canvas_x - canvas_width / 2
    min_y = canvas_y - canvas_height / 2

    # 비율계산 (0 ~ 1)
    rel_x = (x - min_x) / canvas_width
    rel_y = (y - min_y) / canvas_height

    # 이미지 좌표계는 y 증가가 아래쪽이므로 반전 필요
    u = int(rel_x * tex_width)
    # u = int((1 - rel_x) * tex_width)  # 좌우 반전
    v = int((1 - rel_y) * tex_height)

    return u, v


def safe_draw(
    img_array, x, y, board_x, board_y, tex_width, tex_height, color=(255, 0, 0)
):
    u, v = world_to_texture(
        x, y, board_x, board_y, tex_width=tex_width, tex_height=tex_height
    )
    if 0 <= u < tex_width and 0 <= v < tex_height:
        img_array[v, u, :] = color
    return img_array


openmanipulator_path = "asset/omx/scene_omx_f_drawing.xml"
env = MuJoCoParserClass(name="omx", rel_xml_path=openmanipulator_path, verbose=False)
if REAL_ROBOT:
    controller = OMX_Controller()
    controller.DXL_ID = [11, 12, 13, 14, 15]
    print("controller.DEVICENAME: ", controller.DEVICENAME)
    controller.BAUDRATE = 1000000
    controller.initialize()

    # Move robot to initial position
    print("Moving robot to initial position...")
    ranges = env.ctrl_ranges
    initial_qpos = np.append(q0[0:4], q0[-1])
    initial_qpos_bytes = rads_to_bytes(initial_qpos, ranges)

texid = None
for i in range(env.model.ntex):
    name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_TEXTURE, i)
    if name == "drawing_tex":
        texid = i
        tex_width = env.model.tex_width[i]
        tex_height = env.model.tex_height[i]
        tex_channel = env.model.tex_nchannel[i]
        tex_adr = env.model.tex_adr[i]
        img_array = (
            np.ones((tex_width, tex_height, 3), dtype=np.uint8) * 255
        )  # 하얀 배경

        # sanity check
        expected_len = tex_width * tex_height * tex_channel
        assert env.model.tex_data.shape[0] >= tex_adr + expected_len

        print(f"drawing texture texid: {texid}, tex_wd: {tex_width} x {tex_height}")
        break

idx = 0

env.init_viewer(
    title="Drawing",
    transparent=False,
    azimuth=120,
    distance=0.4,
    elevation=-42.4,
    lookat=(0.3, 0.0, 0.005),
    use_rgb_overlay=True,
    loc_rgb_overlay="bottom right",
)
print(
    f"window renderer: {glGetString(GL_RENDERER).decode()}\nwindow vendor: {glGetString(GL_VENDOR).decode()}"
)
env.reset()  # reset
env.forward(q=q0, joint_names=joint_names)  # initial position
board_pos = env.data.xpos[
    mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "obj_board")
]

# Create progress bar
pbar = tqdm(total=len(qpos_to_draw), desc="Drawing progress", unit="step")

# Start timer for drawing
drawing_start_time = time.time()

step_size = 1
while env.is_viewer_alive():
    # Check if drawing is complete
    if idx >= len(qpos_to_draw):
        break  # Exit the loop when all positions are drawn

    qpos = qpos_to_draw[idx]
    # update real robot
    if REAL_ROBOT:
        controller.read()
        ranges = env.ctrl_ranges
        # print(f"ranges: {ranges}\ncurrent qpos: {controller.dxl_present_position}")
        qpos_to_go = np.append(qpos[0:4], q0[-1])
        qpos_bytes = rads_to_bytes(
            qpos_to_go,
            ranges,
        )
        # controller.safety(controller.dxl_present_position,qpos_bytes,controller.DXL_ID)
        controller.run_multi_motor(qpos_bytes, controller.DXL_ID)

        # Add delay for precise, slow drawing
        # time.sleep(0.005)  # delay per step for precision

    # update simulation
    if REAL_ROBOT:
        controller.read()
        current_qpos_bytes = bytes_to_rads(
            controller.dxl_present_position,
        )
        # current_qpos_bytes[0:4] += 0.02  # sim2real gap
        env.step(
            ctrl=current_qpos_bytes,
            joint_names=joint_names,
            nstep=step_size,
        )
    else:
        if idx >= len(qpos_to_draw):
            print(f"qpos: {qpos}")
        env.step(
            ctrl=np.append(qpos[0:4], q0[-1]),
            joint_names=joint_names,
            nstep=step_size,
        )

    if env.loop_every(tick_every=step_size):
        img_array = np.flipud(
            env.model.tex_data[tex_adr : tex_adr + expected_len].reshape(
                (tex_height, tex_width, tex_channel)
            )
        )

        end_effector_pos = env.data.xpos[
            mujoco.mj_name2id(
                env.model, mujoco.mjtObj.mjOBJ_BODY, "end_effector_target"
            )
        ]

        d1 = mujoco.mj_rayMesh(
            env.model, env.data, board_id, end_effector_pos, np.array([0, 0, 1])
        )
        d2 = mujoco.mj_rayMesh(
            env.model, env.data, board_id, end_effector_pos, np.array([0, 0, -1])
        )
        # print(f"d1: {d1}, d2: {d2}")
        if REAL_ROBOT:
            d = 0.002 + sim2real_gap
        else:
            d = 0.002
        if -1 < d1 < d or -1 < d2 < d:
            x, y, _ = end_effector_pos
            img_array = safe_draw(
                img_array, x, y, board_pos[0], board_pos[1], tex_width, tex_height
            )
            env.viewer.add_rgb_overlay(rgb_img_raw=img_array, fix_ratio=True)
            update_texture(env, img_array, texid, expected_len, tex_adr)
    if env.loop_every(tick_every=step_size):  # tick_every=1 ~ real time

        env.plot_time()
        env.render()

    # Update progress bar
    pbar.update(1)
    idx += 1

# Close progress bar
pbar.close()

# Calculate and display total drawing time
drawing_end_time = time.time()
total_drawing_time = drawing_end_time - drawing_start_time
minutes = int(total_drawing_time // 60)
seconds = total_drawing_time % 60

print("=" * 50)
print("Drawing completed!")
print(f"Total drawing time: {minutes}m {seconds:.2f}s ({total_drawing_time:.2f}s)")
print(f"Total steps: {len(qpos_to_draw)}")
print(f"Average time per step: {total_drawing_time/len(qpos_to_draw)*1000:.2f}ms")
print("=" * 50)

img_array = np.flipud(
    env.model.tex_data[tex_adr : tex_adr + expected_len].reshape(
        (tex_height, tex_width, tex_channel)
    )
)
img = Image.fromarray(img_array)
display(img)
timestamp = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S")
result_dir = os.path.join(script_dir, "drawing", "result")
os.makedirs(result_dir, exist_ok=True)  # Ensure directory exists
filename = os.path.join(result_dir, f"{timestamp}.png")
img.save(filename)
print(f"Image saved to: {filename}")
# Close
env.close_viewer()
