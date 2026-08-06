# -*- coding: utf-8 -*-
"""
KITTI Raw Data 序列 —— 主前车筛选 + 双目测距 + 相对速度 + TTC/安全距离融合预警
================================================================================

本脚本在原“IOU 跟踪 + StereoSGBM 双目测距 + 一维卡尔曼滤波”基础上，针对
“相邻车道车辆误报、前车远离仍显示红框、测距暂时失效后红框滞留”等问题，加入：

1. 基于图像车道梯形 ROI 与横向坐标约束的主前车筛选；
2. 仅对主前车进行 TTC 与安全距离融合预警；
3. 仅当目标存在明显接近趋势时计算 TTC 并允许触发风险等级；
4. TTC <= 1.5 s 时直接告警；其余场景按安全距离和 TTC 分级；
5. 当前帧测距无效时清空该目标的风险状态，避免上一帧红框残留；
6. 使用 image_02/timestamps.txt 的真实相邻帧时间间隔更新卡尔曼滤波。

注意：
- 本程序仅用于公开数据集离线仿真验证，不用于真实车辆控制。
- 车道梯形 ROI 与横向阈值属于工程启发式参数，可根据视频效果进一步调节。

依赖：
    pip install ultralytics opencv-python numpy
"""

import os
import glob
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

# 串口通信：用于把预警等级发送给 STM32/Arduino
try:
    import serial
except ImportError:
    serial = None

#13测试变换车道前车检测，20窄路慢速对向来车
# ============================================================
# 1. 路径配置（保持原路径不变）
# ============================================================

BASE_DIR = r"D:\classDesign"

DATE = "2011_09_26"
SEQ_ID = "0056"

SEQ_NUM = str(int(SEQ_ID))

DRIVE_NAME = f"{DATE}_drive_{SEQ_ID}_sync"

# KITTI Raw 序列根目录
SEQ_ROOT = os.path.join(
    BASE_DIR,
    DRIVE_NAME,
    DATE,
    DRIVE_NAME
)

LEFT_IMAGE_DIR = os.path.join(
    SEQ_ROOT,
    "image_02",
    "data"
)

RIGHT_IMAGE_DIR = os.path.join(
    SEQ_ROOT,
    "image_03",
    "data"
)

LEFT_TIMESTAMP_PATH = os.path.join(
    SEQ_ROOT,
    "image_02",
    "timestamps.txt"
)

OXTS_DIR = os.path.join(
    SEQ_ROOT,
    "oxts",
    "data"
)

# 标定文件路径
CALIB_CAM_TO_CAM_PATH = os.path.join(
    BASE_DIR,
    f"{DATE}_calib_{SEQ_NUM}",
    DATE,
    "calib_cam_to_cam.txt"
)

USE_REAL_HOST_SPEED = True

MODEL_PATH = os.path.join(
    BASE_DIR,
    "yolov8n.pt"
)

# 保存结果目录
SAVE_DIR = os.path.join(
    BASE_DIR,
    "fcw_tracking_results",
    f"drive_{SEQ_ID}"
)

os.makedirs(SAVE_DIR, exist_ok=True)

# 输出视频自动带序列号，避免覆盖
OUTPUT_VIDEO_PATH = os.path.join(
    SAVE_DIR,
    f"fcw_demo_lead_fusion_{SEQ_ID}.mp4"
)

print("当前测试序列：", DRIVE_NAME)
print("左目图像路径：", LEFT_IMAGE_DIR)
print("右目图像路径：", RIGHT_IMAGE_DIR)
print("时间戳路径：", LEFT_TIMESTAMP_PATH)
print("标定文件路径：", CALIB_CAM_TO_CAM_PATH)
print("OXTS 路径：", OXTS_DIR)
print("输出视频路径：", OUTPUT_VIDEO_PATH)

# ============================================================
# 1.1 串口配置：根据电脑设备管理器里的端口号修改 SERIAL_PORT
# ============================================================
SERIAL_ENABLE = True
SERIAL_PORT = "COM3"       # 例如 COM3/COM4/COM6，以设备管理器为准
SERIAL_BAUDRATE = 115200
SERIAL_SEND_EVERY_N_FRAMES = 1  # 每帧发送一次；KITTI 约 10Hz，可保持 1


# ============================================================
# 2. 算法参数配置
# ============================================================

CONF_THRESHOLD = 0.35
TARGET_NAMES = {"car", "truck", "bus"}
PEDESTRIAN_NAMES = {"person"}  # 仅显示，不作为当前“主前车”候选

# --- StereoSGBM + ROI 双目测距参数（保持原设置不变） ---
NUM_DISPARITIES = 192
BLOCK_SIZE = 7
ROI_PROFILE = (0.20, 0.20, 0.25, 0.15)
MIN_BOX_WIDTH = 18
MIN_BOX_HEIGHT = 18
MIN_VALID_DISPARITY = 1.0
MIN_VALID_PIXELS = 35
MIN_DISTANCE_M = 2.0
MAX_DISTANCE_M = 80.0

# --- IOU 跟踪参数 ---
TRACK_IOU_THRESHOLD = 0.3
TRACK_MAX_MISSED = 5

# 时间戳缺失、异常时的回退帧间隔
FRAME_INTERVAL_SEC = 0.1

# --- 安全距离 / TTC 参数 ---
FALLBACK_HOST_SPEED_KMH = 40.0
FALLBACK_HOST_SPEED_MS = FALLBACK_HOST_SPEED_KMH / 3.6
REACTION_TIME_SEC = 1.5
MAX_BRAKE_DECEL = 6.0

TTC_DANGER_THRESHOLD = 1.5
TTC_CAUTION_THRESHOLD = 2.5
TTC_NOTICE_THRESHOLD = 5.0
MIN_CLOSING_SPEED = 0.3  # m/s，小于该值认为目标未明显接近

# 仅在该距离范围内的主前车参与 TTC / 安全距离预警
FCW_MIN_DISTANCE_M = 2.0
FCW_MAX_DISTANCE_M = 35.0

# --- 主前车筛选参数：可根据视频效果微调 ---
# 车道梯形由近端较宽、远端较窄的图像区域近似表示自车所在车道。
LANE_TOP_Y_RATIO = 0.48
LANE_BOTTOM_Y_RATIO = 1.0
LANE_TOP_HALF_WIDTH_RATIO = 0.02
LANE_BOTTOM_HALF_WIDTH_RATIO = 0.18
LANE_CENTER_OFFSET_RATIO = -0.01

# 由 X=(u-cx)Z/fx 得到的横向位置约束，取半车道宽附近。
LEAD_MAX_LATERAL_M = 1.4


# ============================================================
# 3. 数据结构
# ============================================================

@dataclass
class Track:
    track_id: int
    bbox: Tuple[float, float, float, float]
    class_name: str
    missed_frames: int = 0
    updated_this_frame: bool = False

    # 卡尔曼滤波状态：[距离, 距离变化率]
    kf_state: Optional[np.ndarray] = None
    kf_cov: Optional[np.ndarray] = None

    last_raw_distance: Optional[float] = None
    smoothed_distance: Optional[float] = None
    relative_speed: Optional[float] = None  # 正值：距离缩短，正在接近
    ttc: Optional[float] = None
    warning_level: str = "NONE"

    # 主前车筛选状态
    has_valid_distance: bool = False
    in_lane_roi: bool = False
    lateral_x_m: Optional[float] = None
    is_lead: bool = False


# ============================================================
# 4. KITTI Raw 文件解析
# ============================================================

def parse_raw_calib(calib_path: str):
    """解析 calib_cam_to_cam.txt，返回 fx、fy、cx、cy 与有效双目基线。"""
    params = {}
    with open(calib_path, "r", encoding="utf-8") as f:
        for line in f:
            if ":" not in line:
                continue
            key, value = line.strip().split(":", 1)
            try:
                vals = np.array([float(v) for v in value.strip().split()], dtype=np.float64)
            except ValueError:
                continue
            params[key.strip()] = vals

    if "P_rect_02" not in params or "P_rect_03" not in params:
        raise ValueError(f"标定文件缺少 P_rect_02/P_rect_03：{calib_path}")

    p2 = params["P_rect_02"].reshape(3, 4)
    p3 = params["P_rect_03"].reshape(3, 4)

    fx = float(p2[0, 0])
    fy = float(p2[1, 1])
    cx = float(p2[0, 2])
    cy = float(p2[1, 2])

    c_left = -p2[0, 3] / p2[0, 0]
    c_right = -p3[0, 3] / p3[0, 0]
    baseline = float(abs(c_right - c_left))

    return fx, fy, cx, cy, baseline


def load_timestamps(timestamp_path: str) -> List[datetime]:
    """读取 KITTI Raw 图像时间戳；纳秒部分截断至 datetime 支持的微秒精度。"""
    timestamps: List[datetime] = []
    with open(timestamp_path, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            if "." in text:
                main_part, frac_part = text.split(".", 1)
                text = f"{main_part}.{frac_part[:6].ljust(6, '0')}"
            timestamps.append(datetime.strptime(text, "%Y-%m-%d %H:%M:%S.%f"))
    return timestamps


def parse_oxts_speed(oxts_path: str) -> Optional[float]:
    """读取 OXTS 第 9 个字段 vf，单位为 m/s。"""
    if not os.path.exists(oxts_path):
        return None
    try:
        with open(oxts_path, "r", encoding="utf-8") as f:
            values = f.readline().strip().split()
        if len(values) < 9:
            return None
        return float(values[8])
    except (ValueError, OSError, IndexError):
        return None


# ============================================================
# 5. StereoSGBM 视差与单目标距离估计
# ============================================================

def create_stereo_matcher(image_width: int):
    max_allowed = max(16, ((image_width // 16) - 1) * 16)
    num_disparities = min(NUM_DISPARITIES, max_allowed)

    return cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=BLOCK_SIZE,
        P1=8 * BLOCK_SIZE * BLOCK_SIZE,
        P2=32 * BLOCK_SIZE * BLOCK_SIZE,
        disp12MaxDiff=1,
        preFilterCap=31,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def compute_disparity(left_img, right_img, matcher, clahe):
    left_gray = clahe.apply(cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY))
    right_gray = clahe.apply(cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY))
    disparity = matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0
    disparity[disparity <= 0] = np.nan
    return disparity


def estimate_distance(disparity, bbox, fx: float, baseline: float) -> Optional[float]:
    """固定中心 ROI + 有效视差筛选 + IQR + 中值视差的双目距离估计。"""
    img_h, img_w = disparity.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    x1, x2 = np.clip([x1, x2], 0, img_w - 1)
    y1, y2 = np.clip([y1, y2], 0, img_h - 1)

    box_w, box_h = x2 - x1, y2 - y1
    if box_w < MIN_BOX_WIDTH or box_h < MIN_BOX_HEIGHT:
        return None

    crop_l, crop_r, crop_t, crop_b = ROI_PROFILE
    rx1 = int(np.clip(x1 + crop_l * box_w, 0, img_w - 1))
    rx2 = int(np.clip(x2 - crop_r * box_w, 0, img_w - 1))
    ry1 = int(np.clip(y1 + crop_t * box_h, 0, img_h - 1))
    ry2 = int(np.clip(y2 - crop_b * box_h, 0, img_h - 1))

    if rx2 <= rx1 or ry2 <= ry1:
        return None

    roi_disp = disparity[ry1:ry2, rx1:rx2]
    valid_disp = roi_disp[np.isfinite(roi_disp) & (roi_disp > MIN_VALID_DISPARITY)]
    if len(valid_disp) < MIN_VALID_PIXELS:
        return None

    q1, q3 = np.percentile(valid_disp, [25, 75])
    iqr = q3 - q1
    lower = max(MIN_VALID_DISPARITY, q1 - 1.5 * iqr)
    upper = q3 + 1.5 * iqr
    filtered = valid_disp[(valid_disp >= lower) & (valid_disp <= upper)]
    if len(filtered) < MIN_VALID_PIXELS:
        filtered = valid_disp

    disparity_median = float(np.median(filtered))
    if disparity_median <= 0:
        return None

    distance = fx * baseline / disparity_median
    if not (MIN_DISTANCE_M <= distance <= MAX_DISTANCE_M):
        return None
    return distance


# ============================================================
# 6. 简单 IOU 跟踪器
# ============================================================

def compute_iou(box_a, box_b) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class SimpleIOUTracker:
    """基于 IOU 的轻量级多目标跟踪器。"""

    def __init__(self):
        self.tracks: Dict[int, Track] = {}
        self.next_id = 1

    def update(self, detections: List[Dict]) -> List[Track]:
        # 每帧先将旧轨迹标记为“未被当前检测更新”。
        for track in self.tracks.values():
            track.updated_this_frame = False

        matched_track_ids = set()
        used_dets = set()

        candidate_pairs = []
        for tid, track in self.tracks.items():
            for di, det in enumerate(detections):
                iou = compute_iou(track.bbox, det["bbox"])
                if iou >= TRACK_IOU_THRESHOLD:
                    candidate_pairs.append((iou, tid, di))

        candidate_pairs.sort(key=lambda item: item[0], reverse=True)

        for _, tid, di in candidate_pairs:
            if tid in matched_track_ids or di in used_dets:
                continue
            track = self.tracks[tid]
            track.bbox = detections[di]["bbox"]
            track.class_name = detections[di]["class_name"]
            track.missed_frames = 0
            track.updated_this_frame = True
            matched_track_ids.add(tid)
            used_dets.add(di)

        # 未匹配检测框新建轨迹，并标记为当前帧已更新。
        for di, det in enumerate(detections):
            if di in used_dets:
                continue
            self.tracks[self.next_id] = Track(
                track_id=self.next_id,
                bbox=det["bbox"],
                class_name=det["class_name"],
                updated_this_frame=True,
            )
            self.next_id += 1

        # 仅对当前帧未更新的旧轨迹累计丢失帧数。
        for tid in list(self.tracks.keys()):
            track = self.tracks[tid]
            if not track.updated_this_frame:
                track.missed_frames += 1
            if track.missed_frames > TRACK_MAX_MISSED:
                del self.tracks[tid]

        return list(self.tracks.values())


# ============================================================
# 7. 一维卡尔曼滤波（状态：[距离，距离变化率]）
# ============================================================

def kf_init(distance: float) -> Tuple[np.ndarray, np.ndarray]:
    state = np.array([[distance], [0.0]], dtype=np.float64)
    cov = np.eye(2, dtype=np.float64) * 5.0
    return state, cov


def kf_predict_update(state: np.ndarray, cov: np.ndarray, measured_distance: float, dt: float):
    """匀速模型卡尔曼滤波：状态=[距离, 距离变化率]。"""
    F = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float64)
    Q = np.array([[0.05, 0.0], [0.0, 0.5]], dtype=np.float64)
    H = np.array([[1.0, 0.0]], dtype=np.float64)
    R = np.array([[0.3]], dtype=np.float64)

    state_pred = F @ state
    cov_pred = F @ cov @ F.T + Q

    z = np.array([[measured_distance]], dtype=np.float64)
    innovation = z - H @ state_pred
    innovation_cov = H @ cov_pred @ H.T + R
    gain = cov_pred @ H.T @ np.linalg.inv(innovation_cov)

    state_new = state_pred + gain @ innovation
    cov_new = (np.eye(2) - gain @ H) @ cov_pred
    return state_new, cov_new


# ============================================================
# 8. 主前车筛选、TTC 与安全距离融合预警
# ============================================================

def compute_safe_distance(v_host: float) -> float:
    """最小制动安全距离：D1+D2+D3。"""
    return v_host * 1.03 + (v_host ** 2) / 14.715


def get_lane_polygon(image_width: int, image_height: int) -> np.ndarray:
    """根据图像尺寸生成自车道近似梯形 ROI。"""
    center_x = image_width / 2.0
    y_top = int(image_height * LANE_TOP_Y_RATIO)
    y_bottom = int(image_height * LANE_BOTTOM_Y_RATIO)
    top_half_width = int(image_width * LANE_TOP_HALF_WIDTH_RATIO)
    bottom_half_width = int(image_width * LANE_BOTTOM_HALF_WIDTH_RATIO)

    return np.array([
        [int(center_x - top_half_width), y_top],
        [int(center_x + top_half_width), y_top],
        [int(center_x + bottom_half_width), y_bottom],
        [int(center_x - bottom_half_width), y_bottom],
    ], dtype=np.int32)


def bbox_bottom_center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """返回检测框底边中心点，用于判断车辆是否位于本车道区域。"""
    x1, _, x2, y2 = bbox
    return (x1 + x2) / 2.0, y2


def is_point_in_lane_roi(point: Tuple[float, float], lane_polygon: np.ndarray) -> bool:
    """点位于梯形内部或边界上时返回 True。"""
    return cv2.pointPolygonTest(lane_polygon.astype(np.float32), point, False) >= 0


def compute_lateral_position(u_center: float, distance: float, fx: float, cx: float) -> float:
    """由图像横坐标、深度和标定参数估计目标横向位置 X。"""
    return (u_center - cx) * distance / fx


def select_lead_vehicle(
    tracks: List[Track],
    lane_polygon: np.ndarray,
    fx: float,
    cx: float,
) -> Optional[Track]:
    """
    主前车筛选：
    1) 仅使用当前帧检测更新且有有效双目距离的车辆；
    2) 车辆底边中心需落在自车道梯形 ROI 内；
    3) 横向位置需满足 |X| <= LEAD_MAX_LATERAL_M；
    4) 在候选中选择纵向距离最小者。
    """
    candidates: List[Track] = []

    for track in tracks:
        track.is_lead = False
        track.in_lane_roi = False
        track.lateral_x_m = None

        if not track.updated_this_frame:
            continue
        if not track.has_valid_distance or track.smoothed_distance is None:
            continue
        if track.class_name not in TARGET_NAMES:
            continue
        if not (FCW_MIN_DISTANCE_M <= track.smoothed_distance <= FCW_MAX_DISTANCE_M):
            continue

        u_center, v_bottom = bbox_bottom_center(track.bbox)
        in_lane = is_point_in_lane_roi((u_center, v_bottom), lane_polygon)
        lateral_x = compute_lateral_position(u_center, track.smoothed_distance, fx, cx)

        track.in_lane_roi = in_lane
        track.lateral_x_m = lateral_x

        if in_lane and abs(lateral_x) <= LEAD_MAX_LATERAL_M:
            candidates.append(track)

    if not candidates:
        return None

    lead = min(candidates, key=lambda track: track.smoothed_distance)
    lead.is_lead = True
    return lead


def determine_warning_level(
    distance: float,
    relative_speed: float,
    d_safe: float,
) -> Tuple[str, Optional[float]]:
    """
    主前车风险判定。

    规则：
    - 目标未明显接近：SAFE，不计算 TTC；
    - TTC <= 1.5 s：DANGER；
    - TTC <= 2.5 s 或距离不大于安全距离：CAUTION；
    - TTC <= 5.0 s：NOTICE；
    - 其他：SAFE。

    安全距离条件只在目标存在明显接近趋势时生效，避免前车远离仍触发红框。
    """
    if relative_speed <= MIN_CLOSING_SPEED:
        return "SAFE", None

    ttc = distance / relative_speed

    if ttc <= TTC_DANGER_THRESHOLD:
        return "DANGER", ttc
    if ttc <= TTC_CAUTION_THRESHOLD:
        return "CAUTION", ttc
    if distance <= d_safe:
        return "CAUTION", ttc
    if ttc <= TTC_NOTICE_THRESHOLD:
        return "NOTICE", ttc
    return "SAFE", ttc



# ============================================================
# 8.1 串口下发：电脑端只发送预警结果，单片机只负责声光执行
# ============================================================

WARNING_LEVEL_TO_CODE = {
    "NONE": 0,
    "SAFE": 1,
    "NOTICE": 2,
    "CAUTION": 3,
    "DANGER": 4,
}


def open_serial_port():
    """打开串口；失败时返回 None，不影响离线视频生成。"""
    if not SERIAL_ENABLE:
        print("[串口] SERIAL_ENABLE=False，未启用单片机通信。")
        return None
    if serial is None:
        print("[串口] 未安装 pyserial，请先执行：pip install pyserial")
        return None
    try:
        ser = serial.Serial(SERIAL_PORT, SERIAL_BAUDRATE, timeout=0.02)
        print(f"[串口] 已打开 {SERIAL_PORT}, baud={SERIAL_BAUDRATE}")
        return ser
    except Exception as exc:
        print(f"[串口] 打开失败：{SERIAL_PORT}，原因：{exc}")
        print("[串口] 程序将继续生成视频，但不会驱动单片机。")
        return None


def build_serial_packet(lead_track: Optional[Track]) -> str:
    """
    串口协议：level,distance_cm,ttc_x10\n
    level：0 NONE, 1 SAFE, 2 NOTICE, 3 CAUTION, 4 DANGER
    distance_cm：主前车距离，单位 cm；无主前车为 0
    ttc_x10：TTC*10，例如 1.5s 发送 15；无 TTC 时发送 999
    """
    if lead_track is None:
        return "0,0,999\n"

    level_code = WARNING_LEVEL_TO_CODE.get(lead_track.warning_level, 0)
    if lead_track.smoothed_distance is None:
        distance_cm = 0
    else:
        distance_cm = int(round(lead_track.smoothed_distance * 100.0))
        distance_cm = max(0, min(9999, distance_cm))

    if lead_track.ttc is None:
        ttc_x10 = 999
    else:
        ttc_x10 = int(round(lead_track.ttc * 10.0))
        ttc_x10 = max(0, min(999, ttc_x10))

    return f"{level_code},{distance_cm},{ttc_x10}\n"


def send_warning_to_mcu(ser, lead_track: Optional[Track]):
    """向单片机发送一帧预警状态。"""
    if ser is None:
        return
    packet = build_serial_packet(lead_track)
    try:
        ser.write(packet.encode("ascii"))
    except Exception as exc:
        print(f"[串口] 发送失败：{exc}")

# ============================================================
# 9. 可视化
# ============================================================

LEVEL_COLORS = {
    "SAFE": (0, 200, 0),
    "NOTICE": (255, 180, 0),
    "CAUTION": (0, 200, 255),
    "DANGER": (0, 0, 255),
    "NONE": (180, 180, 180),
}


def draw_tracks(
    frame: np.ndarray,
    tracks: List[Track],
    d_safe: float,
    v_host: float,
    used_fallback: bool,
    lane_polygon: np.ndarray,
    lead_track: Optional[Track],
) -> np.ndarray:
    """绘制车道 ROI、主前车标识、距离、相对速度、TTC 和告警等级。"""
    vis = frame.copy()

    # 绘制主车道近似梯形。
    cv2.polylines(vis, [lane_polygon.reshape((-1, 1, 2))], True, (255, 255, 0), 1, cv2.LINE_AA)

    source_text = "FALLBACK" if used_fallback else "OXTS GPS/IMU"
    lead_text = f"Lead ID: {lead_track.track_id}" if lead_track is not None else "Lead ID: NONE"
    cv2.putText(
        vis,
        f"Host: {v_host * 3.6:.1f} km/h [{source_text}] | Safe: {d_safe:.1f} m | {lead_text}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    for track in tracks:
        x1, y1, x2, y2 = map(int, track.bbox)

        # 非当前帧更新目标、无有效测距目标或非主前车均不显示风险色。
        if track.is_lead and track.has_valid_distance:
            color = LEVEL_COLORS.get(track.warning_level, LEVEL_COLORS["NONE"])
        else:
            color = LEVEL_COLORS["NONE"]

        thickness = 3 if track.is_lead else 1
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)

        role = "LEAD" if track.is_lead else "OTHER"
        if not track.updated_this_frame:
            role = "LOST"

        if track.has_valid_distance and track.smoothed_distance is not None:
            speed_text = (
                f"vrel={track.relative_speed:.2f}m/s"
                if track.relative_speed is not None else "vrel=NA"
            )
            ttc_text = f"TTC={track.ttc:.2f}s" if track.ttc is not None else "TTC=NA"
            x_text = f"X={track.lateral_x_m:.2f}m" if track.lateral_x_m is not None else "X=NA"
            label_1 = f"ID{track.track_id} {track.class_name} {track.smoothed_distance:.1f}m [{role}]"
            label_2 = f"{speed_text} {ttc_text} [{track.warning_level}] {x_text}"
        else:
            label_1 = f"ID{track.track_id} {track.class_name} [DIST=NA] [{role}]"
            label_2 = "[NONE]"

        cv2.putText(
            vis,
            label_1,
            (x1, max(20, y1 - 28)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            label_2,
            (x1, max(40, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )

    return vis


# ============================================================
# 10. 主流程
# ============================================================

def main():
    for path in [LEFT_IMAGE_DIR, RIGHT_IMAGE_DIR, CALIB_CAM_TO_CAM_PATH]:
        if not os.path.exists(path):
            print(f"[错误] 路径不存在：{path}")
            return
    if not os.path.exists(MODEL_PATH):
        print(f"[错误] 模型文件不存在：{MODEL_PATH}")
        return

    fx, fy, cx, cy, baseline = parse_raw_calib(CALIB_CAM_TO_CAM_PATH)
    print(f"[标定参数] fx={fx:.2f}px, cx={cx:.2f}px, baseline={baseline:.4f}m")

    if USE_REAL_HOST_SPEED:
        if not os.path.exists(OXTS_DIR):
            print(f"[警告] OXTS_DIR 路径不存在：{OXTS_DIR}，将回退使用固定自车速度。")
        print(
            f"[安全距离] 使用 OXTS vf；缺失帧回退至 {FALLBACK_HOST_SPEED_KMH:.1f} km/h"
        )
    else:
        print(f"[安全距离] 使用固定自车速度：{FALLBACK_HOST_SPEED_KMH:.1f} km/h")

    left_paths = sorted(glob.glob(os.path.join(LEFT_IMAGE_DIR, "*.png")))
    right_paths = sorted(glob.glob(os.path.join(RIGHT_IMAGE_DIR, "*.png")))
    if len(left_paths) == 0 or len(left_paths) != len(right_paths):
        print(f"[错误] 左右目图像数量不匹配或为空：left={len(left_paths)}, right={len(right_paths)}")
        return

    timestamps: Optional[List[datetime]] = None
    if os.path.exists(LEFT_TIMESTAMP_PATH):
        try:
            loaded = load_timestamps(LEFT_TIMESTAMP_PATH)
            if len(loaded) == len(left_paths):
                timestamps = loaded
                print(f"[时间戳] 成功读取 {len(timestamps)} 个时间戳，使用真实相邻帧间隔。")
            else:
                print(
                    f"[警告] 时间戳数量={len(loaded)}，图像数量={len(left_paths)}，"
                    "回退使用固定帧间隔。"
                )
        except (OSError, ValueError) as exc:
            print(f"[警告] 时间戳读取失败：{exc}，回退使用固定帧间隔。")
    else:
        print(f"[警告] 时间戳文件不存在：{LEFT_TIMESTAMP_PATH}，回退使用固定帧间隔。")

    print(f"[数据] 共 {len(left_paths)} 帧，开始处理……")
    print("[加载] YOLO 模型……")
    model = YOLO(MODEL_PATH)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    first_img = cv2.imread(left_paths[0])
    if first_img is None:
        print(f"[错误] 无法读取首帧：{left_paths[0]}")
        return

    frame_h, frame_w = first_img.shape[:2]
    matcher = create_stereo_matcher(frame_w)
    lane_polygon = get_lane_polygon(frame_w, frame_h)
    tracker = SimpleIOUTracker()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    if timestamps is not None and len(timestamps) > 1:
        average_dt = (timestamps[-1] - timestamps[0]).total_seconds() / (len(timestamps) - 1)
        video_fps = 1.0 / average_dt if average_dt > 0 else 1.0 / FRAME_INTERVAL_SEC
    else:
        video_fps = 1.0 / FRAME_INTERVAL_SEC

    writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, video_fps, (frame_w, frame_h))
    if not writer.isOpened():
        print(f"[错误] 无法创建视频文件：{OUTPUT_VIDEO_PATH}")
        return

    # 打开串口：如果失败，不影响离线视频运行。
    ser = open_serial_port()

    for frame_idx, (left_path, right_path) in enumerate(zip(left_paths, right_paths)):
        left_img = cv2.imread(left_path)
        right_img = cv2.imread(right_path)
        if left_img is None or right_img is None:
            print(f"[跳过] 读取图像失败：{os.path.basename(left_path)}")
            continue

        if timestamps is not None and frame_idx > 0:
            dt = (timestamps[frame_idx] - timestamps[frame_idx - 1]).total_seconds()
            if dt <= 0.0 or dt > 0.5:
                dt = FRAME_INTERVAL_SEC
        else:
            dt = FRAME_INTERVAL_SEC

        # --- OXTS 自车速度与安全距离 ---
        v_host: Optional[float] = None
        if USE_REAL_HOST_SPEED:
            frame_id = os.path.splitext(os.path.basename(left_path))[0]
            oxts_path = os.path.join(OXTS_DIR, frame_id + ".txt")
            v_host = parse_oxts_speed(oxts_path)

        used_fallback = v_host is None
        if used_fallback:
            v_host = FALLBACK_HOST_SPEED_MS
        d_safe = compute_safe_distance(v_host)

        # --- 车辆检测与 IOU 跟踪 ---
        disparity = compute_disparity(left_img, right_img, matcher, clahe)
        results = model(left_img, conf=CONF_THRESHOLD, verbose=False)[0]

        detections: List[Dict] = []
        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            class_name = results.names[cls_id]
            if class_name not in TARGET_NAMES and class_name not in PEDESTRIAN_NAMES:
                continue
            bbox = tuple(map(float, box.xyxy[0].tolist()))
            detections.append({"bbox": bbox, "class_name": class_name})

        tracks = tracker.update(detections)

        # --- 对当前检测更新到的轨迹进行距离滤波 ---
        for track in tracks:
            # 每帧重置瞬时测距和主前车状态，避免上一帧状态滞留。
            track.has_valid_distance = False
            track.is_lead = False
            track.in_lane_roi = False
            track.lateral_x_m = None
            track.ttc = None
            track.warning_level = "NONE"

            if not track.updated_this_frame:
                track.relative_speed = None
                continue

            raw_distance = estimate_distance(disparity, track.bbox, fx, baseline)
            if raw_distance is None:
                track.last_raw_distance = None
                track.relative_speed = None
                continue

            if track.kf_state is None or track.kf_cov is None:
                track.kf_state, track.kf_cov = kf_init(raw_distance)
            else:
                track.kf_state, track.kf_cov = kf_predict_update(
                    track.kf_state,
                    track.kf_cov,
                    raw_distance,
                    dt,
                )

            track.last_raw_distance = raw_distance
            track.smoothed_distance = float(track.kf_state[0, 0])
            # 距离缩短时状态速度为负，取相反数定义为正的相对接近速度。
            track.relative_speed = float(-track.kf_state[1, 0])
            track.has_valid_distance = True

        # --- 筛选主前车，仅主前车参与风险评估 ---
        lead_track = select_lead_vehicle(tracks, lane_polygon, fx, cx)

        if lead_track is not None:
            lead_track.warning_level, lead_track.ttc = determine_warning_level(
                lead_track.smoothed_distance,
                lead_track.relative_speed,
                d_safe,
            )

        # 下发给单片机：NONE/SAFE/NOTICE/CAUTION/DANGER -> 0/1/2/3/4
        if frame_idx % SERIAL_SEND_EVERY_N_FRAMES == 0:
            send_warning_to_mcu(ser, lead_track)

        vis_frame = draw_tracks(
            left_img,
            tracks,
            d_safe,
            v_host,
            used_fallback,
            lane_polygon,
            lead_track,
        )
        writer.write(vis_frame)
        cv2.imshow("FCW Realtime Display", vis_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if frame_idx % 1 == 0:
            lead_info = "NONE"
            if lead_track is not None:
                lead_info = (
                    f"ID{lead_track.track_id}, d={lead_track.smoothed_distance:.1f}m, "
                    f"vrel={lead_track.relative_speed:.2f}m/s, "
                    f"{lead_track.warning_level}"
                )
            print(
                f"  帧 {frame_idx}/{len(left_paths)} | dt={dt:.4f}s | "
                f"目标数={len(tracks)} | 自车={v_host * 3.6:.1f}km/h | "
                f"d_safe={d_safe:.1f}m | lead={lead_info}"
            )

    writer.release()
    cv2.destroyAllWindows()
    if ser is not None:
        ser.close()
        print("[串口] 已关闭。")
    print(f"\n完成。标注视频已保存到：{OUTPUT_VIDEO_PATH}")


if __name__ == "__main__":
    main()
