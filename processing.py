import h5py
import os
import glob
import numpy as np
from scipy.signal import butter, filtfilt
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import math
from scipy.spatial.transform import Rotation as R

# MODE = 0            # ALL_LABEL_ALL_SENSOR
# MODE = 1            # ALL_LABEL_LOWER_SENSOR
# MODE = 2            # ALL_LABEL_UPPER_SENSOR
MODE = 3            # Transient_LABEL_ALL_SENSOR
# MODE = 4            # Transient_LABEL_ALL_SENSOR_NO_STOPWALKING
# MODE = 5            # Transient_LABEL_LOWER_SENSOR
# MODE = 6            # Transient_LABEL_UPPER_SENSOR
# MODE = 7            # Add "Start Walking" 

############################################################################################################################################
############################################################################################################################################
#################################################  DATA Loading ############################################################################
############################################################################################################################################
############################################################################################################################################

### 여러 개의 trial(1~10)을 포함한 h5파일을 여러 개 포함하고 있는 폴더에 대한 Loading
def Load_files(folder_path, pattern='cropped_*.h5'):
    all_trials = []

    file_list = sorted(glob.glob(os.path.join(folder_path, pattern)))
    print(f"Total {len(file_list)} files are now loaded...")

    for file_path in file_list:
        with h5py.File(file_path, "r") as f:
            for trial_name in f.keys():
                trial = f[trial_name]

                # 개별 센서별 데이터 모두 불러오기
                trial_data = {
                    'file': os.path.basename(file_path),

                    'trial': trial_name,
                    
                    'time': trial['time'][:],
                    
                    'emgL1': trial['emgL1'][:],
                    'emgL2': trial['emgL2'][:],
                    'emgL3': trial['emgL3'][:],
                    'emgL4': trial['emgL4'][:],
                    'emgR1': trial['emgR1'][:],
                    'emgR2': trial['emgR2'][:],
                    'emgR3': trial['emgR3'][:],
                    'emgR4': trial['emgR4'][:],

                    'imu1':  trial['imu1'][:],
                    'imu2':  trial['imu2'][:],
                    'imu3':  trial['imu3'][:],
                    'imu4':  trial['imu4'][:],
                    'imu5':  trial['imu5'][:],
                    'imu6':  trial['imu6'][:],
                    'imu7':  trial['imu7'][:],
                    'imu8':  trial['imu8'][:],
                    'imu9':  trial['imu9'][:],
                    'imu10':  trial['imu10'][:],

                    'rhip':  trial['rhip'][:],
                    'rknee':  trial['rknee'][:],
                    'lhip':  trial['lhip'][:],
                    'lknee':  trial['lknee'][:],
                    'trunk':  trial['trunk'][:],
                    'rshoulder':  trial['rshoulder'][:],
                    'relbow':  trial['relbow'][:],
                    'lshoulder':  trial['lshoulder'][:],
                    'lelbow':  trial['lelbow'][:],

                    'rising_ok': trial['rising_ok'][:],
                    'falling_ok': trial['falling_ok'][:]
                }

                all_trials.append(trial_data)

    print(f"Total {len(all_trials)} trials are completely loaded !!")
    return all_trials






############################################################################################################################################
############################################################################################################################################
####################################################  Label  ###############################################################################
############################################################################################################################################
############################################################################################################################################

def Generate_labels(trial_data, fs=100):
    """
    trial_data: dict with 'rising_ok', 'falling_ok', 'time'
    fs: sampling frequency (Hz)
    k_ms: transition phase duration after OK button (in ms)
    """

    button_rising = np.squeeze(trial_data['rising_ok'])
    button_falling = np.squeeze(trial_data['falling_ok'])
    T = len(button_rising)

    # OK 버튼 눌림 시점 (rising & falling edges)
    rising_ok_indices = np.where(np.diff(button_rising.astype(int)) == 1)[0] + 1
    falling_ok_indices = np.where(np.diff(button_falling.astype(int)) == 1)[0] + 1

    if len(rising_ok_indices) != 4:
        raise ValueError(f"Expected 4 Rising_OK button presses, but found {len(rising_ok_indices)}")
    if len(falling_ok_indices) != 4:
        raise ValueError(f"Expected 4 Falling_OK button presses, but found {len(falling_ok_indices)}")
    
    # index를 모두 합치고 오름차순으로 정렬
    sorted_indices = np.sort(np.concatenate((rising_ok_indices, falling_ok_indices)))

    # Default 실험 고정 순서 (Training Set)
    if (MODE == 0 or MODE == 1 or MODE == 2 or MODE == 7):
        phase_seq = [
            'Stand',
            'Stand-to-Sit',
            'Sit',
            'Sit-to-Stand',
            'Stand',
            'Stand-to-Walk',
            'Walk',
            'Walk-to-Stand',
            'Stand'
        ]
    elif (MODE == 3 or MODE == 5 or MODE == 6):
        phase_seq = [
            'Stand',
            'Stand-to-Sit',
            'Sit',
            'Sit-to-Stand',
            'Stand',
            'Walk',
            'Walk',
            'Stop Walking',
            'Stand'
        ]
    elif (MODE == 4):
        phase_seq = [
            'Stand',
            'Stand-to-Sit',
            'Sit',
            'Sit-to-Stand',
            'Stand',
            'Walk',
            'Walk',
            'Walk',
            'Stand'
        ]

    labels = np.empty(T, dtype=object)
    idx = 0
    phase = 0

    # i = 0,1,2,..,7
    for i, press_idx in enumerate(sorted_indices):
        labels[idx:press_idx] = phase_seq[phase]
        idx = press_idx
        phase = phase + 1

    # MODE == 7인 경우, Walk의 앞부분 600ms정도를 Stand-to-Walk로 변경
    if  (MODE == 7):
        # "Walk"가 처음 나오는 인덱스를 찾음
        first_walk_index = np.where(labels == "Walk")[0][0]

        # "Walk"인 값만 바꾸기 위해 카운트
        walk_count = 0
        i = first_walk_index
        while i < len(labels) and walk_count < 100:
            if labels[i] == "Walk":
                labels[i] = "Stand-to-Walk"
                walk_count += 1
            i += 1

    # Last phase
    labels[idx:] = phase_seq[phase]

    return labels    


if (MODE == 0 or MODE == 1 or MODE == 2 or MODE == 7):
    phase_to_int = {
        'Stand': 0,
        'Stand-to-Sit': 1,
        'Sit': 2,
        'Sit-to-Stand': 3,
        'Stand-to-Walk': 4,
        'Walk': 5,
        'Walk-to-Stand': 6
    }
elif (MODE == 3 or MODE == 5 or MODE == 6):
    phase_to_int = {
        'Stand': 0,
        'Stand-to-Sit': 1,
        'Sit': 2,
        'Sit-to-Stand': 3,
        'Walk': 4,
        'Stop Walking': 5,
    }
elif (MODE == 4):
    phase_to_int = {
        'Stand': 0,
        'Stand-to-Sit': 1,
        'Sit': 2,
        'Sit-to-Stand': 3,
        'Walk': 4,
    }



    # phase_to_int = {
    #     'Stand': -1,
    #     'Stand-to-Sit': 0,
    #     'Sit': -1,
    #     'Sit-to-Stand': 1,
    #     'Stand-to-Walk': 2,
    #     'Walk': -1,
    #     'Walk-to-Stand': 3,
    #     'No-Transition': -1
    # }


def Convert_labels_to_int(trial_data, mapping):
    label_str = trial_data['label']
    label_int = np.array([mapping[label] for label in label_str])
    trial_data['label_int'] = label_int
    return trial_data








############################################################################################################################################
############################################################################################################################################
####################################################  EMG Filters  #########################################################################
############################################################################################################################################
############################################################################################################################################

# emgL1: Left Vastus Medialis (L_VM)
# emgL2: Left Medial Gastrocnemius (L_MG)
# emgL3: Left Trapezius (L_TR)
# emgL4: Left Erector Spinae (L_ES)
# emgR1: Right Vastus Medialis (R_VM)
# emgR2: Right Medial Gastrocnemius (R_MG)
# emgR3: Right Trapezius (R_TR)
# emgR4: Right Erector Spinae (R_ES)

def lowpass_filter(data, fc=5, fs=100, order=4):
    nyquist_freq = 0.5 * fs
    norm_cutoff = fc / nyquist_freq
    b, a = butter(order, norm_cutoff, btype='low')
    return filtfilt(b, a, data)


def normalize_emg(data, method='zscore', mean_val=None, std_val=None, max_val=None):
    if method == 'zscore':
        if mean_val is None:
            mean_val = np.mean(data)
        if std_val is None:
            std_val = np.std(data)
        return (data - mean_val) / (std_val + 1e-8)
    elif method == 'max':
        if max_val is None:
            max_val = np.max(data)
        return data / (max_val + 1e-8)
    else:
        raise ValueError("method must be 'zscore' or 'max'")


def process_all_emg(all_trials, fs=100, lpf_fc=5, norm_method='zscore'):
    processed_trials = []
    emg_keys = ['emgL1', 'emgL2', 'emgL3', 'emgL4', 'emgR1', 'emgR2', 'emgR3', 'emgR4']
    imu_keys = ['imu1', 'imu2', 'imu3', 'imu4', 'imu5', 'imu6', 'imu7', 'imu8', 'imu9', 'imu10', 'rhip', 'rknee', 'lhip', 'lknee', 'trunk', 'rshoulder', 'relbow', 'lshoulder', 'lelbow']

    for trial in all_trials:
        trial_copy = {}

        for key, value in trial.items():
            if key in ['file', 'trial']:
                trial_copy[key] = value

            elif key in emg_keys:
                raw = np.squeeze(value)
                envelope = lowpass_filter(raw, fc=lpf_fc, fs=fs)
                norm = normalize_emg(envelope, method=norm_method)

                trial_copy[key] = raw                   # raw EMG 저장 (예: 'emgL1')
                trial_copy[f"{key}_filt"] = envelope    # envelope 저장 (예: 'emgL1_filt')
                trial_copy[f"{key}_norm"] = norm        # normalized 저장 (예: 'emgL1_norm')

            elif key in imu_keys:
                trial_copy[key] = value                 # 그대로

            elif key == 'time':
                trial_copy[key] = [i * 10 for i in range(len(np.squeeze(value)))]       # time에 대해, 0부터 시작으로 변경

            else:
                trial_copy[key] = np.squeeze(value)     # button_ok 등

        processed_trials.append(trial_copy)

    return processed_trials




############################################################################################################################################
############################################################################################################################################
####################################################  IMU Calibration  #####################################################################
############################################################################################################################################
############################################################################################################################################

# IMU1: Right Hip (RH)
# IMU2: Right Knee (RK)
# IMU3: Left Hip (LH)
# IMU4: Left Knee (LK)
# IMU5: Pelvis (PEL)
# IMU6: Trunk (TR)
# IMU7: Right Upperarm (RS)
# IMU8: Right Forearm (RE)
# IMU9: Left Upperarm (LS)
# IMU10: Left Forearm (LE)


# Quaternion Method 
class Quaternion:
    @staticmethod
    def conjugate(q):
        return np.array([q[0], -q[1], -q[2], -q[3]])
    
    @staticmethod
    def inverse(q):
        return np.array([q[0], -q[1], -q[2], -q[3]])

    @staticmethod
    def multiply(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    @staticmethod
    def to_rotmat(q):
        w, x, y, z = q
        return np.array([
            [1 - 2*y**2 - 2*z**2,     2*x*y - 2*w*z,     2*x*z + 2*w*y],
            [2*x*y + 2*w*z,     1 - 2*x**2 - 2*z**2,     2*y*z - 2*w*x],
            [2*x*z - 2*w*y,     2*y*z + 2*w*x,     1 - 2*x**2 - 2*y**2]
        ])
    
    @staticmethod
    def to_rotmat_flatten_6D(q):        ### 1st and 2nd columns of Rotation matrix
        w, x, y, z = q
        return np.array(
            [1 - 2*y**2 - 2*z**2,     2*x*y + 2*w*z,     2*x*z - 2*w*y,    2*x*y - 2*w*z,     1 - 2*x**2 - 2*z**2,     2*y*z + 2*w*x]
        )
    

class RotationMatrix:
    @staticmethod
    def to_euler_zyx(R, degree=True):
        """
        ZYX 순서 (Yaw → Pitch → Roll) 기준 오일러각을 반환
        입력:
            R: 3x3 회전행렬 (numpy array or list of lists)
        출력:
            [roll, pitch, yaw]: in radians or degrees
        """
        r11, r12, r13 = R[0]
        r21, r22, r23 = R[1]
        r31, r32, r33 = R[2]

        # 안정성 고려
        if abs(r31) <= 0.999999:
            pitch = -math.asin(r31)
            cos_pitch = math.cos(pitch)
            roll = math.atan2(r32 / cos_pitch, r33 / cos_pitch)
            yaw = math.atan2(r21 / cos_pitch, r11 / cos_pitch)
        else:
            # Gimbal lock 발생
            pitch = math.pi / 2 if r31 <= -1 else -math.pi / 2
            roll = 0.0
            yaw = math.atan2(-r12, r22)

        if degree:
            roll = math.degrees(roll)
            pitch = math.degrees(pitch)
            yaw = math.degrees(yaw)

        return np.array([roll, pitch, yaw])
    

quat_default = {
    "RH_joint": np.array([np.sqrt(2)/2, 0,  np.sqrt(2)/2, 0]),              # y, +90
    "RK_joint": np.array([np.sqrt(2)/2, 0,  np.sqrt(2)/2, 0]),              # y, +90
    "LH_joint": np.array([np.sqrt(2)/2, 0, -np.sqrt(2)/2, 0]),              # y, -90
    "LK_joint": np.array([np.sqrt(2)/2, 0, -np.sqrt(2)/2, 0]),              # y, -90
    "Pelvis":   np.array([1, 0, 0, 0]),                                     # Base
    "Trunk":    np.array([0, 0, 1, 0]),                                     # y, +180
    "RS_joint": np.array([np.sqrt(2)/2, 0,  np.sqrt(2)/2, 0]),              # y, +90
    "RE_joint": np.array([np.sqrt(2)/2, 0,  np.sqrt(2)/2, 0]),              # y, +90
    "LS_joint": np.array([np.sqrt(2)/2, 0, -np.sqrt(2)/2, 0]),              # y, -90
    "LE_joint": np.array([np.sqrt(2)/2, 0, -np.sqrt(2)/2, 0]),              # y, -90
}


quat_default_rel = {
    "RH_joint": np.array([np.sqrt(2)/2, 0,  np.sqrt(2)/2, 0]),              # y, +90
    "RK_joint": np.array([1, 0, 0, 0]),             
    "LH_joint": np.array([np.sqrt(2)/2, 0, -np.sqrt(2)/2, 0]),              # y, -90
    "LK_joint": np.array([1, 0, 0 ,0]),            
    "Pelvis":   np.array([1, 0, 0, 0]),                                     # Base
    "Trunk":    np.array([0, 0, 1, 0]),                                     # y, +180
    "RS_joint": np.array([np.sqrt(2)/2, 0, -np.sqrt(2)/2, 0]),              # y, -90
    "RE_joint": np.array([1, 0, 0, 0]),             
    "LS_joint": np.array([np.sqrt(2)/2, 0,  np.sqrt(2)/2, 0]),              # y, +90
    "LE_joint": np.array([1, 0, 0, 0]),           
}  


quat_default_EULER = {
    "RH_joint": np.array([1, 0, 0, 0]),            
    "RK_joint": np.array([1, 0, 0, 0]),             
    "LH_joint": np.array([1, 0, 0, 0]),          
    "LK_joint": np.array([1, 0, 0 ,0]),            
    "Pelvis":   np.array([1, 0, 0, 0]),            # Base
    "Trunk":    np.array([1, 0, 0, 0]),                                  
    "RS_joint": np.array([1, 0, 0, 0]),              
    "RE_joint": np.array([1, 0, 0, 0]),             
    "LS_joint": np.array([1, 0, 0, 0]),       
    "LE_joint": np.array([1, 0, 0, 0]),           
}  



# 각 센서의 첫 상태(Stand)를 기준[1,0,0,0]으로 하고 이후 쿼터니언을 계산  
def CalibrateIMU(trial):
    quat_raw = {}
    quat_raw["RH_joint"] = np.transpose(trial["imu1"])          # Make (4,T) -> (T,4)
    quat_raw["RK_joint"] = np.transpose(trial["imu2"])
    quat_raw["LH_joint"] = np.transpose(trial["imu3"])
    quat_raw["LK_joint"] = np.transpose(trial["imu4"])
    quat_raw["Pelvis"]   = np.transpose(trial["imu5"])
    quat_raw["Trunk"]    = np.transpose(trial["imu6"])
    quat_raw["RS_joint"] = np.transpose(trial["imu7"])
    quat_raw["RE_joint"] = np.transpose(trial["imu8"])
    quat_raw["LS_joint"] = np.transpose(trial["imu9"])
    quat_raw["LE_joint"] = np.transpose(trial["imu10"])

    quat_initial_inv = {
        "RH_joint": Quaternion.inverse(quat_raw["RH_joint"][0]),
        "RK_joint": Quaternion.inverse(quat_raw["RK_joint"][0]),
        "LH_joint": Quaternion.inverse(quat_raw["LH_joint"][0]),
        "LK_joint": Quaternion.inverse(quat_raw["LK_joint"][0]),
        "Pelvis"  : Quaternion.inverse(quat_raw["Pelvis"][0]),
        "Trunk"   : Quaternion.inverse(quat_raw["Trunk"][0]),
        "RS_joint": Quaternion.inverse(quat_raw["RS_joint"][0]),
        "RE_joint": Quaternion.inverse(quat_raw["RE_joint"][0]),
        "LS_joint": Quaternion.inverse(quat_raw["LS_joint"][0]),
        "LE_joint": Quaternion.inverse(quat_raw["LE_joint"][0])
    }

    quat_fromInitial = {}
    for joint, quatRaw in quat_raw.items():
        q_fromInitial_seq = []
        for q in quatRaw:
            q_fromInitial = Quaternion.multiply(quat_initial_inv[joint], q)
            q_fromInitial_seq.append(q_fromInitial)
        quat_fromInitial[joint] = np.array(q_fromInitial_seq)

    trial["imu1"] = quat_fromInitial["RH_joint"]
    trial["imu2"] = quat_fromInitial["RK_joint"]
    trial["imu3"] = quat_fromInitial["LH_joint"]
    trial["imu4"] = quat_fromInitial["LK_joint"]
    trial["imu5"] = quat_fromInitial["Pelvis"]
    trial["imu6"] = quat_fromInitial["Trunk"]
    trial["imu7"] = quat_fromInitial["RS_joint"]
    trial["imu8"] = quat_fromInitial["RE_joint"]
    trial["imu9"] = quat_fromInitial["LS_joint"]
    trial["imu10"] = quat_fromInitial["LE_joint"]



# Pelvis를 Base[1,0,0,0]로 잡고 Base에 대한 상대 쿼터니언을 적용
def CalibrateIMU_2(trial):
    quat_raw = {}
    quat_raw["RH_joint"] = np.transpose(trial["imu1"])          # Make (4,T) -> (T,4)
    quat_raw["RK_joint"] = np.transpose(trial["imu2"])
    quat_raw["LH_joint"] = np.transpose(trial["imu3"])
    quat_raw["LK_joint"] = np.transpose(trial["imu4"])
    quat_raw["Pelvis"]   = np.transpose(trial["imu5"])
    quat_raw["Trunk"]    = np.transpose(trial["imu6"])
    quat_raw["RS_joint"] = np.transpose(trial["imu7"])
    quat_raw["RE_joint"] = np.transpose(trial["imu8"])
    quat_raw["LS_joint"] = np.transpose(trial["imu9"])
    quat_raw["LE_joint"] = np.transpose(trial["imu10"])

    quat_corr = {
        "RH_joint": np.array([1, 0, 0, 0]),              # y, +90
        "RK_joint": np.array([1, 0, 0, 0]),              # y, +90
        "LH_joint": np.array([1, 0, 0, 0]),              # y, -90
        "LK_joint": np.array([1, 0, 0, 0]),              # y, -90
        "Pelvis":   np.array([1, 0, 0, 0]),              # Base
        "Trunk":    np.array([1, 0, 0, 0]),              # y, +180
        "RS_joint": np.array([1, 0, 0, 0]),              # y, +90
        "RE_joint": np.array([1, 0, 0, 0]),              # y, +90
        "LS_joint": np.array([1, 0, 0, 0]),              # y, -90
        "LE_joint": np.array([1, 0, 0, 0]),              # y, -90
    }

    quat_initial_base_inv = Quaternion.inverse(quat_raw["Pelvis"][0])

    # Base(Pelvis)를 기준으로 Stand 상태에서 Ideal한 쿼터니언 값이 나오도록 보정하는 correction term 계산
    for joint, _ in quat_corr.items():
        quat_corr[joint] = Quaternion.multiply(quat_default[joint], Quaternion.inverse(Quaternion.multiply(quat_initial_base_inv, quat_raw[joint][0])))

    # Base(Pelvis)는 [1,0,0,0]으로 고정, 다른 joint들은 Base를 기준으로 측정된 쿼터니언 값으로 변환
    quat_fromBase = {}
    for joint, quatRaw in quat_raw.items():
        q_fromBase_seq = []
        for idx, q in enumerate(quatRaw):
            q_fromBase = Quaternion.multiply(Quaternion.inverse(quat_raw["Pelvis"][idx]), q)
            ### Correction Term ###
            if (joint in quat_corr.keys()):
                q_fromBase = Quaternion.multiply(quat_corr[joint], q_fromBase)

            q_fromBase_seq.append(q_fromBase)
        quat_fromBase[joint] = np.array(q_fromBase_seq)

    trial["imu1"] = quat_fromBase["RH_joint"]
    trial["imu2"] = quat_fromBase["RK_joint"]
    trial["imu3"] = quat_fromBase["LH_joint"]
    trial["imu4"] = quat_fromBase["LK_joint"]
    trial["imu5"] = quat_fromBase["Pelvis"]
    trial["imu6"] = quat_fromBase["Trunk"]
    trial["imu7"] = quat_fromBase["RS_joint"]
    trial["imu8"] = quat_fromBase["RE_joint"]
    trial["imu9"] = quat_fromBase["LS_joint"]
    trial["imu10"] = quat_fromBase["LE_joint"]



def normalize_quaternion_series(quat_seq):
    """
    quat_seq: np.ndarray of shape (T, 4)
    Returns: normalized np.ndarray of shape (T, 4)
    """
    norms = np.linalg.norm(quat_seq, axis=1, keepdims=True)
    normalized = quat_seq / norms
    return normalized


def normalize_quaternion(q):
    """
    q: np.ndarray or list of shape (4,) - 단일 쿼터니언 [w, x, y, z] 또는 [x, y, z, w]
    return: 정규화된 np.ndarray of shape (4,)
    """
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-8:
        raise ValueError("Quaternion norm is too close to zero. Cannot normalize.")
    return q / norm


def quat_to_rot6d(quat_seq):
    """
    quat_seq: (T, 4) numpy array (w, x, y, z)
    returns: (T, 6) numpy array (6D rotation representation)
    """
    # scipy는 (x, y, z, w) 순서를 사용함
    quat_seq_xyzw = quat_seq[:, [1, 2, 3, 0]]  # (T, 4)

    # 쿼터니언 → 회전행렬
    r = R.from_quat(quat_seq_xyzw)
    rot_mats = r.as_matrix()  # (T, 3, 3)

    # 앞 두 열 추출 → 6D
    rot6d = rot_mats[:, :, 0:2]  # (T, 3, 2)
    rot6d = rot6d.reshape(-1, 6)  # (T, 6)

    return rot6d


def rot6d_to_matrix(rot6d):
    a1 = rot6d[:, 0:3]
    a2 = rot6d[:, 3:6]

    b1 = a1 / np.linalg.norm(a1, axis=1, keepdims=True)
    a2_proj = np.sum(b1 * a2, axis=1, keepdims=True) * b1
    b2 = a2 - a2_proj
    b2 = b2 / np.linalg.norm(b2, axis=1, keepdims=True)
    b3 = np.cross(b1, b2)

    rot_mat = np.stack([b1, b2, b3], axis=2)  # (T, 3, 3)
    return rot_mat


# Pelvis를 Base[1,0,0,0]로 잡고 prev_joint에 대한 상대 쿼터니언 적용
def CalibrateIMU_3(trial):
    quat_raw = {}
    quat_raw["RH_joint"] = np.transpose(trial["imu1"])          # Make (4,T) -> (T,4)
    quat_raw["RK_joint"] = np.transpose(trial["imu2"])
    quat_raw["LH_joint"] = np.transpose(trial["imu3"])
    quat_raw["LK_joint"] = np.transpose(trial["imu4"])
    quat_raw["Pelvis"]   = np.transpose(trial["imu5"])
    quat_raw["Trunk"]    = np.transpose(trial["imu6"])
    quat_raw["RS_joint"] = np.transpose(trial["imu7"])
    quat_raw["RE_joint"] = np.transpose(trial["imu8"])
    quat_raw["LS_joint"] = np.transpose(trial["imu9"])
    quat_raw["LE_joint"] = np.transpose(trial["imu10"])

    quat_corr = {
        "RH_joint": np.array([1, 0, 0, 0]),              # y, +90
        "RK_joint": np.array([1, 0, 0, 0]),              # y, +90
        "LH_joint": np.array([1, 0, 0, 0]),              # y, -90
        "LK_joint": np.array([1, 0, 0, 0]),              # y, -90
        "Pelvis":   np.array([1, 0, 0, 0]),              # Base
        "Trunk":    np.array([1, 0, 0, 0]),              # y, +180
        "RS_joint": np.array([1, 0, 0, 0]),              # y, +90
        "RE_joint": np.array([1, 0, 0, 0]),              # y, +90
        "LS_joint": np.array([1, 0, 0, 0]),              # y, -90
        "LE_joint": np.array([1, 0, 0, 0]),              # y, -90
    }

    prev_joint = {
        "RH_joint": "Pelvis",
        "RK_joint": "RH_joint",
        "LH_joint": "Pelvis",
        "LK_joint": "LH_joint",
        "Pelvis":   "Pelvis",       # Base
        "Trunk":    "Pelvis",
        "RS_joint": "Trunk",
        "RE_joint": "RS_joint",
        "LS_joint": "Trunk",
        "LE_joint": "LS_joint",
    }

    # Previous Joint를 기준으로 Stand 상태에서 Ideal한 쿼터니언 값이 나오도록 보정하는 correction term 계산
    for joint, _ in quat_corr.items():
        quat_corr[joint] = Quaternion.multiply(quat_default_rel[joint], Quaternion.inverse(Quaternion.multiply(Quaternion.inverse(quat_raw[prev_joint[joint]][0]), quat_raw[joint][0])))

    # Base(Pelvis)는 [1,0,0,0]으로 고정, 다른 joint들은 Previous Joint를 기준으로 측정된 상대 쿼터니언 값으로 변환
    quat_rel = {}
    for joint, quatRaw in quat_raw.items():
        q_rel_seq = []
        for idx, q in enumerate(quatRaw):
            q_rel = Quaternion.multiply(Quaternion.inverse(quat_raw[prev_joint[joint]][idx]), q) 

            ### Correction Term ###
            if (joint in quat_corr.keys()):
                q_rel = Quaternion.multiply(quat_corr[joint], q_rel)

            q_rel_seq.append(q_rel)
        quat_rel[joint] = np.array(q_rel_seq)

    trial["imu1"] = quat_rel["RH_joint"]
    trial["imu2"] = quat_rel["RK_joint"]
    trial["imu3"] = quat_rel["LH_joint"]
    trial["imu4"] = quat_rel["LK_joint"]
    trial["imu5"] = quat_rel["Pelvis"]
    trial["imu6"] = quat_rel["Trunk"]
    trial["imu7"] = quat_rel["RS_joint"]
    trial["imu8"] = quat_rel["RE_joint"]
    trial["imu9"] = quat_rel["LS_joint"]
    trial["imu10"] = quat_rel["LE_joint"]



# Pelvis를 Base[1,0,0,0]로 잡고 prev_joint에 대한 상대 쿼터니언 적용
def CalibrateIMU_4(trial):
    quat_raw = {}
    quat_raw["RH_joint"] = np.transpose(trial["imu1"])          # Make (4,T) -> (T,4)
    quat_raw["RK_joint"] = np.transpose(trial["imu2"])
    quat_raw["LH_joint"] = np.transpose(trial["imu3"])
    quat_raw["LK_joint"] = np.transpose(trial["imu4"])
    quat_raw["Pelvis"]   = np.transpose(trial["imu5"])
    quat_raw["Trunk"]    = np.transpose(trial["imu6"])
    quat_raw["RS_joint"] = np.transpose(trial["imu7"])
    quat_raw["RE_joint"] = np.transpose(trial["imu8"])
    quat_raw["LS_joint"] = np.transpose(trial["imu9"])
    quat_raw["LE_joint"] = np.transpose(trial["imu10"])

    quat_corr = {
        "RH_joint": np.array([1, 0, 0, 0]),              # y, +90
        "RK_joint": np.array([1, 0, 0, 0]),              # y, +90
        "LH_joint": np.array([1, 0, 0, 0]),              # y, -90
        "LK_joint": np.array([1, 0, 0, 0]),              # y, -90
        "Pelvis":   np.array([1, 0, 0, 0]),              # Base
        "Trunk":    np.array([1, 0, 0, 0]),              # y, +180
        "RS_joint": np.array([1, 0, 0, 0]),              # y, +90
        "RE_joint": np.array([1, 0, 0, 0]),              # y, +90
        "LS_joint": np.array([1, 0, 0, 0]),              # y, -90
        "LE_joint": np.array([1, 0, 0, 0]),              # y, -90
    }

    prev_joint = {
        "RH_joint": "Pelvis",
        "RK_joint": "RH_joint",
        "LH_joint": "Pelvis",
        "LK_joint": "LH_joint",
        "Pelvis":   "Pelvis",       # Base
        "Trunk":    "Pelvis",
        "RS_joint": "Trunk",
        "RE_joint": "RS_joint",
        "LS_joint": "Trunk",
        "LE_joint": "LS_joint",
    }

    # Previous Joint를 기준으로 Stand 상태에서 Ideal한 쿼터니언 값이 나오도록 보정하는 correction term 계산
    for joint, _ in quat_corr.items():
        quat_corr[joint] = Quaternion.multiply(quat_default_rel[joint], Quaternion.inverse(Quaternion.multiply(Quaternion.inverse(quat_raw[prev_joint[joint]][0]), quat_raw[joint][0])))

    # Base(Pelvis)는 [1,0,0,0]으로 고정, 다른 joint들은 Previous Joint를 기준으로 측정된 상대 쿼터니언 값으로 변환
    quat_rel = {}
    for joint, quatRaw in quat_raw.items():
        q_rel_seq = []
        for idx, q in enumerate(quatRaw):
            q_rel = Quaternion.multiply(Quaternion.inverse(quat_raw[prev_joint[joint]][idx]), q) 

            ### Correction Term ###
            if (joint in quat_corr.keys()):
                q_rel = Quaternion.multiply(quat_corr[joint], q_rel)

            q_rel_seq.append(q_rel)
        quat_rel[joint] = np.array(q_rel_seq)


    trial["imu1"] = quat_rel["RH_joint"]
    trial["imu2"] = quat_rel["RK_joint"]
    trial["imu3"] = quat_rel["LH_joint"]
    trial["imu4"] = quat_rel["LK_joint"]
    trial["imu5"] = quat_rel["Pelvis"]
    trial["imu6"] = quat_rel["Trunk"]
    trial["imu7"] = quat_rel["RS_joint"]
    trial["imu8"] = quat_rel["RE_joint"]
    trial["imu9"] = quat_rel["LS_joint"]
    trial["imu10"] = quat_rel["LE_joint"]

    trial["imu1"] = quat_to_rot6d(trial["imu1"])
    trial["imu2"] = quat_to_rot6d(trial["imu2"])
    trial["imu3"] = quat_to_rot6d(trial["imu3"])
    trial["imu4"] = quat_to_rot6d(trial["imu4"])
    trial["imu5"] = quat_to_rot6d(trial["imu5"])
    trial["imu6"] = quat_to_rot6d(trial["imu6"])
    trial["imu7"] = quat_to_rot6d(trial["imu7"])
    trial["imu8"] = quat_to_rot6d(trial["imu8"])
    trial["imu9"] = quat_to_rot6d(trial["imu9"])
    trial["imu10"] = quat_to_rot6d(trial["imu10"])





import numpy as np
from scipy.spatial.transform import Rotation as R

def quaternion_to_smooth_euler(q_seq, order='zyx', degrees=True):
    """
    (w, x, y, z) 순서의 쿼터니언 시계열 (T, 4) → 불연속 없는 오일러 각 시계열 (T, 3)

    Args:
        q_seq (np.ndarray): (T, 4) 형태의 쿼터니언 시계열 (w, x, y, z)
        order (str): 오일러 각 회전 순서 (default: 'zyx')
        degrees (bool): 출력 단위 (True = 도, False = 라디안)

    Returns:
        euler_angles (np.ndarray): (T, 3) 형태의 부드러운 오일러 각 시계열
    """
    q_seq = np.asarray(q_seq)
    T = q_seq.shape[0]

    # 1. (w, x, y, z) → (x, y, z, w)로 변환
    q_seq = q_seq[:, [1, 2, 3, 0]]

    # 2. 정규화 (필요 시)
    q_seq = q_seq / np.linalg.norm(q_seq, axis=1, keepdims=True)

    # 3. 부호 일관성 유지 (프레임 간 dot product 기준)
    for t in range(1, T):
        if np.dot(q_seq[t], q_seq[t - 1]) < 0:
            q_seq[t] = -q_seq[t]

    # 4. 쿼터니언 → 오일러 각
    rot = R.from_quat(q_seq)
    eulers = rot.as_euler(order, degrees=degrees)

    # 5. unwrap으로 불연속 제거
    if degrees:
        eulers = np.unwrap(np.deg2rad(eulers), axis=0)
        eulers = np.rad2deg(eulers)
    else:
        eulers = np.unwrap(eulers, axis=0)

    return eulers




# Pelvis를 Base[1,0,0,0]로 잡고 prev_joint에 대한 상대 쿼터니언 적용 -> Euler angle (ZYX euler angle -> B3 = Rx * Ry * Rz * B )
def CalibrateIMU_5(trial):
    quat_raw = {}
    quat_raw["RH_joint"] = normalize_quaternion_series(np.transpose(trial["imu1"]))          # Make (4,T) -> (T,4)
    quat_raw["RK_joint"] = normalize_quaternion_series(np.transpose(trial["imu2"]))  
    quat_raw["LH_joint"] = normalize_quaternion_series(np.transpose(trial["imu3"]))  
    quat_raw["LK_joint"] = normalize_quaternion_series(np.transpose(trial["imu4"]))  
    quat_raw["Pelvis"]   = normalize_quaternion_series(np.transpose(trial["imu5"]))  
    quat_raw["Trunk"]    = normalize_quaternion_series(np.transpose(trial["imu6"]))  
    quat_raw["RS_joint"] = normalize_quaternion_series(np.transpose(trial["imu7"]))  
    quat_raw["RE_joint"] = normalize_quaternion_series(np.transpose(trial["imu8"]))  
    quat_raw["LS_joint"] = normalize_quaternion_series(np.transpose(trial["imu9"]))  
    quat_raw["LE_joint"] = normalize_quaternion_series(np.transpose(trial["imu10"]))  

    quat_corr = {
        "RH_joint": np.array([1, 0, 0, 0]),              # y, +90
        "RK_joint": np.array([1, 0, 0, 0]),              # y, +90
        "LH_joint": np.array([1, 0, 0, 0]),              # y, -90
        "LK_joint": np.array([1, 0, 0, 0]),              # y, -90
        "Pelvis":   np.array([1, 0, 0, 0]),              # Base
        "Trunk":    np.array([1, 0, 0, 0]),              # y, +180
        "RS_joint": np.array([1, 0, 0, 0]),              # y, +90
        "RE_joint": np.array([1, 0, 0, 0]),              # y, +90
        "LS_joint": np.array([1, 0, 0, 0]),              # y, -90
        "LE_joint": np.array([1, 0, 0, 0]),              # y, -90
    }

    prev_joint = {
        "RH_joint": "Pelvis",
        "RK_joint": "RH_joint",
        "LH_joint": "Pelvis",
        "LK_joint": "LH_joint",
        "Pelvis":   "Pelvis",       # Base
        "Trunk":    "Pelvis",
        "RS_joint": "Trunk",
        "RE_joint": "RS_joint",
        "LS_joint": "Trunk",
        "LE_joint": "LS_joint",
    }

    # Previous Joint를 기준으로 Stand 상태에서 Ideal한 쿼터니언 값이 나오도록 보정하는 correction term 계산
    for joint, _ in quat_corr.items():
        quat_corr[joint] = Quaternion.multiply(quat_default_rel[joint], Quaternion.inverse(Quaternion.multiply(Quaternion.inverse(quat_raw[prev_joint[joint]][0]), quat_raw[joint][0])))

    # Base(Pelvis)는 [1,0,0,0]으로 고정, 다른 joint들은 Previous Joint를 기준으로 측정된 상대 쿼터니언 값으로 변환
    quat_rel = {}
    for joint, quatRaw in quat_raw.items():
        q_rel_seq = []
        for idx, q in enumerate(quatRaw):
            q_rel = Quaternion.multiply(Quaternion.inverse(quat_raw[prev_joint[joint]][idx]), q) 

            ### Correction Term ###
            if (joint in quat_corr.keys()):
                q_rel = Quaternion.multiply(quat_corr[joint], q_rel)

            q_rel_seq.append(q_rel)
        quat_rel[joint] = np.array(q_rel_seq)


    trial["imu1"] = quat_rel["RH_joint"]
    trial["imu2"] = quat_rel["RK_joint"]
    trial["imu3"] = quat_rel["LH_joint"]
    trial["imu4"] = quat_rel["LK_joint"]
    trial["imu5"] = quat_rel["Pelvis"]
    trial["imu6"] = quat_rel["Trunk"]
    trial["imu7"] = quat_rel["RS_joint"]
    trial["imu8"] = quat_rel["RE_joint"]
    trial["imu9"] = quat_rel["LS_joint"]
    trial["imu10"] = quat_rel["LE_joint"]

    trial["imu1"] = quaternion_to_smooth_euler(trial["imu1"])
    trial["imu2"] = quaternion_to_smooth_euler(trial["imu2"])
    trial["imu3"] = quaternion_to_smooth_euler(trial["imu3"])
    trial["imu4"] = quaternion_to_smooth_euler(trial["imu4"])
    trial["imu5"] = quaternion_to_smooth_euler(trial["imu5"])
    trial["imu6"] = quaternion_to_smooth_euler(trial["imu6"])
    trial["imu7"] = quaternion_to_smooth_euler(trial["imu7"])
    trial["imu8"] = quaternion_to_smooth_euler(trial["imu8"])
    trial["imu9"] = quaternion_to_smooth_euler(trial["imu9"])
    trial["imu10"] = quaternion_to_smooth_euler(trial["imu10"])



# Pelvis를 Base[1,0,0,0]로 잡고 prev_joint에 대한 상대 쿼터니언 적용 -> Euler angle (ZYX euler angle -> B3 = Rx * Ry * Rz * B ) "No Library version"
def CalibrateIMU_6(trial):
    quat_raw = {}
    quat_raw["RH_joint"] = np.transpose(trial["imu1"])          # Make (4,T) -> (T,4)
    quat_raw["RK_joint"] = np.transpose(trial["imu2"])  
    quat_raw["LH_joint"] = np.transpose(trial["imu3"])  
    quat_raw["LK_joint"] = np.transpose(trial["imu4"])  
    quat_raw["Pelvis"]   = np.transpose(trial["imu5"])  
    quat_raw["Trunk"]    = np.transpose(trial["imu6"])  
    quat_raw["RS_joint"] = np.transpose(trial["imu7"])  
    quat_raw["RE_joint"] = np.transpose(trial["imu8"])  
    quat_raw["LS_joint"] = np.transpose(trial["imu9"])  
    quat_raw["LE_joint"] = np.transpose(trial["imu10"]) 

    quat_corr = {
        "RH_joint": np.array([1, 0, 0, 0]),              # y, +90
        "RK_joint": np.array([1, 0, 0, 0]),              # y, +90
        "LH_joint": np.array([1, 0, 0, 0]),              # y, -90
        "LK_joint": np.array([1, 0, 0, 0]),              # y, -90
        "Pelvis":   np.array([1, 0, 0, 0]),              # Base
        "Trunk":    np.array([1, 0, 0, 0]),              # y, +180
        "RS_joint": np.array([1, 0, 0, 0]),              # y, +90
        "RE_joint": np.array([1, 0, 0, 0]),              # y, +90
        "LS_joint": np.array([1, 0, 0, 0]),              # y, -90
        "LE_joint": np.array([1, 0, 0, 0]),              # y, -90
    }

    prev_joint = {
        "RH_joint": "Pelvis",
        "RK_joint": "RH_joint",
        "LH_joint": "Pelvis",
        "LK_joint": "LH_joint",
        "Pelvis":   "Pelvis",       # Base
        "Trunk":    "Pelvis",
        "RS_joint": "Trunk",
        "RE_joint": "RS_joint",
        "LS_joint": "Trunk",
        "LE_joint": "LS_joint",
    }

    # Previous Joint를 기준으로 Stand 상태에서 Ideal한 쿼터니언 값이 나오도록 보정하는 correction term 계산
    for joint, _ in quat_corr.items():
        quat_corr[joint] = Quaternion.multiply(quat_default_rel[joint], Quaternion.inverse(Quaternion.multiply(Quaternion.inverse(quat_raw[prev_joint[joint]][0]), quat_raw[joint][0])))

    # Base(Pelvis)는 [1,0,0,0]으로 고정, 다른 joint들은 Previous Joint를 기준으로 측정된 상대 쿼터니언 값으로 변환
    quat_rel = {}
    for joint, quatRaw in quat_raw.items():
        q_rel_seq = []
        for idx, q in enumerate(quatRaw):
            q_rel = Quaternion.multiply(Quaternion.inverse(quat_raw[prev_joint[joint]][idx]), q) 

            ### Correction Term ###
            if (joint in quat_corr.keys()):
                q_rel = normalize_quaternion(Quaternion.multiply(quat_corr[joint], q_rel))

            q_rel_seq.append(q_rel)
        quat_rel[joint] = np.array(q_rel_seq)

    # Quaternion -> Rotation Matrix
    rot_rel = {}
    for joint, quatRel in quat_rel.items():
        rot_rel_seq = []
        for q in quatRel:
            rot_rel_temp = Quaternion.to_rotmat(q)
            rot_rel_seq.append(rot_rel_temp)
        rot_rel[joint] = np.array(rot_rel_seq)

    # Rotation Matrix -> ZYX Euler Angle
    euler_rel = {}
    for joint, rotRel in rot_rel.items():
        euler_rel_seq = []
        for rot in rotRel:
            euler_rel_temp = RotationMatrix.to_euler_zyx(rot)
            euler_rel_seq.append(euler_rel_temp)
        euler_rel[joint] = np.array(euler_rel_seq)

    trial["imu1"] = euler_rel["RH_joint"]
    trial["imu2"] = euler_rel["RK_joint"]
    trial["imu3"] = euler_rel["LH_joint"]
    trial["imu4"] = euler_rel["LK_joint"]
    trial["imu5"] = euler_rel["Pelvis"]
    trial["imu6"] = euler_rel["Trunk"]
    trial["imu7"] = euler_rel["RS_joint"]
    trial["imu8"] = euler_rel["RE_joint"]
    trial["imu9"] = euler_rel["LS_joint"]
    trial["imu10"] = euler_rel["LE_joint"]





# Pelvis를 Base[1,0,0,0]로 잡고 prev_joint에 대한 상대 쿼터니언 적용 -> Euler angle (ZYX euler angle -> B3 = Rx * Ry * Rz * B ) "No Library version"
def CalibrateIMU_7(trial):
    quat_raw = {}
    quat_raw["RH_joint"] = np.transpose(trial["imu1"])          # Make (4,T) -> (T,4)
    quat_raw["RK_joint"] = np.transpose(trial["imu2"])  
    quat_raw["LH_joint"] = np.transpose(trial["imu3"])  
    quat_raw["LK_joint"] = np.transpose(trial["imu4"])  
    quat_raw["Pelvis"]   = np.transpose(trial["imu5"])  
    quat_raw["Trunk"]    = np.transpose(trial["imu6"])  
    quat_raw["RS_joint"] = np.transpose(trial["imu7"])  
    quat_raw["RE_joint"] = np.transpose(trial["imu8"])  
    quat_raw["LS_joint"] = np.transpose(trial["imu9"])  
    quat_raw["LE_joint"] = np.transpose(trial["imu10"]) 

    prev_joint = {
        "RH_joint": "Pelvis",
        "RK_joint": "RH_joint",
        "LH_joint": "Pelvis",
        "LK_joint": "LH_joint",
        "Pelvis":   "Pelvis",       # Base
        "Trunk":    "Pelvis",
        "RS_joint": "Trunk",
        "RE_joint": "RS_joint",
        "LS_joint": "Trunk",
        "LE_joint": "LS_joint",
    }

    # Stand 상태에서 각 IMU값이 [1,0,0,0]이도록 World frame상에서의 보정
    quat_world_calib = {}
    for joint, quatRaw in quat_raw.items():
        q_world_seq = []
        for idx, q in enumerate(quatRaw):
            q_world = Quaternion.multiply(Quaternion.inverse(quat_raw[joint][0]), q)
            q_world_seq.append(q_world)
        quat_world_calib[joint] = np.array(q_world_seq)

    # Base(Pelvis)는 [1,0,0,0]으로 고정, 다른 joint들은 Previous Joint를 기준으로 측정된 상대 쿼터니언 값으로 변환
    quat_rel = {}
    for joint, quatWorld in quat_world_calib.items():
        q_rel_seq = []
        for idx, q in enumerate(quatWorld):
            q_rel = Quaternion.multiply(Quaternion.inverse(quat_world_calib[prev_joint[joint]][idx]), q) 
            q_rel_seq.append(q_rel)
        quat_rel[joint] = np.array(q_rel_seq)

    # Quaternion -> Rotation Matrix
    rot_rel = {}
    for joint, quatRel in quat_rel.items():
        rot_rel_seq = []
        for q in quatRel:
            rot_rel_temp = Quaternion.to_rotmat(q)
            rot_rel_seq.append(rot_rel_temp)
        rot_rel[joint] = np.array(rot_rel_seq)

    # Rotation Matrix -> ZYX Euler Angle
    euler_rel = {}
    for joint, rotRel in rot_rel.items():
        euler_rel_seq = []
        for rot in rotRel:
            euler_rel_temp = RotationMatrix.to_euler_zyx(rot)
            euler_rel_seq.append(euler_rel_temp)
        euler_rel[joint] = np.array(euler_rel_seq)

    trial["imu1"] = euler_rel["RH_joint"]
    trial["imu2"] = euler_rel["RK_joint"]
    trial["imu3"] = euler_rel["LH_joint"]
    trial["imu4"] = euler_rel["LK_joint"]
    trial["imu5"] = euler_rel["Pelvis"]
    trial["imu6"] = euler_rel["Trunk"]
    trial["imu7"] = euler_rel["RS_joint"]
    trial["imu8"] = euler_rel["RE_joint"]
    trial["imu9"] = euler_rel["LS_joint"]
    trial["imu10"] = euler_rel["LE_joint"]


############################################################################################################################################
############################################################################################################################################
######################################################  ANN TRAINING  ######################################################################
############################################################################################################################################
############################################################################################################################################

def split_trials_train_val_test(trial_keys, val_ratio=0.2, test_ratio=0.2, seed=42):
    np.random.seed(seed)
    trial_keys = np.random.permutation(trial_keys).tolist()

    n_total = len(trial_keys)
    print(f"Total trial number: {n_total}")
    n_test = int(n_total * test_ratio)
    n_val = int((n_total - n_test) * val_ratio)

    test_keys = trial_keys[:n_test]
    val_keys = trial_keys[n_test:n_test + n_val]
    train_keys = trial_keys[n_test + n_val:]

    return train_keys, val_keys, test_keys


def build_dataset_custom_keys(
    all_trials,
    input_keys,
    trial_names_to_use=None,
    window_size=20,
    stride=1,
    pred = 0
):
    """
    Build an LSTM dataset using custom input keys.
    
    Parameters:
    - all_trials: list of trial dicts
    - input_keys: list of strings (e.g., ['emgL1_norm', 'imu1', 'imu2'])
    - trial_names_to_use: optional list of (file, trial) keys to include
    - window_size: length of time window
    - stride: step size between windows
    
    Returns:
    - x: (N, window_size, num_features)
    - y: (N,)
    """
    x_list = []
    y_list = []

    for trial in all_trials:
        if trial_names_to_use is not None:
            if (trial['file'], trial['trial']) not in trial_names_to_use:
                continue

        inputs = []                     # 한 trial에 대한 총 시계열 길이만큼 들어감
        for k in input_keys:
            arr = np.squeeze(trial[k])  # (T,) or (4, T) → (T,), (T, 4)
            
            if arr.ndim == 1:
                arr = arr[:, np.newaxis]  # (T, 1)
            elif arr.ndim == 2:
                if arr.shape[0] < arr.shape[1]:  # (4, T) → transpose
                    arr = arr.T  # → (T, 4)
            else:
                raise ValueError(f"Unsupported shape {arr.shape} for key '{k}'")

            inputs.append(arr)

        input_stack = np.concatenate(inputs, axis=-1)  # (T, D)
        labels = trial['label_int']
        T = len(labels)

        for start in range(0, T - window_size - max(pred, 0), stride):
            x_window = input_stack[start:start + window_size]      # (window_size, D)
            y_label  = labels[start + window_size + pred]          # 중앙 프레임 라벨
            # y_label  = labels[start + (window_size//2) + pred]          # 중앙 프레임 라벨
            x_list.append(x_window)
            y_list.append(y_label)

    x = np.array(x_list)
    y = np.array(y_list)
    return x, y


def build_dataset_from_trial_keys(trial_dict, trial_keys, input_keys, window_size=200, stride=20, pred=0):
    selected_trials = [trial_dict[k] for k in trial_keys]
    x, y = build_dataset_custom_keys(
        all_trials=selected_trials,
        input_keys=input_keys,
        window_size=window_size,
        stride=stride,
        pred=pred
    )
    return x, y


# y = -1인 라벨에 대해서는 [0, 0, ... , 0]으로 OHE, 나머지는 나머지에 대해서만 OHE (전체 라벨 수 = OHE 비트 수 = "-1"이 아닌 라벨의 개수)
def to_categorical_with_mask(y, num_classes):
    y = np.array(y)
    mask = (y != -1)
    y_ohe = np.zeros((len(y), num_classes))
    y_ohe[mask] = to_categorical(y[mask], num_classes=num_classes)

    # mask는 sample_weight 등으로 사용 가능 (-1인 라벨들은 False->0.0, -1이 아닌 라벨들은 True->1.0)
    return y_ohe, mask.astype(float)    


# Masking된 라벨을 제외한 데이터들에 대해 accuracy 리턴
def masked_accuracy(y_true, y_pred):
    # y_true: one-hot encoded (batch_size, num_classes)
    # y_pred: probabilities

    # mask: [True, False, True, ...]
    mask = tf.reduce_sum(y_true, axis=-1) > 0

    y_true_class = tf.argmax(y_true, axis=-1)
    y_pred_class = tf.argmax(y_pred, axis=-1)

    correct = tf.cast(tf.equal(y_true_class, y_pred_class), tf.float32)
    mask = tf.cast(mask, tf.float32)

    masked_correct = correct * mask         # Masking되지 않은 y에 대해 correct한 개수
    total = tf.reduce_sum(mask)             # 평가 대상이 되는 전체 샘플 수

    return tf.cond(
        tf.equal(total, 0.0),                               # 조건문
        lambda: tf.constant(0.0),                           # 조건문이 True이면 실행
        lambda: tf.reduce_sum(masked_correct) / total       # 조건문이 False이면 실행
    )







