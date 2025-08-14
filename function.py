import h5py
import os
import glob
import numpy as np


# ---- LABEL ---- #
# 0: Stand 
# 1: Stand-to-Sit (APA)
# 2: Sit
# 3: Sit-to-Stand (APA)
# 4: Gait Initiation (APA / Left foot)
# 5: Left Foot Forward
# 6: Left Swing (APA)
# 7: Right Foot Forward
# 8: Right Swing (APA)
# 9: Gait Termination (APA / Left foot)


# ---------------------------------------------------- Data Loading ---------------------------------------------------- #
def Load_files(folder_path, pattern='cropped_*.h5'):
    all_DS = []

    file_list = sorted(glob.glob(os.path.join(folder_path, pattern)))
    print(f"Total {len(file_list)} files are now loaded...")

    for file_path in file_list:
        if 'STEADY_gait_init' in file_path:
            Label = 0
        elif 'APA_Stand-to-Sit' in file_path:
            Label = 1
        elif 'STEADY_Sit-to-Stand' in file_path:
            Label = 2
        elif 'APA_Sit-to-Stand' in file_path:
            Label = 3
        elif 'APA_gait_init' in file_path:
            Label = 4
        elif 'STEADY_RightSwing' in file_path:
            Label = 5
        elif 'APA_LeftSwing' in file_path:
            Label = 6
        elif 'STEADY_LeftSwing' in file_path:
            Label = 7
        elif 'APA_RightSwing' in file_path:
            Label = 8
        elif 'APA_GaitTermination' in file_path:
            Label = 9


        with h5py.File(file_path, "r") as f:
            for trial_name in f.keys():
                trial = f[trial_name]
                time_data = trial['time'][:]

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

                    'imu5':  trial['imu5'][:],
                    'imu6':  trial['imu6'][:],
                    'imu7':  trial['imu7'][:],
                    'imu8':  trial['imu8'][:],
                    'imu9':  trial['imu9'][:],
                    'imu10':  trial['imu10'][:],

                    'rising_ok': trial['rising_ok'][:],
                    'falling_ok': trial['falling_ok'][:],

                    'Label': np.full(time_data.size, Label, dtype=int)
                }

                all_DS.append(trial_data)

    print(f"Total {len(all_DS)} DataSets are completely loaded !!")
    return all_DS



# ---------------------------------------------------- EMG Filtering ---------------------------------------------------- #
# emgL1: Left Rectus Abdominis (L_RA)
# emgL2: Left External Oblique (L_EO)
# emgL3: Left Trapezius (L_TR)
# emgL4: Left Posterior Deltoid (L_PD)
# emgR1: Right Rectus Abdominis (R_RA)
# emgR2: Right External Oblique (R_EO)
# emgR3: Right Trapezius (R_TR)
# emgR4: Right Posterior Deltoid (R_PD)


# 10Hz High-pass filter
def HPF_1NE(data):
    data = np.array(data)
    data_hpf = []

    for idx, u in enumerate(data):
        if (idx == 0):
            y1 = 0
            u1 = 0
            y = u
        else:
            y = 0.969540972048579 * u - 0.969540972048579 * u1 +  0.939081944097158 * y1
        y1 = y
        u1 = u
        data_hpf.append(y)
    return data_hpf

# 200Hz Low-pass filter
def LPF_1NE(data):
    data = np.array(data)
    data_lpf = []

    for idx, u in enumerate(data):
        if (idx == 0):
            y1 = 0
            u1 = 0
            y = u
        else:
            y = 0.385869545095038 * u + 0.385869545095038 * u1 +  0.228260909809925 * y1
        y1 = y
        u1 = u
        data_lpf.append(y)
    return data_lpf

# Rectification filter
def Rectify_1NE(data):
    data = np.array(data)
    data_rect = []

    for idx, u in enumerate(data):
        if (u >= 0):
            y = u
        else:
            y = -u

        data_rect.append(y)
    return data_rect

# Normalization filter
def Normalize_1NE(data):
    data = np.array(data)
    data_norm = []

    for idx, u in enumerate(data):
        if (u < 0):
            y = -u
        data_norm.append(y)

    return data_norm

# RMS filter
def RMS_1NE(data, num=5):
    data = np.array(data)
    data_MA = []

    for idx in range(len(data)):
        if idx < num:
            window = data[0:idx+1]  # 최소 1개는 포함
        else:
            window = data[idx - num + 1:idx + 1]

        rms = np.sqrt(np.mean(window**2, axis=0))
        data_MA.append(rms)

    return np.array(data_MA)

def PostProcess_EMG(all_DataSet):
    processed_DS = []
    emg_keys = ['emgL1', 'emgL2', 'emgL3', 'emgL4', 'emgR1', 'emgR2', 'emgR3', 'emgR4']
    imu_keys = ['imu5', 'imu6', 'imu7', 'imu8', 'imu9', 'imu10']

    for DS in all_DataSet:
        DS_copy = {}

        for key, value in DS.items():
            if key in ['file', 'trial']:
                DS_copy[key] = value

            elif key in emg_keys:
                raw = np.squeeze(value)
                raw_rect = Rectify_1NE(raw)
                raw_hpf = HPF_1NE(raw_rect)
                raw_lpf = LPF_1NE(raw_hpf)
                raw_rms = RMS_1NE(raw_lpf)
                # raw_hpf = HPF_1NE(raw)
                # raw_lpf = LPF_1NE(raw_hpf)
                # raw_rect = Rectify_1NE(raw_lpf)
                # raw_rms = RMS_1NE(raw_rect)
                DS_copy[key] = raw_rect                     # EMG raw 저장 (예: 'emgL1')
                DS_copy[f"{key}_Envelope"] = raw_rms        # EMG envelope 저장 (예: 'emgL1_Envelope')

            elif key in imu_keys:
                DS_copy[key] = value

            elif key == 'time':
                DS_copy[key] = [i * 10 for i in range(len(np.squeeze(value)))]       # time에 대해, 0부터 시작으로 변경

            else:
                DS_copy[key] = np.squeeze(value)        # button_ok 등
            
        processed_DS.append(DS_copy)

    return processed_DS




# ---------------------------------------------------- AI Learning ---------------------------------------------------- #
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
    all_DS,
    input_keys,
    trial_names_to_use=None,
    window_size=20,
    stride=1,
    pred = 0
):
    """
    Build an LSTM dataset using custom input keys.
    
    Parameters:
    - all_DS: list of trial dicts
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

    for trial in all_DS:
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
        labels = trial['Label']
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
        all_DS=selected_trials,
        input_keys=input_keys,
        window_size=window_size,
        stride=stride,
        pred=pred
    )
    return x, y



phase_to_int = {
    'Stand': 0,
    'Stand-to-Sit (APA)': 1,
    'Sit': 2,
    'Sit-to-Stand (APA)': 3 ,
    'Gait Initiation (APA / Left foot)': 4,
    'Left Foot Forward': 5,
    'Left Swing (APA)': 6,
    'Right Foot Forward': 7,
    'Right Swing (APA)': 8,
    'Gait Termination (APA / Left foot)': 9
}



