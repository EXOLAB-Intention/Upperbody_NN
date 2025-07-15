from Functions import *
from Model import *
import wandb
from wandb.integration.keras import WandbCallback
# from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint # 다른 버젼인데, wandb 0.16~ 0.17 에서만 됨
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import random



# MODE = 0            # ALL_LABEL_ALL_SENSOR
# MODE = 1            # ALL_LABEL_LOWER_SENSOR
# MODE = 2            # ALL_LABEL_UPPER_SENSOR
MODE = 3            # Transient_LABEL_ALL_SENSOR
# MODE = 4            # Transient_LABEL_ALL_SENSOR_NO_STOPWALKING
# MODE = 5            # Transient_LABEL_LOWER_SENSOR
# MODE = 6            # Transient_LABEL_UPPER_SENSOR
# MODE = 7            # Add "Start Walking" 
# MODE = 8          # Transient_LABEL_ALL_SENSOR_WITH_VELOCITY

Model = 0          # LSTM  (0.9333)
# Model = 1           # CNN (0.9120)
# Model = 2           # TCN (0.9133)
# Model = 3           # LSTM + CNN (0.9593)
# Model = 4           # CNN + LSTM 
# Model = 5           # Transformer

IMU_MODE = 0             # Quaternion
# IMU_MODE = 1             # Euler Angle
# IMU_MODE = 2             # 6D Representation



def DataLoader(folder_paths=['DataFile/data_250612', 'DataFile/data_250616', 'DataFile/data_250618']):
    all_trials = []

    for folder_path in folder_paths:
        trials = Load_files(folder_path, pattern='cropped_*.h5')
        trials = [trial for trial in trials if 'test' not in trial['file']]     # Temporary remove Test set
        all_trials.extend(trials)

    ## Labeling ##
    for trial in all_trials:
        trial['label'] = Generate_labels(trial)
        Convert_labels_to_int(trial, phase_to_int)

    ## EMG filtering ##
    all_trials = process_all_emg(all_trials, lpf_fc=5, norm_method='max')

    trial_num = len(all_trials)
    print(f"Total trials: {trial_num}")

    return all_trials


def IMUCalibration(all_trials):
    for trial in all_trials:
        if (IMU_MODE == 0):
            # CalibrateIMU(trial)
            # CalibrateIMU_2(trial)
            CalibrateIMU_3(trial)
        elif (IMU_MODE == 1):
            # CalibrateIMU_5(trial)
            # CalibrateIMU_6(trial)
            # CalibrateIMU_7(trial)
            CalibrateIMU_vel(trial)
        elif (IMU_MODE == 2):
            CalibrateIMU_4(trial)

    print("IMU Calibration is Done !!!")    


def PlotIMUData(all_trials):
    trial_num = len(all_trials)
    idx1 = random.randint(0, trial_num-1) 
    idx2 = random.randint(0, trial_num-1)  
    while idx2 == idx1:
        idx2 = random.randint(0, trial_num-1)

    IMU_SEL = 'imu1'

    if (IMU_SEL == 'imu1'):
        EULER_SEL = 'rhip'
        NAME = 'RH'
    elif (IMU_SEL == 'imu2'):
        EULER_SEL = 'rknee'
        NAME = 'RK_vel'
    elif (IMU_SEL == 'imu3'):
        EULER_SEL = 'lhip'
        NAME = 'LH_vel'
    elif (IMU_SEL == 'imu4'):
        EULER_SEL = 'lknee'
        NAME = 'LK_vel'
    elif (IMU_SEL == 'imu5'):
        EULER_SEL = 'trunk'
        NAME = 'Pelvis_vel'
    elif (IMU_SEL == 'imu6'):
        EULER_SEL = 'trunk'
        NAME = 'Trunk_vel'
    elif (IMU_SEL == 'imu7'):
        EULER_SEL = 'rshoulder'
        NAME = 'RS_vel'
    elif (IMU_SEL == 'imu8'):
        EULER_SEL = 'relbow'
        NAME = 'RE_vel'
    elif (IMU_SEL == 'imu9'):
        EULER_SEL = 'lshoulder'
        NAME = 'LS_vel'
    elif (IMU_SEL == 'imu10'):
        EULER_SEL = 'lelbow'
        NAME = 'LE_vel'


    ### Quaternion ###
    if (IMU_MODE == 0):
        plt.figure(figsize=(16,6))

        plt.subplot(1, 2, 1)
        plt.plot(all_trials[idx1]['time'], all_trials[idx1][IMU_SEL][:, 0], label='w')
        plt.plot(all_trials[idx1]['time'], all_trials[idx1][IMU_SEL][:, 1], label='x')
        plt.plot(all_trials[idx1]['time'], all_trials[idx1][IMU_SEL][:, 2], label='y')
        plt.plot(all_trials[idx1]['time'], all_trials[idx1][IMU_SEL][:, 3], label='z')
        plt.figtext(0.5, 0.91, NAME, fontsize=12, ha='center')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(all_trials[idx2]['time'], all_trials[idx2][IMU_SEL][:, 0], label='w')
        plt.plot(all_trials[idx2]['time'], all_trials[idx2][IMU_SEL][:, 1], label='x')
        plt.plot(all_trials[idx2]['time'], all_trials[idx2][IMU_SEL][:, 2], label='y')
        plt.plot(all_trials[idx2]['time'], all_trials[idx2][IMU_SEL][:, 3], label='z')
        plt.figtext(0.5, 0.91, NAME, fontsize=12, ha='center')
        plt.legend()
        plt.grid(True)

        plt.suptitle(f"Dataset Comparison: {idx1} vs {idx2}", fontsize=16, y=1.02)
        plt.show()


    ### Euler Angle ###
    elif (IMU_MODE == 1):
        plt.figure(figsize=(16,8))

        plt.subplot(2, 2, 1)
        plt.plot(all_trials[idx1]['time'], all_trials[idx1][IMU_SEL][:, 0], label='Roll')
        plt.plot(all_trials[idx1]['time'], all_trials[idx1][IMU_SEL][:, 1], label='Pitch')
        plt.plot(all_trials[idx1]['time'], all_trials[idx1][IMU_SEL][:, 2], label='Yaw')
        plt.figtext(0.5, 0.91, NAME, fontsize=12, ha='center')
        plt.title(f"{idx1} - Euler_Python")
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 2, 2)
        plt.plot(all_trials[idx2]['time'], all_trials[idx2][IMU_SEL][:, 0], label='Roll')
        plt.plot(all_trials[idx2]['time'], all_trials[idx2][IMU_SEL][:, 1], label='Pitch')
        plt.plot(all_trials[idx2]['time'], all_trials[idx2][IMU_SEL][:, 2], label='Yaw')
        plt.figtext(0.5, 0.91, NAME, fontsize=12, ha='center')
        plt.title(f"{idx2} - Euler_Python")
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 2, 3)
        plt.plot(all_trials[idx1]['time'], all_trials[idx1][EULER_SEL][2, :], label='Roll')
        plt.plot(all_trials[idx1]['time'], all_trials[idx1][EULER_SEL][1, :], label='Pitch')
        plt.plot(all_trials[idx1]['time'], all_trials[idx1][EULER_SEL][0, :], label='Yaw')
        plt.figtext(0.5, 0.91, NAME, fontsize=12, ha='center')
        plt.title(f"{idx1} - Euler_MATLAB")
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 2, 4)
        plt.plot(all_trials[idx2]['time'], all_trials[idx2][EULER_SEL][2, :], label='Roll')
        plt.plot(all_trials[idx2]['time'], all_trials[idx2][EULER_SEL][1, :], label='Pitch')
        plt.plot(all_trials[idx2]['time'], all_trials[idx2][EULER_SEL][0, :], label='Yaw')
        plt.figtext(0.5, 0.91, NAME, fontsize=12, ha='center')
        plt.title(f"{idx2} - Euler_MATLAB")
        plt.legend()
        plt.grid(True)

        plt.suptitle(f"Dataset Comparison: {idx1} vs {idx2}", fontsize=16, y=1.02)
        plt.show()


    ### 6D Representation ###
    elif (IMU_MODE == 2):
        plt.figure(figsize=(16,6))

        plt.subplot(1, 2, 1)
        plt.plot(all_trials[idx1]['time'], all_trials[idx1][IMU_SEL][:, 0], label='R1')
        plt.plot(all_trials[idx1]['time'], all_trials[idx1][IMU_SEL][:, 1], label='R2')
        plt.plot(all_trials[idx1]['time'], all_trials[idx1][IMU_SEL][:, 2], label='R3')
        plt.plot(all_trials[idx1]['time'], all_trials[idx1][IMU_SEL][:, 3], label='R4')
        plt.plot(all_trials[idx1]['time'], all_trials[idx1][IMU_SEL][:, 4], label='R5')
        plt.plot(all_trials[idx1]['time'], all_trials[idx1][IMU_SEL][:, 5], label='R6')
        plt.figtext(0.5, 0.91, NAME, fontsize=12, ha='center')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(all_trials[idx2]['time'], all_trials[idx2][IMU_SEL][:, 0], label='R1')
        plt.plot(all_trials[idx2]['time'], all_trials[idx2][IMU_SEL][:, 1], label='R2')
        plt.plot(all_trials[idx2]['time'], all_trials[idx2][IMU_SEL][:, 2], label='R3')
        plt.plot(all_trials[idx2]['time'], all_trials[idx2][IMU_SEL][:, 3], label='R4')
        plt.plot(all_trials[idx2]['time'], all_trials[idx2][IMU_SEL][:, 4], label='R5')
        plt.plot(all_trials[idx2]['time'], all_trials[idx2][IMU_SEL][:, 5], label='R6')
        plt.figtext(0.5, 0.91, NAME, fontsize=12, ha='center')
        plt.legend()
        plt.grid(True)

        plt.suptitle(f"Dataset Comparison: {idx1} vs {idx2}", fontsize=16, y=1.02)
        plt.show()


def CheckTotalData(all_trials):
    trial_num = len(all_trials)
    idx = random.randint(0, trial_num-1) 

    trial = all_trials[idx]         # trial_1, trial_10, trial_2, ... 순이어서 [1] -> trial_10 데이터임
    time = trial['time']
    label_int = trial['label_int']
    rising_ok = trial['rising_ok']
    falling_ok = trial['falling_ok']

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

    # 위쪽: 라벨
    axes[0].plot(time, trial['label_int'], drawstyle='steps-post', color='black', linewidth=2.5)
    axes[0].plot(time, rising_ok * max(trial['label_int']), linestyle='--', alpha=0.6, color='red', label='Labeling Start')
    axes[0].plot(time, falling_ok * max(trial['label_int']), linestyle='--', alpha=0.6, color='blue', label='Labeling Stop')
    axes[0].set_yticks(list(phase_to_int.values()))
    axes[0].set_yticklabels(list(phase_to_int.keys()))
    axes[0].set_ylabel("Phase Label")
    axes[0].legend(loc='upper right')
    axes[0].grid(True)

    # 중간: EMG
    if (MODE == 0 or MODE == 3 or MODE == 4 or MODE == 7 or MODE == 8):
        axes[1].plot(time, trial['emgL1_norm'], label='emg_L-VM')
        axes[1].plot(time, trial['emgL2_norm'], label='emg_L-MG')
        axes[1].plot(time, trial['emgL3_norm'], label='emg_L-TR')
        axes[1].plot(time, trial['emgL4_norm'], label='emg_L-ES')
        axes[1].plot(time, trial['emgR1_norm'], label='emg_R-VM')
        axes[1].plot(time, trial['emgR2_norm'], label='emg_R-MG')
        axes[1].plot(time, trial['emgR3_norm'], label='emg_R-TR')
        axes[1].plot(time, trial['emgR4_norm'], label='emg_R-ES')
    elif (MODE == 1 or MODE == 5):
        axes[1].plot(time, trial['emgL1_norm'], label='emg_L-VM')
        axes[1].plot(time, trial['emgL2_norm'], label='emg_L-MG')
        axes[1].plot(time, trial['emgR1_norm'], label='emg_R-VM')
        axes[1].plot(time, trial['emgR2_norm'], label='emg_R-MG')
    elif (MODE == 2 or MODE == 6):
        axes[1].plot(time, trial['emgL3_norm'], label='emg_L-TR')
        axes[1].plot(time, trial['emgL4_norm'], label='emg_L-ES')
        axes[1].plot(time, trial['emgR3_norm'], label='emg_R-TR')
        axes[1].plot(time, trial['emgR4_norm'], label='emg_R-ES')
    axes[1].set_ylabel("EMG")
    axes[1].set_xlabel("Time[msec]")
    axes[1].legend()
    axes[1].grid(True)

    # 아래쪽: IMU
    if (MODE == 0 or MODE == 3 or MODE == 4 or MODE == 7):
        axes[2].plot(time, trial['imu1'][:,3], label='imu1_RH')
        axes[2].plot(time, trial['imu2'][:,3], label='imu2_RK')
        axes[2].plot(time, trial['imu3'][:,3], label='imu3_LH')
        axes[2].plot(time, trial['imu4'][:,3], label='imu4_LK')
        axes[2].plot(time, trial['imu5'][:,3], label='imu5_Pelvis')
        axes[2].plot(time, trial['imu6'][:,3], label='imu6_Trunk')
        axes[2].plot(time, trial['imu7'][:,3], label='imu7_RS')
        axes[2].plot(time, trial['imu8'][:,3], label='imu8_RE')
        axes[2].plot(time, trial['imu9'][:,3], label='imu9_LS')
        axes[2].plot(time, trial['imu10'][:,3], label='imu10_LE')
    elif (MODE == 1 or MODE == 5):
        axes[2].plot(time, trial['imu1'][:,3], label='imu1_RH')
        axes[2].plot(time, trial['imu2'][:,3], label='imu2_RK')
        axes[2].plot(time, trial['imu3'][:,3], label='imu3_LH')
        axes[2].plot(time, trial['imu4'][:,3], label='imu4_LK')
        axes[2].plot(time, trial['imu5'][:,3], label='imu5_Pelvis')
        axes[2].plot(time, trial['imu6'][:,3], label='imu6_Trunk')
    elif (MODE == 2 or MODE == 6):
        axes[2].plot(time, trial['imu5'][:,3], label='imu5_Pelvis')
        axes[2].plot(time, trial['imu6'][:,3], label='imu6_Trunk')
        axes[2].plot(time, trial['imu7'][:,3], label='imu7_RS')
        axes[2].plot(time, trial['imu8'][:,3], label='imu8_RE')
        axes[2].plot(time, trial['imu9'][:,3], label='imu9_LS')
        axes[2].plot(time, trial['imu10'][:,3], label='imu10_LE')
    elif (MODE == 8):
        axes[2].plot(time, trial['imu1_vel'][:,2], label='imu1_RH')
        axes[2].plot(time, trial['imu2_vel'][:,2], label='imu2_RK')
        axes[2].plot(time, trial['imu3_vel'][:,2], label='imu3_LH')
        axes[2].plot(time, trial['imu4_vel'][:,2], label='imu4_LK')
        axes[2].plot(time, trial['imu5_vel'][:,2], label='imu5_Pelvis')
        axes[2].plot(time, trial['imu6_vel'][:,2], label='imu6_Trunk')
        axes[2].plot(time, trial['imu7_vel'][:,2], label='imu7_RS')
        axes[2].plot(time, trial['imu8_vel'][:,2], label='imu8_RE')
        axes[2].plot(time, trial['imu9_vel'][:,2], label='imu9_LS')
        axes[2].plot(time, trial['imu10_vel'][:,2], label='imu10_LE')
    axes[2].set_ylabel("IMU")
    axes[2].set_xlabel("Time[msec]")
    axes[2].legend(loc='upper left')
    axes[2].grid(True)

    plt.suptitle(f"{trial['file']} - {trial['trial']}")
    plt.tight_layout()
    plt.show()




def DataPreprocessing(all_trials, config):  
    trial_keys = [(trial_data['file'], trial_data['trial']) for trial_data in all_trials]      # (filename, trial_number) 형태
    trial_dict = {k: t for k, t in zip(trial_keys, all_trials)}                                # {(filename, trial_number): trial_data, ...} 형태
    if (MODE == 0 or MODE == 3 or MODE == 4 or MODE == 7):
        input_keys = ['emgL1_norm','emgL2_norm','emgL3_norm','emgL4_norm', 'emgR1_norm','emgR2_norm','emgR3_norm','emgR4_norm', 'imu1', 'imu2', 'imu3', 'imu4', 'imu5', 'imu6', 'imu7', 'imu8', 'imu9', 'imu10']
    elif (MODE == 1 or MODE == 5):
        input_keys = ['emgL1_norm','emgL2_norm', 'emgR1_norm','emgR2_norm', 'imu1', 'imu2', 'imu3', 'imu4', 'imu5', 'imu6']
    elif (MODE == 2 or MODE == 6):
        input_keys = ['emgL3_norm','emgL4_norm', 'emgR3_norm','emgR4_norm', 'imu5', 'imu6', 'imu7', 'imu8', 'imu9', 'imu10']
    elif (MODE == 8):
        input_keys = ['emgL1_norm','emgL2_norm','emgL3_norm','emgL4_norm', 'emgR1_norm','emgR2_norm','emgR3_norm','emgR4_norm', 
                    'imu1', 'imu2', 'imu3', 'imu4', 'imu5', 'imu6', 'imu7', 'imu8', 'imu9', 'imu10',
                    'imu1_vel', 'imu2_vel', 'imu3_vel', 'imu4_vel', 'imu5_vel', 'imu6_vel', 'imu7_vel', 'imu8_vel', 'imu9_vel', 'imu10_vel']

    train_keys, val_keys, test_keys = split_trials_train_val_test(trial_keys, val_ratio=0.2, test_ratio=0.01)
    # 해결: list of list → list of tuple로 변환   
    train_keys = [tuple(k) for k in train_keys]     
    val_keys = [tuple(k) for k in val_keys]
    test_keys = [tuple(k) for k in test_keys]

    windowSize = config.windowSize
    Stride = config.stride
    x_train, y_train = build_dataset_from_trial_keys(trial_dict, train_keys, input_keys, window_size = windowSize, stride = Stride, pred = 0)
    x_val,   y_val   = build_dataset_from_trial_keys(trial_dict, val_keys, input_keys, window_size = windowSize, stride = Stride, pred = 0)
    x_test,  y_test  = build_dataset_from_trial_keys(trial_dict, test_keys, input_keys, window_size = windowSize, stride = Stride, pred = 0)

    num_classes = len(np.unique(y_train[y_train != -1]))  
    y_train_ohe, train_mask = to_categorical_with_mask(y_train, num_classes)
    y_val_ohe, val_mask = to_categorical_with_mask(y_val, num_classes)
    y_test_ohe, test_mask = to_categorical_with_mask(y_test, num_classes)

    print(f"Total classes: {num_classes}")
    print("Train:", x_train.shape, "Val:", x_val.shape, "Test:", x_test.shape)

    return x_train, y_train_ohe, x_val, y_val_ohe, x_test, y_test_ohe



def DataPreprocessing2(all_trials, params):  
    trial_keys = [(trial_data['file'], trial_data['trial']) for trial_data in all_trials]      # (filename, trial_number) 형태
    trial_dict = {k: t for k, t in zip(trial_keys, all_trials)}                                # {(filename, trial_number): trial_data, ...} 형태
    if (MODE == 0 or MODE == 3 or MODE == 4 or MODE == 7):
        input_keys = ['emgL1_norm','emgL2_norm','emgL3_norm','emgL4_norm', 'emgR1_norm','emgR2_norm','emgR3_norm','emgR4_norm', 'imu1', 'imu2', 'imu3', 'imu4', 'imu5', 'imu6', 'imu7', 'imu8', 'imu9', 'imu10']
    elif (MODE == 1 or MODE == 5):
        input_keys = ['emgL1_norm','emgL2_norm', 'emgR1_norm','emgR2_norm', 'imu1', 'imu2', 'imu3', 'imu4', 'imu5', 'imu6']
    elif (MODE == 2 or MODE == 6):
        input_keys = ['emgL3_norm','emgL4_norm', 'emgR3_norm','emgR4_norm', 'imu5', 'imu6', 'imu7', 'imu8', 'imu9', 'imu10']
    elif (MODE == 8):
        input_keys = ['emgL1_norm','emgL2_norm','emgL3_norm','emgL4_norm', 'emgR1_norm','emgR2_norm','emgR3_norm','emgR4_norm', 
                    'imu1', 'imu2', 'imu3', 'imu4', 'imu5', 'imu6', 'imu7', 'imu8', 'imu9', 'imu10',
                    'imu1_vel', 'imu2_vel', 'imu3_vel', 'imu4_vel', 'imu5_vel', 'imu6_vel', 'imu7_vel', 'imu8_vel', 'imu9_vel', 'imu10_vel']

    train_keys, val_keys, test_keys = split_trials_train_val_test(trial_keys, val_ratio=0.2, test_ratio=0.01)
    # 해결: list of list → list of tuple로 변환   
    train_keys = [tuple(k) for k in train_keys]     
    val_keys = [tuple(k) for k in val_keys]
    test_keys = [tuple(k) for k in test_keys]

    windowSize = params["WindowSize"]
    Stride = params["Stride"]
    x_train, y_train = build_dataset_from_trial_keys(trial_dict, train_keys, input_keys, window_size = windowSize, stride = Stride, pred = 0)
    x_val,   y_val   = build_dataset_from_trial_keys(trial_dict, val_keys, input_keys, window_size = windowSize, stride = Stride, pred = 0)
    x_test,  y_test  = build_dataset_from_trial_keys(trial_dict, test_keys, input_keys, window_size = windowSize, stride = Stride, pred = 0)

    num_classes = len(np.unique(y_train[y_train != -1]))  
    y_train_ohe, train_mask = to_categorical_with_mask(y_train, num_classes)
    y_val_ohe, val_mask = to_categorical_with_mask(y_val, num_classes)
    y_test_ohe, test_mask = to_categorical_with_mask(y_test, num_classes)

    print(f"Total classes: {num_classes}")
    print("Train:", x_train.shape, "Val:", x_val.shape, "Test:", x_test.shape)

    return x_train, y_train_ohe, x_val, y_val_ohe, x_test, y_test_ohe



def TrainingModel(x_train, y_train, x_val, y_val, input_shape, num_classes, config):    
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

    if (Model == 0):
        model = build_LSTM_classifier(
            input_shape=input_shape,
            num_classes=num_classes,
            dropout_rate=config.dropout_rate,
            lstm_units=config.lstm_units,
            dense_units=config.dense_units,
            learning_rate=config.learning_rate,
            use_batchnorm=True
        )

    model.summary() # 모델 구조 출력


    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=100,
        batch_size=config.batch_size,
        # callbacks=[
        #     WandbMetricsLogger(),           # wandb에 metric 기록
        #     WandbModelCheckpoint("model")   # 모델 체크포인트 저장
        # ]
        callbacks=[
            early_stop,
            # WandbCallback(log_graph=False) # 그래프 로깅(모델구조, 레이어, 연결 등) 비활성화 (이거 키면 안돌아감)
        ]
    )

    return model, history


def TrainingModel2(x_train, y_train, x_val, y_val, input_shape, num_classes, params):    
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

    if (Model == 0):
        model = build_LSTM_classifier(
            input_shape=input_shape,
            num_classes=num_classes,
            dropout_rate=params["Dropout"],
            lstm_units=params["LSTM_units"],
            dense_units=params["Dense_units"],
            learning_rate=params["LearningRate"],
            use_batchnorm=True
        )

    model.summary() # 모델 구조 출력


    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=100,
        batch_size=params["BatchSize"],
        # callbacks=[
        #     WandbMetricsLogger(),           # wandb에 metric 기록
        #     WandbModelCheckpoint("model")   # 모델 체크포인트 저장
        # ]
        callbacks=[
            early_stop,
            # WandbCallback(log_graph=False) # 그래프 로깅(모델구조, 레이어, 연결 등) 비활성화 (이거 키면 안돌아감)
        ]
    )

    return model, history



def PlotAccuracy(history, param, idx):
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.title("Classification Accuracy")

    # 설명 텍스트
    if param:
        info = (
            f"Model: LSTM{param['LSTM_units']}, Dense{param['Dense_units']}\n"
            f"Stride: {param['Stride']}\n"
            f"Window size: {param['WindowSize']}\n"
            f"Learning rate: {param['LearningRate']}\n"
            f"Dropout rate: {param['Dropout']}\n"
            f"Batch size: {param['BatchSize']}"
        )

        plt.text(
            0.98, 0.4, info,                     # ← x는 오른쪽에 고정, y는 legend보다 약간 위
            transform=plt.gca().transAxes,        # ← 축 기준 좌표 (axes 기준)
            fontsize=9,
            ha='right', va='top',
            family='monospace',                   # ← 글자 정렬 깔끔하게
            linespacing=1.3,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')
        )

    # 🔽 여기가 저장하는 부분
    plt.tight_layout()
    plt.savefig(f"Result/accuracy_plot_{idx}.png", dpi=300)
    plt.close()  # 저장 후 창 닫기 (메모리 절약)




def PlotHyperparamComparison(train_acc_list, val_acc_list):
    train_acc_keys = list(train_acc_list.keys())
    train_acc_values = list(train_acc_list.values())
    val_acc_keys = list(val_acc_list.keys())
    val_acc_values = list(val_acc_list.values())

    plt.figure()
    plt.bar(train_acc_keys, train_acc_values)
    plt.xlabel("Model")
    plt.ylabel("Training Accuracy")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.title("Training Accuracy Comparison")
    plt.tight_layout()
    plt.savefig(f"Result/Training/Training_acc_comparison.png", dpi=300)
    plt.close()  # 저장 후 창 닫기 (메모리 절약)

    plt.figure()
    plt.bar(val_acc_keys, val_acc_values)
    plt.xlabel("Model")
    plt.ylabel("Training Accuracy")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.title("Validation Accuracy Comparison")
    plt.tight_layout()
    plt.savefig(f"Result/Validation/Validation_acc_comparison.png", dpi=300)
    plt.close()  # 저장 후 창 닫기 (메모리 절약)