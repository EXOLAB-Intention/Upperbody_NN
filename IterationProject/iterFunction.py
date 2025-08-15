from function import *
from Model import *
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import random
import os


def DataLoader(folder_paths=['DataFile/250813']):
    all_DS = []

    for folder_path in folder_paths:
        DS = Load_files(folder_path, pattern='cropped_*.h5')
        all_DS.extend(DS)

    ### EMG Filtering ###
    all_DS = PostProcess_EMG(all_DS)

    dataset_num = len(all_DS)
    print(f"Total Datasets: {dataset_num}")

    return all_DS


def IMUCalibration(all_DS):
    GetCorrectionTermIMU('DataFile/250813/cropped_Tpose.h5')
    for trial in all_DS:
        CalibrateIMU_3(trial)
    print("IMU Calibration is Done !!!")    


def PlotIMUData(all_DS):
    dataset_num = len(all_DS)
    idx = random.randint(0, dataset_num-1)    # Choose
    print(f"Index: {idx}\n")

    plt.figure(figsize=(12,16))

    plt.subplot(3, 2, 1)
    plt.plot(all_DS[idx]['time'], all_DS[idx]['imu5'][:, 0], label='w')
    plt.plot(all_DS[idx]['time'], all_DS[idx]['imu5'][:, 1], label='x')
    plt.plot(all_DS[idx]['time'], all_DS[idx]['imu5'][:, 2], label='y')
    plt.plot(all_DS[idx]['time'], all_DS[idx]['imu5'][:, 3], label='z')
    plt.xlim(all_DS[idx]['time'][0], all_DS[idx]['time'][-1])  # 필요시 여백 크기 조절 가능
    plt.title("[Pelvis]")
    plt.legend(loc='lower right')
    plt.grid(True)

    plt.subplot(3, 2, 2)
    plt.plot(all_DS[idx]['time'], all_DS[idx]['imu6'][:, 0], label='w')
    plt.plot(all_DS[idx]['time'], all_DS[idx]['imu6'][:, 1], label='x')
    plt.plot(all_DS[idx]['time'], all_DS[idx]['imu6'][:, 2], label='y')
    plt.plot(all_DS[idx]['time'], all_DS[idx]['imu6'][:, 3], label='z')
    plt.xlim(all_DS[idx]['time'][0], all_DS[idx]['time'][-1])  # 필요시 여백 크기 조절 가능
    plt.title("[Trunk]")
    plt.legend(loc='lower right')
    plt.grid(True)

    plt.subplot(3, 2, 3)
    plt.plot(all_DS[idx]['time'], all_DS[idx]['imu7'][:, 0], label='w')
    plt.plot(all_DS[idx]['time'], all_DS[idx]['imu7'][:, 1], label='x')
    plt.plot(all_DS[idx]['time'], all_DS[idx]['imu7'][:, 2], label='y')
    plt.plot(all_DS[idx]['time'], all_DS[idx]['imu7'][:, 3], label='z')
    plt.xlim(all_DS[idx]['time'][0], all_DS[idx]['time'][-1])  # 필요시 여백 크기 조절 가능
    plt.title("[RS]")
    plt.legend(loc='lower right')
    plt.grid(True)

    plt.subplot(3, 2, 4)
    plt.plot(all_DS[idx]['time'], all_DS[idx]['imu8'][:, 0], label='w')
    plt.plot(all_DS[idx]['time'], all_DS[idx]['imu8'][:, 1], label='x')
    plt.plot(all_DS[idx]['time'], all_DS[idx]['imu8'][:, 2], label='y')
    plt.plot(all_DS[idx]['time'], all_DS[idx]['imu8'][:, 3], label='z')
    plt.xlim(all_DS[idx]['time'][0], all_DS[idx]['time'][-1])  # 필요시 여백 크기 조절 가능
    plt.title("[RE]")
    plt.legend(loc='lower right')
    plt.grid(True)

    plt.subplot(3, 2, 5)
    plt.plot(all_DS[idx]['time'], all_DS[idx]['imu9'][:, 0], label='w')
    plt.plot(all_DS[idx]['time'], all_DS[idx]['imu9'][:, 1], label='x')
    plt.plot(all_DS[idx]['time'], all_DS[idx]['imu9'][:, 2], label='y')
    plt.plot(all_DS[idx]['time'], all_DS[idx]['imu9'][:, 3], label='z')
    plt.xlim(all_DS[idx]['time'][0], all_DS[idx]['time'][-1])  # 필요시 여백 크기 조절 가능
    plt.title("[LS]")
    plt.legend(loc='lower right')
    plt.grid(True)

    plt.subplot(3, 2, 6)
    plt.plot(all_DS[idx]['time'], all_DS[idx]['imu10'][:, 0], label='w')
    plt.plot(all_DS[idx]['time'], all_DS[idx]['imu10'][:, 1], label='x')
    plt.plot(all_DS[idx]['time'], all_DS[idx]['imu10'][:, 2], label='y')
    plt.plot(all_DS[idx]['time'], all_DS[idx]['imu10'][:, 3], label='z')
    plt.xlim(all_DS[idx]['time'][0], all_DS[idx]['time'][-1])  # 필요시 여백 크기 조절 가능
    plt.title("[LE]")
    plt.legend(loc='lower right')
    plt.grid(True)

    plt.show()


def PlotEMGData(all_DS):
    dataset_num = len(all_DS)
    idx = random.randint(0, dataset_num-1)    # Choose
    print(f"Index: {idx}\n")

    plt.figure(figsize=(12,15))

    plt.subplot(4, 2, 1)
    plt.plot(all_DS[idx]['time'], all_DS[idx]['emgL1'], label='Raw')
    plt.plot(all_DS[idx]['time'], all_DS[idx]['emgL1_Envelope'], label='Envelope')
    plt.title(f"{all_DS[idx]['file']} - {all_DS[idx]['trial']} [L_RA]")
    plt.legend()
    plt.grid(True)

    plt.subplot(4, 2, 2)
    plt.plot(all_DS[idx]['time'], all_DS[idx]['emgL2'], label='Raw')
    plt.plot(all_DS[idx]['time'], all_DS[idx]['emgL2_Envelope'], label='Envelope')
    plt.title("[L_EO]")
    plt.legend()
    plt.grid(True)

    plt.subplot(4, 2, 3)
    plt.plot(all_DS[idx]['time'], all_DS[idx]['emgL3'], label='Raw')
    plt.plot(all_DS[idx]['time'], all_DS[idx]['emgL3_Envelope'], label='Envelope')
    plt.title("[L_TR]")
    plt.legend()
    plt.grid(True)

    plt.subplot(4, 2, 4)
    plt.plot(all_DS[idx]['time'], all_DS[idx]['emgL4'], label='Raw')
    plt.plot(all_DS[idx]['time'], all_DS[idx]['emgL4_Envelope'], label='Envelope')
    plt.title("[L_PD]")
    plt.legend()
    plt.grid(True)

    plt.subplot(4, 2, 5)
    plt.plot(all_DS[idx]['time'], all_DS[idx]['emgR1'], label='Raw')
    plt.plot(all_DS[idx]['time'], all_DS[idx]['emgR1_Envelope'], label='Envelope')
    plt.title("[R_RA]")
    plt.legend()
    plt.grid(True)

    plt.subplot(4, 2, 6)
    plt.plot(all_DS[idx]['time'], all_DS[idx]['emgR2'], label='Raw')
    plt.plot(all_DS[idx]['time'], all_DS[idx]['emgR2_Envelope'], label='Envelope')
    plt.title("[R_EO]")
    plt.legend()
    plt.grid(True)

    plt.subplot(4, 2, 7)
    plt.plot(all_DS[idx]['time'], all_DS[idx]['emgR3'], label='Raw')
    plt.plot(all_DS[idx]['time'], all_DS[idx]['emgR3_Envelope'], label='Envelope')
    plt.title("[R_TR]")
    plt.legend()
    plt.grid(True)

    plt.subplot(4, 2, 8)
    plt.plot(all_DS[idx]['time'], all_DS[idx]['emgR4'], label='Raw')
    plt.plot(all_DS[idx]['time'], all_DS[idx]['emgR4_Envelope'], label='Envelope')
    plt.title("[R_PD]")
    plt.legend()
    plt.grid(True)

    plt.show()



def DataPreprocessing(all_trials, params):  
    trial_keys = [(trial_data['file'], trial_data['trial']) for trial_data in all_trials]      # (filename, trial_number) 형태
    trial_dict = {k: t for k, t in zip(trial_keys, all_trials)}                                # {(filename, trial_number): trial_data, ...} 형태
    
    input_keys = ['emgL1_Envelope','emgL2_Envelope','emgL3_Envelope','emgL4_Envelope', 'emgR1_Envelope','emgR2_Envelope','emgR3_Envelope','emgR4_Envelope', 'imu5', 'imu6', 'imu7', 'imu8', 'imu9', 'imu10']

    train_keys, val_keys, test_keys = split_trials_train_val_test(trial_keys, val_ratio=0.1, test_ratio=0.1)
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
    y_train_ohe = to_categorical(y_train, num_classes)
    y_val_ohe = to_categorical(y_val, num_classes)
    y_test_ohe = to_categorical(y_test, num_classes)

    print(f"Total classes: {num_classes}")
    print("Train:", x_train.shape, "Val:", x_val.shape, "Test:", x_test.shape)

    return x_train, y_train_ohe, x_val, y_val_ohe, x_test, y_test_ohe



def TrainingModel(x_train, y_train, x_val, y_val, input_shape, num_classes, params):    
    # early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

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
        epochs=params["Epoch"],
        batch_size=params["BatchSize"],
    )

    return model, history



def PlotAccuracy(history, param, idx, train_acc, val_acc):
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.title("Classification Accuracy")

    training_acc = train_acc * 100
    validation_acc = val_acc * 100

    # 설명 텍스트
    if param:
        info = (
            f"Model: LSTM{param['LSTM_units']}, Dense{param['Dense_units']}\n"
            f"Stride: {param['Stride']}\n"
            f"Window size: {param['WindowSize']}\n"
            f"Learning rate: {param['LearningRate']}\n"
            f"Dropout rate: {param['Dropout']}\n"
            f"Batch size: {param['BatchSize']}\n"
            f"Epoch: {param['Epoch']}\n"
            f"Training Accuracy; {training_acc}%\n"
            f"Validation Accuracy; {validation_acc}%"
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
    os.makedirs("IterationProject/Result", exist_ok=True)  # 폴더 없으면 생성
    plt.savefig(f"IterationProject/Result/accuracy_plot_{idx}.png", dpi=300)
    plt.close()  # 저장 후 창 닫기 (메모리 절약)




def PlotHyperparamComparison(train_acc_list, val_acc_list, chunk_size=30):
    """
    train_acc_list, val_acc_list : {global_idx: accuracy} 형태의 dict (동일 순서 가정)
    chunk_size : 파트당 막대 개수
    """
    # 0) 키 순서: 정렬/집합 X, 그대로 사용
    keys = list(train_acc_list.keys())
    # (안전망) val 키 순서가 다르면 경고만 출력하고, 교집합 기준으로 맞춥니다.
    val_keys = list(val_acc_list.keys())
    if keys != val_keys:
        print("[WARN] train/val 키 순서가 다릅니다. 교집합 기준으로 정렬 없이 맞춥니다.")
        keys = [k for k in keys if k in val_acc_list]

    # 1) 청크 분할(원래 순서 유지)
    chunks = [keys[i:i+chunk_size] for i in range(0, len(keys), chunk_size)]

    # 2) 폴더 준비
    os.makedirs("IterationProject/Result/Training", exist_ok=True)
    os.makedirs("IterationProject/Result/Validation", exist_ok=True)

    # 3) 파트별 저장
    for part_idx, keys_chunk in enumerate(chunks, start=1):
        # ---- Training ----
        train_vals = [train_acc_list[k] for k in keys_chunk]
        x = np.arange(len(keys_chunk)) * 2

        plt.figure(figsize=(16, 8))
        plt.bar(x, train_vals, width=1.0)
        plt.xticks(x, keys_chunk, rotation=45)
        plt.xlabel("Model index (global)")
        plt.ylabel("Training Accuracy")
        plt.grid(axis='y', linestyle='--', alpha=0.5)

        rng = (keys_chunk[0], keys_chunk[-1])
        plt.title(f"Training Accuracy Comparison (Part {part_idx}: idx {str(rng[0]+1)}–{str(rng[1]+1)})")

        # 최고값 라벨
        max_idx = int(np.nanargmax(train_vals))
        max_val = float(train_vals[max_idx])
        plt.text(x[max_idx], max_val, f"{max_val:.4f}",
                 ha='center', va='bottom', fontsize=10, fontweight='bold', color='red')

        plt.tight_layout()
        plt.savefig(f"IterationProject/Result/Training/Train_part{part_idx}_idx{str(rng[0]+1)}–{str(rng[1]+1)}.png", dpi=300)
        plt.close()

        # ---- Validation ----
        val_vals = [val_acc_list[k] for k in keys_chunk]
        x = np.arange(len(keys_chunk)) * 2

        plt.figure(figsize=(16, 8))
        plt.bar(x, val_vals, width=1.0)
        plt.xticks(x, keys_chunk, rotation=45)
        plt.xlabel("Model index (global)")
        plt.ylabel("Validation Accuracy")
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.title(f"Validation Accuracy Comparison (Part {part_idx}: idx {str(rng[0]+1)}–{str(rng[1]+1)})")


        # 최고값 라벨
        max_idx = int(np.nanargmax(val_vals))
        max_val = float(val_vals[max_idx])
        plt.text(x[max_idx], max_val, f"{max_val:.4f}",
                 ha='center', va='bottom', fontsize=10, fontweight='bold', color='red')

        plt.tight_layout()
        plt.savefig(f"IterationProject/Result/Validation/Val_part{part_idx}_idx{str(rng[0]+1)}–{str(rng[1]+1)}.png", dpi=300)
        plt.close()