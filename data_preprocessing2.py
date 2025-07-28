from tensorflow.keras.utils import to_categorical
from functions import *

def preprocess_data2(all_trial_data_processed):
    # input_keys = ['emgR1_norm','emgR2_norm','emgR3_norm','emgR4_norm','emgL1_norm','emgL2_norm','emgL3_norm','emgL4_norm','hipR_sag','hipL_sag','kneeR_sag','kneeL_sag', 'torso_sag'
    #               ,'hipR_vel_sag','hipL_vel_sag','kneeR_vel_sag','kneeL_vel_sag', 'torso_vel_sag']
    input_keys = ['emgR1_filt','emgR2_filt','emgR3_filt','emgR4_filt','emgL1_filt','emgL2_filt','emgL3_filt','emgL4_filt','elbowR_sag','elbowL_sag','shoulderR_sag','shoulderL_sag', 'trunk_sag'
                ,'elbowR_vel_sag','elbowL_vel_sag','shoulderR_vel_sag','shoulderL_vel_sag', 'trunk_vel_sag']
    window_size = 50
    stride = 5
    pred = 0

    filtered_trials = [trial for trial in all_trial_data_processed if 'test' not in trial['file']]
    trial_keys = [(t['file'], t['trial']) for t in filtered_trials]
    trial_dict = {k: t for k, t in zip(trial_keys, filtered_trials)}



    train_keys, val_keys, _ = split_trials_train_val_test(trial_keys, test_ratio=0, val_ratio=0.2)
    test_trials = [trial for trial in all_trial_data_processed if 'test' in trial['file']]
    test_keys = [(t['file'], t['trial']) for t in test_trials]
    test_dict = {k: t for k, t in zip(test_keys, test_trials)}
    # 해결: list of list → list of tuple로 변환
    train_keys = [tuple(k) for k in train_keys]
    val_keys = [tuple(k) for k in val_keys]
    test_keys = [tuple(k) for k in test_keys]

    X_train, y_train = build_dataset_from_trial_keys(trial_dict, train_keys, input_keys, output='label_int', window_size=window_size, stride=stride, pred=pred)
    X_val, y_val = build_dataset_from_trial_keys(trial_dict, val_keys, input_keys, output='label_int', window_size=window_size, stride=stride, pred=pred)
    X_test, y_test = build_dataset_from_trial_keys(test_dict, test_keys, input_keys, output='label_int', window_size=window_size, stride=stride, pred=pred)
    test_data_list = build_dataset_per_trial(test_dict, test_keys, input_keys, output='label_int', window_size=window_size, stride=stride, pred=pred)

    num_classes = len(np.unique(y_train))  # 7개일 경우
    print(num_classes)
    # y_train_oh = to_categorical(y_train, num_classes)
    # y_val_oh = to_categorical(y_val, num_classes)
    # y_test_oh = to_categorical(y_test, num_classes)
    print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)
    print("Train x shape", X_train.shape, "Train y shape", y_train.shape)

    return X_train, y_train, X_val, y_val, X_test, y_test, test_data_list, input_keys