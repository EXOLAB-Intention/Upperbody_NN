from AdvancedFunction import *
from Model import *
from Training import *
import wandb
import itertools

# ITERATION_MODE = 0          # Use wandb
ITERATION_MODE = 1          # Use Local Python

if (ITERATION_MODE == 1):
    parameters = {"WindowSize":     [20, 30, 40, 50],
                  "Stride":         [1, 3, 5, 10],
                  "LearningRate":   [0.0001, 0.001, 0.005],
                  "Dropout":        [0.2, 0.3],
                  "LSTM_units":     [[64, 32], [32, 16], [64], [32], [16]],
                  "Dense_units":    [[32, 16], [32], [16]],
                  "BatchSize":      [64, 128]}

    # 1개 테스트용 코드
    # parameters = {"WindowSize":   [20],
    #             "Stride":         [5],
    #             "LearningRate":   [0.001],
    #             "Dropout":        [0.2],
    #             "LSTM_units":     [[16]],
    #             "Dense_units":    [[16]],
    #             "BatchSize":      [64]}
    
    # 파라미터 이름과 값 목록 분리
    param_keys = list(parameters.keys())
    param_values = list(parameters.values())

    # 모든 조합 생성
    all_combinations = list(itertools.product(*param_values))

    # 반복하면서 딕셔너리 형태로 변환
    param_list = []
    for combo in all_combinations:
        params = dict(zip(param_keys, combo))
        param_list.append(params) 


### Data Loading
if (ITERATION_MODE == 1):
        all_trials = DataLoader(folder_paths=['../DataFile/data_250612', '../DataFile/data_250616', '../DataFile/data_250618'])
        IMUCalibration(all_trials)


def main():
    if (ITERATION_MODE == 0):
        wandb.init(project="IterationProject")
        config = wandb.config
        
        # Data Loading
        # all_trials = DataLoader(folder_paths=['../DataFile/data_250612', '../DataFile/data_250616', '../DataFile/data_250618'])
        # IMUCalibration(all_trials)
        # x_train, y_train_ohe, x_val, y_val_ohe, x_test, y_test_ohe = DataPreprocessing(all_trials, config)

        # Data Information Check
        input_shape = x_train.shape[1:]      # (window_size, num_features)
        num_classes = y_train_ohe.shape[1]   # one-hot Label Dimension 
        print(f"Input Shape = {input_shape}")
        print(f"Number of Classes = {num_classes}")

        # Model
        model, history = TrainingModel(x_train, y_train_ohe, x_val, y_val_ohe, input_shape, num_classes, config)

        # Result


    elif (ITERATION_MODE == 1):
        train_acc_list = {}
        val_acc_list = {}
        for idx, param in enumerate(param_list):
            # Data Loading
            # all_trials = DataLoader(folder_paths=['../DataFile/data_250612', '../DataFile/data_250616', '../DataFile/data_250618'])
            # IMUCalibration(all_trials)
            x_train, y_train_ohe, x_val, y_val_ohe, x_test, y_test_ohe = DataPreprocessing2(all_trials, param)

            # Data Check
            # PlotIMUData(all_trials)
            # CheckTotalData(all_trials)

            # Data Information Checkz
            input_shape = x_train.shape[1:]      # (window_size, num_features)
            num_classes = y_train_ohe.shape[1]   # one-hot Label Dimension 

            # Model
            model, history = TrainingModel2(x_train, y_train_ohe, x_val, y_val_ohe, input_shape, num_classes, param)

            # Save Accuracy
            train_acc_list[
                f"window={param['WindowSize']}, stride={param['Stride']}, LR={param['LearningRate']}, Dropout={param['Dropout']}, LSTM_units={param['LSTM_units']}, Dense_units={param['Dense_units']}, BatchSize={param['BatchSize']}"] = max(history.history['accuracy'])

            val_acc_list[
                f"window={param['WindowSize']}, stride={param['Stride']}, LR={param['LearningRate']}, Dropout={param['Dropout']}, LSTM_units={param['LSTM_units']}, Dense_units={param['Dense_units']}, BatchSize={param['BatchSize']}"] = max(history.history['val_accuracy'])
            
            # Result
            PlotAccuracy(history, param, idx+1)

        ### Final HyperParameter Comparison
        PlotHyperparamComparison(train_acc_list, val_acc_list)



if __name__ == "__main__":
    main()