from iterFunction import *
from Model import *
import itertools
import os


parameters = {"WindowSize":       [10, 30, 50, 80],
                "Stride":         [1, 5],
                "LearningRate":   [0.001],
                "Dropout":        [0.2],
                "LSTM_units":     [[16], [32], [64], [32, 16], [64, 32], [64, 32, 16]],
                "Dense_units":    [[16], [32], [64]],
                "Epoch":          [10, 15, 20],  
                "BatchSize":      [16, 32, 64, 128]}

# 파라미터 이름과 값 목록 분리
param_keys = list(parameters.keys())
param_values = list(parameters.values())

# 모든 조합 생성
all_combinations = list(itertools.product(*param_values))

# 반복하면서 딕셔너리 형태로 변환 (filtering: Dense가 LSTM 마지막 layer보다 작게)
param_list = []
for combo in all_combinations:
    params = dict(zip(param_keys, combo))
    if params["Dense_units"][0] <= params["LSTM_units"][-1]:
        param_list.append(params)
print(f"총 조합 수: {len(param_list)}")

# TXT 파일로 파라미터 저장
os.makedirs("IterationProject/Result", exist_ok=True)
with open("IterationProject/Result/params_list.txt", "w", encoding="utf-8") as f:
    for idx, params in enumerate(param_list, start=1):
        f.write(f"#{idx} - {params}\n")

### Data Loading
all_DS = DataLoader(folder_paths=['DataFile/250813'])
IMUCalibration(all_DS)


def main():
    train_acc_list = {}
    val_acc_list = {}
    for idx, param in enumerate(param_list):
        # Data Loading
        x_train, y_train_ohe, x_val, y_val_ohe, x_test, y_test_ohe = DataPreprocessing(all_DS, param)


        # Data Check
        # PlotIMUData(all_DS)
        # CheckTotalData(all_DS)


        # Data Information Check
        input_shape = x_train.shape[1:]      # (window_size, num_features)
        num_classes = y_train_ohe.shape[1]   # one-hot Label Dimension 


        # Model
        print(
            f"\n[Index]: {idx}\n"
            f"Parameters:\n"
            f"  WindowSize: {param['WindowSize']}\n"
            f"  Stride: {param['Stride']}\n"
            f"  LSTM Layer: {param['LSTM_units']}\n"
            f"  Dense Layer: {param['Dense_units']}\n"
            f"  Epoch: {param['Epoch']}\n"
            f"  BatchSize: {param['BatchSize']}\n"
        )
        model, history = TrainingModel(x_train, y_train_ohe, x_val, y_val_ohe, input_shape, num_classes, param)


        # Save Accuracy
        train_acc_list[
            f"{idx}"] = max(history.history['accuracy'])

        val_acc_list[
            f"{idx}"] = max(history.history['val_accuracy'])
        

        # Result
        PlotAccuracy(history, param, idx+1 , max(history.history['accuracy']), max(history.history['val_accuracy']))

    ### Final HyperParameter Comparison
    PlotHyperparamComparison(train_acc_list, val_acc_list, chunk_size=30)



if __name__ == "__main__":
    main()