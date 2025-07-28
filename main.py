from data_loader import load_processed_data
from data_preprocessing import preprocess_data
from model import build_lstm_classifier_advanced
from train import train_model
import wandb


# 데이터 준비
all_trial_data_processed = load_processed_data(folder_path='./data_250602/')

# 데이터 전처리 및 데이터 분할(train,val,test)
X_train, y_train, X_val, y_val, X_test, y_test, test_data_list, input_keys= preprocess_data(all_trial_data_processed)

print("y_train shape:", y_train.shape)
print("y_train sample:", y_train[:5])

print("y_val shape:", y_val.shape)
print("y_val sample:", y_val[:5])

print("y_test shape:", y_test.shape)
print("y_test sample:", y_test[:5])

def main():
    # wandb sweep/agent가 실행할 때 config를 자동으로 넘겨줌
    # sweep/agent 실행 시 config를 직접 넘기지 않음

    wandb.init(project="intention_nn")
    config = wandb.config

    # 데이터 정보
    input_shape = X_train.shape[1:]     # (window_size, num_features)
    print(input_shape)
    num_classes = 7

    model = train_model(X_train, y_train, X_val, y_val, input_shape, num_classes, config)

if __name__ == "__main__":
    main()
