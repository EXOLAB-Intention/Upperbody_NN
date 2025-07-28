import wandb
from wandb.integration.keras import WandbCallback
# from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint # 다른 버젼인데, wandb 0.16~ 0.17 에서만 됨
from tensorflow.keras.callbacks import EarlyStopping


def train_model(X_train, y_train, X_val, y_val, input_shape, num_classes, config):    
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

    # 모델 생성 (예시)
    from model import build_lstm_classifier_advanced
    model = build_lstm_classifier_advanced(
        input_shape=input_shape,
        num_classes=num_classes,
        dropout_rate=config.dropout_rate,
        lstm_units=config.lstm_units,
        dense_units=[32],
        learning_rate=config.learning_rate,
        use_batchnorm=True
    )

    model.summary() # 모델 구조 출력


    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
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

    return model