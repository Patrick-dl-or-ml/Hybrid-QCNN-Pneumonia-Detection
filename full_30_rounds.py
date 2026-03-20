import os
import pennylane as qml
import logging

# 这一行屏蔽 TensorFlow 本身的系统日志（0为全部，1为不显示INFO，2为不显示INFO和WARNING）
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 这一行屏蔽 Python 层面的 TensorFlow 警告
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)

# 实验配置
NUM_ROUNDS = 30
EPOCHS = 20
BATCH_SIZE = 32
IMG_SIZE = (128, 128)
CSV_FILENAME = "experiment_30_rounds_results.csv"

print("📂 正在加载数据集 (Train, Val, Test)...")
data_dir = './chest_xray'


def get_dataset(subset):
    ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, subset),
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='binary'
    )
    normalization_layer = tf.keras.layers.Rescaling(1. / 255)
    return ds.map(lambda x, y: (normalization_layer(x), y))


train_ds = get_dataset('train')
val_ds = get_dataset('val')
test_ds = get_dataset('test')

# 量子层定义
n_qubits = 2
dev = qml.device("default.qubit", wires=n_qubits)


@qml.qnode(dev, interface="tf")
def qnode(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='X')
    for i in range(3):
        qml.RX(weights[i, 0], wires=0)
        qml.RX(weights[i, 1], wires=1)
        qml.CNOT(wires=[0, 1])
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]


weight_shapes = {"weights": (3, 2)}


# 工厂函数：每次调用都生成一个全新、没训练过的模型
def build_base_layers():
    return [
        tf.keras.layers.Input(shape=(128, 128, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((4, 4)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((4, 4)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(2, activation='relu')
    ]


def create_classical_model():
    layers = build_base_layers()
    layers.append(tf.keras.layers.Dense(1, activation='sigmoid'))
    return tf.keras.models.Sequential(layers)


def create_hybrid_model():
    layers = build_base_layers()
    layers.append(qml.qnn.KerasLayer(qnode, weight_shapes, output_dim=n_qubits))
    layers.append(tf.keras.layers.Dense(1, activation='sigmoid'))
    return tf.keras.models.Sequential(layers)


# 初始化表格表头
if not os.path.exists(CSV_FILENAME):
    with open(CSV_FILENAME, "w") as f:
        f.write("Round,Model_Type,Train_Loss,Val_Loss,Test_Accuracy,Test_AUROC\n")

print(f"\n🚀 准备开始 30 轮对比马拉松！")

for round_num in range(1, NUM_ROUNDS + 1):
    print(f"\n" + "=" * 50)
    print(f"🌟 第 {round_num}/{NUM_ROUNDS} 轮实验正式开始")
    print("=" * 50)

    models_to_test = {
        "Classical_CNN": create_classical_model(),
        "Hybrid_QCNN": create_hybrid_model()
    }

    for model_name, model in models_to_test.items():
        print(f"\n---> [{model_name}] 开始训练...")
        model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )

        # 训练 20 个 Epochs
        history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)

        # 直接在独立测试集上出成绩
        print(f"\n🔍 [{model_name}] 正在独立测试集上评估...")
        test_loss, test_acc, test_auc = model.evaluate(test_ds, verbose=0)
        print(f"✅ {model_name} 本轮得分 -> Test Acc: {test_acc:.4f}, AUROC: {test_auc:.4f}")

        # 实时存入 Excel (CSV)
        with open(CSV_FILENAME, "a") as f:
            f.write(
                f"{round_num},{model_name},{history.history['loss'][-1]:.4f},{history.history['val_loss'][-1]:.4f},{test_acc:.4f},{test_auc:.4f}\n")

print(f"\n🎉 恭喜！30轮实验跑完了，可以拿 {CSV_FILENAME} 里的数据去算 p-value 写论文了！")