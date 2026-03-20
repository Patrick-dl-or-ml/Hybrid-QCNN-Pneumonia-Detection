import os
import tensorflow as tf
import pennylane as qml

print("正在加载 Test 测试集数据...")
data_dir = './chest_xray'
test_dir = os.path.join(data_dir, 'test')
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

# 1. 加载测试集
test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='binary'
)
normalization_layer = tf.keras.layers.Rescaling(1./255)
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

# 2. 重新定义完全一样的量子层
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
qlayer = qml.qnn.KerasLayer(qnode, weight_shapes, output_dim=n_qubits)

# 3. 拼装混合模型
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(128, 128, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((4, 4)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((4, 4)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(2, activation='relu'),
    qlayer,
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# 4. 编译模型（最关键的是加入 AUC 指标）
model.compile(
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')] # 对标论文 AUROC
)

# 5. 加载刚才训练好的权重
weights_path = "./models/hybrid_qcnn_weights.weights.h5"
print(f"\n正在加载已训练的权重: {weights_path}")
model.load_weights(weights_path)

# 6.
print("\n开始在 624 张独立测试集上进行评估...")
loss, accuracy, auroc = model.evaluate(test_ds)

print("\n" + "="*40)
print("最终测试集成绩 (Test Set Results)")
print("="*40)
print(f"Accuracy (准确率): {accuracy:.4f}")
print(f"AUROC (曲线下面积): {auroc:.4f}")
print("="*40)