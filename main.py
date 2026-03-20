import os
import tensorflow as tf
import pennylane as qml
import matplotlib.pyplot as plt

# ==========================================
# 1. 路径设置
# ==========================================
data_dir = './chest_xray'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')

# 论文参数设置
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

# ==========================================
# 2. 数据预处理: 加载、缩放与归一化
# ==========================================
print("正在加载数据集...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary'
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary'
)

# 归一化处理: 将 0-255 像素值转为 0-1
normalization_layer = tf.keras.layers.Rescaling(1. / 255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# ==========================================
# 3. 定义量子层 (PennyLane) [cite: 144-154]
# ==========================================
n_qubits = 2
dev = qml.device("default.qubit", wires=n_qubits)


# 显式声明接口为 tf，确保梯度和张量兼容
@qml.qnode(dev, interface="tf")
def qnode(inputs, weights):
    # 修复点: 使用官方的 AngleEmbedding，自动处理 (32, 2) 的 Batch 维度
    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='X')

    # 3层变分层 (论文图 3 的结构)
    for i in range(3):
        qml.RX(weights[i, 0], wires=0)
        qml.RX(weights[i, 1], wires=1)
        qml.CNOT(wires=[0, 1])

    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]


weight_shapes = {"weights": (3, 2)}
qlayer = qml.qnn.KerasLayer(qnode, weight_shapes, output_dim=n_qubits)
# ==========================================
# 4. 构建混合模型
# ==========================================
# 直接使用 tf.keras 绕过 import bug
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(128, 128, 3)),

    # 卷积层 1
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((4, 4)),

    # 卷积层 2
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((4, 4)),

    tf.keras.layers.Flatten(),

    # 全连接层
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(2, activation='relu'),

    # 量子层
    qlayer,

    # 输出层
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ==========================================
# 5. 开始训练!
# ==========================================
print("\n开始训练混合量子神经网络...")

# 自动保存权重的回调函数
checkpoint_path = "./models/hybrid_qcnn_weights.weights.h5"
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    verbose=1
)

# 训练模型 (先跑20个 Epoch 看看收敛情况)
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=[cp_callback]
)

# ==========================================
# 6. 绘制训练结果曲线并保存
# ==========================================
print("\n正在生成训练曲线图...")

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(1, len(acc) + 1)

plt.figure(figsize=(12, 5))

# 画 Loss 曲线
plt.subplot(1, 2, 1)
plt.plot(epochs_range, loss, label='Training Loss', color='#1f77b4', linewidth=2)
plt.plot(epochs_range, val_loss, label='Validation Loss', color='#ff7f0e', linewidth=2)
plt.title('Training and Validation Loss (Hybrid QCNN)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# 画 Accuracy 曲线
plt.subplot(1, 2, 2)
plt.plot(epochs_range, acc, label='Training Accuracy', color='#1f77b4', linewidth=2)
plt.plot(epochs_range, val_acc, label='Validation Accuracy', color='#ff7f0e', linewidth=2)
plt.title('Training and Validation Accuracy (Hybrid QCNN)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('./models/training_history.png', dpi=300)
print("图像已保存至 ./models/training_history.png")
plt.show()