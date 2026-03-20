import tensorflow as tf
import pennylane as qml
import numpy as np

print("=============================")
print("🚀 环境依赖测试开始...")
print("=============================")

# 1. 测试包是否都能成功导入并打印版本
print(f"✅ TensorFlow 版本: {tf.__version__}")
print(f"✅ PennyLane 版本: {qml.__version__}")
print(f"✅ NumPy 版本: {np.__version__}")

print("\n-----------------------------")
print("🧪 测试 PennyLane 量子线路...")
print("-----------------------------")

try:
    # 2. 创建一个简单的单比特量子设备
    dev = qml.device('default.qubit', wires=1)


    # 定义一个极其基础的量子节点
    @qml.qnode(dev)
    def test_circuit(x):
        qml.RX(x, wires=0)  # 绕 X 轴旋转
        return qml.expval(qml.PauliZ(0))  # 测量 Z 方向期望值


    # 3. 输入一个角度运行线路
    angle = np.pi / 4
    result = test_circuit(angle)

    print(f"✅ 量子线路运行成功！当输入角度为 π/4 时，Z 测量期望值为: {result:.4f}")
    print("\n🎉 太棒了，你的 qnn 环境一切正常，随时可以开搞混合量子 CNN 项目！")

except Exception as e:
    print(f"\n❌ 量子线路测试失败，捕获到以下错误:")
    print(e)
    print("\n💡 没关系，可能是包的版本不匹配或者缺了啥，把上面的报错信息发给我，我们一起来调。")