① 项目标题与简介
BCI-Motor-Decoding: Neural Kinematics Reconstruction
基于岭回归（Ridge Regression）与卡尔曼滤波（Kalman Filter）的运动皮层神经解码实现。
本项目利用非人灵长类动物（Monkey Jenkins）的运动皮层（M1）神经元发放数据，通过多步流水线实现手部运动速度与位置的高精度重建。
② 核心功能 (Key Features)
神经特征工程：支持神经信号的 Gaussian 平滑、时间仓化（Binning）以及时延特征（Lagged features）构建。
速度解码器：基于岭回归（Ridge Regression）的线性解码模型。
位置优化流水线：对比了三种从速度推算位置的方法：
直接积分法（Direct Integration）
手动调参卡尔曼滤波（Manual KF）
EM 算法自动调参卡尔曼滤波（Expectation-Maximization KF）
可视化评估：自动生成 2D 轨迹对比图、R² 分数矩阵及欧几里得误差分析图。
③ 数据集说明 (Dataset)
来源：DANDI Archive (ID: 000138 - Monkey Jenkins).
格式：NWB (Neurodata Without Borders).
内容：包含 M1 区多通道单位放电（Spike Times）和同步的手部运动速度（Hand Velocity）。
④ 算法流程 (Pipeline)
Preprocessing: 对神经放电进行 50ms 窗口采样及高斯平滑。
Feature Augmentation: 引入 4 个历史时延步长（共 200ms 历史信息）。
Velocity Decoding: 使用 Ridge 回归训练 𝑋和𝑌方向的速度解码器。
State-Space Filtering: 将解码速度输入卡尔曼滤波器，修正位置估计，抑制累积误差。
⑤ 运行结果展示 (Results)
直接在这里贴出你生成的那张全维度对比图：
![alt text](results/kalman_position_decoding_complete.png)
关键指标摘要：
速度解码 (Test Set): X-R² ≈ 0.80, Y-R² ≈ 0.70
最佳位置解码: X-R² 达到 0.92 (通过分段积分与 KF-EM 优化)
⑥ 环境配置 (Installation)
code
Bash
git clone https://github.com/your-username/BCI-Motor-Decoding.git
cd BCI-Motor-Decoding
pip install -r requirements.txt
依赖项包括：pynwb, pykalman, scikit-learn, scipy, matplotlib。
