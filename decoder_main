# %%
from pynwb import NWBHDF5IO
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import random
import warnings

warnings.filterwarnings('ignore')

filepath = r"D:\Datasets\dandi datasets\000138\sub-Jenkins\sub-Jenkins_ses-large_desc-train_behavior+ecephys.nwb"
print("📂 加载数据并完美对齐时间轴...")

with NWBHDF5IO(filepath, 'r') as io:
    nwbfile = io.read()
    num_units = len(nwbfile.units)
    spike_data_dict = {i: np.asarray(nwbfile.units.get_unit_spike_times(i)) for i in range(num_units)}

    hand_vel_raw = np.asarray(nwbfile.processing['behavior'].data_interfaces['hand_vel'].data[:])
    hand_timestamps = np.asarray(nwbfile.processing['behavior'].data_interfaces['hand_vel'].timestamps[:])

    all_trial_starts = np.asarray(nwbfile.trials.start_time[:])
    all_trial_stops = np.asarray(nwbfile.trials.stop_time[:])
    num_trials = len(nwbfile.trials)

bin_size_ms = 50
dt = bin_size_ms / 1000.0

bin_edges = np.arange(hand_timestamps[0], hand_timestamps[-1], dt)
bin_centers = bin_edges[:-1] + dt / 2
num_bins = len(bin_centers)

print("🧠 构建神经特征 X...")
spike_counts_matrix = np.zeros((num_bins, num_units))
for unit_idx in range(num_units):
    spikes = spike_data_dict[unit_idx]
    counts, _ = np.histogram(spikes, bins=bin_edges)
    spike_counts_matrix[:, unit_idx] = counts

spikes_smoothed = gaussian_filter1d(spike_counts_matrix, sigma=2.0, axis=0)

print("🖐️ 构建速度标签 Y (使用插值绝对对齐)...")
vel_x = np.interp(bin_centers, hand_timestamps, hand_vel_raw[:, 0])
vel_y = np.interp(bin_centers, hand_timestamps, hand_vel_raw[:, 1])
vel_binned = np.column_stack([vel_x, vel_y])

# 新增：对速度积分得到真实位置（用于训练和评估）
print("📍 计算真实位置标签...")
pos_x = np.cumsum(vel_x * dt)
pos_y = np.cumsum(vel_y * dt)
pos_binned = np.column_stack([pos_x, pos_y])


print("️ 构建时延特征...")
n_lags = 4
X_features = np.zeros((num_bins, num_units * n_lags))
for i in range(n_lags, num_bins):
    X_features[i] = spikes_smoothed[i - n_lags:i].flatten()

print("🔀 【关键步骤】: 随机打乱 Trial 顺序以抵抗电极漂移...")
shuffled_indices = list(range(num_trials))
random.seed(42)
random.shuffle(shuffled_indices)

train_limit = int(0.8 * num_trials)
train_indices = set(shuffled_indices[:train_limit])
test_indices = set(shuffled_indices[train_limit:])

X_train, Y_train_vel, Y_train_pos = [], [], []
X_test, Y_test_vel, Y_test_pos = [], [], []

for trial_idx in range(num_trials):
    start_bin = int((all_trial_starts[trial_idx] - hand_timestamps[0]) / dt)
    end_bin = int((all_trial_stops[trial_idx] - hand_timestamps[0]) / dt)

    start_bin += 5
    end_bin -= 5

    if end_bin > start_bin + n_lags:
        X_trial = X_features[start_bin:end_bin]
        Y_trial_vel = vel_binned[start_bin:end_bin]
        Y_trial_pos = pos_binned[start_bin:end_bin].copy()  # 使用副本

        # ✅ 关键修复：对每个trial的位置进行归一化（相对于trial起点）
        Y_trial_pos[:, 0] -= Y_trial_pos[0, 0]
        Y_trial_pos[:, 1] -= Y_trial_pos[0, 1]

        speeds = np.linalg.norm(Y_trial_vel, axis=1)
        active_mask = speeds > 10.0

        if np.sum(active_mask) > 0:
            if trial_idx in train_indices:
                X_train.append(X_trial[active_mask])
                Y_train_vel.append(Y_trial_vel[active_mask])
                Y_train_pos.append(Y_trial_pos[active_mask])
            else:
                X_test.append(X_trial[active_mask])
                Y_test_vel.append(Y_trial_vel[active_mask])
                Y_test_pos.append(Y_trial_pos[active_mask])

X_train = np.vstack(X_train)
Y_train_vel = np.vstack(Y_train_vel)
Y_train_pos = np.vstack(Y_train_pos)
X_test = np.vstack(X_test)
Y_test_vel = np.vstack(Y_test_vel)
Y_test_pos = np.vstack(Y_test_pos)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("🤖 训练速度解码器...")
decoder_vel = Ridge(alpha=10.0)
decoder_vel.fit(X_train_scaled, Y_train_vel)

Y_train_vel_pred = decoder_vel.predict(X_train_scaled)
Y_test_vel_pred = decoder_vel.predict(X_test_scaled)

# 速度解码评估
train_r2_vel_x = r2_score(Y_train_vel[:, 0], Y_train_vel_pred[:, 0])
train_r2_vel_y = r2_score(Y_train_vel[:, 1], Y_train_vel_pred[:, 1])
test_r2_vel_x = r2_score(Y_test_vel[:, 0], Y_test_vel_pred[:, 0])
test_r2_vel_y = r2_score(Y_test_vel[:, 1], Y_test_vel_pred[:, 1])

print(f"\n{'=' * 60}")
print(f"🌟 速度解码报告")
print(f"{'=' * 60}")
print(f"【训练集】拟合能力:")
print(f"   - X 速度 R² = {train_r2_vel_x:.4f}")
print(f"   - Y 速度 R² = {train_r2_vel_y:.4f}")
print(f"\n【测试集】泛化能力:")
print(f"   - X 速度 R² = {test_r2_vel_x:.4f}")
print(f"   - Y 速度 R² = {test_r2_vel_y:.4f}")
print(f"{'=' * 60}")

# =========================================================================
# 🔄 方案3：卡尔曼滤波融合（速度 → 位置）
# =========================================================================

print("\n🔄 卡尔曼滤波融合：速度解码 → 位置估计...")
from pykalman import KalmanFilter

# ✅ 修复方法1：正确的位置积分
print("\n📊 方法1：直接积分（基线）")

# 按trial分别积分，避免跨trial累积
pos_integrated_x = np.zeros_like(Y_test_vel_pred[:, 0])
pos_integrated_y = np.zeros_like(Y_test_vel_pred[:, 1])

# 重建测试集的trial结构
test_trial_boundaries = []
current_idx = 0
for trial_idx in range(num_trials):
    if trial_idx in test_indices:
        start_bin = int((all_trial_starts[trial_idx] - hand_timestamps[0]) / dt) + 5
        end_bin = int((all_trial_stops[trial_idx] - hand_timestamps[0]) / dt) - 5

        speeds = np.linalg.norm(vel_binned[start_bin:end_bin], axis=1)
        active_mask = speeds > 10.0
        n_active = np.sum(active_mask)

        if n_active > 0:
            test_trial_boundaries.append((current_idx, current_idx + n_active))
            current_idx += n_active

# 对每个trial分别积分（从0开始）
for start_idx, end_idx in test_trial_boundaries:
    trial_vel_x = Y_test_vel_pred[start_idx:end_idx, 0]
    trial_vel_y = Y_test_vel_pred[start_idx:end_idx, 1]

    pos_integrated_x[start_idx:end_idx] = np.cumsum(trial_vel_x * dt)
    pos_integrated_y[start_idx:end_idx] = np.cumsum(trial_vel_y * dt)

int_r2_x = r2_score(Y_test_pos[:, 0], pos_integrated_x)
int_r2_y = r2_score(Y_test_pos[:, 1], pos_integrated_y)
int_rmse_x = np.sqrt(mean_squared_error(Y_test_pos[:, 0], pos_integrated_x))
int_rmse_y = np.sqrt(mean_squared_error(Y_test_pos[:, 1], pos_integrated_y))

print(f"直接积分法位置解码:")
print(f"   X: R² = {int_r2_x:.4f}, RMSE = {int_rmse_x:.2f} cm")
print(f"   Y: R² = {int_r2_y:.4f}, RMSE = {int_rmse_y:.2f} cm")

# ✅ 修复方法2：调整卡尔曼滤波器参数
print("\n📊 方法2：卡尔曼滤波平滑（优化参数）")

# 降低观测噪声，增加对预测速度的信任
kf_x = KalmanFilter(
    n_dim_state=2,
    n_dim_obs=1,
    initial_state_mean=[0, 0],
    initial_state_covariance=[[1, 0], [0, 1]],
    transition_matrices=[[1, dt], [0, 1]],
    observation_matrices=[[0, 1]],
    transition_covariance=[[0.01, 0], [0, 1]],  # 降低过程噪声
    observation_covariance=[[10]]  # 降低观测噪声
)

kf_y = KalmanFilter(
    n_dim_state=2,
    n_dim_obs=1,
    initial_state_mean=[0, 0],
    initial_state_covariance=[[1, 0], [0, 1]],
    transition_matrices=[[1, dt], [0, 1]],
    observation_matrices=[[0, 1]],
    transition_covariance=[[0.01, 0], [0, 1]],
    observation_covariance=[[10]]
)

# 按trial分别平滑，避免跨trial影响
pos_kf_x = np.zeros_like(Y_test_vel_pred[:, 0])
pos_kf_y = np.zeros_like(Y_test_vel_pred[:, 1])

for start_idx, end_idx in test_trial_boundaries:
    trial_vel_pred_x = Y_test_vel_pred[start_idx:end_idx, 0]
    trial_vel_pred_y = Y_test_vel_pred[start_idx:end_idx, 1]

    smoothed_x, _ = kf_x.smooth(trial_vel_pred_x)
    smoothed_y, _ = kf_y.smooth(trial_vel_pred_y)

    pos_kf_x[start_idx:end_idx] = smoothed_x[:, 0]
    pos_kf_y[start_idx:end_idx] = smoothed_y[:, 0]

kf_r2_x = r2_score(Y_test_pos[:, 0], pos_kf_x)
kf_r2_y = r2_score(Y_test_pos[:, 1], pos_kf_y)
kf_rmse_x = np.sqrt(mean_squared_error(Y_test_pos[:, 0], pos_kf_x))
kf_rmse_y = np.sqrt(mean_squared_error(Y_test_pos[:, 1], pos_kf_y))

print(f"卡尔曼滤波位置解码:")
print(f"   X: R² = {kf_r2_x:.4f}, RMSE = {kf_rmse_x:.2f} cm")
print(f"   Y: R² = {kf_r2_y:.4f}, RMSE = {kf_rmse_y:.2f} cm")

# ✅ 修复方法3：EM算法（按trial训练）
print("\n📊 方法3：EM算法自动调参卡尔曼滤波")

# 使用训练集学习参数（也按trial处理）
train_trial_boundaries = []
current_idx = 0
for trial_idx in range(num_trials):
    if trial_idx in train_indices:
        start_bin = int((all_trial_starts[trial_idx] - hand_timestamps[0]) / dt) + 5
        end_bin = int((all_trial_stops[trial_idx] - hand_timestamps[0]) / dt) - 5

        speeds = np.linalg.norm(vel_binned[start_bin:end_bin], axis=1)
        active_mask = speeds > 10.0
        n_active = np.sum(active_mask)

        if n_active > 0:
            train_trial_boundaries.append((current_idx, current_idx + n_active))
            current_idx += n_active

# 在训练集上使用EM（使用第一个trial的数据）
if len(train_trial_boundaries) > 0:
    first_train_start, first_train_end = train_trial_boundaries[0]
    kf_x_em = KalmanFilter(
        n_dim_state=2,
        n_dim_obs=1,
        initial_state_mean=[0, 0],
        transition_matrices=[[1, dt], [0, 1]],
        observation_matrices=[[0, 1]]
    )

    kf_y_em = KalmanFilter(
        n_dim_state=2,
        n_dim_obs=1,
        initial_state_mean=[0, 0],
        transition_matrices=[[1, dt], [0, 1]],
        observation_matrices=[[0, 1]]
    )

    # EM学习
    kf_x_em = kf_x_em.em(Y_train_vel_pred[first_train_start:first_train_end, 0].reshape(-1, 1), n_iter=50)
    kf_y_em = kf_y_em.em(Y_train_vel_pred[first_train_start:first_train_end, 1].reshape(-1, 1), n_iter=50)

    # 在测试集上应用
    pos_em_x = np.zeros_like(Y_test_vel_pred[:, 0])
    pos_em_y = np.zeros_like(Y_test_vel_pred[:, 1])

    for start_idx, end_idx in test_trial_boundaries:
        smoothed_em_x, _ = kf_x_em.smooth(Y_test_vel_pred[start_idx:end_idx, 0].reshape(-1, 1))
        smoothed_em_y, _ = kf_y_em.smooth(Y_test_vel_pred[start_idx:end_idx, 1].reshape(-1, 1))

        pos_em_x[start_idx:end_idx] = smoothed_em_x[:, 0]
        pos_em_y[start_idx:end_idx] = smoothed_em_y[:, 0]

    em_r2_x = r2_score(Y_test_pos[:, 0], pos_em_x)
    em_r2_y = r2_score(Y_test_pos[:, 1], pos_em_y)
    em_rmse_x = np.sqrt(mean_squared_error(Y_test_pos[:, 0], pos_em_x))
    em_rmse_y = np.sqrt(mean_squared_error(Y_test_pos[:, 1], pos_em_y))
else:
    em_r2_x = em_r2_y = -999
    em_rmse_x = em_rmse_y = 999

print(f"EM自动调参卡尔曼滤波:")
print(f"   X: R² = {em_r2_x:.4f}, RMSE = {em_rmse_x:.2f} cm")
print(f"   Y: R² = {em_r2_y:.4f}, RMSE = {em_rmse_y:.2f} cm")

# =========================================================================
# 📊 综合性能对比报告
# =========================================================================

print(f"\n{'=' * 70}")
print(f" 位置解码综合性能对比")
print(f"{'=' * 70}")
print(f"方法                  | X-R²    | Y-R²    | X-RMSE  | Y-RMSE")
print(f"{'-' * 70}")
print(f"直接积分法           | {int_r2_x:.4f}  | {int_r2_y:.4f}  | {int_rmse_x:6.2f} | {int_rmse_y:6.2f}")
print(f"卡尔曼滤波(手动调参) | {kf_r2_x:.4f}  | {kf_r2_y:.4f}  | {kf_rmse_x:6.2f} | {kf_rmse_y:6.2f}")
print(f"卡尔曼滤波(EM自动)   | {em_r2_x:.4f}  | {em_r2_y:.4f}  | {em_rmse_x:6.2f} | {em_rmse_y:6.2f}")
print(f"{'=' * 70}")

# 选择最佳方法
best_method_x = max([("积分", int_r2_x), ("KF手动", kf_r2_x), ("KF-EM", em_r2_x)], key=lambda x: x[1])
best_method_y = max([("积分", int_r2_y), ("KF手动", kf_r2_y), ("KF-EM", em_r2_y)], key=lambda x: x[1])

print(f"\n⭐ 最佳方法:")
print(f"   X方向: {best_method_x[0]} (R² = {best_method_x[1]:.4f})")
print(f"   Y方向: {best_method_y[0]} (R² = {best_method_y[1]:.4f})")
print(f"{'=' * 70}")

# =========================================================================
# 📊 可视化：多维度对比
# =========================================================================

print("\n 生成可视化结果...")

fig = plt.figure(figsize=(20, 12))

# 图1-2：速度解码（保留原图）
ax1 = fig.add_subplot(3, 3, 1)
plot_length = min(200, len(Y_test_vel))
time_axis = np.arange(plot_length) * bin_size_ms
ax1.plot(time_axis, Y_test_vel[:plot_length, 0], 'k-', linewidth=2, label='True', alpha=0.8)
ax1.plot(time_axis, Y_test_vel_pred[:plot_length, 0], 'r--', linewidth=1.5, label='Predicted')
ax1.set_title(f'X Velocity (R² = {test_r2_vel_x:.2f})', fontsize=12, fontweight='bold')
ax1.set_xlabel('Time (ms)')
ax1.set_ylabel('Velocity X (cm/s)')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(3, 3, 2)
ax2.plot(time_axis, Y_test_vel[:plot_length, 1], 'k-', linewidth=2, label='True', alpha=0.8)
ax2.plot(time_axis, Y_test_vel_pred[:plot_length, 1], 'b--', linewidth=1.5, label='Predicted')
ax2.set_title(f'Y Velocity (R² = {test_r2_vel_y:.2f})', fontsize=12, fontweight='bold')
ax2.set_xlabel('Time (ms)')
ax2.set_ylabel('Velocity Y (cm/s)')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# 图3-5：X方向位置对比（三种方法）
ax3 = fig.add_subplot(3, 3, 3)
ax3.plot(time_axis, Y_test_pos[:plot_length, 0], 'k-', linewidth=2, label='True', alpha=0.8)
ax3.plot(time_axis, pos_integrated_x[:plot_length], 'g--', linewidth=1.2, label='Integration')
ax3.plot(time_axis, pos_kf_x[:plot_length], 'r--', linewidth=1.2, label='KF (Manual)')
ax3.plot(time_axis, pos_em_x[:plot_length], 'm--', linewidth=1.2, label='KF (EM)')
ax3.set_title(f'X Position Comparison', fontsize=12, fontweight='bold')
ax3.set_xlabel('Time (ms)')
ax3.set_ylabel('Position X (cm)')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# 图4-5：Y方向位置对比
ax4 = fig.add_subplot(3, 3, 4)
ax4.plot(time_axis, Y_test_pos[:plot_length, 1], 'k-', linewidth=2, label='True', alpha=0.8)
ax4.plot(time_axis, pos_integrated_y[:plot_length], 'g--', linewidth=1.2, label='Integration')
ax4.plot(time_axis, pos_kf_y[:plot_length], 'r--', linewidth=1.2, label='KF (Manual)')
ax4.plot(time_axis, pos_em_y[:plot_length], 'm--', linewidth=1.2, label='KF (EM)')
ax4.set_title(f'Y Position Comparison', fontsize=12, fontweight='bold')
ax4.set_xlabel('Time (ms)')
ax4.set_ylabel('Position Y (cm)')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

# 图6：2D轨迹对比（积分法）
ax5 = fig.add_subplot(3, 3, 5)
trajectory_len = min(500, len(Y_test_pos))
ax5.plot(Y_test_pos[:trajectory_len, 0], Y_test_pos[:trajectory_len, 1],
         'k-', linewidth=2, label='True', alpha=0.8)
ax5.plot(pos_integrated_x[:trajectory_len], pos_integrated_y[:trajectory_len],
         'g--', linewidth=1.5, label='Integration')
ax5.plot(Y_test_pos[0, 0], Y_test_pos[0, 1], 'go', markersize=10, label='Start')
ax5.plot(Y_test_pos[trajectory_len-1, 0], Y_test_pos[trajectory_len-1, 1], 'gs',
         markersize=10, label='End (True)')
ax5.set_title(f'2D Trajectory - Integration (R²={int_r2_x:.2f}/{int_r2_y:.2f})',
              fontsize=11, fontweight='bold')
ax5.set_xlabel('Position X (cm)')
ax5.set_ylabel('Position Y (cm)')
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)
ax5.set_aspect('equal')

# 图7：2D轨迹对比（卡尔曼手动）
ax6 = fig.add_subplot(3, 3, 6)
ax6.plot(Y_test_pos[:trajectory_len, 0], Y_test_pos[:trajectory_len, 1],
         'k-', linewidth=2, label='True', alpha=0.8)
ax6.plot(pos_kf_x[:trajectory_len], pos_kf_y[:trajectory_len],
         'r--', linewidth=1.5, label='KF (Manual)')
ax6.plot(Y_test_pos[0, 0], Y_test_pos[0, 1], 'go', markersize=10, label='Start')
ax6.set_title(f'2D Trajectory - KF Manual (R²={kf_r2_x:.2f}/{kf_r2_y:.2f})',
              fontsize=11, fontweight='bold')
ax6.set_xlabel('Position X (cm)')
ax6.set_ylabel('Position Y (cm)')
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.3)
ax6.set_aspect('equal')

# 图8：2D轨迹对比（EM自动）
ax7 = fig.add_subplot(3, 3, 7)
ax7.plot(Y_test_pos[:trajectory_len, 0], Y_test_pos[:trajectory_len, 1],
         'k-', linewidth=2, label='True', alpha=0.8)
ax7.plot(pos_em_x[:trajectory_len], pos_em_y[:trajectory_len],
         'm--', linewidth=1.5, label='KF (EM)')
ax7.plot(Y_test_pos[0, 0], Y_test_pos[0, 1], 'go', markersize=10, label='Start')
ax7.set_title(f'2D Trajectory - KF EM (R²={em_r2_x:.2f}/{em_r2_y:.2f})',
              fontsize=11, fontweight='bold')
ax7.set_xlabel('Position X (cm)')
ax7.set_ylabel('Position Y (cm)')
ax7.legend(fontsize=8)
ax7.grid(True, alpha=0.3)
ax7.set_aspect('equal')

# 图9：误差分析
ax8 = fig.add_subplot(3, 3, 8)
error_integration = np.linalg.norm(
    np.column_stack([Y_test_pos[:plot_length, 0] - pos_integrated_x[:plot_length],
                     Y_test_pos[:plot_length, 1] - pos_integrated_y[:plot_length]]),
    axis=1)
error_kf = np.linalg.norm(
    np.column_stack([Y_test_pos[:plot_length, 0] - pos_kf_x[:plot_length],
                     Y_test_pos[:plot_length, 1] - pos_kf_y[:plot_length]]),
    axis=1)
error_em = np.linalg.norm(
    np.column_stack([Y_test_pos[:plot_length, 0] - pos_em_x[:plot_length],
                     Y_test_pos[:plot_length, 1] - pos_em_y[:plot_length]]),
    axis=1)

ax8.plot(time_axis, error_integration, 'g-', linewidth=1.5, label='Integration Error')
ax8.plot(time_axis, error_kf, 'r-', linewidth=1.5, label='KF (Manual) Error')
ax8.plot(time_axis, error_em, 'm-', linewidth=1.5, label='KF (EM) Error')
ax8.set_title(f'Position Error Over Time', fontsize=12, fontweight='bold')
ax8.set_xlabel('Time (ms)')
ax8.set_ylabel('Euclidean Error (cm)')
ax8.legend(fontsize=9)
ax8.grid(True, alpha=0.3)

# 图10：综合性能柱状图
ax9 = fig.add_subplot(3, 3, 9)
methods = ['Integration', 'KF\n(Manual)', 'KF\n(EM)']
r2_scores_x = [int_r2_x, kf_r2_x, em_r2_x]
r2_scores_y = [int_r2_y, kf_r2_y, em_r2_y]

x_pos = np.arange(len(methods))
width = 0.35

ax9.bar(x_pos - width/2, r2_scores_x, width, label='X R²', color='steelblue', alpha=0.8)
ax9.bar(x_pos + width/2, r2_scores_y, width, label='Y R²', color='coral', alpha=0.8)
ax9.set_xticks(x_pos)
ax9.set_xticklabels(methods, fontsize=9)
ax9.set_ylabel('R² Score', fontsize=11)
ax9.set_title('Position Decoding Performance', fontsize=12, fontweight='bold')
ax9.legend(fontsize=9)
ax9.grid(True, alpha=0.3, axis='y')
ax9.set_ylim(0, 1.0)

# 添加数值标签
for i, v in enumerate(r2_scores_x):
    ax9.text(i - width/2, v + 0.02, f'{v:.2f}', ha='center', fontsize=8, fontweight='bold')
for i, v in enumerate(r2_scores_y):
    ax9.text(i + width/2, v + 0.02, f'{v:.2f}', ha='center', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig('kalman_position_decoding_complete.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 70)
print("✅ 位置解码完成！")
print("=" * 70)
print("📊 生成的图像:")
print("   - bci_ultimate_victory.png (速度解码)")
print("   - kalman_position_decoding_complete.png (位置解码完整对比)")
print("\n 关键发现:")
print(f"   1. 速度解码 R²: X={test_r2_vel_x:.3f}, Y={test_r2_vel_y:.3f}")
print(f"   2. 最佳位置解码 R²: X={best_method_x[1]:.3f}, Y={best_method_y[1]:.3f}")
print(f"   3. 卡尔曼滤波相比积分法提升: X={kf_r2_x-int_r2_x:+.3f}, Y={kf_r2_y-int_r2_y:+.3f}")
print("\n 建议:")
if kf_r2_x > int_r2_x or kf_r2_y > int_r2_y:
    print("   - 卡尔曼滤波有效减少了位置漂移！")
    print("   - 可用于实时BCI控制系统")
else:
    print("   - 尝试调整卡尔曼滤波参数 (transition_covariance, observation_covariance)")
    print("   - 考虑使用在线版本 (filter 而非 smooth) 用于实时应用")
print("=" * 70)
