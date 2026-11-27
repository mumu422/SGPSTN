"""SGPSTN 训练与评估入口。

Semantic Guided Probabilistic Spatiotemporal Network (SGPSTN)

数据说明：
1. 交通流数据取自公开数据集 PEMS04 / PEMS08 / METR-LA，需自行下载（见 data/README.md）。
2. 天气、节假日、事故等外部语义特征按统计规律仿真生成，不是实测观测；
   固定随机种子即可复现，字段定义见 README 的“外部语义特征”一节。
3. 本仓库只做训练与评估，不含任何绘图代码；--out 目录只写模型权重和训练日志。
"""
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from tensorflow.keras.layers import Layer, Dense, Input, LSTM, Bidirectional, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
import tensorflow_probability as tfp
tfd = tfp.distributions

sensor_ids = []  # 全局变量存储传感器ID


# ==================== 1. 专业数据加载（添加语义数据） ====================
def load_pems_data(flow_path, dist_path, seq_length=12, test_size=0.2, num_sensors=200, seed=42):
    """加载并预处理PEMS数据（支持npz和csv格式），添加仿真语义特征"""
    print(f"正在加载流量数据: {flow_path}")
    try:
        if flow_path.endswith('.npz'):
            # 读取npz文件
            npz = np.load(flow_path)
            flow_values = npz['data']
            if flow_values.ndim == 3:
                num_sensors = min(num_sensors, flow_values.shape[1])
                flow_values = flow_values[:, :num_sensors, 0]
            else:
                num_sensors = min(num_sensors, flow_values.shape[1])
                flow_values = flow_values[:, :num_sensors]
            sensor_ids = [str(i) for i in range(num_sensors)]
        else:
            # 读取csv文件
            flow_data = pd.read_csv(flow_path, sep=',', header=0)
            sensor_ids = [str(id) for id in flow_data.columns.tolist()]
            num_sensors = min(num_sensors, len(sensor_ids))
            sensor_ids = sensor_ids[:num_sensors]
            flow_values = flow_data[sensor_ids].values.astype(float)
    except Exception as e:
        print(f"加载流量数据错误: {e}")
        raise

    # 加载距离数据构建邻接矩阵
    print(f"\n正在加载距离数据: {dist_path}")
    try:
        # 尝试读取距离数据
        dist_data = pd.read_csv(dist_path, sep=',', header=0)
        print(f"距离数据前5行:\n{dist_data.head()}")

        # 确保距离数据有正确的列名
        if 'from' not in dist_data.columns or 'to' not in dist_data.columns or 'cost' not in dist_data.columns:
            print("距离数据列名不包含'from','to'和'distance'，尝试自动分配列名...")
            if len(dist_data.columns) >= 3:
                dist_data.columns = ['from', 'to', 'cost'] + dist_data.columns[3:].tolist()
                print(f"新列名: {dist_data.columns.tolist()}")
            else:
                raise ValueError("距离数据需要至少包含'from','to'和'cost'三列")
    except Exception as e:
        print(f"加载距离数据错误: {e}")
        raise

    # 创建邻接矩阵
    adj_matrix = np.zeros((num_sensors, num_sensors))
    sensor_to_idx = {sensor: idx for idx, sensor in enumerate(sensor_ids)}

    # 打印调试信息
    print(f"\n构建邻接矩阵...")
    print(f"传感器ID到索引映射示例:")
    for i, sensor_id in enumerate(sensor_ids[:5]):
        print(f"  {sensor_id} -> {i}")

    # 构建邻接矩阵
    valid_edges = 0
    missing_sensors = set()

    for _, row in dist_data.iterrows():
        try:
            # 确保ID转换为字符串并去除可能的浮点表示(.0)
            from_id = str(int(float(row['from']))) if '.' in str(row['from']) else str(row['from'])
            to_id = str(int(float(row['to']))) if '.' in str(row['to']) else str(row['to'])

            if from_id in sensor_to_idx and to_id in sensor_to_idx:
                i = sensor_to_idx[from_id]
                j = sensor_to_idx[to_id]
                distance = float(row['cost'])
                adj_matrix[i, j] = 1 / (distance + 1e-5)
                valid_edges += 1
            else:
                if from_id not in sensor_to_idx:
                    missing_sensors.add(from_id)
                if to_id not in sensor_to_idx:
                    missing_sensors.add(to_id)
        except Exception as e:
            print(f"处理距离数据行时出错: {e}")
            continue

    # 如果有效边为0，使用备用方案（基于地理位置的邻接矩阵）
    if valid_edges == 0:
        print("\n警告: 没有找到有效的距离边，使用基于地理位置的备用邻接矩阵")
        print(f"缺失的传感器: {list(missing_sensors)[:10]}{'...' if len(missing_sensors) > 10 else ''}")

        # 生成随机地理位置（实际应用中应使用真实坐标）
        np.random.seed(42)
        locations = np.random.rand(num_sensors, 2)

        # 计算欧氏距离
        for i in range(num_sensors):
            for j in range(num_sensors):
                if i != j:
                    dist = np.linalg.norm(locations[i] - locations[j])
                    adj_matrix[i, j] = 1 / (dist + 1e-5)

        valid_edges = num_sensors * (num_sensors - 1)

    # 增强距离数据处理
    if valid_edges < num_sensors * 0.1:  # 如果边太少
        print("\n警告: 有效距离边不足，添加k近邻连接")
        from sklearn.neighbors import NearestNeighbors
        knn = NearestNeighbors(n_neighbors=5)
        knn.fit(flow_values.T)  # 使用流量模式相似性
        distances, indices = knn.kneighbors()

        for i in range(num_sensors):
            for j, dist in zip(indices[i], distances[i]):
                if i != j and adj_matrix[i, j] == 0:
                    adj_matrix[i, j] = 1 / (dist + 1e-5)
                    valid_edges += 1

    # 对称化邻接矩阵
    adj_matrix = 0.5 * (adj_matrix + adj_matrix.T)
    max_val = np.max(adj_matrix)
    if max_val > 0:
        adj_matrix = adj_matrix / max_val
    else:
        print("警告: 邻接矩阵全零，使用单位矩阵")
        adj_matrix = np.eye(num_sensors)

    # 构建时间序列样本
    X, y = [], []
    for i in range(len(flow_values) - seq_length):
        X.append(flow_values[i:i + seq_length])
        y.append(flow_values[i + seq_length])
    X, y = np.array(X), np.array(y)

    # 数据归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    orig_shape = X.shape
    X = scaler.fit_transform(X.reshape(-1, orig_shape[-1])).reshape(orig_shape)
    y = scaler.transform(y)

    # 生成时间戳（假设数据是5分钟间隔）
    start_time = pd.Timestamp('2022-01-01 00:00:00')
    timestamps = [start_time + pd.Timedelta(minutes=5 * i) for i in range(len(flow_values))]

    # =========== 生成完整的语义和事件数据（仿真，非实测） ===========
    def generate_semantic_data_with_events(timestamps, seed_local=seed):
        """
        按统计规律仿真生成 8 维外部语义特征（与论文 3.1 一致）：
        [天气强度, 节假日标记, 事故标记, 时刻编码, 季节编码, 暴雨标志, 事故标志, 节假日标志]
        注意：均为合成变量，不改变真实流量，也不代表实测气象或事故记录。
        """
        rng = np.random.default_rng(seed_local)
        num_samples = len(timestamps)

        # 基本语义特征
        weather = rng.uniform(0, 1, (num_samples, 1))
        holiday = rng.integers(0, 2, (num_samples, 1))
        accident = rng.integers(0, 2, (num_samples, 1))
        time_of_day = np.sin(
            2 * np.pi * np.array([ts.hour + ts.minute / 60 for ts in timestamps]).reshape(-1, 1) / 24) * 0.5 + 0.5
        season = np.sin(2 * np.pi * np.array([ts.dayofyear for ts in timestamps]).reshape(-1, 1) / 365) * 0.5 + 0.5

        # 事件标记
        rainstorm = np.zeros((num_samples, 1))
        accident_event = np.zeros((num_samples, 1))
        holiday_event = np.zeros((num_samples, 1))

        # 暴雨：夏季 0.3，其余 0.05
        for i, ts in enumerate(timestamps):
            if ts.month in [6, 7, 8]:  # 夏季
                rain_prob = 0.3
            else:
                rain_prob = 0.05

            rainstorm[i] = 1 if rng.random() < rain_prob else 0

        # 事故：早晚高峰 0.1，平峰 0.02
        for i, ts in enumerate(timestamps):
            if 7 <= ts.hour <= 9 or 17 <= ts.hour <= 19:  # 早晚高峰
                accident_prob = 0.1
            else:
                accident_prob = 0.02

            accident_event[i] = 1 if rng.random() < accident_prob else 0

        # 节假日：按日历日期赋值（示例年份的春节与国庆）
        holiday_dates = [
            # 春节（假设为1月25日，仅作示例）
            pd.Timestamp('2022-01-25'), pd.Timestamp('2022-01-26'), pd.Timestamp('2022-01-27'),
            # 国庆节（10月1日至10月7日）
            pd.Timestamp('2022-10-01'), pd.Timestamp('2022-10-02'), pd.Timestamp('2022-10-03'),
            pd.Timestamp('2022-10-04'), pd.Timestamp('2022-10-05'), pd.Timestamp('2022-10-06'),
            pd.Timestamp('2022-10-07'),
        ]

        for i, ts in enumerate(timestamps):
            holiday_event[i] = 1 if ts.date() in [h.date() for h in holiday_dates] else 0

        # 组合所有特征
        semantic_data = np.concatenate([
            weather, holiday, accident, time_of_day, season,
            rainstorm, accident_event, holiday_event
        ], axis=-1)

        # 创建事件字典用于后续分析
        external_events = {
            'rainstorm': rainstorm.flatten(),
            'accident': accident_event.flatten(),
            'holiday': holiday_event.flatten()
        }

        return semantic_data, external_events

    # 生成语义和事件数据
    semantic_all, external_events = generate_semantic_data_with_events(timestamps, seed_local=seed)

    # =========== 统一切分训练集和测试集，顺序一致 ===========
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    semantic_train, semantic_test = semantic_all[:split_idx], semantic_all[split_idx:split_idx + len(X_test)]

    # 也切分外部事件数据
    events_train = {}
    events_test = {}
    for key in external_events:
        events_train[key] = external_events[key][:split_idx]
        events_test[key] = external_events[key][split_idx:split_idx + len(X_test)]

    print(f"语义数据维度: 训练集={semantic_train.shape}, 测试集={semantic_test.shape}")
    print(f"外部事件数据已加载: {list(external_events.keys())}")

    print(f"\n数据加载完成:")
    print(f"时间步数: {len(flow_values)}")
    print(f"样本数: {len(X)} (训练: {len(X_train)}, 测试: {len(X_test)})")
    print(f"邻接矩阵密度: {np.mean(adj_matrix > 0):.2%}")
    print(f"有效距离边数: {valid_edges}")

    return (X_train, semantic_train), (
    X_test, semantic_test), y_train, y_test, adj_matrix, scaler, sensor_ids, events_test


# ==================== 2. 增强的图卷积层 ====================
class GraphConvWithUncertainty(Layer):
    """集成距离信息的图卷积层（修复序列化问题）"""

    def __init__(self, units, adj_matrix, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.adj_matrix = adj_matrix  # 存储原始numpy数组

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            name='graph_kernel'
        )
        self.uncertainty_kernel = self.add_weight(
            shape=(input_shape[-1], 1),
            initializer='orthogonal',
            name='uncertainty_kernel'
        )
        # 在build中转换为Tensor
        self.adj_matrix_tensor = tf.constant(self.adj_matrix, dtype=tf.float32)

    def call(self, inputs):
        # 空间特征传播
        spatial_feat = tf.einsum('ij,btj->bti', self.adj_matrix_tensor, inputs)
        output = tf.einsum('bti,io->bto', spatial_feat, self.kernel)

        # 不确定性估计
        uncertainty = tf.einsum('bti,io->bto', inputs, self.uncertainty_kernel)
        uncertainty = tf.nn.softplus(uncertainty)

        return tf.nn.relu(output), uncertainty

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'adj_matrix': self.adj_matrix.tolist()  # 将numpy数组转为列表
        })
        return config

    @classmethod
    def from_config(cls, config):
        # 从配置中重建层
        adj_matrix = np.array(config.pop('adj_matrix'))
        return cls(adj_matrix=adj_matrix, **config)


# ==================== 3. 语义引导的并行注意力层 ====================
class SemanticGuidedParallelAttention(Layer):
    """语义引导的并行注意力机制（创新点实现）"""

    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.attention_weights_time = None
        self.attention_weights_space = None

    def build(self, input_shapes):
        # 输入: [时间特征, 空间特征, 语义向量]
        time_shape, space_shape, semantic_shape = input_shapes

        # 时间分支参数
        self.W_time_q = self.add_weight(shape=(time_shape[-1], self.units),
                                        initializer='glorot_uniform', name='W_time_q')
        self.W_time_k = self.add_weight(shape=(semantic_shape[-1], self.units),
                                        initializer='glorot_uniform', name='W_time_k')
        self.W_time_v = self.add_weight(shape=(semantic_shape[-1], self.units),
                                        initializer='glorot_uniform', name='W_time_v')

        # 空间分支参数
        self.W_space_q = self.add_weight(shape=(space_shape[-1], self.units),
                                         initializer='glorot_uniform', name='W_space_q')
        self.W_space_k = self.add_weight(shape=(semantic_shape[-1], self.units),
                                         initializer='glorot_uniform', name='W_space_k')
        self.W_space_v = self.add_weight(shape=(semantic_shape[-1], self.units),
                                         initializer='glorot_uniform', name='W_space_v')

        # 语义融合参数
        self.W_fuse = self.add_weight(shape=(2 * self.units, self.units),
                                      initializer='glorot_uniform', name='W_fuse')
        self.b_fuse = self.add_weight(shape=(self.units,), initializer='zeros', name='b_fuse')

        # 动态权重参数
        self.W_dynamic = self.add_weight(shape=(semantic_shape[-1], 1),
                                         initializer='glorot_uniform', name='W_dynamic')

        super().build(input_shapes)

    def call(self, inputs):
        H_t, H_v, c = inputs  # 时间特征, 空间特征, 语义向量

        # 1. 时间注意力分支
        Q_time = tf.einsum('bi,ij->bj', H_t, self.W_time_q)  # [batch, units]
        K_time = tf.einsum('bi,ij->bj', c, self.W_time_k)  # [batch, units]
        V_time = tf.einsum('bi,ij->bj', c, self.W_time_v)  # [batch, units]

        # 计算注意力分数 (缩放点积)
        attn_scores_time = tf.einsum('bi,bi->b', Q_time, K_time) / tf.sqrt(tf.cast(self.units, tf.float32))
        attn_weights_time = tf.nn.softmax(attn_scores_time, axis=-1)

        # 应用注意力权重
        attn_time = tf.einsum('b,bj->bj', attn_weights_time, V_time)

        # 2. 空间注意力分支
        Q_space = tf.einsum('bi,ij->bj', H_v, self.W_space_q)  # [batch, units]
        K_space = tf.einsum('bi,ij->bj', c, self.W_space_k)  # [batch, units]
        V_space = tf.einsum('bi,ij->bj', c, self.W_space_v)  # [batch, units]

        # 计算注意力分数 (缩放点积)
        attn_scores_space = tf.einsum('bi,bi->b', Q_space, K_space) / tf.sqrt(tf.cast(self.units, tf.float32))
        attn_weights_space = tf.nn.softmax(attn_scores_space, axis=-1)

        # 应用注意力权重
        attn_space = tf.einsum('b,bj->bj', attn_weights_space, V_space)

        # 3. 自适应特征融合
        combined = tf.concat([attn_time, attn_space], axis=-1)
        fused = tf.einsum('bi,ij->bj', combined, self.W_fuse) + self.b_fuse
        fused = tf.nn.relu(fused)

        # 4. 语义引导的动态权重分配（创新点核心）
        # 根据语义上下文动态调整时空特征重要性
        semantic_weight = tf.einsum('bi,ij->bj', c, self.W_dynamic)
        semantic_weight = tf.sigmoid(semantic_weight)  # 映射到[0,1]范围

        # 最终输出 = 时间特征 * 权重 + 空间特征 * (1 - 权重) + 融合特征
        output = semantic_weight * H_t + (1 - semantic_weight) * H_v + fused

        # 5. 针对交通突发场景的优化（如事故、拥堵）
        # 当事故概率高时，增加时间特征的权重
        accident_prob = c[:, 2]  # 假设语义向量的第三维是事故概率
        accident_factor = tf.where(accident_prob > 0.8, 0.3, 0.0)
        semantic_weight = tf.minimum(semantic_weight + accident_factor, 1.0)

        # 存储注意力权重以便后续分析
        self.attention_weights_time = attn_weights_time
        self.attention_weights_space = attn_weights_space

        return output

    def get_attention_weights(self):
        """返回注意力权重"""
        return self.attention_weights_time, self.attention_weights_space

    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units})
        return config


# ==================== 4. 概率解码层 ====================
class ProbabilisticDecoder(Layer):
    def __init__(self, num_nodes, num_components=3, **kwargs):
        super().__init__(**kwargs)
        self.num_components = num_components
        self.num_nodes = num_nodes

    def build(self, input_shape):
        # 输出维度变为 num_nodes * num_components，对应 K 个分量的参数
        self.means_layer = Dense(self.num_nodes * self.num_components, name='means_layer')
        self.scales_layer = Dense(self.num_nodes * self.num_components, activation='softplus', name='scales_layer')
        self.weights_layer = Dense(self.num_nodes * self.num_components, activation='softmax', name='weights_layer')
        super().build(input_shape)

    def call(self, inputs):
        means = self.means_layer(inputs)
        scales = self.scales_layer(inputs)
        weights = self.weights_layer(inputs)
        # 形状变为 [batch, num_nodes, num_components, 1]
        means = tf.reshape(means, [-1, self.num_nodes, self.num_components, 1])
        scales = tf.reshape(scales, [-1, self.num_nodes, self.num_components, 1])
        weights = tf.reshape(weights, [-1, self.num_nodes, self.num_components, 1])

        return tf.concat([means, scales, weights], axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_nodes': self.num_nodes,
            'num_components': self.num_components
        })
        return config


# ==================== 5. 完整模型构建 ====================
def build_pems_model(seq_length, num_nodes, adj_matrix, semantic_dim=5):
    inputs = Input(shape=(seq_length, num_nodes))
    semantic_input = Input(shape=(semantic_dim,), name='semantic_input')

    # 时间特征提取
    time_feat = Bidirectional(LSTM(64, return_sequences=True))(inputs)
    time_feat = LSTM(128)(time_feat)  # [batch, 128]

    # 空间特征提取
    space_feat, space_uncertainty = GraphConvWithUncertainty(
        units=64, adj_matrix=adj_matrix)(inputs)
    space_feat = LSTM(128)(space_feat)  # [batch, 128]
    last_uncertainty = space_uncertainty[:, -1]  # [batch, 1]

    # 【修改点】将不确定特征 U 与语义特征拼接，作为上下文向量 C'
    # 把 [batch, 1] 的 U 扩展成 [batch, semantic_dim] 或直接拼接
    uncertainty_expanded = Dense(semantic_dim, activation='relu')(last_uncertainty)
    context_c = Concatenate(axis=-1)([semantic_input, uncertainty_expanded]) # [batch, semantic_dim*2]

    # 语义引导的并行注意力
    # 注意：原来的 SemanticGuidedParallelAttention 的 build 里需要接受 context_c 的维度
    # 如果 context_c 维度变了，SemanticGuidedParallelAttention 内部会自动适应 semantic_shape[-1]
    fused_feat = SemanticGuidedParallelAttention(units=128, name='semantic_guided_parallel_attention')(
        [time_feat, space_feat, context_c])

    # 最终概率解码（直接由 fused_feat 输出，不再拼接未进入注意力的 U）
    dist_params = ProbabilisticDecoder(num_nodes=num_nodes, num_components=3)(fused_feat)

    model = Model([inputs, semantic_input], outputs=dist_params)
    return model


# ==================== 7. 辅助函数 ====================
# 修正后的分布创建函数
# ==================== 7. 辅助函数 ====================
def create_mixture_distribution(params, num_nodes, num_components=3):
    # params 形状: [batch, num_nodes, num_components, 3]
    means = params[..., 0]
    scales = tf.nn.softplus(params[..., 1]) + 1e-6
    weights = tf.nn.softmax(params[..., 2], axis=-1)

    # 构造拉普拉斯分量
    component_dist = tfd.Laplace(loc=means, scale=scales)

    # 真正的拉普拉斯混合模型
    mix_dist = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=weights),
        components_distribution=component_dist
    )

    return tfd.Independent(mix_dist, reinterpreted_batch_ndims=1)


# ==================== 8. 主程序 ====================
def parse_args():
    parser = argparse.ArgumentParser(description='SGPSTN 训练与评估')
    parser.add_argument('--flow', default='data/pems08.npz', help='流量数据 npz/csv 路径')
    parser.add_argument('--dist', default='data/distance.csv', help='传感器距离表路径')
    parser.add_argument('--out', default='runs/exp1', help='输出目录（权重、日志）')
    parser.add_argument('--num-sensors', type=int, default=170, help='使用的传感器数量')
    parser.add_argument('--seq-length', type=int, default=12, help='输入历史步长（5 min/步）')
    parser.add_argument('--horizon', type=int, default=6, help='预测步长，6 对应 30 min')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42, help='语义生成与初始化的随机种子')
    return parser.parse_args()


# ==================== 8. 主程序 ====================
def main():
    args = parse_args()
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    SAVE_DIR = args.out
    os.makedirs(SAVE_DIR, exist_ok=True)

    time_step = args.horizon * 5  # 6 步 × 5 min = 30 min
    seq_length = args.seq_length

    print(f"\n预测时域: {time_step} min（{args.horizon} 步）")

    # 加载数据
    print("加载数据...")
    data = load_pems_data(
        flow_path=args.flow,
        dist_path=args.dist,
        seq_length=seq_length,
        num_sensors=args.num_sensors,
        seed=args.seed
    )

    # 验证返回值
    if len(data) != 8:  # 现在返回8个值
        raise ValueError("load_pems_data() 应该返回8个值")

    (X_train, semantic_train), (
        X_test, semantic_test), y_train, y_test, adj_matrix, scaler, sensor_ids, events_test = data

    # 打印数据形状用于调试
    print("\n数据形状验证:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"semantic_train: {semantic_train.shape}")
    print(f"semantic_test: {semantic_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")
    print(f"adj_matrix: {adj_matrix.shape}")
    print(f"传感器数量: {len(sensor_ids)}")

    # 定义模型参数
    seq_length = X_train.shape[1]
    num_nodes = X_train.shape[2]
    semantic_dim = semantic_train.shape[1]

    print(f"\n模型参数:")
    print(f"序列长度: {seq_length}")
    print(f"节点数: {num_nodes}")
    print(f"语义维度: {semantic_dim}")

    # 构建模型
    print("\n构建模型...")
    model = build_pems_model(
        seq_length=seq_length,
        num_nodes=num_nodes,
        adj_matrix=adj_matrix,
        semantic_dim=semantic_dim
    )

    # 打印模型结构
    model.summary()

    # 编译模型（定义损失函数和指标）
    def nll_loss(y_true, y_pred):
        y_true = tf.reshape(y_true, [-1, num_nodes])
        # 记得传入 num_components=3
        dist = create_mixture_distribution(y_pred, num_nodes=num_nodes, num_components=3)
        return -tf.reduce_mean(dist.log_prob(y_true))

    def mae_metric(y_true, y_pred):
        """自定义MAE指标：使用混合分布的期望"""
        y_true = tf.reshape(y_true, [-1, num_nodes])

        # 从输出中分离 mean, scale, weight (形状均为 [batch, num_nodes, num_components])
        means = y_pred[..., 0]
        weights = y_pred[..., 2]  # 注意：如果你的输出排列是 [mean, scale, weight]，这里取 2

        # 加上 softmax 保证权重和为 1 (虽然在 decoder 里激活过了，但这里更严谨)
        weights = tf.nn.softmax(weights, axis=-1)

        # 计算混合分布的期望： sum(weight * mean)
        y_pred_expect = tf.reduce_sum(weights * means, axis=-1)  # 结果形状: [batch, num_nodes]

        return tf.reduce_mean(tf.abs(y_pred_expect - y_true))

    model.compile(
        optimizer=Adam(learning_rate=args.lr),
        loss=nll_loss,
        metrics=[mae_metric]
    )

    # 训练模型
    print("\n开始训练模型...")
    history = model.fit(
        [X_train, semantic_train],
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.2,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint(os.path.join(SAVE_DIR, f'best_model_{time_step}min.h5'),
                            save_best_only=True)
        ],
        verbose=2
    )

    # 评估模型
    print("\n评估模型...")
    model.load_weights(os.path.join(SAVE_DIR, f'best_model_{time_step}min.h5'))

    # ==================== 语义分项消融分析 ====================
    print("\n=== 开始语义分项消融分析 ===")

    y_test_inv = scaler.inverse_transform(y_test)
    from sklearn.metrics import r2_score

    scenarios = {
        'Only_Weather': [0, 5],  # 仅天气
        'Only_Holiday': [1, 7],  # 仅节假日
        'Only_Accident': [2, 6],  # 仅事故
        'No_Weather': [1, 2, 3, 4, 6, 7],  # 移除天气
        'No_Accident': [0, 1, 3, 4, 5, 7]  # 移除事故
    }

    for name, keep_indices in scenarios.items():
        # 把不相关特征的列置为 0
        sem_mask = np.zeros_like(semantic_test)
        sem_mask[:, keep_indices] = semantic_test[:, keep_indices]

        # 预测
        pred_params_masked = model.predict([X_test, sem_mask], batch_size=128, verbose=0)
        dist_masked = create_mixture_distribution(pred_params_masked, num_nodes=num_nodes, num_components=3)
        y_pred_mean_masked = dist_masked.mean().numpy()

        # 反归一化
        y_pred_mean_inv_masked = scaler.inverse_transform(y_pred_mean_masked)

        # 计算指标 (加上 R2)
        mae_masked = mean_absolute_error(y_test_inv, y_pred_mean_inv_masked)
        rmse_masked = np.sqrt(mean_squared_error(y_test_inv, y_pred_mean_inv_masked))
        r2_masked = r2_score(y_test_inv, y_pred_mean_inv_masked)

        print(f"{name} - MAE: {mae_masked:.4f}, RMSE: {rmse_masked:.4f}, R2: {r2_masked:.4f}")

    # 预测测试集
    pred_params = model.predict([X_test, semantic_test], batch_size=128, verbose=1)
    dist = create_mixture_distribution(pred_params, num_nodes=num_nodes)
    y_pred_mean = dist.mean().numpy()
    samples = dist.sample(100).numpy()  # 采样100次
    y_pred_std = np.std(samples, axis=0)  # 计算样本标准差

    # 计算测试集的 NLL
    test_nll = -np.mean(dist.log_prob(y_test).numpy())

    # 反归一化
    y_test_inv = scaler.inverse_transform(y_test)
    y_pred_mean_inv = scaler.inverse_transform(y_pred_mean)
    # 标准差恢复到原始流量单位：除以缩放系数
    y_pred_std_inv = y_pred_std / scaler.scale_

    # ==================== 不确定性分层统计 ====================
    # 每个“测试时刻—传感器”预测点作为一条观测
    uncertainty_flat = y_pred_std_inv.reshape(-1)
    absolute_error_flat = np.abs(
        y_pred_mean_inv - y_test_inv
    ).reshape(-1)

    # 同时筛选，保证不确定性和误差逐点对应
    valid = (
            np.isfinite(uncertainty_flat)
            & np.isfinite(absolute_error_flat)
            & (uncertainty_flat >= 0)
    )

    layer_data = pd.DataFrame({
        "不确定性": uncertainty_flat[valid],
        "绝对误差": absolute_error_flat[valid]
    })

    if layer_data.empty:
        raise ValueError("没有可用于分层统计的有效预测点")

    # 按不确定性的三分位数分组；重复分位点自动合并
    if layer_data["不确定性"].nunique() == 1:
        layer_data["组号"] = 0
    else:
        layer_data["组号"] = pd.qcut(
            layer_data["不确定性"],
            q=3,
            labels=False,
            duplicates="drop"
        )

    group_ids = sorted(layer_data["组号"].unique())
    group_count = len(group_ids)

    if group_count == 3:
        group_names = ["低不确定性", "中不确定性", "高不确定性"]
    elif group_count == 2:
        group_names = ["较低不确定性", "较高不确定性"]
    else:
        group_names = ["单组（无法区分不确定性层级）"]

    rows = []
    for group_id, group_name in zip(group_ids, group_names):
        group = layer_data[layer_data["组号"] == group_id]
        errors = group["绝对误差"].to_numpy()

        rows.append({
            "分组": group_name,
            "样本量": len(group),
            "不确定性最小值": group["不确定性"].min(),
            "不确定性最大值": group["不确定性"].max(),
            "平均不确定性": group["不确定性"].mean(),
            "MAE": errors.mean(),
            "RMSE": np.sqrt(np.mean(errors ** 2))
        })

    stratified_results = pd.DataFrame(rows)

    print("\n=== 完整测试集的不确定性分层统计 ===")
    print(stratified_results.to_string(index=False))

    # 计算评估指标
    test_mae = mean_absolute_error(y_test_inv, y_pred_mean_inv)
    test_rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_mean_inv))
    test_variance = np.mean(y_pred_std_inv ** 2)
    test_mape = np.mean(np.abs((y_test_inv - y_pred_mean_inv) / (y_test_inv + 1e-6))) * 100

    flattened_y_test = y_test_inv.flatten()
    flattened_y_pred = y_pred_mean_inv .flatten()

    # 过滤NaN值
    valid_indices = ~np.isnan(flattened_y_test) & ~np.isnan(flattened_y_pred)
    test_corr, _ = pearsonr(flattened_y_test[valid_indices], flattened_y_pred[valid_indices])

    print(f"\n测试集性能指标 ({time_step}分钟):")
    print(f"NLL: {test_nll:.4f}")
    print(f"MAE: {test_mae:.4f}")
    print(f"RMSE: {test_rmse:.4f}")
    print(f"Variance: {test_variance:.4f}")
    print(f"MAPE: {test_mape:.4f}%")
    print(f"相关系数: {test_corr:.4f}")

if __name__ == "__main__":
    main()