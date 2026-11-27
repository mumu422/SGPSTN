# SGPSTN：语义引导的概率时空网络

Semantic Guided Probabilistic Spatiotemporal Network (SGPSTN)

本仓库给出论文《面向路网运维的语义引导概率时空交通流预测》的模型实现、训练评估入口与外部语义特征生成规则，用于复现与核对。

**重要说明（数据性质）**

- **交通流数据是真实的公开数据集**：PEMS04、PEMS08、METR-LA，需按 `data/README.md` 自行下载，仓库不分发数据本体。
- **天气、节假日、事故等外部语义特征是程序仿真生成的**，不是实测气象或事故记录，也不是论文所称的“多源实测数据”。它们用于检验语义引导机制，固定随机种子即可逐位复现。
- 语义标记不改变真实流量，因此本仓库不构成具有事故冲击真值的交通仿真。

## 目录结构

```
SGPSTN/
├─ src/
│  ├─ train.py        训练、评估、消融与不确定性分析的完整入口
│  └─ semantic.py     8 维外部语义特征生成规则（独立实现，便于核对）
├─ data/
│  └─ README.md       数据下载来源与放置方式（数据本体不入库）
├─ docs/
│  └─ differences.md  代码实现与论文描述的差异清单
├─ requirements.txt
├─ .gitignore
└─ LICENSE
```

## 环境

Python 3.9 及以上，深度学习框架 TensorFlow 2.x + TensorFlow Probability。已验证的组合见 `requirements.txt`。

```bash
pip install -r requirements.txt
```

GPU 非必需；CPU 可跑通，但训练较慢。

## 数据准备

1. 按 `data/README.md` 下载 `pems08.npz` 与 `distance.csv`（或 PEMS04 / METR-LA 对应文件）。
2. 放入 `data/` 目录，保持如下结构：

```
data/
├─ pems08.npz        维度 (时间步, 节点, 特征)，本实现取通道 0 作为流量
└─ distance.csv      传感器距离表，至少含 from / to / cost 三列
```

3. 时间粒度 5 min，输入窗口 `L=12`（60 min），预测步长 `H=6`（30 min）。

## 快速开始

```bash
# 默认：PEMS08、170 节点、L=12、H=6、batch=32、100 epochs
python src/train.py --flow data/pems08.npz --dist data/distance.csv --out runs/exp1

# 换数据集、换随机种子
python src/train.py --flow data/pems04.npz --dist data/distance.csv4 --num-sensors 307 --seed 2026
```

## 许可

代码采用 MIT 许可，见 `LICENSE`。数据集版权归原始发布方所有，使用请遵循其条款。
