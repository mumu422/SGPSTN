"""外部语义特征生成（仿真）。

论文 3.1 中 8 维外部语义特征采用程序仿真生成，非实测气象或事故记录。
本模块独立实现同一套生成规则，便于单独核对与复用：

    idx 0  weather_intensity  天气强度，U(0,1) 连续值
    idx 1  holiday_flag       节假日标记，Bernoulli(0.5)
    idx 2  accident_flag      事故标记，Bernoulli(0.5)
    idx 3  time_of_day        时刻编码，sin 变换至 [0,1]
    idx 4  season             季节编码，按年积日 sin 变换至 [0,1]
    idx 5  rainstorm          暴雨标志，夏季(6-8月) p=0.30，其余 p=0.05
    idx 6  accident_event     事故标志，早晚高峰(7:00-9:00, 17:00-19:00) p=0.10，平峰 p=0.02
    idx 7  holiday_event      节假日标志，按日历节假日日期命中

所有随机抽取由 numpy.random.default_rng(seed) 驱动，固定 seed 即可完全复现。
同一时刻所有节点共享同一份语义向量，不区分站点空间差异；语义标记不改变真实流量。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_NAMES = [
    'weather_intensity', 'holiday_flag', 'accident_flag',
    'time_of_day', 'season', 'rainstorm', 'accident_event', 'holiday_event',
]

# 示例年份的节假日（春节、国庆），仅用于演示规则
HOLIDAY_DATES = [
    '2022-01-25', '2022-01-26', '2022-01-27',
    '2022-10-01', '2022-10-02', '2022-10-03', '2022-10-04',
    '2022-10-05', '2022-10-06', '2022-10-07',
]


def make_semantic_features(timestamps, seed: int = 42) -> np.ndarray:
    """按固定种子生成 (T, 8) 的语义特征矩阵。"""
    rng = np.random.default_rng(seed)
    ts = pd.to_datetime(pd.Series(timestamps))
    n = len(ts)

    weather = rng.uniform(0.0, 1.0, (n, 1))
    holiday = rng.integers(0, 2, (n, 1)).astype(float)
    accident = rng.integers(0, 2, (n, 1)).astype(float)

    hour = ts.dt.hour.to_numpy() + ts.dt.minute.to_numpy() / 60.0
    tod = 0.5 + 0.5 * np.sin(2 * np.pi * hour / 24.0)
    season = 0.5 + 0.5 * np.sin(2 * np.pi * ts.dt.dayofyear.to_numpy() / 365.0)

    month = ts.dt.month.to_numpy()
    rain_p = np.where(np.isin(month, [6, 7, 8]), 0.30, 0.05)
    rainstorm = (rng.random(n) < rain_p).astype(float)

    peak = ((hour >= 7) & (hour <= 9)) | ((hour >= 17) & (hour <= 19))
    acc_p = np.where(peak, 0.10, 0.02)
    accident_event = (rng.random(n) < acc_p).astype(float)

    dates = ts.dt.strftime('%Y-%m-%d')
    holiday_event = dates.isin(HOLIDAY_DATES).to_numpy().astype(float)

    return np.column_stack([
        weather.ravel(), holiday.ravel(), accident.ravel(),
        tod, season, rainstorm, accident_event, holiday_event,
    ]).astype(np.float32)


if __name__ == '__main__':
    demo = pd.date_range('2016-07-01', periods=5, freq='5min')
    out = make_semantic_features(demo, seed=42)
    print(out.shape)
    for name, value in zip(FEATURE_NAMES, out[0]):
        print(f'{name}: {value:.4f}')
