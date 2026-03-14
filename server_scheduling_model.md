# server scheduling model

## 1. 代码实现结论
- Cloud/Edge 服务器上的 CE task 不是“单机串行队列”模型。
- 也不是显式的 capacity-constrained resource allocation 模型。
- 更接近“按分配结果并行执行，再用负载规模和通信量修正执行时间”的近似模型。

## 2. 关键实现位置
- `src/Problems.cpp`
- 关键逻辑：
```cpp
if (cevar[i] < Cnum)
    CloudLoad[cevar[i]].push_back(i);
else
    EdgeLoad[cevar[i]].push_back(i);
```
```cpp
size_t edge_load_size = EdgeLoad[edge].size();
double edge_extra = (edge_load_size >= 6) ? 0.05 * (double)(edge_load_size - 5) : 0.0;
ce_et_val = ce_st_val + comm_up + comm_down + comp_time * (1.0 + edge_extra);
```
```cpp
int dur = (int)(CE_ET[idx] - CE_ST[idx]);
energy += EnergyList[8] * dur;
```

## 3. ST_i^CE / ET_i^CE 如何决定
- `CE_ST` 先由 CE precedence、制造侧约束、数据依赖约束共同决定。
- `CE_ET = CE_ST + 上传通信 + 计算 + 下载通信`。
- 若落在 edge，上式中的计算时间还会乘以随该 edge 当前任务数增长的额外系数。
- 代码没有给同一 cloud/edge server 维护显式时间轴队列，也没有在服务器内部逐任务串行排队。

## 4. capacity_e / bandwidth / utilization 的作用
- `bandwidth` 通过上传/下载通信时间进入 `CE_ST/CE_ET`。
- `capacity_e` 在当前仓库代码中没有找到一个与“同服并发上限”严格对应的离散排队实现。
- `server utilization` 更像是通过能耗项和 edge 负载惩罚被近似体现，而不是标准资源约束。

## 5. 3 个 CE tasks 同服示例
假设 3 个任务 `A/B/C` 都映射到同一 edge server，且它们的前驱都已满足：
- 三者都可在各自依赖释放后开始，不会因为“服务器上已有 2 个任务在跑”而被显式顺延。
- `ST` 主要由前驱完工时间、制造约束、通信起点决定。
- `ET` 由 `ST + comm_up + comm_down + comp*(1+edge_extra)` 决定。
- 若该 edge 上共有 3 个任务，则 `edge_extra=0`；若共有 6 个及以上任务，才出现额外放大。

## 6. 论文补写建议
- 不应写成严格的单服务器串行调度模型。
- 也不应写成具有明确定义并发容量上限的经典并行机模型。
- 更准确的技术表述应是：CE 层采用依赖驱动的完成时间近似计算，同服负载主要通过通信/负载惩罚影响完成时间，而非显式队列调度。
