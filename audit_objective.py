from audit_lib import *


def write_epsilon_schedule() -> dict:
    epsilon0 = 0.2
    epsilon_min = 0.02
    epsilon_k = 0.01
    gmax = 10000
    rows = []
    for g in [0, 100, 500, 1000, 2000, 5000, 8000, 10000]:
        eps = max(epsilon_min, epsilon0 * math.exp(-epsilon_k * g / gmax))
        rows.append([g, eps])
    write_csv(ROOT / "epsilon_schedule.csv", ["generation", "epsilon"], rows)
    return {"epsilon0": epsilon0, "epsilon_min": epsilon_min, "epsilon_k": epsilon_k, "Gmax": gmax}


def write_processing_time_dependency() -> None:
    parsed = parse_data_matrix(ROOT / "data" / SCALES["T100"]["data_file"], "T100")
    rows = []
    picked = 0
    for op_idx in range(len(parsed["avail_devices"])):
        if picked >= 5:
            break
        job = op_idx // SCALES["T100"]["mopt"] + 1
        op = op_idx % SCALES["T100"]["mopt"] + 1
        proc = parsed["mtask_time"][op_idx]
        for dev in parsed["avail_devices"][op_idx]:
            rows.append([job, op, dev, proc])
        picked += 1
    write_csv(ROOT / "processing_time_device_dependency.csv", ["job", "op", "eligible_device", "processing_time"], rows)


def write_makespan_audit() -> None:
    rows = []
    for scale in SCALES:
        for seed in [1, 2, 3]:
            rows.append([scale, seed, "REQUIRES RERUN", "REQUIRES RERUN", "REQUIRES RERUN", "REQUIRES RERUN"])
    write_csv(ROOT / "makespan_audit.csv", ["problem", "seed", "max_ET_CE", "max_ET_MFG", "overall_completion_time", "mismatch_flag"], rows)


def write_server_model() -> None:
    text = """# server scheduling model

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
"""
    (ROOT / "server_scheduling_model.md").write_text(text, encoding="utf-8")


def write_device_gene_semantics() -> None:
    text = """# device gene semantics

## 当前实现语义
- `m_i` 在代码里不是固定 operation ID 上的设备基因。
- 在 `src/Problems.cpp` 中，设备段先跟随 `op_order`（排序后的工序执行次序）逐位读取，再写回 `mvar[op]`。
- 因此设备基因位 `i` 的真实语义是“执行序 rank = i 的工序选哪台设备”。

关键片段：
```cpp
for (int i = 0; i < ops; i++) {
    int op = op_order[i];
    int idx = (int)(var[CE_Tnum * 2 + ops + i] * avail);
    mvar[op] = AvailDeviceList[op][idx];
}
```

## 具体例子
- 假设两个工序 `op7` 和 `op12`。
- 当排序结果 `op_order = [op7, op12, ...]` 时，设备段第 0 位控制 `op7`。
- 如果 sequencing 改成 `op_order = [op12, op7, ...]`，设备段第 0 位就改为控制 `op12`。
- 也就是说，同一个 `m_i` 会因为 sequencing 改变而映射到不同 operation。

## 对 CC 分解合理性的影响
- 这会让 sequence block 与 device block 强耦合。
- device 子块并不是在固定任务集合上独立优化，而是依赖 sequence block 先给出 rank-to-operation 映射。
- 因此论文若把 device block 说成“对固定工序设备分配的独立子问题”，该表述不严格。

## 是否存在 alternative encoding
- 在当前主实现代码中，未找到按 operation ID 直接编码设备选择的 alternative encoding。
"""
    (ROOT / "device_gene_semantics.md").write_text(text, encoding="utf-8")


def main() -> None:
    write_epsilon_schedule()
    write_processing_time_dependency()
    write_makespan_audit()
    write_server_model()
    write_device_gene_semantics()


if __name__ == "__main__":
    main()
