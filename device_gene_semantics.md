# device gene semantics

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
