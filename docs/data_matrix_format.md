# data_matrix 文件格式（由代码反向确认）

读取入口：`src/Multimethod.cpp:229` 的 `MultiMet::Initial()`。  
数据通过 `operator>>` 连续读取，因此本质是“按 token 顺序”而不是严格按行。

## 1) 维度参数来源与强约束

- 参数由主程序传入构造函数：`src/main.cpp:307`、`src/main.cpp:331`
- 构造函数签名：`include/Multimethod.h:27`
- 关键维度：
  - `Cnum`：Cloud 数量
  - `Enum`：Edge 数量
  - `Dnum`：Device 数量
  - `CE_Tnum`：Cloud-Edge 任务数
  - `M_Jnum`：Job 数
  - `M_OPTnum`：每 Job 操作数
- 重要约束（由 `src/Problems.cpp` 的索引方式决定）：
  - 实际运行应满足 `CE_Tnum == M_Jnum`
  - 操作总数 `ops = M_Jnum * M_OPTnum`
  - 决策变量维度 `Nvar = 2 * CE_Tnum + 2 * ops`

## 2) 文件 section 顺序（必须严格一致）

## Section A: `EtoD_Distance`

- 读取：`src/Multimethod.cpp:245`
- token 数：`Enum * Dnum`
- 逻辑形状：`[Enum][Dnum]`
- 含义：Edge 到 Device 的距离/链路代价
- 取值约束：应为正数（代码中作分母使用，见 `src/Problems.cpp:251`）

## Section B: `DtoD_Distance`

- 读取：`src/Multimethod.cpp:251`
- token 数：`Dnum * Dnum`
- 逻辑形状：`[Dnum][Dnum]`
- 含义：Device 到 Device 的距离/链路代价
- 建议约束：对称、对角最小（常见为常量基线），避免异常通信时延

## Section C: `MTask_Time`

- 读取：`src/Multimethod.cpp:257`
- token 数：`M_Jnum * M_OPTnum`
- 逻辑形状：`[M_Jnum * M_OPTnum]`
- 含义：每个操作的处理时长

## Section D: `CETask_Property`（每任务一条记录）

- 读取：`src/Multimethod.cpp:260`
- 记录数：`CE_Tnum`
- 单条记录 token 结构：
  - `Computation`（double）
  - `Communication`（double）
  - `k_pre` + `k_pre` 个 `Precedence` 任务索引
  - `k_inter` + `k_inter` 个 `Interact` 任务索引
  - `k_start` + `k_start` 个 `Start_Pre` 任务索引
  - `k_end` + `k_end` 个 `End_Pre` 任务索引
  - `Job_Constraints`（int，语义见 `src/Problems.cpp:349`、`src/Problems.cpp:354`）
- 索引范围：上述四类依赖索引均应在 `[0, CE_Tnum-1]`
- DAG 约束：`Precedence` 至少应无环（推荐只引用更早任务索引）

## Section E: `AvailDeviceList`（每操作一条）

- 读取：`src/Multimethod.cpp:301`
- 记录数：`M_Jnum * M_OPTnum`
- 单条记录 token 结构：
  - `k_dev` + `k_dev` 个 device 索引
- 索引范围：`[0, Dnum-1]`
- 约束：建议 `k_dev >= 1`，避免空可用集

## Section F: `AvailEdgeServerList`（每任务一条）

- 读取：`src/Multimethod.cpp:315`
- 记录数：`CE_Tnum`
- 单条记录 token 结构：
  - `k_edge` + `k_edge` 个 edge 索引
- 索引范围：`[0, Enum-1]`
- 约束：建议 `k_edge >= 1`

## Section G: `EnergyList`

- 读取：`src/Multimethod.cpp:327`
- token 数：`11`
- 含义：利用率档位 0..10 的能耗参数

## 3) `CETask` 字段在求解中的使用

定义：`include/Problems.h:22`

- `Computation` / `Communication`
  - 影响 CE 层计算时间和通信时间：`src/Problems.cpp:281`、`src/Problems.cpp:292`
- `Precedence` / `Start_Pre` / `End_Pre` / `Interact`
  - 影响 CE 任务开始/结束时刻约束：`src/Problems.cpp:314`-`src/Problems.cpp:360`
- `Job_Constraints`
  - 约束是否与本任务对应 Job 的最后一道操作完工时间绑定：`src/Problems.cpp:349`、`src/Problems.cpp:354`
- `AvailEdgeServerList`
  - 仅在该列表内选择 edge：`src/Problems.cpp:54`-`src/Problems.cpp:56`

## 4) `data_matrix_100.txt` 实测统计（按 100/100/300/100/5 解析）

- `EtoD_Distance`：`[100 x 300]`，min=1000，max=1299
- `DtoD_Distance`：`[300 x 300]`，min=500，max=799，对角线常量 500
- `MTask_Time`：500 个值，取值 {1,2,3,4,5}
- `Computation`：100..199
- `Communication`：10..14
- 四类依赖长度：全部为 0（密度为 0）
- `Job_Constraints`：全部为 0
- `AvailDeviceList`：每操作固定 5 个设备
- `AvailEdgeServerList`：每任务固定 3 个 edge
- `EnergyList`：`0 1 2 3 4 5 6 7 8 9 10`

