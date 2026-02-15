# CED_Schedule 性能优化计划

## 概述

优化 C++17 元启发式求解器（GA/DE/GDE/CCHIHH/QPHH/CGA）用于云-边-端任务调度的执行速度。  
**预期总提升：3-5x 加速（8核机器）**

### 约束条件
- Windows + MSVC 2022, C++17
- 必须保持 `--seed` 确定性可复现
- 同 seed 同输出（串行执行语义保留）
- 所有求解器正常工作

---

## 瓶颈分析

| # | 瓶颈 | 影响范围 | 预估提升 |
|---|------|----------|----------|
| 1 | **串行适应度评估** — `Evaluation()` 循环完全串行，仅1个 Workspace 阻塞并行化 | ~60-80% 运行时间 | 2-5x (多核) |
| 2 | **编译器优化不足** — 仅用 `/O2`，缺 `/GL /arch:AVX2 /fp:fast /LTCG` | 全局 | 10-30% |
| 3 | **热路径堆分配** — `select()`, `Subgradient()`, `APSO_3()` 每次调用都 `new/delete` | 频繁 GC 压力 | 5-15% |
| 4 | **QPHH 互斥锁** — `EvalVarSafe()` 用 mutex 包裹 Eval，OpenMP 形同虚设 | QPHH 求解器 | 2-4x |
| 5 | **CED_Schedule 微优化** — `log2()` 转录、重复除法、vector churn | 每次评估 | 5-10% |

---

## 任务依赖图

```
Wave 1 (立即并行执行，无依赖):
├── Task 1: 编译器标志优化
├── Task 2: 每线程 Workspace 池
├── Task 3: 线程安全 RNG 验证
├── Task 7: 消除热路径堆分配
└── Task 8: CED_Schedule 微优化

Wave 2 (Wave 1 完成后):
├── Task 4: 将可变评估状态迁入 Workspace
└── Task 6: 修复 QPHH 并行评估

Wave 3 (Wave 2 完成后):
└── Task 5: OpenMP 并行化 Evaluation()

Wave 4 (全部完成后):
└── Task 9: 构建验证 + 基准测试
```

| Task | 依赖 | 被阻塞 | 原因 |
|------|------|--------|------|
| Task 1: 编译器标志 | 无 | Task 9 | 纯 CMakeLists.txt 修改，独立 |
| Task 2: Workspace 池 | 无 | Task 4, 5, 6 | 并行评估的基础设施 |
| Task 3: 线程安全 RNG | 无 | Task 5, 6 | OpenMP 启用前必须保证线程安全 |
| Task 4: 迁移可变状态 | Task 2 | Task 5 | 需要 Workspace 结构变更 |
| Task 5: 并行 Evaluation | Task 2, 3, 4 | Task 9 | 需要线程安全 workspace + RNG + 缓冲区 |
| Task 6: QPHH 去锁 | Task 2, 3 | Task 9 | 需要每线程 workspace 池 |
| Task 7: 堆分配消除 | 无 | Task 9 | 独立优化 |
| Task 8: 微优化 | 无 | Task 9 | 独立优化 |
| Task 9: 验证 + 基准 | Task 1-8 | 无 | 必须在所有变更后验证 |

---

## Wave 1：无依赖，立即并行执行

### Task 1: MSVC 编译器标志优化

- **文件**: `CMakeLists.txt`
- **难度**: 简单
- **预估提升**: 10-30%

#### 具体修改

1. **MSVC 编译选项**（当前仅 `/O2`）:
   - `/O2` → `/Ox`（最大优化）
   - 添加 `/GL`（全程序优化）
   - 添加 `/arch:AVX2`（SIMD 向量化）
   - 添加 `/fp:fast`（快速浮点运算）

2. **MSVC 链接选项**:
   - 添加 `/LTCG`（链接时代码生成）via `target_link_options`

3. **GCC/Clang 兼容**（可选）:
   - 添加 `-O3 -march=native -flto` 等效标志

#### 验收标准
- [x] `cmake --build build --config Release` 编译成功无错误
- [ ] `CED_Schedule.exe --bench_eval 1000 --seed 1` 运行无崩溃
- [ ] 与基线相比，`--bench_eval 50000` 有可测量的速度提升

---

### Task 2: 每线程 Workspace 池

- **文件**: `include/Workspace.h`, `include/Multimethod.h`
- **难度**: 中等
- **前置**: 无
- **阻塞**: Task 4, 5, 6

#### 具体修改

1. **在 `Workspace.h` 中添加 `WorkspacePool` 类**:
   ```cpp
   class WorkspacePool {
   public:
       void init(int num_threads, /* Workspace resize params */);
       Workspace& get(int thread_id);
       int size() const;
   private:
       std::vector<Workspace> pool_;
   };
   ```

2. **在 `MultiMet` 类中添加成员**:
   ```cpp
   WorkspacePool ws_pool;  // 与现有 workspace 并存
   ```

3. **在 MultiMet 构造函数中初始化**:
   - 池大小 = `omp_get_max_threads()`
   - 每个 Workspace 调用 `resize()` 配置正确的维度

4. **保持现有 `workspace` 成员不变**，用于串行代码路径的向后兼容。

#### 验收标准
- [ ] 编译成功
- [ ] `ws_pool.get(0)` 返回有效的 Workspace 引用
- [ ] 现有串行代码行为不变

---

### Task 3: 线程安全 RNG 验证

- **文件**: `include/Rng.h`, `src/QPHH.cpp`
- **难度**: 简单
- **前置**: 无
- **阻塞**: Task 5, 6

#### 具体修改

1. **验证 `CED_Schedule()` 中无 `rand()` 调用**:
   - `CED_Schedule()` 是纯确定性函数（给定输入），不使用 RNG → 并行安全
   - `Evaluation()` 循环体仅调用 `EVAL_COMPAT` → 并行安全

2. **QPHH 已有 `thread_local std::mt19937`**（`QPHH.cpp:31-33`）:
   ```cpp
   inline std::mt19937& tls_rng() {
       static thread_local std::mt19937 eng(std::random_device{}());
       return eng;
   }
   ```
   已满足线程安全需求。

3. **注意**：不替换所有 70 处 `rand()` 调用 — 仅确保并行化路径（评估循环、QPHH 初始化）线程安全即可。

#### 验收标准
- [ ] 确认 `CED_Schedule()` 内无 `rand()` 调用
- [ ] 确认 `Evaluation()` 循环体内无 `rand()` 调用
- [ ] 现有串行路径行为不变

---

### Task 7: 消除热路径堆分配

- **文件**: `include/Multimethod.h`, `src/Multimethod.cpp`
- **难度**: 简单
- **预估提升**: 5-15%

#### 具体修改

1. **`select()` 函数**（Multimethod.cpp 第 842-878 行）:
   - 当前：每次调用 `new double[Popsize]` 分配 `rfitness` 和 `cfitness`
   - 修改：在 `MultiMet` 中添加成员 `std::vector<double> sel_rfitness, sel_cfitness`
   - 在构造函数中 `resize(Popsize)`
   - `select()` 中直接使用成员，删除 `new/delete`

2. **`Subgradient()` 函数**（Multimethod.cpp 第 1031-1065 行）:
   - 当前：每次调用分配 `delta` 和 `var` 数组
   - 修改：在 `MultiMet` 中添加成员 `std::vector<double> subgrad_delta`，`std::vector<double*> subgrad_var`（或等效扁平数组）
   - 在构造函数中预分配

3. **`APSO_3()` 函数**（Multimethod.cpp 第 1200 行）:
   - 当前：每次调用 `new double[Nvar]` 分配 `d`
   - 修改：添加成员 `std::vector<double> apso_d`，构造函数中 `resize(Nvar)`

4. **`SPSO()` 函数**（Multimethod.cpp 第 1071 行）:
   - 当前：每次调用分配 `subgrad` 数组
   - 修改：添加成员 `std::vector<double> spso_subgrad`，构造函数中 `resize(Nvar)`

#### 验收标准
- [ ] `select()`, `Subgradient()`, `APSO_3()`, `SPSO()` 中无 `new`/`delete`
- [ ] `--solver GA --seed 1 --generations 5` 输出与基线一致
- [ ] `--bench_eval 10000` 有可测量提升

---

### Task 8: CED_Schedule 微优化

- **文件**: `src/Problems.cpp`, `include/Workspace.h`
- **难度**: 简单
- **预估提升**: 5-10%

#### 具体修改

1. **替换 `log2(1.0 + x)` 调用**（Problems.cpp 第 257 行附近）:
   ```cpp
   // 当前
   rate = log2(1.0 + SNR);
   // 优化后
   static const double inv_ln2 = 1.0 / std::log(2.0);
   rate = std::log1p(SNR) * inv_ln2;
   ```
   `log1p` 更数值稳定且可能更快。

2. **预计算通信因子**:
   ```cpp
   // 当前：每次评估重复计算
   CETask_Property[i].Communication * 10 / (1000 * rate)
   // 优化：数据加载时预计算
   comm_factor[i] = CETask_Property[i].Communication * 0.01;  // 10/1000 = 0.01
   // 评估时
   comm_factor[i] / rate
   ```

3. **预留 CloudLoad/EdgeLoad 容量**:
   - 在 `Workspace::resize()` 中为 `CloudLoad[i]` 和 `EdgeLoad[i]` 调用 `reserve()`
   - 避免 `push_back` 导致的动态重分配
   - 预估每个服务器处理的任务数上限，以此为 reserve 大小

#### 验收标准
- [ ] `--bench_eval 10000 --seed 1` 显示可测量提升
- [ ] `--seed 1 --generations 5` 输出不变
- [ ] 无精度损失（`log1p` 精度 ≥ `log2`）

---

## Wave 2：依赖 Wave 1 (Task 2 + 3)

### Task 4: 将可变评估状态迁入 Workspace

- **文件**: `include/Workspace.h`, `include/Multimethod.h`, `src/Problems.cpp`, `src/Multimethod.cpp`
- **难度**: 高（需仔细追踪跨文件数据流）
- **前置**: Task 2
- **阻塞**: Task 5

#### 背景

`CloudLoad`, `EdgeLoad`, `ST`, `ET`, `CE_ST`, `CE_ET` 当前是 `MultiMet` 的成员，但每次 `CED_Schedule()` 调用都会写入这些缓冲区。为实现并行评估，每个线程需要自己的副本。

#### 具体修改

1. **在 `Workspace` 中添加新成员**:
   ```cpp
   struct Workspace {
       // ... 现有成员 ...
       
       // 从 MultiMet 迁移的评估可变状态
       std::vector<int>* CloudLoad;   // [Cnum]
       std::vector<int>* EdgeLoad;    // [Enum]
       double* ST;                     // [M_Jnum * M_OPTnum]
       double* ET;                     // [M_Jnum * M_OPTnum]
       double* CE_ST;                  // [CE_Tnum]
       double* CE_ET;                  // [CE_Tnum]
       
       void resize(/* 更新参数列表 */);
   };
   ```

2. **更新 `Workspace::resize()`**:
   - 分配 `CloudLoad`（大小 `Cnum`）、`EdgeLoad`（大小 `Enum`）
   - 分配 `ST`、`ET`（大小 `M_Jnum * M_OPTnum`）
   - 分配 `CE_ST`、`CE_ET`（大小 `CE_Tnum`）

3. **更新 `CED_Schedule()` 签名**:
   - 使用 Workspace 拥有的缓冲区，不再访问 `MultiMet` 的成员
   - 更新 `EVAL_COMPAT` 宏传递正确的 Workspace

4. **保持 `MultiMet` 成员作为别名**:
   - 指向默认 `workspace` 中对应的缓冲区，保证读取评估结果的代码向后兼容

#### 关键注意事项
- 这是最容易出错的任务 — 需要仔细跟踪 `CloudLoad`/`EdgeLoad` 在 `Problems.cpp` 中的所有读写位置
- 修改后必须验证串行结果不变

#### 验收标准
- [ ] `--solver GDE --seed 1 --generations 100` 输出与修改前基线完全一致
- [ ] 编译无警告
- [ ] Workspace 正确拥有所有可变评估状态

---

### Task 6: 修复 QPHH 并行评估（去除互斥锁）

- **文件**: `src/QPHH.cpp`, `include/QPHH.h`
- **难度**: 中等
- **前置**: Task 2, 3
- **阻塞**: Task 9

#### 背景

`EvalVarSafe()`（QPHH.cpp 第 122-126 行）用 `std::lock_guard<std::mutex>` 包裹 `Eval()` 调用，导致 `#pragma omp parallel for`（第 163 行和第 251 行）实际串行执行。

#### 具体修改

1. **替换 `EvalVarSafe()` 实现**:
   ```cpp
   // 当前
   double QPHH_Solver::EvalVarSafe(const double* var) const {
       std::lock_guard<std::mutex> lock(g_eval_mutex);
       return solver->Eval(var);
   }
   
   // 优化后
   double QPHH_Solver::EvalVarSafe(const double* var) const {
       int tid = omp_get_thread_num();
       Workspace& ws = solver->ws_pool.get(tid);
       return solver->EvalWithWorkspace(var, ws);
   }
   ```

2. **在 `MultiMet` 中添加 `EvalWithWorkspace()` 方法**:
   ```cpp
   double EvalWithWorkspace(const double* var, Workspace& ws);
   ```
   - 与 `Eval()` 逻辑相同，但使用传入的 Workspace 而非成员 `workspace`

3. **删除全局 `g_eval_mutex`**（QPHH.cpp 第 36 行）

#### 验收标准
- [ ] `--solver QHH --seed 1 --generations 100` 运行无崩溃
- [ ] 多线程下墙钟时间明显减少
- [ ] 无 mutex，无数据竞争

---

## Wave 3：依赖 Wave 2

### Task 5: OpenMP 并行化 Evaluation()

- **文件**: `src/Multimethod.cpp`
- **难度**: 高（核心性能路径，必须正确）
- **前置**: Task 2, 3, 4
- **阻塞**: Task 9
- **预估提升**: 2-5x（多核机器）

#### 具体修改

1. **并行化 `Evaluation()` 循环**（Multimethod.cpp 第 812-830 行）:
   ```cpp
   void MultiMet::Evaluation(int p_start, int p_end) {
       #pragma omp parallel for schedule(static)
       for (int i = p_start; i < p_end; i++) {
           int tid = omp_get_thread_num();
           Workspace& ws = ws_pool.get(tid);
           pop_fit[i] = EvalWithWorkspace(pop[i], ws);
           #pragma omp atomic
           eval_count++;
       }
   }
   ```

2. **并行化 `ComputeReferenceValues()` 初始评估循环**:
   - 同样的模式：`#pragma omp parallel for` + per-thread Workspace

3. **保持所有算子循环（crossover, mutation, DE）串行**:
   - 这些循环使用 `rand()`，并行化会改变 RNG 序列
   - 评估循环是最大的瓶颈，仅并行化评估已足够

4. **eval_count 原子更新**:
   - 使用 `#pragma omp atomic` 保证计数正确

#### 关键注意事项
- `schedule(static)` 确保确定性线程分配（比 `dynamic` 可预测）
- 每个线程使用独立的 Workspace，无共享写入 → 无数据竞争
- 评估函数 `CED_Schedule()` 是纯函数（给定输入确定输出）→ 并行安全

#### 验收标准
- [ ] `--solver GDE --seed 1 --generations 100` 输出与基线一致
- [ ] `--bench_eval 50000` 多线程比单线程快
- [ ] `OMP_NUM_THREADS=1,2,4,8` 下分别测试，展示扩展性
- [ ] 无数据竞争（可用 ThreadSanitizer 或手工验证）

---

## Wave 4：最终验证

### Task 9: 构建验证 + 基准测试

- **文件**: 无（仅执行命令）
- **难度**: 简单
- **前置**: Task 1-8

#### 执行步骤

1. **清洁构建**:
   ```bash
   cmake -S . -B build -G "Visual Studio 17 2022" -A x64
   cmake --build build --config Release
   ```

2. **确定性测试**（与优化前基线对比）:
   ```bash
   # 每个求解器运行短测试
   CED_Schedule.exe --solver GA   --seed 1 --generations 100 --data_dir ./data --data_file data_matrix_100.txt
   CED_Schedule.exe --solver DE   --seed 1 --generations 100 --data_dir ./data --data_file data_matrix_100.txt
   CED_Schedule.exe --solver GDE  --seed 1 --generations 100 --data_dir ./data --data_file data_matrix_100.txt
   CED_Schedule.exe --solver CCHIHH --seed 1 --generations 100 --data_dir ./data --data_file data_matrix_100.txt
   CED_Schedule.exe --solver QHH  --seed 1 --generations 100 --data_dir ./data --data_file data_matrix_100.txt
   CED_Schedule.exe --solver CGA  --seed 1 --generations 100 --data_dir ./data --data_file data_matrix_100.txt
   ```
   **输出的 best_fit 必须与优化前一致**（Task 5 并行化后允许微小浮点差异）。

3. **单线程基准测试**（编译器优化 + 微优化效果）:
   ```bash
   set OMP_NUM_THREADS=1
   CED_Schedule.exe --bench_eval 50000 --seed 1 --data_dir ./data --data_file data_matrix_100.txt
   ```

4. **多线程扩展性测试**:
   ```bash
   set OMP_NUM_THREADS=1
   CED_Schedule.exe --solver GDE --seed 1 --generations 1000 --data_dir ./data --data_file data_matrix_100.txt
   
   set OMP_NUM_THREADS=2
   CED_Schedule.exe --solver GDE --seed 1 --generations 1000 --data_dir ./data --data_file data_matrix_100.txt
   
   set OMP_NUM_THREADS=4
   CED_Schedule.exe --solver GDE --seed 1 --generations 1000 --data_dir ./data --data_file data_matrix_100.txt
   
   set OMP_NUM_THREADS=8
   CED_Schedule.exe --solver GDE --seed 1 --generations 1000 --data_dir ./data --data_file data_matrix_100.txt
   ```

5. **记录结果**:
   - 每个线程数下的墙钟时间
   - 计算加速比（speedup = T1 / Tn）
   - 记录是否有求解器崩溃

#### 验收标准
- [ ] 所有 6 个求解器编译运行无崩溃
- [ ] 确定性测试通过（同 seed 同输出）
- [ ] 单线程 `--bench_eval 50000` 比基线快 >10%
- [ ] 4线程下 `--generations 1000` 墙钟时间比单线程快 >2x
- [ ] 结果记录完整

---

## 提交策略

| 提交序号 | 内容 | 对应任务 |
|---------|------|---------|
| Commit 1 | 编译器标志优化 | Task 1 |
| Commit 2 | 消除热路径堆分配 | Task 7 |
| Commit 3 | CED_Schedule 微优化 | Task 8 |
| Commit 4 | WorkspacePool + 线程安全 RNG + 可变状态迁移 | Task 2, 3, 4 |
| Commit 5 | OpenMP 并行 Evaluation + QPHH 去锁 | Task 5, 6 |

每次提交后验证：`--seed 1 --generations 5` 输出正确。

---

## 成功标准总结

| 指标 | 目标 |
|------|------|
| MSVC 2022 构建 | 无错误无警告 |
| 确定性 | `--solver GDE --seed 1 --generations 100` 输出与基线一致 |
| 单线程加速 | `--bench_eval 50000` 比基线快 >10% |
| 多线程加速 | 4 核下 `--generations 1000` 墙钟时间 >2x |
| 求解器完整性 | GA, DE, GDE, CCHIHH, QHH, CGA 全部无崩溃 |
