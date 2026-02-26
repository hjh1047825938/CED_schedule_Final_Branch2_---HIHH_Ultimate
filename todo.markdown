# CED_Schedule 全量性能优化 TODO v2（严格可复现）

## 摘要
目标：在不改变同 seed/同参数结果轨迹的前提下，优化 `GA/DE/GDE/CCHIHH/GA-SLHH/QHH/IMOMA/CGA` 运行速度。  
本版本已完成核心实现与首轮回归。

## 已落地变更

### WP0 基线冻结与回归门禁
- 新增 `scripts/bench_regression.py`
  - `collect`：采集基线（日志序列、final、耗时）
  - `compare`：对比 `best_fit` 序列与 final
  - `perf`：每条命令跑 N 次取中位数
- 新增 `results/perf_baseline/.gitkeep`

### WP1 构建配置默认严格可复现
- 修改 `CMakeLists.txt`
  - 新增 `CED_STRICT_REPRO`（默认 `ON`）
  - 新增 `CED_FAST_MATH`（默认 `OFF`）
  - MSVC 下默认使用 `/fp:precise`
  - 仅在 `CED_STRICT_REPRO=OFF` 且 `CED_FAST_MATH=ON` 时启用 `/fp:fast`

### WP2 共用评估路径降开销
- 修改 `include/Workspace.h`
  - 新增 `comm_factor_ready`
- 修改 `src/Problems.cpp`
  - `comm_factor` 改为惰性一次初始化（按 workspace 生命周期缓存）
- 修改 `src/Multimethod.cpp`
  - `Evaluation()` 使用线程局部计数 + reduction，减少每次评估原子增量开销
  - 保持 `ComputeReferenceValues()` 仍走 `EvaluFunc(...)` 路径

### WP3 标准 GA/DE/GDE 热点复用
- 修改 `include/Multimethod.h`
  - 新增 `migration_buffer`/`migration_fit_buffer`
  - 新增 `gde_trial_buffer`/`gde_v_buffer`
- 修改 `src/Multimethod.cpp`
  - `GDE()` 复用缓冲，去除每个个体的临时 vector 分配
  - `RingMigration()` 去除每轮 `new[]/delete[]`

### WP4 CCHIHH 专项优化（保持搜索逻辑）
- 修改 `include/CC_HIHH.h`
  - `ComputeState` 改为 `std::array<double, 7>`
  - `ContextualBanditSelector` 接口适配固定状态向量
  - 增加迁移复用缓冲成员
- 修改 `src/CC_HIHH_singleisland.cpp`
  - 状态向量改固定数组，消除代际分配
  - `MigrationWithinBlock()` 改复用缓冲，移除 `new double[]`

### WP5 QPHH 专项优化
- 修改 `include/QPHH.h`
  - 增加 `rank/apply/idx` 复用缓冲与线程 scratch
- 修改 `src/QPHH.cpp`
  - `RunIteration()` 复用 `rank_idx/apply_ids/idx`
  - OpenMP 分支下复用每线程 `order/dev_idx/tmp_var`

### WP6 GA-SLHH 专项优化
- 修改 `include/GA_SLHH.h`
  - 增加 `idx/hashes/counts/trans` 复用缓冲
  - 增加本地搜索复用缓冲
- 修改 `src/GA_SLHH.cpp`
  - `RunGeneration()` 与哈希统计复用缓冲
  - `LocalSearch()` 复用 `best/cand`，减少 trial 中反复构造
  - `ComputeProbabilities()` 复用 `counts/trans_*` 缓冲

### WP7 IMOMA / CGA 专项优化
- 修改 `include/IMOMA.h` + `src/IMOMA.cpp`
  - 非支配排序、front、拥挤度排序缓冲复用
- 修改 `include/CGA.h` + `src/CGA.cpp`
  - `EvaluateIndividual()` 复用 `rt` 缓冲
  - `ApplyCatastrophe()` 复用索引缓冲

## 回归与验证

### 构建
```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

### 基线采集
```powershell
python scripts/bench_regression.py --exe build/Release/CED_Schedule.exe --data_dir data --data_file data_matrix_100.txt --out_dir results/perf_baseline --omp_threads 1 collect
```

### 一致性比对
```powershell
python scripts/bench_regression.py --exe build/Release/CED_Schedule.exe --data_dir data --data_file data_matrix_100.txt --out_dir results/perf_baseline --omp_threads 1 --baseline results/perf_baseline/baseline.json compare
```

### 性能中位数
```powershell
python scripts/bench_regression.py --exe build/Release/CED_Schedule.exe --data_dir data --data_file data_matrix_100.txt --out_dir results/perf_baseline --omp_threads 1 --runs 3 perf
```

## 当前实测结果（本次实现）

### compare（与新基线）
- GA: OK
- DE: OK
- GDE: OK
- CCHIHH: OK
- GA-SLHH: OK
- QHH: OK
- IMOMA: OK
- CGA: OK

### perf median（runs=3, OMP_NUM_THREADS=1）
- GA: 0.439s
- DE: 0.650s
- GDE: 1.266s
- CCHIHH: 2.295s
- GA-SLHH: 1.483s
- QHH: 9.857s
- IMOMA: 2.629s
- CGA: 2.835s

## 重点场景（你给的 CCHIHH 命令）
- 命令可正常运行，示例结果：
  - `The best solution = 0.340934`
  - `Time = 21.346 s`

## 后续建议
1. 若要严格证明“与优化前完全一致”，请先对优化前二进制执行同一 `collect`，再用当前版本做 `compare`。  
2. 如需多 seed 终验，建议对 `seed=1,7,13` 分别执行 `collect/compare/perf` 并汇总到 `results/perf_baseline/seed_*`。
