# 基于 AI Agent 的算法能力工厂原型系统

本项目是针对 [docs/requirements.pdf](docs/requirements.pdf) 的实现原型，目标是验证“能力抽取 -> 能力复刻 -> 能力验证 -> 能力沉淀”的完整闭环。系统围绕两个可运行场景展开：

- `Top-K 推荐`：基于 MovieLens 风格隐式反馈数据，自动生成可运行推荐算法代码并验证。
- `表格二分类`：基于客户流失样例数据，自动生成可运行分类算法代码并验证。

系统默认支持 `硅基流动（SiliconFlow）OpenAI 兼容 API`，同时保留 `mock provider` 作为无外部模型时的本地兜底方案。

## 1. 项目背景和目标

传统算法开发通常依赖人工完成需求理解、方案选择、代码实现、调试、验证和经验沉淀。本项目尝试将这一过程拆解为多个可组合的 Agent 阶段，并用一个轻量原型系统演示以下闭环：

```text
自然语言能力描述
  -> 能力理解
  -> 图谱/知识检索
  -> 候选算法搜索
  -> 代码生成
  -> 自动验证
  -> 自动修复
  -> 结果回写知识图谱
```

本项目的目标不是追求最强模型效果，而是验证一个可运行、可复现、可扩展的“算法能力工厂”最小闭环。

## 2. 系统架构和模块设计

### 2.1 总体架构

```text
User / CLI / Web / API
  -> CapabilityFactoryService
     -> LLM Provider
     -> KnowledgeGraphStore
     -> BeamSearchPlanner
     -> CandidateGenerator
     -> CandidateValidator
     -> RepairAgent
     -> ReportWriter
     -> GitRepositoryManager
```

### 2.2 核心模块

- `app/workflow.py`
  端到端编排入口，串联理解、检索、搜索、生成、验证、修复、回写和报告输出。
- `app/llm.py`
  负责能力理解与方案解释，支持 `SiliconFlow` 和 `mock` 两种模式。
- `app/search.py`
  使用 `Beam Search` 在多个候选算法之间做搜索和多样性选择。
- `app/codegen.py`
  通过安全模板生成可运行代码。
- `app/plugin_loader.py`
  通过 manifest 加载算法插件和验证 profile 插件。
- `app/profile_library.py`
  提供不同任务类型的验证配置，目前包含推荐与分类两类 profile。
- `app/validation.py`
  负责 AST 安全扫描、数据快照、子进程沙箱验证调度。
- `app/sandbox_worker.py`
  在独立子进程中执行候选代码，支持 CPU/内存资源限制。
- `app/knowledge.py`
  负责知识图谱加载、检索、布局导出、结果回写、失败经验沉淀。
- `app/repo_mining.py`
  从真实代码仓库中抽取函数、类、依赖、README 片段等能力信息。
- `app/gitops.py`
  暴露 Git 状态与快照能力，用于版本上下文管理。
- `app/main.py` / `app/cli.py`
  提供 Web、API、CLI 三种交互入口。

### 2.3 目录说明

- `app/`：核心代码
- `data/`：示例数据
- `knowledge/`：知识图谱 schema、种子数据、抽取结果、导出结果
- `generated/`：生成的候选算法代码
- `reports/`：验证报告
- `tests/`：自动化测试
- `scripts/run_demo.sh`：统一运行脚本

## 3. 能力知识图谱 schema 和示例

### 3.1 节点类型

当前图谱包含以下主要节点：

- `Capability`
- `TaskType`
- `Algorithm`
- `Metric`
- `Constraint`
- `CodeTemplate`
- `FailurePattern`
- `ValidationResult`
- `Repository`
- `SourceFile`
- `CodeSymbol`
- `Dependency`

### 3.2 关系类型

当前图谱包含以下主要关系：

- `supports`
- `uses`
- `evaluated_by`
- `fits`
- `implemented_by`
- `suggests_fix`
- `verifies`
- `evaluated`
- `contains`
- `defines`
- `depends_on`

### 3.3 图谱文件

- Schema 说明：[`knowledge/schema.md`](knowledge/schema.md)
- 种子图谱：[`knowledge/seed_catalog.json`](knowledge/seed_catalog.json)
- 导出 JSON：[`knowledge/exports/knowledge_graph.json`](knowledge/exports/knowledge_graph.json)
- GraphML：[`knowledge/exports/knowledge_graph.graphml`](knowledge/exports/knowledge_graph.graphml)
- 仓库抽取结果示例：[`knowledge/extracted/ZJUI.json`](knowledge/extracted/ZJUI.json)

### 3.4 当前内置能力

推荐场景：

- `popularity`
- `itemcf`
- `genre_affinity`

分类场景：

- `majority_class`
- `risk_score`

## 4. Agent 工作流设计

本项目不是一次 prompt 直接生成代码，而是显式拆分为多个阶段：

1. `IntentParser`
   将自然语言请求解析为 `CapabilitySpec`，输出任务类型、数据集、约束、目标指标、偏好算法等结构化信息。
2. `KnowledgeRetriever`
   从知识图谱中检索能力节点、算法节点、指标节点、约束节点和失败经验。
3. `BeamSearchPlanner`
   对候选算法进行排序和扩展，兼顾分数与多样性，输出候选集合和搜索轨迹。
4. `Codegen Agent`
   依据插件模板生成安全、可运行的 Python 候选代码。
5. `Verifier Agent`
   先做 AST 安全扫描，再将候选代码送入子进程沙箱执行验证。
6. `Repair Agent`
   在验证失败后执行多轮模板回退修复。
7. `Writer Agent`
   输出 JSON / Markdown 报告，并把验证结果和失败经验回写到知识图谱。

### 4.1 为什么采用多 Agent

- 各阶段职责明确，便于单独替换或增强。
- 搜索、生成、验证、修复可以保留可追踪的中间结果。
- 便于满足题目中的“多智能体协作”“候选方案比较”“失败经验沉淀”等要求。

## 5. 环境配置和运行方法

### 5.1 环境要求

- Python `3.11+`
- Git

### 5.2 依赖安装

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
cp .env.example .env
```

### 5.3 硅基流动配置

在 `.env` 中配置：

```bash
LLM_PROVIDER=siliconflow
OPENAI_API_KEY=你的硅基流动 API Key
OPENAI_BASE_URL=https://api.siliconflow.cn/v1
OPENAI_MODEL=deepseek-ai/DeepSeek-V3
OPENAI_TIMEOUT_SECONDS=60
BEAM_WIDTH=3
MAX_REPAIR_ATTEMPTS=2
SANDBOX_TIMEOUT_SECONDS=20
SANDBOX_MEMORY_LIMIT_MB=512
```

如果没有可用 API Key，系统会自动退回到 `mock` 模式。

### 5.4 一键运行脚本

项目提供统一脚本：

```bash
./scripts/run_demo.sh serve
./scripts/run_demo.sh recommend
./scripts/run_demo.sh churn
./scripts/run_demo.sh test
```

含义如下：

- `serve`：启动 Web/API 服务
- `recommend`：运行推荐场景示例
- `churn`：运行客户流失分类示例
- `test`：运行测试

### 5.5 Web / CLI / API 用法

启动服务：

```bash
./scripts/run_demo.sh serve
```

访问地址：

- Web UI：`http://127.0.0.1:8000`
- FastAPI Docs：`http://127.0.0.1:8000/docs`

CLI 示例：

```bash
./.venv/bin/python -m app.cli run-task \
  --task-name movie_homepage \
  --task-type top_k_recommendation \
  --dataset-name movielens_sample \
  --description "Build a Top-K movie recommendation workflow for a homepage feed. Use implicit feedback, prioritize recall and ndcg, and keep it runnable on local cpu." \
  --top-k 5 \
  --constraints "implicit feedback" "local cpu" \
  --candidate-count 3
```

```bash
./.venv/bin/python -m app.cli run-task \
  --task-name customer_churn \
  --task-type tabular_classification \
  --dataset-name churn_sample \
  --description "Build a customer churn classification workflow. Use local cpu, compare baseline and score-based methods, and optimize f1." \
  --top-k 5 \
  --constraints "local cpu" \
  --candidate-count 2
```

真实仓库知识抽取：

```bash
./.venv/bin/python -m app.cli ingest-repo --repo-path .
```

Git 状态与快照：

```bash
./.venv/bin/python -m app.cli git-status
./.venv/bin/python -m app.cli git-snapshot --message "feat: capability factory update"
```

## 6. 示例数据和测试任务说明

### 6.1 推荐数据

- [`data/movielens_sample/interactions.csv`](data/movielens_sample/interactions.csv)
- [`data/movielens_sample/items.csv`](data/movielens_sample/items.csv)

处理规则：

- 将评分 `>= 4` 视为正反馈
- 使用 `leave-last-one-out` 做验证拆分
- 指标包括 `Recall@K`、`NDCG@K`、`Coverage`

### 6.2 流失分类数据

- [`data/churn_sample/customers.csv`](data/churn_sample/customers.csv)

数据特点：

- 包含 `train/test` 切分
- 目标列为 `churned`
- 指标包括 `Accuracy`、`Precision`、`Recall`、`F1`

### 6.3 示例任务

推荐任务：

- “为电影首页构建 Top-K 推荐能力，使用隐式反馈，优先 recall 和 ndcg，本地 CPU 可运行。”

分类任务：

- “为客户流失预测构建表格分类能力，比较基线与打分法，优化 F1。”

## 7. 生成算法代码示例

生成的候选代码会输出到 `generated/<task_id>/` 目录。

仓库内已提供推荐场景样例：

- [`generated/sample-topk-movielens/itemcf-1.py`](generated/sample-topk-movielens/itemcf-1.py)
- [`generated/sample-topk-movielens/popularity-2.py`](generated/sample-topk-movielens/popularity-2.py)
- [`generated/sample-topk-movielens/genre_affinity-3.py`](generated/sample-topk-movielens/genre_affinity-3.py)

系统当前会自动生成以下接口形态：

推荐算法：

- `fit(train_df)`
- `recommend(user_id, k)`
- `batch_recommend(user_ids, k)`

分类算法：

- `fit(train_df)`
- `predict(features_df)`
- `batch_predict(features_df)`

## 8. 验证结果和报告样例

报告输出在 `reports/tasks/` 下，格式同时包含 JSON 和 Markdown。

仓库内已有样例：

- [`reports/tasks/sample-topk-movielens.json`](reports/tasks/sample-topk-movielens.json)
- [`reports/tasks/sample-topk-movielens.md`](reports/tasks/sample-topk-movielens.md)

报告内容包括：

- 原始请求
- 结构化意图
- 多 Agent 执行轨迹
- 候选算法比较结果
- 指标分数
- 延迟与峰值内存
- 修复历史
- Git 版本上下文
- 知识图谱回写 ID

## 9. 遇到的挑战和解决方案

### 9.1 真实 LLM 输出不稳定

问题：

- 远程 LLM 可能超时、字段缺失、返回格式不稳定。

解决：

- 使用结构化 JSON 输出约束；
- 对解析结果做归一化；
- 远程失败时自动回退到 `mock provider`。

### 9.2 单场景系统扩展性不足

问题：

- 仅支持推荐任务时，无法覆盖题目中“跨场景迁移”的要求。

解决：

- 增加 `tabular_classification` 任务类型；
- 增加 `CustomerChurnDatasetLoader`；
- 增加独立验证 profile；
- 使用统一工作流复用能力理解、搜索、生成、验证、报告、回写模块。

### 9.3 候选方案选择缺少可解释搜索过程

问题：

- 单次 prompt 难以清晰体现“为什么选这个算法”。

解决：

- 引入 `Beam Search`；
- 输出搜索轨迹和候选得分；
- 结合知识图谱检索结果与算法插件元数据做排序。

### 9.4 安全执行风险

问题：

- 直接执行生成代码可能带来系统调用、网络访问或无限运行风险。

解决：

- 先做 AST 安全扫描；
- 再使用子进程沙箱执行；
- 在支持的平台上限制 CPU 时间和地址空间内存。

### 9.5 失败经验不易复用

问题：

- 验证失败如果只写日志，很难在下一次生成时复用。

解决：

- 将失败模式写回知识图谱；
- 在后续检索阶段一并返回失败经验节点。

## 10. 后续可扩展方向

- 将验证指标进一步拆成可单独注册的 `metric plugin`
- 增加文本分类、异常检测等更多任务类型
- 将任务记录和图谱落盘到 SQLite / Neo4j
- 将模板生成升级为“模板 + LLM 补全”的混合代码生成
- 引入 MCTS 或更复杂的搜索策略
- 增加自动性能调优策略，而不只是性能分析
- 增加多用户任务队列与持久化

## 功能完成度对照

### 基础要求

- 完整项目代码库：已完成
- 示例行业数据、文档或模拟业务材料：已完成
- 知识图谱 schema 设计说明：已完成
- 知识抽取结果文件：已完成
- Agent 工作流代码：已完成
- 至少 1 个完整示例：已完成，当前包含 2 个任务类型
- 自动验证脚本：已完成，集成在 `validation.py + sandbox_worker.py + scripts/run_demo.sh`
- 验证报告样例：已完成
- README：已完成
- `requirements.txt`：已完成
- 使用示例和测试用例：已完成

### 进阶要求

- Web / CLI / API：已完成
- 多轮代码修复：已完成
- 多候选算法自动比较：已完成
- 知识图谱可视化：已完成
- 自动生成验证报告：已完成
- 验证结果回写知识库 / 图谱：已完成
- 不同任务类型验证配置：已完成
- 插件式接入算法模板 / 验证 profile：已完成

### 加分项

- 图搜索 / Beam Search：已完成（Beam Search）
- 多智能体协作：已完成（显式多阶段 Agent）
- 真实代码仓库抽取：已完成
- 代码安全检查和沙箱执行：已完成
- 自动分析失败案例并沉淀经验：已完成
- 算法能力版本管理：已完成（Git 状态 + capability version）
- 自然语言解释方案设计依据：已完成
- 跨场景迁移：已完成（推荐 + 表格分类）
- 自动生成接口文档 / 部署配置：已完成（FastAPI docs + Dockerfile）
- 资源消耗分析：已完成

说明：

- `MCTS` 未实现，但题目只要求在图搜索 / Beam Search / MCTS 中任选其一即可，当前已用 `Beam Search` 覆盖。
- 自动“性能优化”目前实现的是“性能与资源分析”，若需要更强的自动调优能力，可在后续迭代中继续补充。

## Git 版本管理说明

项目已经纳入 Git 管理，并内置了运行时版本上下文记录能力：

- 任务报告会记录当前 `branch / head_commit / dirty state`
- 系统会为每次任务生成 `capability_version`
- 可通过 API / CLI 查询当前 Git 状态
- 可通过 CLI 创建快照提交

## 测试

执行：

```bash
./scripts/run_demo.sh test
```

当前状态：

- `19` 个测试全部通过
- 覆盖推荐、分类、插件加载、知识回写、API、仓库抽取、验证与工作流
