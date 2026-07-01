# AgentDojo Benchmark Runner

这是一个基于 [AgentDojo](https://github.com/ethz-spylab/agentdojo) 的代理安全评测项目，用于在多个任务套件上运行 LLM Agent，记录 utility/security 指标，并支持 Spotlighting 防御、OpenRouter 限流、多模型接入、后台运行、断点续跑和结果汇总。

## 目录结构

```text
.
├── main.py                     # CLI 入口，加载 .env 后调用 runner.main
├── runner.py                   # 核心评测流程与命令行参数定义
├── pipeline_builder.py         # 构建 AgentDojo AgentPipeline
├── llm_factory.py              # 创建 OpenAI/Ollama/Anthropic/Google/OpenRouter LLM
├── pipeline_hooks.py           # 工具调用与工具结果 hook
├── defenses.py                 # 自定义防御预处理器
├── rate_limiter.py             # OpenRouter 多 key、RPM、每日额度控制
├── token_usage.py              # token 用量记录
├── aggregate_results.py        # 汇总已有 JSON 结果，不重新跑模型
├── requirements.txt            # Python 依赖
├── scripts/
│   ├── start_background.py     # 跨平台后台启动
│   ├── stop_background.py      # 跨平台停止后台任务
│   ├── start-background.ps1    # Windows PowerShell 后台启动封装
│   └── stop-background.ps1     # Windows PowerShell 停止封装
├── .run/                       # 后台进程 PID 与 stdout/stderr 日志，已被 git 忽略
└── logs_my_agentdojo/          # AgentDojo 结果、token 统计、限流状态，已被 git 忽略
```

## 环境准备

建议使用 Python 3.10+。

### Windows PowerShell

```powershell
cd D:\Python_Program\Final_project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux / Git Bash

```bash
cd /path/to/Final_project
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> 如果运行时提示 `ModuleNotFoundError: No module named 'dotenv'`，说明当前终端使用的 Python 环境还没有安装依赖，请先激活虚拟环境并执行 `pip install -r requirements.txt`。

## 环境变量配置

项目启动时会自动读取根目录下的 `.env` 文件：

```python
load_dotenv(BASE_DIR / ".env", override=True)
```

可以在项目根目录创建 `.env`，按需填写以下变量。不要提交真实密钥。

```env
# OpenAI
OPENAI_API_KEY=sk-...

# Ollama，本地默认地址通常不需要改
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_API_KEY=ollama

# Anthropic
ANTHROPIC_API_KEY=sk-ant-...

# Google Gemini
GOOGLE_API_KEY=...

# OpenRouter，单 key 或多 key 二选一
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_API_KEYS=sk-or-key1,sk-or-key2,sk-or-key3
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# OpenRouter 本地限流与重试
OPENROUTER_RPM=20
OPENROUTER_DAILY_LIMIT=50
OPENROUTER_DAILY_LIMIT_ACTION=wait
OPENROUTER_RETRY_MAX_ATTEMPTS=10
OPENROUTER_RETRY_BASE_SECONDS=60
OPENROUTER_RETRY_MAX_SECONDS=300
OPENROUTER_RETRY_DAILY_LIMIT=50
OPENROUTER_ERROR_LOG_PATH=logs_my_agentdojo/openrouter_error_log.jsonl
```

`OPENROUTER_DAILY_LIMIT_ACTION` 支持：

- `wait`：所有 key 达到每日额度后等待到 24 小时窗口重置。
- `pause`：达到每日额度后中止当前运行，之后用相同命令继续断点续跑。

## 命令行参数

主入口是：

```bash
python main.py [OPTIONS]
```

常用参数来自 `runner.main`：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--model` | `ollama:qwen2.5:7b` | 模型，格式为 `provider:model_name` |
| `--suites` | `workspace banking travel slack` | AgentDojo 套件，可传多个 |
| `--benchmark-version` | `v1.2` | AgentDojo benchmark 版本 |
| `--attack-name` | `important_instructions` | 注入攻击名称 |
| `--run-attack` / `--no-run-attack` | `--run-attack` | 是否运行攻击评测 |
| `--user-tasks` | 全部 user tasks | 只运行指定 user task，可传多个 |
| `--injection-tasks` | 全部 injection tasks | 只运行指定 injection task，可传多个 |
| `--max-injection-tasks` | 不限制 | 只取前 N 个 injection tasks，适合快速 smoke test |
| `--reasoning-effort` | `medium` | OpenAI reasoning 模型的 reasoning effort |
| `--thinking-budget-tokens` | 空 | Anthropic thinking token budget |
| `--logdir` | `./logs_my_agentdojo` | 结果输出目录 |
| `--record-token-usage` / `--no-record-token-usage` | `--record-token-usage` | 是否记录 token 使用 |
| `--force-rerun` / `--no-force-rerun` | `--no-force-rerun` | 是否强制重跑已有结果 |
| `--daily-limit-action` | `wait` | OpenRouter 每日额度耗尽时的行为，`wait` 或 `pause` |
| `--defense` | `spotlighting` | 防御模式，支持 `spotlighting`、`task_shield`、`spotlighting_task_shield` 或 `no_defense` |

安装依赖后也可以查看自动生成的帮助：

```bash
python main.py --help
```

## 启动方式

### 1. 默认前台启动

使用默认配置运行所有套件：

```bash
python main.py
```

默认等价于：

```bash
python main.py --model ollama:qwen2.5:7b --attack-name important_instructions --defense spotlighting
```

Task Shield 单独启用或与 Spotlighting 同时启用：

```bash
python main.py --defense task_shield --suites workspace
python main.py --defense spotlighting_task_shield --suites workspace
```

### 2. Ollama 本地模型启动

先启动 Ollama 并准备模型：

```bash
ollama pull qwen2.5:7b
ollama serve
```

然后运行：

```bash
python main.py --model ollama:qwen2.5:7b
```

只跑一个套件做快速验证：

```bash
python main.py --model ollama:qwen2.5:7b --suites workspace --max-injection-tasks 2
```

### 3. OpenAI 模型启动

`.env` 中配置：

```env
OPENAI_API_KEY=sk-...
```

运行：

```bash
python main.py --model openai:gpt-4o-mini-2024-07-18 --suites workspace
```

使用 reasoning 模型时可以传 reasoning effort：

```bash
python main.py --model openai:o4-mini --reasoning-effort medium --suites workspace
```

### 4. OpenRouter 启动

`.env` 中配置一个或多个 key：

```env
OPENROUTER_API_KEYS=sk-or-key1,sk-or-key2
OPENROUTER_RPM=20
OPENROUTER_DAILY_LIMIT=50
OPENROUTER_DAILY_LIMIT_ACTION=pause
```

运行：

```bash
python main.py --model openrouter:google/gemini-2.0-flash-exp:free --suites workspace --daily-limit-action pause
```

OpenRouter 运行时会写入：

- `logs_my_agentdojo/openrouter_rate_limit_state.json`
- `logs_my_agentdojo/openrouter_error_log.jsonl`

如果每日额度耗尽并选择 `pause`，程序会停止；等额度恢复后使用相同命令再次运行，默认会跳过已经完成的结果文件。

### 5. Anthropic 启动

`.env` 中配置：

```env
ANTHROPIC_API_KEY=sk-ant-...
```

运行：

```bash
python main.py --model anthropic:claude-3-5-sonnet-latest --suites workspace --thinking-budget-tokens 1024
```

### 6. Google Gemini 启动

`.env` 中配置：

```env
GOOGLE_API_KEY=...
```

运行：

```bash
python main.py --model google:gemini-1.5-flash --suites workspace
```

代码里 Google client 每次调用后会 sleep 60 秒，用于降低触发限流的概率。

### 7. 无攻击 baseline 启动

只评估 utility，不运行注入攻击：

```bash
python main.py --no-run-attack --suites workspace
```

### 8. 关闭防御启动

用于和 Spotlighting 防御做对照：

```bash
python main.py --defense no_defense --suites workspace
```

### 9. 指定任务启动

只运行指定 user task：

```bash
python main.py --suites workspace --user-tasks user_task_0
```

指定多个 user task 和 injection task：

```bash
python main.py --suites workspace --user-tasks user_task_0 user_task_1 --injection-tasks injection_task_0 injection_task_1
```

如果不确定 task id，先用较小范围运行一个 suite，程序启动时会打印该 suite 的可用工具与运行进度；也可以在安装 AgentDojo 后通过 Python 读取 suite 的 `user_tasks` / `injection_tasks`。

### 10. 强制重跑

默认 `force_rerun=False`，已有完整结果会被复用。需要覆盖已有结果时：

```bash
python main.py --force-rerun --suites workspace
```

### 11. 自定义日志目录启动

```bash
python main.py --logdir ./logs_experiment_001 --suites workspace
```

## 后台运行

长时间评测建议使用后台启动脚本。

### Windows PowerShell 后台启动

```powershell
.\scripts\start-background.ps1 -Python python -- --model ollama:qwen2.5:7b --suites workspace
```

指定 run dir：

```powershell
.\scripts\start-background.ps1 -Python python -RunDir .run-openrouter -- --model openrouter:google/gemini-2.0-flash-exp:free --suites workspace --daily-limit-action pause
```

### 跨平台 Python 后台启动

```bash
python scripts/start_background.py -- --model ollama:qwen2.5:7b --suites workspace
```

指定 run dir：

```bash
python scripts/start_background.py --run-dir .run-openrouter -- --model openrouter:google/gemini-2.0-flash-exp:free --suites workspace
```

后台启动后会生成：

```text
.run/main.pid
.run/main.out.log
.run/main.err.log
```

如果 PID 仍在运行，重复执行启动脚本会提示 `Already running with PID ...`，不会重复启动。

## 查看后台日志

Windows PowerShell：

```powershell
Get-Content .\.run\main.out.log -Wait
Get-Content .\.run\main.err.log -Wait
```

macOS / Linux：

```bash
tail -f .run/main.out.log
tail -f .run/main.err.log
```

## 停止后台运行

Windows PowerShell：

```powershell
.\scripts\stop-background.ps1
```

指定 run dir：

```powershell
.\scripts\stop-background.ps1 -RunDir .run-openrouter
```

跨平台 Python：

```bash
python scripts/stop_background.py
python scripts/stop_background.py --run-dir .run-openrouter
```

## 断点续跑机制

每次运行前，程序会扫描目标日志目录中已经存在的结果文件：

```text
logs_my_agentdojo/
└── <suite>/
    └── <pipeline_name>/
        └── <suite_name>/
            └── <user_task_id>/
                └── <attack_name>/
                    └── <injection_task_id>.json
```

完整结果会被跳过；不完整或损坏的 JSON 会被改名为：

```text
*.partial-YYYYMMDD-HHMMSS.json.bak
```

因此，普通续跑只需要重新执行相同命令：

```bash
python main.py --model ollama:qwen2.5:7b --suites workspace
```

只有明确想覆盖已有结果时才使用 `--force-rerun`。

## 结果汇总

不重新运行模型，只汇总已有 JSON 日志：

```bash
python aggregate_results.py
```

指定日志目录：

```bash
python aggregate_results.py --logdir logs_my_agentdojo
```

只看某个 suite：

```bash
python aggregate_results.py --suite workspace
```

包含无攻击 baseline：

```bash
python aggregate_results.py --include-baseline
```

输出 JSON：

```bash
python aggregate_results.py --json
```

组合示例：

```bash
python aggregate_results.py --logdir logs_experiment_001 --suite workspace --json
```

## Token 用量记录

默认开启 token 记录：

```bash
python main.py --record-token-usage
```

每个 suite 下会生成：

```text
token_usage.jsonl
token_usage_summary.json
```

关闭 token 记录：

```bash
python main.py --no-record-token-usage
```

## 推荐工作流

1. 先跑一个小 smoke test：

   ```bash
   python main.py --suites workspace --max-injection-tasks 1
   ```

2. 确认模型、密钥、依赖和日志都正常后，后台跑完整实验：

   ```bash
   python scripts/start_background.py -- --model ollama:qwen2.5:7b
   ```

3. 运行中查看日志：

   ```bash
   tail -f .run/main.out.log
   ```

4. 结束后汇总：

   ```bash
   python aggregate_results.py
   ```

5. 做防御对照实验：

   ```bash
   python main.py --defense no_defense --logdir logs_no_defense
   python main.py --defense spotlighting --logdir logs_spotlighting
   python aggregate_results.py --logdir logs_no_defense
   python aggregate_results.py --logdir logs_spotlighting
   ```

## 常见问题

### 1. `ModuleNotFoundError: No module named 'dotenv'`

当前 Python 环境没有安装依赖：

```bash
pip install -r requirements.txt
```

确认 `python` 和 `pip` 来自同一个环境：

```bash
python -m pip --version
python -c "import sys; print(sys.executable)"
```

### 2. OpenRouter 提示没有 key

使用 `openrouter:` 模型前必须配置：

```env
OPENROUTER_API_KEY=sk-or-...
```

或：

```env
OPENROUTER_API_KEYS=sk-or-key1,sk-or-key2
```

### 3. OpenRouter 免费额度耗尽

如果 `--daily-limit-action pause`，程序会退出并保留已完成结果。等额度恢复后重新执行相同命令即可续跑。

如果 `--daily-limit-action wait`，程序会等待到 24 小时窗口重置后继续运行。

### 4. 后台任务无法重复启动

如果 `.run/main.pid` 对应进程仍在运行，启动脚本会直接提示已有进程。先停止：

```bash
python scripts/stop_background.py
```

如果进程已不存在，停止脚本会清理陈旧 PID 文件。

### 5. 想看 CLI 支持的真实参数

安装依赖后运行：

```bash
python main.py --help
```

### 6. PowerShell 激活虚拟环境被执行策略拦截

可以临时允许当前进程执行脚本：

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\.venv\Scripts\Activate.ps1
```

## 注意事项

- `.env`、`.run/`、`logs_my_agentdojo/` 已在 `.gitignore` 中忽略，不要提交密钥或大体积运行日志。
- 默认模型是本地 Ollama 的 `qwen2.5:7b`，需要确保 Ollama 服务已启动。
- `--force-rerun` 会重新生成结果，普通续跑不要加这个参数。
- 不同 provider 的模型名必须使用 `provider:model_name` 格式，例如 `openai:gpt-4o-mini-2024-07-18`、`ollama:qwen2.5:7b`、`openrouter:google/gemini-2.0-flash-exp:free`。
