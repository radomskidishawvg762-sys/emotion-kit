# emotion-kit

情绪编排中间层。它坐在任意 OpenAI 兼容对话模型前面，把每一轮用户消息走完一条固定流水线：识别策略标签、用置信度防抖维持粘滞状态、选出该状态的策略包、按状态权重召回记忆，再带上改写后的系统提示、思考深度和工具策略去调用主模型。

**这是行为策略模拟，不是主观情绪测量。** `calm` / `anxious` / `frustrated` / `excited` 只决定回复怎么组织、思考几步、允许哪些工具。库不会判断用户“真实感受到了什么”，提示词里也要求模型不要这么说。

This is behavioral strategy simulation, not a claim about subjective emotion. The four labels select phrasing, deliberation depth, and tool policy.

## 状态

v1 固定四个标签，每个都有完整策略包（`config/orchestration.yaml`）：

| 标签 | 中文 | 思考深度 | 工具策略 |
| --- | --- | --- | --- |
| `calm` | 平静 | 1 | allow: search, calculator, calendar, memory_lookup；max 4 |
| `anxious` | 焦虑 | 3 | allow: search, memory_lookup, calculator；deny: send_message, code_exec, delete；max 2 |
| `frustrated` | 受挫 | 2 | allow: search, memory_lookup；deny: send_message, delete, code_exec；max 2 |
| `excited` | 兴奋 | 2 | allow: search, calculator, calendar, memory_lookup, draft；deny: delete；max 3 |

思考深度是主模型调用次数：深度 1 只生成最终回复；深度 N 先做 N−1 轮内部草稿，再生成最终回复。中间件不执行工具。`ToolPolicy.allows(name)` 留给宿主拦截，同一份允许/拒绝列表和 `max_tool_calls` 会写进系统提示。

## 安装

Python 3.11+。

```bash
uv sync
```

或：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

运行时依赖只有 PyYAML。测试依赖 pytest（`uv sync` 会装 dev extra；pip 用上面的 `.[dev]`）。

## 环境变量

| 变量 | 作用 | 默认 |
| --- | --- | --- |
| `OPENAI_API_KEY` | 有密钥就走真实接口；没有就走 Mock，流水线步骤相同 | 空 |
| `OPENAI_BASE_URL` | OpenAI 兼容地址，**需要包含 `/v1`** | `https://api.openai.com/v1` |
| `OPENAI_MODEL` | 主对话模型 | `gpt-4o-mini` |
| `EMOTION_KIT_MAIN_MODEL` | 主对话模型，优先于 `OPENAI_MODEL` | 同上 |
| `EMOTION_KIT_DETECT_MODEL` | 分类模型，可以和主模型不同 | 主模型 |
| `EMOTION_KIT_MODE` | `rule`、`llm` 或 `auto` | 配置文件里的 `auto` |
| `EMOTION_KIT_CONFIG_DIR` | 配置目录 | 见下方查找顺序 |

真实接口示例：

```bash
export OPENAI_API_KEY=sk-...
export OPENAI_BASE_URL=https://api.openai.com/v1
export EMOTION_KIT_MAIN_MODEL=gpt-4o-mini
export EMOTION_KIT_DETECT_MODEL=gpt-4o-mini
uv run python examples/cli_demo.py
uv run python examples/cli_demo.py --interactive
```

其它兼容服务同样设置 `OPENAI_BASE_URL` 和 `OPENAI_API_KEY`。检测和主对话可以指向不同模型名，客户端是同一个。

不设置密钥时，Mock 适配器离线跑完检测、防抖、策略、记忆和多轮思考：

```bash
uv run python examples/cli_demo.py
# 或
python examples/cli_demo.py --mock
```

## 命令行

`examples/cli_demo.py` 默认跑 5 轮脚本对话后退出，不需要输入：

1. 焦虑表述 → 高置信度，立即切到 `anxious`，召回发布会清单
2. 没有关键词的短句 → 置信度不够，状态保持 `anxious`
3. 失败 / 受挫 → 切到 `frustrated`，召回上次环境变量失败
4. 兴奋表述 → 切到 `excited`
5. 平静收尾 → 切到 `calm`

```bash
uv run python examples/cli_demo.py --mode rule
uv run python examples/cli_demo.py --mode llm --mock
uv run python examples/cli_demo.py --verbose
uv run python examples/cli_demo.py --config ./config --interactive
```

`--mock` 在已经配置密钥时仍强制离线。`--verbose` 打印内部草稿和最终系统提示。

## 测试

不访问网络，也不需要密钥。

```bash
uv run pytest
# 或已激活虚拟环境时
pytest
```

覆盖检测器（规则 / LLM / auto）、粘滞状态防抖、四份策略包与 JSON 覆盖加载、记忆加权召回、流水线（含脚本对话和思考深度调用次数）、Mock 与 OpenAI 兼容客户端的请求组装。

## 在代码里用

```python
from emotion_kit import build_orchestrator

orchestrator = build_orchestrator(force_mock=True, mode="rule")
result = orchestrator.turn("我有点焦虑，明天发布会来不及了，怎么办？")

result.detection.emotion      # anxious
result.detection.confidence
result.state.reason           # immediate_high_confidence / pending_debounce / ...
result.strategy.thinking_depth
result.strategy.tool_policy.allows("send_message")
result.memories               # 按当前状态加权的 top-k
result.reply
```

换配置或只改一个字段：

```python
orchestrator = build_orchestrator(
    config_dir="config",
    force_mock=True,
    overrides={"states": {"calm": {"thinking_depth": 2}}},
)
```

有密钥时去掉 `force_mock=True` 即可走真实模型。检测客户端和主客户端可以分开传：`build_orchestrator(client=..., detect_client=...)`。

## 架构

```text
user message
    -> detector (rule | llm | auto)
    -> sticky state (confidence debounce)
    -> strategy pack for the active label
    -> weighted memory recall
    -> main model x thinking_depth
         system = base + prompt delta + tool policy + memories + deliberation notes
```

| 模块 | 职责 |
| --- | --- |
| `emotion_kit/detector.py` | 规则打分，或让分类模型返回 JSON |
| `emotion_kit/state.py` | 粘滞状态。低置信度忽略；高置信度立即切换；中等置信度要连续出现 |
| `emotion_kit/strategy.py` | 读取 YAML/JSON 策略包，支持目录和字段覆盖 |
| `emotion_kit/memory.py` | 条目含 id、text、tags、metadata、可选 emotion affinity；按状态权重打分 |
| `emotion_kit/adapters.py` | `OpenAICompatibleClient` 与离线 `MockClient` |
| `emotion_kit/pipeline.py` | 把上面五步串起来 |
| `examples/cli_demo.py` | 脚本 / 交互演示 |

### 检测模式

- `rule`：只做关键词和正则。中英模式都在 `config/rules.yaml`。
- `llm`：向检测模型要一个 JSON 对象：`emotion`、`confidence`、`evidence`。
- `auto`：规则结果足够清晰（有证据且置信度 ≥ `rule_accept_confidence`）就直接用。否则有客户端时再问分类模型；调用失败则退回规则结果。没有客户端时 `auto` 等于 `rule`。

规则置信度是 `best / (best + second + 0.08)`。单一类别接近 0.93，两类打架会掉到 0.5 附近，从而触发防抖而不是每轮来回切。

### 防抖

默认值在 `orchestration.yaml` 的 `debounce`：

- 低于 `min_confidence`（0.45）：忽略，不打断当前连续计数
- 达到 `immediate_confidence`（0.90）：立刻切换
- 介于 `switch_confidence`（0.62）和立即阈值之间：同一新标签要连续出现 `consecutive_turns`（2）次才切换

### 记忆

`score = tag_overlap * w_tag + affinity * w_affinity + recency * w_recency`。权重按状态分开，不必加总为 1；随包的表加总为 1，分量也都归一化到 0–1。没有 affinity 的条目在该项上记 0。召回的 top-k 写进系统提示，只作为背景。

### 配置查找顺序

1. `build_orchestrator(config_dir=...)` 或 CLI `--config`
2. `EMOTION_KIT_CONFIG_DIR`
3. 当前目录下的 `config/`
4. 仓库里的 `config/`
5. 安装包内的 `emotion_kit/data/`（与 `config/` 内容一致，给非可编辑安装用）

目录里要有 `orchestration.yaml`、`rules.yaml`、`memories.yaml`。同名 `.json` 也可以，`.yaml` 优先。

## 布局

```text
emotion_kit/
  detector.py
  state.py
  strategy.py
  memory.py
  adapters.py
  pipeline.py
  data/                  # 与 config/ 同步的打包副本
config/
  orchestration.yaml
  rules.yaml
  memories.yaml
examples/cli_demo.py
tests/
```

## 免责声明

本库把消息路由到预设的回复策略，用来演示“检测 → 粘滞状态 → 策略 → 记忆 → 主模型”这条中间层。标签名称借用了情绪词，但输出不是情绪识别、心理评估或医疗判断。不要用它声称知道用户的主观感受。
