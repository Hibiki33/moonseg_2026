---
name: grok-subagent
description: |
  Use the locally installed Grok Build CLI as a controlled external sub-agent.
  Suitable for second opinions, independent reviews, lightweight implementation,
  search, or verification. Prefer the project's Grok usage docs if they exist.
---

# Grok Subagent Skill

You can use the locally installed **Grok Build** CLI as an external sub-agent.

## When to use

- You need an independent second opinion (review, critique, risk finding)
- You want to offload implementation or exploration to a different model family
- You need Grok’s live search (X / Web) capabilities
- You want to keep the parent agent’s context clean by outsourcing noisy work

**Do not use** when:
- The task is simple enough for you to handle directly
- The work requires heavy, tightly controlled multi-file edits
- The user explicitly wants only the current model / Codex

## Prerequisite check (recommended before calling)

```bash
command -v grok
grok --version
grok models
```

Inspect the actual error before concluding that authentication is missing. A successful model listing does not verify every downstream tool or network endpoint. If authentication is explicitly missing, ask the user to run `grok` once to log in.

## Recommended invocation methods (in order of preference)

### 1. Read-only review / second opinion (safest, prefer this)

For unattended research that needs shell-based network reads, see the observed `plan`-mode limitations in the dated experience notes below.

```bash
grok -p "Your task description. Output only conclusions, evidence (file + line numbers), and suggestions. Do not modify any files." \
  --cwd . \
  --permission-mode plan \
  --output-format plain
```

### 2. Normal delegation (edits allowed)

```bash
grok -p "Concrete task. After finishing, provide a summary and the main changes." \
  --cwd . \
  --output-format plain
```

### 3. Specify model and effort

```bash
grok -p "Task" \
  --model grok \
  --effort high \
  --cwd .
```

Choose a model identifier from `grok models`; do not assume the CLI product name is a valid model identifier.

## Calling principles

1. **Parent agent owns the final judgment**: Treat Grok’s output as input only. You must verify important conclusions and changes yourself.
2. **Default to read-only tasks**: Keep the delegated task read-only unless implementation is authorized. Select an execution mode that supports the required tools; a prompt saying “read-only” is not a filesystem permission boundary.
3. **Give clear boundaries**: In the prompt, specify constraints such as “do not modify files”, “return structured conclusions only”, “include file and line number evidence”, etc.
4. **Summarize the result**: Compress Grok’s output into key points before reporting to the user. Do not forward the raw output verbatim.
5. **Prefer project docs**: If the project contains `docs/grok-usage.md` (or similar), follow its conventions and examples.

## Example prompt templates

**Review:**
```
As an independent reviewer, examine the uncommitted changes in the current workspace.
List the 3–5 most important issues. For each issue include:
- Severity
- File path and line number
- Description of the problem
- Suggestion
Do not modify any files.
```

**Implementation:**
```
Implement the following requirement in the current project: …
When finished, provide:
1. List of main files changed
2. Brief implementation summary
3. Anything the user should be aware of
```

## Failure handling

- If the `grok` command fails, check authentication and PATH first.
- While a task is running, wait and inspect its session/tool progress before diagnosing a hang. Use timeouts for individual network requests; do not stop and restart a healthy research task just because its output is temporarily quiet.
- Exit code zero is not proof of task completion. Check for the requested final deliverable; resume incomplete sessions instead of marking them successful.
- Report errors honestly to the user. Do not pretend the call succeeded.

## Output requirements

After calling Grok, report to the user using this structure:

- What you asked Grok to do
- Grok’s main conclusions (summarized)
- Your own verification (agree / partially agree / disagree)
- Recommended next steps

## Effort levels

Grok Build supports the following effort levels via `--effort`:

| Level  | Flag              | When to use                                                  |
| ------ | ----------------- | ------------------------------------------------------------ |
| low    | `--effort low`    | Quick checks, simple lookups, low-cost exploration           |
| medium | `--effort medium` | Default balanced choice for most reviews and moderate tasks  |
| high   | `--effort high`   | Complex reviews, multi-file analysis, higher-quality reasoning |
| xhigh  | `--effort xhigh`  | Hardest tasks, deep investigation, maximum reasoning (higher token use) |

### Example with effort

```bash
# High effort review
grok -p "Review the auth changes for security issues. Provide file:line evidence." \
  --effort high \
  --permission-mode plan \
  --cwd .

# Lower cost exploration
grok -p "Summarize the architecture of this project." \
  --effort medium \
  --cwd .
```

## 本仓库实测经验（2026-10-21）

以下来自五组月面数据调研，实测 CLI 为 `grok 1.0.5 (5115b46bc9)`。当时 `grok models` 列出 `grok-4.6`、`grok-4.5` 和默认别名 `grok`；本次使用 `grok --effort high`。这些是当前环境记录，不保证其他版本或账号行为相同。

### 1. 退出码为 0，也可能根本没有最终报告

第一轮五个进程均返回 0，但输出只有数百字节的“我继续核对……”等进度说明。导出会话后发现末尾仍是工具调用，没有完整结论。最初调度器按 `returncode == 0` 标成完成，这是错误的完成判断，后来已修正。

任务状态与进程状态分开记录：进程退出只记录 `exit_code`；实际工作依次记录 `running`、必要时的 `needs_followup`、`report_received`、`reviewed`。完成条件应检查最终报告是否覆盖所要求的字段、证据和未确认项。文件大小或输出停止增长都只能辅助观察，不能替代内容检查。

`--output-format plain` 本次会把进度说明和最终正文接在同一份输出里；原始输出应留档，整理进仓库时再去掉过程文字。不要把最后一句“接下来我会……”当成最终答复。

### 2. 无交互调用的模式和工具白名单需要实际核对

本次 `--permission-mode plan` 调研在 shell 网络读取等工具调用处结束，未形成最终报告；不能因此推断外网不可达。一次改为 `--tools WebSearch,WebFetch` 仍未完成，导出记录还出现 `IntegrationSearch`、`UseTool` 等工具路由。不能假定搜索界面显示的名称就是足够完整的 CLI 工具白名单。

在本会话已授权标准命令执行的前提下，恢复原会话并使用 `--no-plan --permission-mode bypassPermissions --always-approve`，最终取得了五份完整报告。**这是本次成功使用的选项组合，不是已经隔离验证过的单一故障原因。** 不要将该组合不加判断地套到其他环境：它会自动放行工具，提示词中的“只读”不会限制文件写入权限，任务范围仍以用户授权为准。

本次不需要改变主代理的沙箱设置，也没有因某个网站的 TLS 错误或 403 就认定整机无法联网。访问失败应记下具体域名、接口与错误；元数据可读、文件下载失败是两种不同状态。

### 3. 并发低一点，等结果，不抢着替它做

用户要求低并发时，本次采用最多两个 Grok 任务，并加 `--no-subagents` 防止内部再扩出一批 agent。这个参数限制子 agent，不等于限制每次搜索包含的查询数；有需要时也应在提示词里约束网络请求并发。

运行中的 Grok 有时几分钟不打印新进度，但会话中仍在做归档 Range 读取或目录统计。此时应等待、查看会话进展；不要因为暂时没正文就终止、重复启动或加开任务。主代理最初在报告未返回时自行铺开了大量同主题搜索，不符合用户“让 Grok 调研、没返回就等”的要求。后续改为等待完整报告，再针对关键结论复核。

单个网络请求可以设置 `curl --connect-timeout 20 --max-time 60` 或 Python 请求超时，避免一条链接无限阻塞；这与给整个研究任务设很短的总时限不同。不要把 `--max-turns 80` 当成 80 秒，它限制 agent 回合数。

### 4. 恢复明确的会话，保存提示词和原始输出

先用 `grok sessions list` 找到对应 session，再用 `grok export` 检查是否真的结束。并发时使用显式 `--resume SESSION_ID`；`--continue` 取当前目录最近的会话，容易选错专题。继续调研不需要 `--restore-code`，该选项会恢复代码快照。

下面是本次使用方式的模板。先准备好提示词文件，并将 session ID 替换为实际值；新任务不传 `--resume`。执行模式适用范围见上节。

```bash
grok_task_root=/home/disk02/wjj/moonseg/research/grok
grok_session_id='替换为实际会话ID'
mkdir -p "$grok_task_root"

grok export "$grok_session_id" "$grok_task_root/topic.transcript.md"

grok --resume "$grok_session_id" \
  --prompt-file "$grok_task_root/topic.prompt.txt" \
  --cwd /home/nas_1/wjj/moon \
  --model grok --effort high \
  --no-plan --permission-mode bypassPermissions --always-approve \
  --no-subagents --max-turns 80 --output-format plain \
  > "$grok_task_root/topic.output.md" \
  2> "$grok_task_root/topic.stderr.log"
```

长提示词放文件，避免多层 shell 引号和特殊字符问题。提示词明确最终交付字段、来源要求、可执行范围，以及“必须返回完整报告，证据不足写未知”。明确哪些是允许保存的临时元数据、哪些目录不能修改，比笼统写“不要修改任何文件”更准确。

本次提示词、输出、错误日志、会话 ID 和元数据都放在 disk02 的研究目录；仓库只放整理后的文档。记录模型别名和 CLI 版本即可，不读取或复制本机凭证。

### 5. 报告返回后仍要核对计数和结论强度

这次有三个具体例子值得保留：

- POLAR-Sim 的 3,985 个检测目录文本包含 12 个 `classes.txt`；排除类别说明后才是 3,973 个候选逐图标签，还没有去副本。文件数不自动等于样本数。
- DFKI 的 736 个源编号及跨 split 交集可以从归档目录重算，但未查看像素和原始来源，不能直接证明只有 736 张独立照片或已经证实泄漏。整理版将其改成待核实的重复风险。
- LuMon 数据卡与评测代码对掩码用途的描述不一致，应保留冲突，不能为了给出确定答案擅自选一边。

优先复核 Grok 已取得的原始 API、归档目录和小段标注输出，避免把整套搜索重做一遍。下载页可打开、HEAD/Range 成功、整包已取得和内容已验证应分别记录。本次只读取元数据及少量片段，没有把它写成数据已批量下载。

原始证据目录：`/home/disk02/wjj/moonseg/research/external_datasets/2026-10-21/`；其中 `status.json` 保存第一轮未形成报告的状态及最终会话 ID。整理后的调查结论见 [外部数据调研](docs/external_datasets/README.md)。
