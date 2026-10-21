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

Use a model identifier returned by `grok models`. Inspect errors before diagnosing an authentication or network problem; a failed endpoint does not imply that all network access is unavailable. If authentication is explicitly missing, ask the user to run `grok` once to log in.

## Recommended invocation methods (in order of preference)

### 1. Read-only review / second opinion

```bash
grok -p "Your task description. Return a complete report with conclusions, evidence, and suggestions. Do not modify project files." \
  --cwd . \
  --permission-mode plan \
  --no-subagents \
  --output-format plain
```

Plan mode may stop at a tool call during unattended execution. Check the session before treating the task as complete or diagnosing a tool failure.

### 2. Normal delegation (edits allowed)

```bash
grok -p "Concrete task. After finishing, provide a summary and the main changes." \
  --cwd . \
  --no-subagents \
  --output-format plain
```

For unattended tasks whose tool execution is already authorized, `--no-plan --permission-mode bypassPermissions --always-approve` allows tools to run without interaction. These flags automatically approve tool execution; a read-only prompt does not enforce filesystem permissions.

For long prompts, use `--prompt-file task.txt` instead of embedding the text in a shell command. Save stdout and stderr separately, along with the session ID, so an incomplete task can be inspected and resumed.

### 3. Specify model and effort

```bash
grok -p "Task" \
  --model grok \
  --effort high \
  --cwd .
```

Replace `grok` if your model listing uses a different identifier. The CLI product name is not necessarily a model identifier. `--max-turns` limits agent turns, not elapsed time.

## Calling principles

1. **Parent agent owns the final judgment**: Verify important claims against primary sources or tool results. Keep verified facts, inferences, and unresolved conflicts distinct.
2. **Default to read-only tasks**: Keep research and review read-only unless implementation is authorized. Specify whether temporary research files may be saved and where.
3. **Give clear boundaries**: State the scope, required deliverable, evidence requirements, and allowed actions. Ask for a complete report, with unknowns identified explicitly.
4. **Keep concurrency bounded**: Follow the user’s limit; otherwise start with one or two tasks. Use `--no-subagents` to prevent nested delegation. Wait for results before checking conclusions; do not duplicate the delegated investigation while it is still running.
5. **Summarize the result**: Preserve the raw output, then extract the conclusions for the user. Plain output may contain progress messages before the final report.
6. **Prefer project docs**: If the project contains `docs/grok-usage.md` (or similar), follow its conventions and examples.

## Example prompt templates

**Review:**

```
As an independent reviewer, examine the uncommitted changes in the current workspace.
List the 3–5 most important issues. For each issue include:
- Severity
- File path and line number
- Description of the problem
- Suggestion
Do not modify project files. Return the completed review, not a plan.
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

- **Command failure**: Check the error, executable path, authentication, and options supported by `grok --help`. Report the specific failure.
- **Exit without a result**: Exit code zero does not prove completion. Check that the requested final deliverable is present; progress messages and pending tool calls are insufficient.
- **Tool interruption**: Export the session and inspect the last tool call. Check the permission mode and any tool allowlist; visible tool names may rely on additional discovery or routing tools.
- **Quiet execution**: Wait and inspect session progress before diagnosing a hang. Apply timeouts to individual network requests rather than restarting a healthy task because output is temporarily quiet.
- **Incomplete task**: Resume the original session by its explicit ID. With concurrent tasks, `--continue` may select the wrong session. Do not use `--restore-code` merely to continue a conversation.

```bash
grok sessions list
grok export SESSION_ID transcript.md
grok --resume SESSION_ID \
  -p "Continue the investigation and return the complete report, including unresolved items." \
  --cwd . \
  --no-subagents \
  --output-format plain
```

Use the execution mode appropriate to the authorized task when resuming. Keep process exit status separate from deliverable completion and review status. Report unresolved failures honestly.

## Output requirements

Before reporting completion, check that the result answers the task and that its key claims match the evidence. Reuse the sources already collected, verify quantities and units, and describe completed actions accurately.

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
