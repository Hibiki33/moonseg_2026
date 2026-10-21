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
which grok && grok --version
grok models >/dev/null 2>&1 && echo "auth ok" || echo "need login"
```

If not authenticated, ask the user to run `grok` once to log in.

## Recommended invocation methods (in order of preference)

### 1. Read-only review / second opinion (safest, prefer this)

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
  --model grok-build \
  --effort high \
  --cwd .
```

## Calling principles

1. **Parent agent owns the final judgment**: Treat Grok’s output as input only. You must verify important conclusions and changes yourself.
2. **Default to read-only**: Use read-only mode unless the user explicitly asks for implementation.
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
- If it times out or hangs, try a narrower task or add a timeout flag if available.
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