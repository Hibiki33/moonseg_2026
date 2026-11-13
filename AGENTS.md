# Workspace and runtime

On this host, `/home/nas_1/wjj/moon` is the source repository for code edits, review, commits and pushes. `/home/disk02/wjj/moonseg/code` is the local execution checkout for training, evaluation and prediction. They are independent Git repositories; changes do not propagate automatically.

- Make source changes in the current workspace. If work was performed in the execution checkout, merge it back before reporting completion.
- After committing, synchronize the execution checkout and verify its commit and tracked files against the source repository. Preserve any divergent edits and avoid replacing code used by active jobs.
- Start training and evaluation from disk02. Keep environments, data, checkpoints, logs and generated run files under `/home/disk02/wjj/moonseg/`, accessed through the existing symlinks.
- Use uv for environment management. Do not commit downloaded weights, datasets, run outputs, local environments or temporary agent files.
- Configure Git author identity as `jiajun <jiajun9264@gmail.com>`. GitHub pushes use the existing SSH configuration for `~/.ssh/id_rsa_wjj`.
- Do not add unittest files or weight checksum calculations. Run necessary checks without committing temporary test code.
- Use only GPUs allocated by the user; preserve `NCCL_P2P_DISABLE=1` for the established GPU pairs. A code change does not itself authorize a new full training run.
