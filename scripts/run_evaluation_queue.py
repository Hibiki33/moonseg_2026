#!/usr/bin/env python3
"""Durable scene-5 evaluation queue for the 2026-10-23 experiments.

Launch under nohup/setsid using .venv/bin/python; --dry-run never starts a process,
queries/allocates GPUs, or writes files. Restarting resumes pending work, adopts
live evaluator PIDs, and never retries failed/interrupted evaluations. A queue
flock prevents concurrent queue parents; GPU flocks coordinate this queue only.
GPU occupancy queries cannot reserve a device against unrelated launchers.
"""
from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import io
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]
TERMINAL = {"completed", "failed", "alias", "blocked", "cancelled"}


class TrainingBlocked(Exception):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def absolute(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def python_path(value):
    # Preserve bin/python itself so Python still recognizes its virtualenv.
    path = absolute(value)
    return path.parent.resolve() / path.name


def process_info(pid):
    """Linux process start ticks prevent confusing a reused PID with a dependency."""
    try:
        fields = Path(f"/proc/{int(pid)}/stat").read_text().rsplit(")", 1)[1].split()
    except FileNotFoundError:
        return None
    return {"start_ticks": int(fields[19]), "state": fields[0]}


def process_alive(pid, start_ticks):
    info = process_info(pid)
    return bool(info and info["start_ticks"] == int(start_ticks) and info["state"] not in {"Z", "X"})


def dependency(run, config):
    run_dir = ROOT / "runs" / run["name"]
    dep = run["dependency"]
    if dep["kind"] == "process_exit" and process_alive(dep["pid"], dep["start_ticks"]):
        return False, "training process is still alive"
    if dep["kind"] == "marker":
        marker = run_dir / "queue_completed.json"
        if not marker.exists():
            status_path = absolute(dep.get("status_path", "runs/queue_2026-10-23.status.json"))
            if status_path.exists():
                for lane in read_json(status_path).get("lanes", {}).values():
                    for job in lane.get("jobs", []):
                        if job.get("name") == run["name"] and (
                                job.get("status") == "failed" or lane.get("status") == "failed"):
                            raise TrainingBlocked(f"Training lane failed: {lane.get('error', job.get('error'))}")
            return False, "waiting for queue_completed.json"
        value = read_json(marker)
        if value.get("status") != "completed" or value.get("exit_code") != 0:
            return False, "training marker is not successful"
        if value.get("expected_epochs") != config["expected_epochs"]:
            raise ValueError(f"Unexpected epoch count in {marker}")
    history = run_dir / "history.jsonl"
    if not history.exists():
        if dep["kind"] == "process_exit":
            raise TrainingBlocked("Training process exited without history")
        return False, "waiting for training history"
    lines = [line for line in history.read_text().splitlines() if line.strip()]
    if not lines:
        if dep["kind"] == "process_exit":
            raise TrainingBlocked("Training process exited with empty history")
        return False, "waiting for training history"
    final = json.loads(lines[-1])
    if final.get("epoch") != config["expected_epochs"]:
        if dep["kind"] == "process_exit":
            raise TrainingBlocked(f"Training process exited at epoch {final.get('epoch')}")
        return False, f"history ends at epoch {final.get('epoch')}"
    if final.get("evaluated_samples") != config["expected_samples"] or final.get("eval_stride") != 1:
        raise ValueError(f"Final history is not full scene-5 validation: {history}")
    if [json.loads(line).get("epoch") for line in lines] != list(range(1, config["expected_epochs"] + 1)):
        raise ValueError(f"History must contain exactly epochs 1..25: {history}")
    return True, "training dependency completed"


def file_identity(path):
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "inode": stat.st_ino}


def checkpoint_metadata(path):
    # Local trusted training artifacts; mmap avoids copying the full state into RAM.
    # Do not import moonseg models or initialize CUDA in the queue parent.
    import torch

    before = file_identity(path)
    try:
        checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    except RuntimeError as error:
        if "mmap can only be used" not in str(error):
            raise
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    epoch, step = checkpoint.get("epoch"), checkpoint.get("global_step")
    if not isinstance(epoch, int) or epoch < 0 or not isinstance(step, int) or step < 1:
        raise ValueError(f"Missing/invalid epoch or global_step in {path}")
    if not isinstance(checkpoint.get("model"), dict):
        raise ValueError(f"No model weights in {path}")
    result = {"epoch": epoch + 1, "global_step": step, "file_identity": before}
    del checkpoint
    if file_identity(path) != before:
        raise ValueError(f"Checkpoint changed while reading metadata: {path}")
    return result


def gpu_snapshot():
    """Fail closed: both inventory and compute-process queries must succeed."""
    def query(arguments):
        output = subprocess.run(["nvidia-smi", *arguments, "--format=csv,noheader,nounits"],
                                check=True, capture_output=True, text=True, timeout=15).stdout
        return list(csv.reader(io.StringIO(output), skipinitialspace=True))

    inventory = query(["--query-gpu=index,uuid"])
    devices = {}
    for row in inventory:
        if len(row) != 2 or not row[1].strip().startswith("GPU-"):
            raise ValueError(f"Invalid GPU inventory row: {row}")
        devices[int(row[0])] = {"uuid": row[1].strip(), "pids": []}
    by_uuid = {device["uuid"]: device for device in devices.values()}
    for row in query(["--query-compute-apps=gpu_uuid,pid"]):
        if len(row) != 2 or row[0].strip() not in by_uuid:
            raise ValueError(f"Invalid GPU process row: {row}")
        by_uuid[row[0].strip()]["pids"].append(int(row[1]))
    return devices


def score_result(path, expected):
    value = read_json(path)
    if value.get("evaluated_samples") != expected or set(value.get("scenes", {})) != {"5"}:
        raise ValueError("Evaluation must cover all 1599 scene-5 samples")
    if value.get("weights") != "model":
        raise ValueError("Evaluation did not use model weights")
    for key in ("mIoU", "mAcc"):
        metric = value.get(key)
        if not isinstance(metric, (float, int)) or not math.isfinite(metric) or not 0 <= metric <= 1:
            raise ValueError(f"Invalid {key}")
    if len(value.get("class_IoU", {})) != 5:
        raise ValueError("Missing class IoUs")
    for metric in value["class_IoU"].values():
        if metric is not None and (not isinstance(metric, (float, int)) or not math.isfinite(metric)
                                   or not 0 <= metric <= 1):
            raise ValueError("Invalid class IoU")
    return value


def validate_config(config):
    if config["gpus"] != [2, 4] or config["max_workers"] not in (1, 2):
        raise ValueError("This queue permits only physical GPUs 2/4 and at most two workers")
    if config["expected_epochs"] != 25 or config["expected_samples"] != 1599:
        raise ValueError("This queue requires 25 epochs and full 1599-sample validation")
    if config["checkpoints"] != ["best_miou.pt", "best_loss.pt"]:
        raise ValueError("Only best_miou.pt and best_loss.pt are supported")
    expected = {"single": {"sliding_window": False, "scales": [1.0], "flip": False},
                "narrowflip": {"sliding_window": False, "scales": [0.9, 1.0, 1.1], "flip": True}}
    if config["profiles"] != expected:
        raise ValueError("Unexpected evaluation inference profiles")
    names = [run["name"] for run in config["runs"]]
    if len(set(names)) != len(names) or any(Path(name).name != name for name in names):
        raise ValueError("Run names must be unique directory names")
    validate_cancellations(config, names)
    if not absolute(config["output_dir"]).resolve().is_relative_to((ROOT / "runs").resolve()):
        raise ValueError("Evaluation output must be under external runs storage")
    if not absolute(config.get("gpu_lock_dir", config["output_dir"])).resolve().is_relative_to((ROOT / "runs").resolve()):
        raise ValueError("GPU lock directory must be under external runs storage")
    for run in config["runs"]:
        if run["dependency"]["kind"] not in {"history", "process_exit", "marker"}:
            raise ValueError("Unknown training dependency")


def job_key(run, checkpoint, profile):
    return f"{run}/{Path(checkpoint).stem}/{profile}"


def validate_cancellations(config, names):
    cancelled = config.get("cancelled_runs", [])
    if len(cancelled) != len(set(cancelled)) or not set(cancelled) <= set(names):
        raise ValueError("cancelled_runs must contain unique planned run names")


def apply_queue_policy(state, config, order):
    """Apply an explicitly migrated plan without losing results or live workers."""
    cancelled = set(config.get("cancelled_runs", []))
    for job in state["jobs"].values():
        if job["run"] in cancelled and job["status"] not in {"completed", "alias", "cancelled"}:
            if job["status"] in {"running", "launching"}:
                raise ValueError("Cannot cancel a live evaluator through queue configuration")
            if "error" in job:
                job["previous_error"] = job.pop("error")
            job.update(previous_status=job["status"], status="cancelled", cancelled_at=now(),
                       cancellation_reason=config.get("cancellation_reason", "Cancelled by explicit plan"))
    if set(order) != set(state["jobs"]) or len(order) != len(state["jobs"]):
        raise ValueError("Queue order must preserve every existing evaluation job")
    state["jobs"] = {key: state["jobs"][key] for key in order}


class Queue:
    def __init__(self, config):
        self.config = config
        self.output = absolute(config["output_dir"]).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.lock = (self.output / "queue.lock").open("a+")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another evaluation queue parent holds queue.lock") from None
        path = self.output / "status.json"
        self.state = read_json(path) if path.exists() else {
            "version": 1, "config": config, "created_at": now(), "jobs": {}, "runs": {}}
        if self.state.get("config") != config:
            raise ValueError("Queue configuration differs from saved status; use a new output_dir")
        self.children, self.gpu_locks = {}, {}
        self.stopping = False
        self.state.update(parent_pid=os.getpid(), parent_start_ticks=process_info(os.getpid())["start_ticks"],
                          status="running", started_at=now())
        for run in config["runs"]:
            for checkpoint in config["checkpoints"]:
                for profile in config["profiles"]:
                    key = job_key(run["name"], checkpoint, profile)
                    self.state["jobs"].setdefault(key, {"run": run["name"], "checkpoint": checkpoint,
                        "profile": profile, "weights": "model", "status": "pending"})
        # A crash in the launch transaction is deliberately not retried.
        for job in self.state["jobs"].values():
            if job["status"] == "launching":
                self.fail(job, "Parent interrupted during launch; manual inspection required")
        apply_queue_policy(self.state, config, [job_key(run["name"], checkpoint, profile)
                           for run in config["runs"] for checkpoint in config["checkpoints"]
                           for profile in config["profiles"]])
        self.save()

    def save(self):
        self.state["updated_at"] = now()
        atomic_json(self.output / "status.json", self.state)
        rows = []
        for key, job in self.state["jobs"].items():
            result = job.get("result", {})
            rows.append({"job": key, **{field: job.get(field) for field in (
                "run", "checkpoint", "epoch", "global_step", "weights", "profile", "status",
                "alias_of", "elapsed_seconds", "score_path", "error", "cancellation_reason")},
                "mIoU": result.get("mIoU"), "mAcc": result.get("mAcc"),
                "class_IoU": result.get("class_IoU"), "evaluated_samples": result.get("evaluated_samples")})
        atomic_json(self.output / "aggregate.json", {"updated_at": now(), "status": self.state["status"],
                                                    "evaluations": rows})

    def fail(self, job, error):
        job.update(status="failed", error=str(error), finished_at=now())
        print(f"FAILED {job_key(job['run'], job['checkpoint'], job['profile'])}: {error}", flush=True)

    def prepare(self, run):
        name = run["name"]
        jobs = [job for job in self.state["jobs"].values() if job["run"] == name]
        if all(job["status"] in TERMINAL for job in jobs):
            return
        try:
            ready, message = dependency(run, self.config)
            self.state["runs"][name] = {"ready": ready, "dependency": message}
            if not ready:
                return
            if all("config_path" in job or job["status"] in TERMINAL for job in jobs):
                return
            run_dir = ROOT / "runs" / name
            base = read_json(run_dir / "resolved_config.json")
            if base["train"]["epochs"] != self.config["expected_epochs"]:
                raise ValueError("Saved config is not a 25-epoch run")
            manifest = read_json(absolute(base["data"]["val_manifest"]))
            records = manifest["samples"] if "samples" in manifest else manifest["records"]
            if sum(int(record["scene"]) == 5 for record in records) != self.config["expected_samples"]:
                raise ValueError("Saved validation manifest does not contain 1599 scene-5 samples")
            metadata = {}
            for checkpoint in self.config["checkpoints"]:
                try:
                    metadata[checkpoint] = checkpoint_metadata(run_dir / checkpoint)
                except Exception as error:
                    for job in jobs:
                        if job["checkpoint"] == checkpoint and job["status"] == "pending":
                            self.fail(job, f"Checkpoint preparation: {error}")
            first, second = self.config["checkpoints"]
            same = len(metadata) == 2 and all(metadata[first][field] == metadata[second][field]
                                             for field in ("epoch", "global_step"))
            import yaml

            for job in jobs:
                if job["status"] in TERMINAL or "config_path" in job:
                    continue
                checkpoint, profile = job["checkpoint"], job["profile"]
                directory = self.output / name / Path(checkpoint).stem / profile
                directory.mkdir(parents=True, exist_ok=True)
                resolved = copy.deepcopy(base)
                resolved.get("train", {}).pop("ema", None)
                resolved["train"].update(output_dir=str(directory), eval_stride=1)
                resolved["data"]["val_scenes"] = [5]
                resolved["inference"] = copy.deepcopy(self.config["profiles"][profile])
                config_path = directory / "config.yaml"
                text = yaml.safe_dump(resolved, sort_keys=False)
                if config_path.exists():
                    if config_path.read_text(encoding="utf-8") != text:
                        raise ValueError(f"Existing evaluation config differs: {config_path}")
                else:
                    with config_path.open("x", encoding="utf-8") as stream:
                        stream.write(text)
                job.update(**metadata[checkpoint], config_path=str(config_path),
                           source_config=str(run_dir / "resolved_config.json"),
                           checkpoint_path=str((run_dir / checkpoint).resolve()),
                           score_path=str(directory / "scores.json"), log_path=str(directory / "evaluation.log"))
                if same and checkpoint == second:
                    job["alias_of"] = job_key(name, first, profile)
                    job["alias_reason"] = "Same model state from the same epoch and global_step"
        except Exception as error:
            self.state["runs"][name] = {"ready": False, "error": str(error)}
            for job in jobs:
                if job["status"] == "pending":
                    if isinstance(error, TrainingBlocked):
                        job.update(status="blocked", error=str(error), finished_at=now())
                    else:
                        self.fail(job, f"Run preparation: {error}")

    def finish(self, job, returncode=None):
        score = Path(job["score_path"])
        partial = score.with_name("scores.partial.json")
        try:
            if returncode not in (None, 0):
                raise ValueError(f"Evaluator exited with code {returncode}; see {job['log_path']}")
            candidate = score if score.exists() else partial
            result = score_result(candidate, self.config["expected_samples"])
            if result.get("checkpoint_epoch") != job["epoch"]:
                raise ValueError("Evaluation checkpoint epoch differs from prepared checkpoint")
            if Path(result.get("checkpoint", "")).resolve() != Path(job["checkpoint_path"]):
                raise ValueError("Evaluation checkpoint path differs from prepared checkpoint")
            if not score.exists():
                os.replace(partial, score)
            job.update(status="completed", result=result, finished_at=now(),
                       elapsed_seconds=time.time() - job.get("started_unix", time.time()), exit_code=returncode)
        except Exception as error:
            self.fail(job, error)

    def refresh(self):
        for key, job in self.state["jobs"].items():
            if job["status"] != "running":
                continue
            child = self.children.get(key)
            if child is not None:
                returncode = child.poll()
                if returncode is None:
                    continue
            else:
                if process_alive(job["pid"], job["start_ticks"]):
                    continue
                returncode = None  # Adopted child: no waitpid exit status is available.
            self.finish(job, returncode)
            self.children.pop(key, None)
            lock = self.gpu_locks.pop(job["gpu"], None)
            if lock:
                lock.close()
        for job in self.state["jobs"].values():
            if job["status"] != "pending" or "alias_of" not in job:
                continue
            original = self.state["jobs"][job["alias_of"]]
            if original["status"] == "completed":
                job.update(status="alias", result=original["result"], score_path=original["score_path"],
                           elapsed_seconds=0, finished_at=now())
            elif original["status"] in {"failed", "blocked", "cancelled"}:
                job.update(status="blocked", error="Canonical evaluation failed; no duplicate retry", finished_at=now())

    def launch(self, key, job, gpu):
        # Recovery queues share the legacy parent's lock directory, including
        # its live workers; separate output roots must not create separate locks.
        lock_dir = absolute(self.config.get("gpu_lock_dir", self.config["output_dir"]))
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock = (lock_dir / f"gpu_{gpu}.lock").open("a+")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            return False
        try:
            try:
                identity = file_identity(Path(job["checkpoint_path"]))
            except OSError as error:
                self.fail(job, f"Checkpoint unavailable after dependency completion: {error}")
                lock.close()
                return False
            if identity != job["file_identity"]:
                self.fail(job, "Checkpoint changed after dependency completion")
                lock.close()
                return False
            score = Path(job["score_path"])
            if score.exists():
                self.finish(job)
                lock.close()
                return False
            partial = score.with_name("scores.partial.json")
            if partial.exists():
                self.fail(job, "Untracked partial score exists; refusing to overwrite or retry")
                lock.close()
                return False
            if Path(job["log_path"]).exists():
                self.fail(job, "Untracked evaluation log exists; refusing to repeat an earlier attempt")
                lock.close()
                return False
            snapshot = gpu_snapshot()
            if gpu not in snapshot:
                raise ValueError(f"Physical GPU {gpu} is absent")
            if snapshot[gpu]["pids"]:
                lock.close()
                return False
            env = os.environ.copy()
            for variable in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
                             "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE", "TORCHELASTIC_RUN_ID"):
                env.pop(variable, None)
            env.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=snapshot[gpu]["uuid"],
                       NCCL_P2P_DISABLE="1", OMP_NUM_THREADS="4", PYTHONUNBUFFERED="1")
            command = [str(python_path(self.config["python"])), "-m", "moonseg.evaluate", "--config",
                       job["config_path"], "--checkpoint", job["checkpoint_path"], "--weights", "model",
                       "--output", str(partial)]
            job.update(status="launching", gpu=gpu, gpu_uuid=snapshot[gpu]["uuid"], command=command,
                       started_at=now(), started_unix=time.time())
            self.save()
            with Path(job["log_path"]).open("xb") as log:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                         stdin=subprocess.DEVNULL, start_new_session=True, close_fds=True,
                                         pass_fds=(lock.fileno(),))
            info = process_info(child.pid)
            job.update(status="running", pid=child.pid, start_ticks=info["start_ticks"] if info else -1)
            self.children[key], self.gpu_locks[gpu] = child, lock
            self.save()
            print(f"START {key}: physical GPU {gpu}, UUID {snapshot[gpu]['uuid']}, PID {child.pid}", flush=True)
            return True
        except (subprocess.SubprocessError, OSError, ValueError) as error:
            lock.close()
            if job["status"] == "launching":
                self.fail(job, f"Launch failed: {error}")
            else:
                # GPU queries are transient and fail closed; pending work can wait safely.
                self.state["gpu_query_error"] = {"at": now(), "error": str(error)}
                print(f"GPU launch deferred: {error}", flush=True)
            return False

    def run(self):
        def stop(signum, frame):
            self.stopping = True
            print("Queue stopping; evaluator children continue and can be adopted on restart", flush=True)

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        while not self.stopping:
            self.refresh()
            for run in self.config["runs"]:
                self.prepare(run)
            self.refresh()  # Resolve newly discovered checkpoint aliases before scheduling.
            running = [job for job in self.state["jobs"].values() if job["status"] == "running"]
            for gpu in self.config["gpus"]:
                if len(running) >= self.config["max_workers"]:
                    break
                if any(job["gpu"] == gpu for job in running):
                    continue
                for key, job in self.state["jobs"].items():
                    if job["status"] == "pending" and "config_path" in job and "alias_of" not in job:
                        if self.launch(key, job, gpu):
                            running.append(job)
                        break
            if all(job["status"] in TERMINAL for job in self.state["jobs"].values()):
                self.state["status"] = "completed_with_failures" if any(
                    job["status"] in {"failed", "blocked"} for job in self.state["jobs"].values()) else "completed"
                self.state["finished_at"] = now()
                self.save()
                return 1 if self.state["status"] == "completed_with_failures" else 0
            self.save()
            deadline = time.monotonic() + self.config["poll_seconds"]
            while not self.stopping and time.monotonic() < deadline:
                time.sleep(min(1, max(0, deadline - time.monotonic())))
        self.state["status"] = "stopped"
        self.save()
        return 0


def main():
    os.chdir(ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="runs/plans/2026-10-23/evaluation.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = read_json(absolute(args.config))
    validate_config(config)
    if args.dry_run:
        plan = {"output_dir": str(absolute(config["output_dir"]).resolve()), "gpus": config["gpus"],
                "max_workers": config["max_workers"], "weights": "model", "profiles": config["profiles"],
                "expected_samples": config["expected_samples"], "runs": []}
        for run in config["runs"]:
            if run["name"] in config.get("cancelled_runs", []):
                plan["runs"].append({**run, "ready": False, "status": "cancelled"})
                continue
            try:
                ready, message = dependency(run, config)
                item = {**run, "ready": ready, "dependency_status": message}
                if ready:
                    item["checkpoints"] = {name: checkpoint_metadata(ROOT / "runs" / run["name"] / name)
                                           for name in config["checkpoints"]}
                plan["runs"].append(item)
            except Exception as error:
                plan["runs"].append({**run, "ready": False, "error": str(error)})
        print(json.dumps(plan, indent=2))
        return 0
    return Queue(config).run()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"Evaluation queue stopped: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        sys.exit(1)
