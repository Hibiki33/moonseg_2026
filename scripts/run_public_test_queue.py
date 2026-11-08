#!/usr/bin/env python3
"""Durable public LuSNAR 2/4/9 research evaluation queue; never trains a model.

Dry-run reads plans, history and CPU checkpoint metadata only, and writes nothing.
Physical GPU locks are shared with the scene-5 evaluation queue. Existing successful
scores are adopted; failed attempts are not automatically repeated.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_evaluation_queue import (Queue as ValidationQueue, TrainingBlocked, absolute, atomic_json,
                                          checkpoint_metadata, file_identity, gpu_snapshot, now,
                                          process_info, read_json, TERMINAL, apply_queue_policy,
                                          validate_cancellations)
from moonseg.evaluate_public_test import (CHECKPOINT_POLICIES, EXPECTED_SAMPLES, MANIFEST_KIND, PROFILES,
                                          SCENE_COUNTS, checkpoint_policy_metadata, validate_final_checkpoint)


def dependency(run):
    """A successful marker takes precedence over stale failed lane statuses."""
    directory = ROOT / "runs" / run["name"]
    dep = run["dependency"]
    epochs = run["expected_epochs"]
    if dep["kind"] == "marker":
        marker = directory / "queue_completed.json"
        if marker.exists():
            value = read_json(marker)
            if value.get("status") != "completed" or value.get("exit_code") != 0:
                raise TrainingBlocked(f"Training completion marker is not successful: {marker}")
            if value.get("expected_epochs") != epochs:
                raise ValueError(f"Unexpected epoch count in {marker}")
        else:
            status = absolute(dep["status_path"])
            if status.exists():
                state = read_json(status)
                if state.get("plan", {}).get("config", "").endswith(f"/{run['name']}.yaml") and state.get("status") == "failed":
                    raise TrainingBlocked(f"Training plan failed: {state.get('error', state.get('exit_code'))}")
                for lane in state.get("lanes", {}).values():
                    for job in lane.get("jobs", []):
                        if job.get("name") == run["name"] and (
                                job.get("status") == "failed" or lane.get("status") == "failed"):
                            raise TrainingBlocked(f"Training lane failed: {lane.get('error', job.get('error'))}")
            return False, f"waiting for completion marker (status: {status})"
    history = directory / "history.jsonl"
    if not history.exists():
        return False, "waiting for training history"
    rows = [json.loads(line) for line in history.read_text().splitlines() if line.strip()]
    if not rows or rows[-1].get("epoch") != epochs:
        return False, f"waiting for epoch {epochs}; last epoch {rows[-1].get('epoch') if rows else None}"
    if [row.get("epoch") for row in rows] != list(range(1, epochs + 1)):
        raise ValueError(f"History must contain exactly epochs 1..{epochs}: {history}")
    if rows[-1].get("evaluated_samples") != 1599 or rows[-1].get("eval_stride") != 1:
        raise ValueError(f"Final history must use full scene-5 validation: {history}")
    return True, f"training finished {epochs} epochs with full scene-5 validation"


def validate_config(config):
    gpus = config["gpus"]
    if (not isinstance(gpus, list) or not gpus or any(type(gpu) is not int or gpu < 0 for gpu in gpus)
            or len(gpus) != len(set(gpus)) or type(config["max_workers"]) is not int
            or config["max_workers"] not in (1, 2)):
        raise ValueError("Use unique nonnegative physical GPU indices and at most two workers")
    policy = config.get("checkpoint_policy", "validation_selected")
    if policy not in CHECKPOINT_POLICIES:
        raise ValueError("Unknown checkpoint policy")
    if config["profiles"] != PROFILES or config["expected_samples"] != EXPECTED_SAMPLES:
        raise ValueError("Only the fixed single/narrowflip profiles over all 5004 samples are permitted")
    if config["weights"] != "model" or config["manifest_kind"] != MANIFEST_KIND:
        raise ValueError("Research manifest and model weights are mandatory")
    if config["gpu_lock_dir"] != "runs/evaluations_2026-10-23":
        raise ValueError("GPU locks must coordinate the existing scene-5 evaluation queue")
    if "expected_gpu_uuids" in config:
        expected = config['expected_gpu_uuids']
        if set(expected) != {str(gpu) for gpu in gpus} or len(set(expected.values())) != len(gpus):
            raise ValueError("Expected one distinct frozen UUID per evaluation GPU")
    output = absolute(config["output_dir"]).resolve()
    if not output.is_relative_to((ROOT / "runs").resolve()) or output == absolute(config["gpu_lock_dir"]).resolve():
        raise ValueError("Use a distinct public-test output directory under external runs storage")
    names = [run["name"] for run in config["runs"]]
    if len(names) != len(set(names)) or any(Path(name).name != name for name in names):
        raise ValueError("Run names must be unique directory names")
    if (not names or len(config["priority_runs"]) != len(set(config["priority_runs"]))
            or not set(config["priority_runs"]) <= set(names)):
        raise ValueError("Expected nonempty runs and unique planned priority comparisons")
    validate_cancellations(config, names)
    for run in config["runs"]:
        epochs = run["expected_epochs"]
        if policy == "final_epoch":
            if type(epochs) is not int or epochs != 25 or run["checkpoint"] != "last.pt":
                raise ValueError("final_epoch evaluations require the fixed 25-epoch last.pt checkpoint")
        elif type(epochs) is not int or epochs <= 0 or run["checkpoint"] != ("best.pt" if epochs == 80 else "best_miou.pt"):
            raise ValueError("Use validation best_miou.pt with an explicit positive stage budget, or legacy best.pt (80 epochs)")
        if run["dependency"]["kind"] not in {"history", "marker"}:
            raise ValueError("Unknown training dependency")
        if run["dependency"]["kind"] == "marker" and not run["dependency"].get("status_path"):
            raise ValueError("Marker dependencies require their own training status path")
    if not 1 <= config["poll_seconds"] <= 60:
        raise ValueError("poll_seconds must be between 1 and 60")


def prepare_checkpoint(run, policy):
    directory = ROOT / "runs" / run["name"]
    config_path = (directory / "resolved_config.json").resolve()
    config = read_json(config_path)
    if config["train"]["epochs"] != run["expected_epochs"]:
        raise ValueError("Original resolved config epoch count differs from plan")
    checkpoint = (directory / run["checkpoint"]).resolve()
    if policy == "final_epoch":
        # Trusted local artifacts only; mmap keeps optimizer tensors off the queue parent's heap.
        import torch
        before = file_identity(checkpoint)
        try:
            value = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
        except RuntimeError as error:
            if "mmap can only be used" not in str(error):
                raise
            value = torch.load(checkpoint, map_location="cpu", weights_only=False)
        validate_final_checkpoint(config, checkpoint, value, run["expected_epochs"])
        metadata = {"epoch": value["epoch"] + 1, "global_step": value["global_step"], "file_identity": before}
        del value
        if file_identity(checkpoint) != before:
            raise ValueError("Checkpoint changed while reading final metadata")
    else:
        metadata = checkpoint_metadata(checkpoint)
        if not 1 <= metadata["epoch"] <= run["expected_epochs"]:
            raise ValueError("Selected checkpoint epoch is outside the completed training run")
        if run["checkpoint"] == "best_miou.pt":
            rows = [json.loads(line) for line in (directory / "history.jsonl").read_text().splitlines() if line.strip()]
            if metadata["epoch"] != rows[-1].get("best_miou_epoch"):
                raise ValueError("Checkpoint epoch differs from the final local-validation selection")
    return config_path, checkpoint, metadata


def job_order(config):
    by_name = {run["name"]: run for run in config["runs"]}
    for profile in PROFILES:
        for name in config["priority_runs"]:
            yield by_name[name], profile
    for run in config["runs"]:
        if run["name"] not in config["priority_runs"]:
            for profile in PROFILES:
                yield run, profile


def score_result(path, job, config):
    value = read_json(path)
    policy = config.get("checkpoint_policy", "validation_selected")
    if value.get("checkpoint_policy", "validation_selected") != policy:
        raise ValueError("Score checkpoint policy does not match this planned evaluation")
    if policy == "final_epoch" and any(value.get(key) != expected for key, expected in checkpoint_policy_metadata(policy).items()):
        raise ValueError("Final scores must identify the fixed-budget public development comparison")
    if (value.get("evaluation_kind") != MANIFEST_KIND or value.get("official_score") is not False
            or value.get("evaluated_samples") != EXPECTED_SAMPLES or value.get("weights") != "model"
            or value.get("scene_sample_counts") != SCENE_COUNTS or "val_loss" in value):
        raise ValueError("Scores must cover all public research samples with model weights and no loss")
    if (value.get("profile") != job["profile"] or value.get("inference") != PROFILES[job["profile"]]
            or value.get("checkpoint_epoch") != job["epoch"] or value.get("global_step") != job["global_step"]
            or value.get("checkpoint_file_identity") != job["file_identity"]
            or Path(value.get("checkpoint", "")).resolve() != Path(job["checkpoint_path"])
            or Path(value.get("manifest", "")).resolve() != absolute(config["manifest"]).resolve()
            or Path(value.get("data_root", "")).resolve() != absolute(config["data_root"]).resolve()):
        raise ValueError("Score provenance does not match this planned evaluation")
    if set(value.get("scenes", {})) != set(SCENE_COUNTS):
        raise ValueError("Expected exactly scenes 2/4/9")
    for metrics in [value, value.get("pooled", {}), value.get("scene_equal_macro", {}), *value["scenes"].values()]:
        for key in ("mIoU", "mAcc"):
            metric = metrics.get(key)
            if isinstance(metric, bool) or not isinstance(metric, (float, int)) or not math.isfinite(metric) or not 0 <= metric <= 1:
                raise ValueError(f"Invalid {key}")
    for scene, count in SCENE_COUNTS.items():
        if value["scenes"][scene].get("evaluated_samples") != count:
            raise ValueError(f"Incomplete scene {scene}")
    for key in ("mIoU", "mAcc"):
        if value[key] != value["pooled"][key] or not math.isclose(
                value["scene_equal_macro"][key], sum(value["scenes"][scene][key] for scene in SCENE_COUNTS) / 3):
            raise ValueError("Inconsistent pooled or equal-scene metrics")
    if len(value.get("class_IoU", {})) != 5 or len(value.get("confusion_matrix", [])) != 5:
        raise ValueError("Missing per-class metrics or pooled confusion matrix")
    return value


class Queue(ValidationQueue):
    """Reuse the validation queue's process refresh, stop handling and scheduler."""

    def __init__(self, config):
        self.config = config
        self.output = absolute(config["output_dir"]).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.lock = (self.output / "public_test_queue.lock").open("a+")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another public-test queue parent is active") from None
        status = self.output / "status.json"
        self.state = read_json(status) if status.exists() else {
            "version": 1, "config": config, "created_at": now(), "jobs": {}, "runs": {}}
        if self.state.get("config") != config:
            raise ValueError("Saved configuration differs; use a new output_dir")
        self.children, self.gpu_locks, self.stopping = {}, {}, False
        self.state.update(parent_pid=os.getpid(), parent_start_ticks=process_info(os.getpid())["start_ticks"],
                          status="running", started_at=now(), evaluation_kind=MANIFEST_KIND, official_score=False,
                          **checkpoint_policy_metadata(config.get("checkpoint_policy", "validation_selected")))
        for run, profile in job_order(config):
            key = f"{run['name']}/{Path(run['checkpoint']).stem}/{profile}"
            self.state["jobs"].setdefault(key, {"run": run["name"], "label": run["label"],
                "checkpoint": run["checkpoint"], "profile": profile, "weights": "model", "status": "pending",
                "checkpoint_policy": config.get("checkpoint_policy", "validation_selected")})
        for job in self.state["jobs"].values():
            if job["status"] == "launching":
                self.fail(job, "Parent interrupted during launch; inspect log/PID before a manual retry")
        apply_queue_policy(self.state, config, [f"{run['name']}/{Path(run['checkpoint']).stem}/{profile}"
                           for run, profile in job_order(config)])
        self.save()

    def save(self):
        self.state["updated_at"] = now()
        atomic_json(self.output / "status.json", self.state)
        rows = []
        for key, job in self.state["jobs"].items():
            result = job.get("result", {})
            rows.append({"job": key, **{field: job.get(field) for field in (
                "run", "label", "checkpoint", "checkpoint_policy", "epoch", "global_step", "weights", "profile", "status",
                "gpu", "pid", "elapsed_seconds", "score_path", "log_path", "error", "cancellation_reason")},
                **{field: result.get(field) for field in (
                    "mIoU", "mAcc", "pooled", "scene_equal_macro", "scenes", "scene_sample_counts", "evaluated_samples")}})
        atomic_json(self.output / "aggregate.json", {"updated_at": now(), "status": self.state["status"],
            "evaluation_kind": MANIFEST_KIND, "official_score": False,
            "metric_scope": "Pooled pixels and equal scene macro; no organizer-weighted official total.",
            **checkpoint_policy_metadata(self.config.get("checkpoint_policy", "validation_selected")),
            "evaluations": rows})

    def prepare(self, run):
        jobs = [job for job in self.state["jobs"].values() if job["run"] == run["name"]]
        if all(job["status"] in TERMINAL or "config_path" in job for job in jobs):
            return
        try:
            ready, message = dependency(run)
            self.state["runs"][run["name"]] = {"ready": ready, "dependency": message}
            if not ready:
                return
            config_path, checkpoint, metadata = prepare_checkpoint(
                run, self.config.get("checkpoint_policy", "validation_selected"))
            for job in jobs:
                if job["status"] in TERMINAL:
                    continue
                output = self.output / run["name"] / Path(run["checkpoint"]).stem / job["profile"]
                output.mkdir(parents=True, exist_ok=True)
                job.update(**metadata, config_path=str(config_path), checkpoint_path=str(checkpoint),
                           score_path=str(output / "scores.json"), log_path=str(output / "evaluation.log"))
        except Exception as error:
            self.state["runs"][run["name"]] = {"ready": False, "error": str(error)}
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
                raise ValueError(f"Evaluator exited {returncode}; see {job['log_path']}")
            result = score_result(score if score.exists() else partial, job, self.config)
            if file_identity(Path(job["checkpoint_path"])) != job["file_identity"]:
                raise ValueError("Checkpoint changed after preparation")
            if not score.exists():
                os.link(partial, score)
                partial.unlink()
            job.update(status="completed", result=result, finished_at=now(), exit_code=returncode,
                       elapsed_seconds=time.time() - job.get("started_unix", time.time()))
        except Exception as error:
            self.fail(job, error)

    def launch(self, key, job, gpu):
        lock = (absolute(self.config["gpu_lock_dir"]) / f"gpu_{gpu}.lock").open("a+")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            return False
        try:
            if file_identity(Path(job["checkpoint_path"])) != job["file_identity"]:
                self.fail(job, "Checkpoint changed after preparation")
                lock.close()
                return False
            score = Path(job["score_path"])
            if score.exists():
                self.finish(job)
                lock.close()
                return False
            if score.with_name("scores.partial.json").exists() or Path(job["log_path"]).exists():
                self.fail(job, "Untracked prior attempt exists; refusing automatic retry/overwrite")
                lock.close()
                return False
            snapshot = gpu_snapshot()
            if gpu not in snapshot:
                raise ValueError(f"Physical GPU {gpu} is absent")
            expected_uuid = self.config.get("expected_gpu_uuids", {}).get(str(gpu))
            if expected_uuid is not None and snapshot[gpu]['uuid'] != expected_uuid:
                raise ValueError(f"Physical GPU {gpu} UUID differs from the approved plan")
            if snapshot[gpu]["pids"]:
                lock.close()
                return False
            env = os.environ.copy()
            for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
                         "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE", "TORCHELASTIC_RUN_ID"):
                env.pop(name, None)
            env.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=snapshot[gpu]["uuid"],
                       NCCL_P2P_DISABLE="1", OMP_NUM_THREADS="4", PYTHONUNBUFFERED="1",
                       PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
            command = [str(absolute(self.config["python"])), "-m", "moonseg.evaluate_public_test", "--config",
                       job["config_path"], "--checkpoint", job["checkpoint_path"], "--weights", "model",
                       "--checkpoint-policy", self.config.get("checkpoint_policy", "validation_selected"),
                       "--data-root", str(absolute(self.config["data_root"])), "--manifest",
                       str(absolute(self.config["manifest"])), "--profile", job["profile"],
                       "--output", str(score.with_name("scores.partial.json"))]
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
                self.state["gpu_query_error"] = {"at": now(), "error": str(error)}
            return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="runs/plans/2026-10-23/public_test.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = read_json(absolute(args.config))
    if config.get("disabled"):
        raise ValueError(f"Plan disabled: {config.get('disabled_reason', 'explicit cancellation')}")
    validate_config(config)
    if args.dry_run:
        plan = {"output_dir": str(absolute(config["output_dir"]).resolve()), "gpus": config["gpus"],
                "max_workers": config["max_workers"], "weights": "model", "profiles": PROFILES,
                "expected_samples": EXPECTED_SAMPLES, "scene_sample_counts": SCENE_COUNTS,
                "evaluation_kind": MANIFEST_KIND, "official_score": False, "runs": [],
                **checkpoint_policy_metadata(config.get("checkpoint_policy", "validation_selected")),
                "job_order": [f"{run['label']}/{profile}" for run, profile in job_order(config)]}
        for run in config["runs"]:
            if run["name"] in config.get("cancelled_runs", []):
                plan["runs"].append({**run, "ready": False, "status": "cancelled"})
                continue
            try:
                ready, message = dependency(run)
                item = {**run, "ready": ready, "dependency_status": message}
                if ready:
                    _, _, item["checkpoint_metadata"] = prepare_checkpoint(
                        run, config.get("checkpoint_policy", "validation_selected"))
                plan["runs"].append(item)
            except Exception as error:
                plan["runs"].append({**run, "ready": False, "error": str(error)})
        print(json.dumps(plan, indent=2))
        return 0
    os.chdir(ROOT)
    absolute(config["gpu_lock_dir"]).mkdir(parents=True, exist_ok=True)
    return Queue(config).run()


if __name__ == "__main__":
    raise SystemExit(main())
