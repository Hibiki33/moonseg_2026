#!/usr/bin/env python3
"""Run matched precision comparisons on released GPUs or an explicitly assigned shared pair."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_fulltrain_plan as queue
from scripts.run_public_test_queue import prepare_checkpoint, score_result
from moonseg.evaluate_public_test import inference_runtime
from moonseg.utils import load_config

MODES = ("bf16", "fp32", "fp32_tf32")
PUBLIC_TEST = {"data_root": "data/research_test/LuSNAR", "manifest": "data/research_test/manifest.json"}


def source_checkpoint(plan):
    config, checkpoint, meta = prepare_checkpoint(
        {"name": plan["source_run"], "checkpoint": "last.pt", "expected_epochs": plan["expected_epochs"]},
        "final_epoch")
    if (meta["file_identity"] != plan["checkpoint_file_identity"]
            or meta["global_step"] != plan["expected_epochs"] * 125
            or config != queue.local_path(plan["source_config"])):
        raise ValueError("Precision comparison source checkpoint/config changed")
    return config, checkpoint, meta


def predecessor_ready(plan):
    dep = plan["predecessor"]
    if dep is None:
        if plan.get("gpus") != [1, 6] or plan.get("temporary_extra_gpus") != [1, 6]:
            raise ValueError("Immediate precision execution requires the explicitly assigned GPU1/6 pair")
        return True, "explicit GPU1/6 assignment; acquire shared locks and verify live GPU idleness"
    original = queue.read_json(queue.local_path(dep["plan_path"]))
    state = queue.read_json(queue.local_path(dep["status_path"]))
    if original != dep["plan_snapshot"] or state.get("plan") != original:
        raise ValueError("Predecessor lane changed after this precision comparison was queued")
    if state.get("status") == "failed":
        raise ValueError("Predecessor baseline lane failed; inspect it before scheduling its GPUs")
    if state.get("status") != "completed":
        if not queue.same_process({"pid": state["parent_pid"], "start_ticks": state["parent_start_ticks"]}):
            raise ValueError("Predecessor supervisor exited before successful completion")
        return False, "waiting for baseline training and both final-test profiles to finish"
    directory = queue.local_path(dep["status_path"]).parent
    marker = directory / "completed.json"
    if not marker.exists():
        return False, "waiting for predecessor completion marker"
    complete = queue.read_json(marker)
    expected_jobs = {job["name"] for job in original["jobs"]}
    if (complete.get("status") != "completed" or complete.get("exit_code") != 0
            or set(complete.get("jobs", [])) != expected_jobs or set(state["jobs"]) != expected_jobs
            or any(job.get("status") != "completed" for job in state["jobs"].values())):
        raise ValueError("Predecessor completion evidence is incomplete")
    expected_stages = {name + suffix for name in expected_jobs for suffix in ("_train", "_single", "_narrowflip")}
    if set(state["stages"]) != expected_stages:
        raise ValueError("Predecessor stage membership differs")
    for stage in state["stages"].values():
        receipt = queue.read_json(queue.local_path(stage["receipt_path"]))
        if (stage.get("status") != "completed" or receipt != stage.get("receipt")
                or receipt.get("status") != "completed" or receipt.get("exit_code") != 0):
            raise ValueError("Predecessor stage lacks a successful matching exit receipt")
        if queue.live_stage(queue.local_path(stage["spec_path"])):
            return False, "waiting for predecessor workers to exit"
    return True, "baseline training and both tests completed; GPU locks/idleness still required"


def validate_plan(plan):
    shared = plan.get("gpus") == [1, 6] and plan.get("temporary_extra_gpus") == [1, 6]
    if (plan.get("version") != 1 or plan.get("recipe") != "inference_precision"
            or (not shared and plan.get("gpus") not in ([2], [3])) or not plan.get("id")
            or plan.get("public_test") != PUBLIC_TEST
            or type(plan.get("poll_seconds")) is not int or not 1 <= plan["poll_seconds"] <= 60):
        raise ValueError("Use a released GPU2/3 or the explicitly assigned GPU1/6 pair")
    gpu = plan["gpus"][0]
    dep = plan["predecessor"]
    if shared:
        uuids = plan.get("expected_gpu_uuids", {})
        if (dep is not None or set(uuids) != {"1", "6"} or len(set(uuids.values())) != 2
                or any(not isinstance(v, str) or not v.startswith("GPU-") for v in uuids.values())):
            raise ValueError("Immediate shared queue needs predecessor:null and both distinct physical UUIDs")
    elif (dep["plan_snapshot"].get("recipe") != "supervised_baseline"
            or dep["plan_snapshot"].get("gpus") != [2, 3]
            or plan.get("expected_gpu_uuids") != {str(gpu): dep["plan_snapshot"]["expected_gpu_uuids"][str(gpu)]}
            or queue.local_path(dep["status_path"]) != queue.local_path(dep["plan_path"]).with_suffix(".state") / "status.json"):
        raise ValueError("Require the frozen baseline lane and its matching physical GPU UUID")
    config_path, checkpoint, meta = source_checkpoint(plan)
    config = load_config(config_path)
    if (config["model"].get("backbone") != "vitl16" or config["model"].get("freeze_backbone") is not False
            or config["train"].get("amp") != "bf16" or config["train"].get("tf32") is not True):
        raise ValueError("This comparison requires the jointly fine-tuned BF16/TF32 DINOv3-L mainline")
    profile = "narrowflip" if gpu == 2 else "single"
    jobs = plan.get("jobs", [])
    modes = MODES
    if "baseline_results" in plan:
        if not shared or set(plan["baseline_results"]) != {"single", "narrowflip"}:
            raise ValueError("Reusing BF16 requires both complete profiles for this shared queue")
        for baseline_profile, filename in plan["baseline_results"].items():
            record = dict(profile=baseline_profile, checkpoint_path=str(checkpoint), **meta,
                          checkpoint_policy="final_epoch")
            value = score_result(queue.local_path(filename), record, PUBLIC_TEST)
            runtime = value.get("inference_runtime", inference_runtime(config))
            if runtime.get("amp") != "bf16" or runtime.get("tf32") is not True:
                raise ValueError("Existing baseline does not use the intended BF16/TF32 precision")
        modes = ("fp32", "fp32_tf32")
    expected = {(p, mode) for p in (("single", "narrowflip") if shared else (profile,)) for mode in modes}
    actual = [(job.get("profile"), job.get("precision")) for job in jobs]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Require each planned precision/profile exactly once; do not rerun a reused BF16 baseline")
    for job in jobs:
        if (job.get("name") != f"{plan['source_run']}_precision_{job['profile']}_{job['precision']}"
                or job.get("config") != plan["source_config"]):
            raise ValueError("Every precision job must share the original checkpoint/config and fixed profile")
    predecessor_ready(plan)
    return {job["name"]: config_path.read_bytes() for job in jobs}


class PrecisionLane(queue.Lane):
    def predecessor_ready(self):
        return predecessor_ready(self.plan)

    def verify_tests(self, job):
        if self.state["stages"].get(job["name"], {}).get("status") != "completed":
            raise ValueError("Missing successful precision evaluator receipt")
        config_path, checkpoint, meta = source_checkpoint(self.plan)
        record = dict(profile=job["profile"], checkpoint_path=str(checkpoint), **meta,
                      checkpoint_policy="final_epoch")
        value = score_result(self.directory / "scores" / job["name"] / "scores.json", record, PUBLIC_TEST)
        runtime = value.get("inference_runtime", {})
        expected = inference_runtime(load_config(config_path), job["precision"])
        if (runtime.get("precision") != job["precision"]
                or any(runtime.get(k) != v for k, v in expected.items())
                or runtime.get("matmul_allow_tf32") != expected["tf32"]
                or runtime.get("cudnn_allow_tf32") != expected["tf32"]
                or not value.get("precision_smoke", {}).get("passed")
                or value.get("elapsed_seconds", 0) <= 0):
            raise ValueError("Evaluator did not use/record the intended inference precision")
        return value

    def launch_job(self, job, locks, gpu):
        config_path, checkpoint, _ = source_checkpoint(self.plan)
        snapshot = self.directory / "configs" / config_path.name
        if config_path.read_bytes() != snapshot.read_bytes():
            raise ValueError("Original resolved training config changed while waiting")
        key = job["name"]
        if key not in self.state["stages"]:
            while True:
                occupancy = queue.gpu_occupancy([gpu])
                if occupancy["gpu_uuids"] != {str(gpu): self.plan["expected_gpu_uuids"][str(gpu)]}:
                    raise ValueError("Physical GPU identity changed")
                if not occupancy["occupied"]:
                    break
                time.sleep(2)
            directory = self.directory / "scores" / key
            directory.mkdir(parents=True, exist_ok=False)
            command = [str(queue.PYTHON), "-m", "moonseg.evaluate_public_test", "--config", str(snapshot),
                "--checkpoint", str(checkpoint), "--checkpoint-policy", "final_epoch", "--weights", "model",
                "--data-root", str(queue.local_path(PUBLIC_TEST["data_root"])),
                "--manifest", str(queue.local_path(PUBLIC_TEST["manifest"])), "--profile", job["profile"],
                "--precision", job["precision"], "--progress-every", "100", "--output", str(directory / "scores.json")]
            self.save(status="public_test", current_job=key)
            self.launch(key, command, directory / "evaluation.log",
                queue.environment([gpu], self.plan["expected_gpu_uuids"], evaluation=True), locks)
            self.state["stages"][key]["gpu"] = gpu
            self.save()

    def finish_job(self, job):
        key = job["name"]
        self.wait_stage(key)
        result = self.verify_tests(job)
        self.state["jobs"][key] = dict(status="completed", results=result, finished_at=queue.now())
        self.save()

    def execute_job(self, job, locks):
        self.launch_job(job, locks, self.plan["gpus"][0])
        self.finish_job(job)

    def run(self):
        if self.plan["gpus"] != [1, 6]:
            return super().run()
        try:
            # Adopt durable workers before reacquiring the locks they inherited.
            for job in self.plan["jobs"]:
                if job["name"] in self.state["stages"]:
                    self.finish_job(job)
            if self.state["status"] == "completed":
                self.complete_marker()
                return 0
            if not self.predecessor_ready()[0]:
                raise ValueError("Immediate shared precision queue unexpectedly has a predecessor")
            with ExitStack() as stack:
                locks = []
                self.save(status="waiting_gpu_locks")
                for gpu in self.plan["gpus"]:
                    for filename in (f"runs/.training_gpu_{gpu}.lock", f"runs/evaluations_2026-10-23/gpu_{gpu}.lock"):
                        handle = stack.enter_context(queue.local_path(filename).open("a+"))
                        while True:
                            try:
                                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                                break
                            except BlockingIOError:
                                time.sleep(self.plan["poll_seconds"])
                        locks.append(handle)
                self.wait_idle()
                pending = [job for job in self.plan["jobs"] if job["name"] not in self.state["jobs"]]
                free, active = [6, 1], {}
                while pending or active:
                    for key, (gpu, job) in list(active.items()):
                        stage = self.state["stages"][key]
                        path = Path(stage["receipt_path"])
                        receipt = queue.read_json(path) if path.exists() else None
                        if ((receipt and receipt.get("status") in {"completed", "failed"})
                                or not queue.live_stage(Path(stage["spec_path"]))):
                            self.finish_job(job)
                            del active[key]
                            free.append(gpu)
                    while pending and free:
                        gpu, job = free.pop(0), pending.pop(0)
                        self.launch_job(job, locks, gpu)
                        active[job["name"]] = (gpu, job)
                    self.save(status="public_test", active_jobs=list(active), pending_jobs=[job["name"] for job in pending])
                    if active:
                        time.sleep(self.plan["poll_seconds"])
            self.save(status="completed", finished_at=queue.now())
            self.complete_marker()
            return 0
        except Exception as error:
            self.save(status="failed", error=str(error), finished_at=queue.now())
            raise


def main():
    os.chdir(ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    path = queue.local_path(args.plan)
    if not path.is_relative_to(queue.local_path("runs/plans")):
        raise ValueError("Keep precision plans under local runs/plans")
    if (path.with_name(path.stem + ".state") / "cancelled.json").exists():
        raise ValueError("This precision plan was cancelled; use its recorded replacement")
    plan = queue.read_json(path)
    contents = validate_plan(plan)
    if args.dry_run:
        print(json.dumps({"plan": str(path), "dependency": predecessor_ready(plan),
                          "gpus": plan["gpus"], "jobs": plan["jobs"]}, indent=2))
        return 0
    return PrecisionLane(path, plan, contents).run()


if __name__ == "__main__":
    raise SystemExit(main())
