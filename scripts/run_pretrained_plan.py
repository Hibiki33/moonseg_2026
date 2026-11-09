#!/usr/bin/env python3
"""Queue one pretrained SAM3 single evaluation on GPU2 after frozen lane_23.

Reuse full-training durable stage workers and shared GPU locks. There is no
training stage, epoch budget, retry or second reserved GPU. --dry-run is a
CPU/read-only admission check and does not query CUDA or GPU occupancy.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_fulltrain_plan as fulltrain
from moonseg.evaluate_pretrained import asset_metadata, validate_config, validate_pretrained_result
from moonseg.evaluate_public_test import PublicTestResearchDataset, file_identity
from moonseg.utils import load_config

PREDECESSOR_PLAN = "runs/plans/2026-11-09/fulltrain/lane_23.json"
PUBLIC_TEST = {"data_root": "data/research_test/LuSNAR", "manifest": "data/research_test/manifest.json"}


def predecessor_plan(plan):
    dep = plan.get("predecessor")
    if (not isinstance(dep, dict) or dep.get("kind") != "fulltrain_lane"
            or dep.get("plan_path") != PREDECESSOR_PLAN
            or dep.get("status_path") != str(Path(PREDECESSOR_PLAN).with_suffix(".state") / "status.json")):
        raise ValueError("SAM3 must wait for the complete approved frozen full-training lane_23")
    original = fulltrain.read_json(fulltrain.local_path(dep["plan_path"]))
    if (original.get("gpus") != [2, 3] or original.get("recipe") is not None
            or original.get("expected_gpu_uuids") != dep.get("expected_gpu_uuids")
            or plan.get("expected_gpu_uuids") != {"2": original["expected_gpu_uuids"]["2"]}
            or [job.get("expected_epochs") for job in original.get("jobs", [])] != [15, 20]
            or dep.get("expected_job_names") != [job["name"] for job in original["jobs"]]):
        raise ValueError("Freeze the complete predecessor 15/20-epoch lane, both physical UUIDs and its job membership")
    return original


def predecessor_ready(plan):
    original = predecessor_plan(plan)
    path = fulltrain.local_path(plan["predecessor"]["status_path"])
    state = fulltrain.read_json(path)
    if state.get("plan") != original:
        raise ValueError("Frozen predecessor plan differs from its durable state")
    if state.get("status") in {"failed", "cancelled", "stopped", "blocked"}:
        raise ValueError("Frozen predecessor lane did not succeed; automatic continuation refused")
    if state.get("status") != "completed":
        if not fulltrain.same_process({"pid": state.get("parent_pid"),
                                       "start_ticks": state.get("parent_start_ticks")}):
            raise ValueError("Frozen predecessor supervisor exited before complete lane success")
        return False, "waiting for all frozen lane_23 training and public-test stages"
    if set(state.get("jobs", {})) != {job["name"] for job in original["jobs"]}:
        raise ValueError("Completed frozen predecessor has incomplete or changed job membership")
    expected_stages = {job["name"] + suffix for job in original["jobs"]
                       for suffix in ("_train", "_single", "_narrowflip")}
    if set(state.get("stages", {})) != expected_stages:
        raise ValueError("Completed frozen predecessor has incomplete or changed stage membership")
    for key, stage in state["stages"].items():
        receipt = fulltrain.read_json(fulltrain.local_path(stage["receipt_path"]))
        if (stage.get("status") != "completed" or stage.get("exit_code") != 0
                or receipt != stage.get("receipt") or receipt.get("status") != "completed"
                or receipt.get("exit_code") != 0 or receipt.get("spec_path") != stage.get("spec_path")
                or (receipt.get("pid"), receipt.get("start_ticks")) != (stage.get("pid"), stage.get("start_ticks"))):
            raise ValueError(f"Frozen predecessor stage lacks a matching successful durable receipt: {key}")
        if (fulltrain.live_stage(fulltrain.local_path(stage["spec_path"]))
                or fulltrain.same_process({"pid": receipt.get("child_pid"),
                                           "start_ticks": receipt.get("child_start_ticks")})):
            return False, "waiting for completed frozen predecessor workers/children to exit"
    # These methods only read artifacts. Do not instantiate a predecessor Lane:
    # that would acquire its supervisor lock and write its durable state.
    verifier = object.__new__(fulltrain.Lane)
    verifier.plan, verifier.state, verifier.directory = original, state, path.parent
    for job in original["jobs"]:
        record = state["jobs"][job["name"]]
        if (record.get("status") != "completed" or record.get("expected_epochs") != job["expected_epochs"]
                or record.get("expected_updates") != job["expected_updates"]
                or record.get("results") != verifier.verify_tests(job)):
            raise ValueError(f"Frozen predecessor final checkpoints/scores are incomplete: {job['name']}")
    marker = path.parent / "completed.json"
    if not marker.exists():
        if fulltrain.same_process({"pid": state.get("parent_pid"), "start_ticks": state.get("parent_start_ticks")}):
            return False, "waiting for frozen predecessor completion marker"
        raise ValueError("Completed frozen predecessor lacks its durable completion marker")
    expected = dict(status="completed", exit_code=0, plan_path=str(fulltrain.local_path(PREDECESSOR_PLAN)),
                    jobs=list(state["jobs"]), finished_at=state["finished_at"])
    if fulltrain.read_json(marker) != expected:
        raise ValueError("Frozen predecessor completion marker differs from verified state")
    return True, "complete frozen lane_23 receipts, final checkpoints and all public-test scores verified"


def plan_python(plan):
    candidate = Path(plan["python"])
    if not candidate.is_absolute() or candidate.name not in {"python", "python3", "python3.12", "python3.13"}:
        raise ValueError("Freeze the absolute Python executable inside the independent local SAM3 venv")
    # A venv's final executable may itself be a symlink to its base interpreter;
    # admission concerns the physical venv directory, not that terminal symlink.
    executable = fulltrain.local_path(candidate.parent) / candidate.name
    if (executable.parent.name != "bin" or not executable.is_file() or not os.access(executable, os.X_OK)
            or not (executable.parent.parent / "pyvenv.cfg").is_file()):
        raise ValueError("SAM3 python must be an executable in a physical disk02 virtual environment")
    return executable


def asset_snapshot(config, python):
    metadata = asset_metadata(config)
    source = Path(metadata["source_manifest"]).parent
    manifest = fulltrain.read_json(source / "SOURCE_MANIFEST.json")
    files = [fulltrain.local_path(source / name) for name in manifest["files"]]
    return {"metadata": metadata, "source_files": {str(path): file_identity(path) for path in files},
            "python": str(python), "python_file_identity": file_identity(python),
            "venv_config_identity": file_identity(python.parent.parent / "pyvenv.cfg")}


def validate_plan(plan):
    if (plan.get("version") != 1 or plan.get("recipe") != "sam3_pretrained_only" or plan.get("gpus") != [2]
            or not isinstance(plan.get("id"), str) or not plan["id"]):
        raise ValueError("Admit only one version-1 inference-only SAM3 evaluation on released GPU2")
    uuids = plan.get("expected_gpu_uuids", {})
    if set(uuids) != {"2"} or not isinstance(uuids["2"], str) or not uuids["2"].startswith("GPU-"):
        raise ValueError("Freeze exactly the physical UUID of GPU2; no other GPU is reserved")
    if type(plan.get("poll_seconds")) is not int or not 1 <= plan["poll_seconds"] <= 60:
        raise ValueError("poll_seconds must be an integer from 1 to 60")
    predecessor_plan(plan)
    python = plan_python(plan)
    jobs = plan.get("jobs", [])
    if len(jobs) != 1 or jobs[0].get("name") != "sam3_pretrained_single" or set(jobs[0]) != {"name", "config"}:
        raise ValueError("Declare exactly one sam3_pretrained_single config job, without training budgets")
    source = fulltrain.local_path(jobs[0]["config"])
    content, config = source.read_bytes(), load_config(source)
    validate_config(config)
    fulltrain.local_path(config["model"]["backbone_weights"])
    fulltrain.local_path(config["model"].get("sam3_repo", "third_party/sam3"))
    asset_snapshot(config, python)
    if plan.get("public_test") != PUBLIC_TEST:
        raise ValueError("Use the fixed complete public scenes 2/4/9 research dataset")
    dataset = PublicTestResearchDataset(fulltrain.local_path(PUBLIC_TEST["data_root"]),
                                       fulltrain.local_path(PUBLIC_TEST["manifest"]), config["data"])
    for record in dataset.records:
        for field in ("image", "label"):
            fulltrain.local_path(dataset.root / record[field])
    if source.read_bytes() != content:
        raise ValueError("SAM3 config changed while validating its source/assets/data")
    return {jobs[0]["name"]: content}


class PretrainedLane(fulltrain.Lane):
    def __init__(self, plan_path, plan, contents):
        self.python = plan_python(plan)
        config = load_config(fulltrain.local_path(plan["jobs"][0]["config"]))
        assets = asset_snapshot(config, self.python)
        super().__init__(plan_path, plan, contents)
        if (self.state.get("pretrained_assets", assets) != assets
                or (self.state["stages"] and "pretrained_assets" not in self.state)):
            raise ValueError("Original SAM3 source/weights/interpreter changed since this inference plan started")
        self.save(pretrained_assets=assets)

    def predecessor_ready(self):
        return predecessor_ready(self.plan)

    def config_path(self, job):
        return self.directory / "configs" / Path(job["config"]).name

    def verify_tests(self, job):
        key = job["name"] + "_single"
        if self.state["stages"].get(key, {}).get("status") != "completed":
            raise ValueError("Missing successful durable pretrained inference stage")
        config_path = self.config_path(job)
        return validate_pretrained_result(fulltrain.read_json(self.directory / "scores/scores.json"),
            load_config(config_path), config_path, fulltrain.local_path(PUBLIC_TEST["data_root"]),
            fulltrain.local_path(PUBLIC_TEST["manifest"]), self.state["pretrained_assets"]["metadata"])

    def execute_job(self, job, locks):
        key, config_path = job["name"] + "_single", self.config_path(job)
        config = load_config(config_path)
        if asset_snapshot(config, self.python) != self.state["pretrained_assets"]:
            raise ValueError("Original SAM3 source/weights/interpreter changed while waiting for lane_23")
        if fulltrain.local_path(job["config"]).read_bytes() != config_path.read_bytes():
            raise ValueError("Source SAM3 config changed after inference plan startup")
        if key not in self.state["stages"]:
            self.wait_idle()
            if (asset_snapshot(config, self.python) != self.state["pretrained_assets"]
                    or fulltrain.local_path(job["config"]).read_bytes() != config_path.read_bytes()):
                raise ValueError("SAM3 assets/config changed while waiting for GPU2 idleness")
            directory = self.directory / "scores"
            if directory.exists() and any(directory.iterdir()):
                raise ValueError("Refusing an untracked prior pretrained inference attempt")
            directory.mkdir(parents=True, exist_ok=True)
            env = fulltrain.environment([2], self.plan["expected_gpu_uuids"], evaluation=True)
            env.update(PYTHON=str(self.python), VIRTUAL_ENV=str(self.python.parent.parent),
                       UV_PROJECT_ENVIRONMENT=str(self.python.parent.parent),
                       PATH=str(self.python.parent) + os.pathsep + env["PATH"])
            command = [str(self.python), "-m", "moonseg.evaluate_pretrained", "--config", str(config_path),
                "--data-root", str(fulltrain.local_path(PUBLIC_TEST["data_root"])),
                "--manifest", str(fulltrain.local_path(PUBLIC_TEST["manifest"])), "--profile", "single",
                "--output", str(directory / "scores.json")]
            self.save(status="public_test", current_job=job["name"])
            self.launch(key, command, directory / "evaluation.log", env, locks)
            self.wait_stage(key)
        result = self.verify_tests(job)
        self.state["jobs"][job["name"]] = dict(status="completed", results=result, finished_at=fulltrain.now())
        self.save()


def main():
    os.chdir(ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    path = fulltrain.local_path(args.plan)
    if not path.is_relative_to(fulltrain.local_path("runs/plans")):
        raise ValueError("Keep the pretrained inference plan and durable state under runs/plans")
    plan = fulltrain.read_json(path)
    contents = validate_plan(plan)
    if args.dry_run:
        ready, message = predecessor_ready(plan)
        print(json.dumps(dict(plan=plan, ready=ready, dependency=message,
            python=str(plan_python(plan)), reserved_gpus=[2], profile="single", checkpoint_policy="pretrained_only",
            state_dir=str(path.with_name(path.stem + ".state")), expected_public_samples=5004), indent=2))
        return 0
    return PretrainedLane(path, plan, contents).run()


if __name__ == "__main__":
    raise SystemExit(main())
