#!/usr/bin/env python3
"""Run one approved full-training lane: predecessor -> train -> last tests.

Legacy frozen lanes use 0/7 (10 then 25 epochs) or 2/3 (15 then 20 epochs).
Explicit joint_fullft lanes use 0/7 or 4/5 respectively and may explicitly
declare predecessor: null to start as soon as their GPUs and locks are free.
Each training starts independently. A durable stage worker retains shared GPU
locks and its actual child exit receipt if this supervisor exits. Restart the
same --plan to adopt successful/in-progress stages; failures are never retried.
--dry-run is CPU/read-only and does not query or initialize CUDA.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
STORAGE = ROOT.parent
PYTHON = (ROOT / ".venv").resolve() / "bin/python"
sys.path.insert(0, str(ROOT))
from moonseg.evaluate_public_test import PROFILES, PublicTestResearchDataset, fulltrain_data_snapshot
from moonseg.train import resolve_train_settings
from moonseg.utils import load_config
from scripts.run_public_test_queue import (checkpoint_runs, prepare_checkpoint,
    score_result, validate_config as validate_public_config)
from scripts.run_training_queue import atomic_json, gpu_occupancy, now, process_identity, same_process


def local_path(value):
    value = (ROOT / value).resolve()
    if not value.is_relative_to(STORAGE) or "/home/nas_1/" in str(value):
        raise ValueError(f"All full-training artifacts must resolve under {STORAGE}: {value}")
    return value


def read_json(value):
    return json.loads(Path(value).read_text())


def live_stage(spec):
    """Cover the Popen-to-parent-status window using the unique stage spec path."""
    found = []
    for directory in Path("/proc").glob("[0-9]*"):
        try:
            argv = (directory / "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if b"--stage-spec" in argv and os.fsencode(spec) in argv:
            identity = process_identity(int(directory.name))
            if identity and identity["state"] not in ("Z", "X"):
                found.append({"pid": int(directory.name), **identity})
    return found


def environment(gpus, uuids, *, evaluation=False):
    env = os.environ.copy()
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
                "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE", "TORCHELASTIC_RUN_ID", "PYTHONHOME"):
        env.pop(key, None)
    # Discard inherited NAS checkout/environment/cache paths from old launchers.
    for key, value in list(env.items()):
        if "/home/nas_1/" in value:
            env.pop(key, None)
    env.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=",".join(uuids[str(g)] for g in gpus),
               PYTHON=str(PYTHON), VIRTUAL_ENV=str(PYTHON.parent.parent), PYTHONPATH=str(ROOT),
               PATH=str(PYTHON.parent) + os.pathsep + env.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", OMP_NUM_THREADS="4",
               PYTHONUNBUFFERED="1", TORCH_EXTENSIONS_DIR=str(STORAGE / "build/torch_extensions"),
               MOONSEG_MSDA_DIR=str((ROOT / "build/ms_deform_attn").resolve()),
               XDG_CACHE_HOME=str(STORAGE / ".cache"), HF_HOME=str(STORAGE / ".cache/huggingface"),
               TORCH_HOME=str(STORAGE / ".cache/torch"), UV_PROJECT_ENVIRONMENT=str(PYTHON.parent.parent))
    if evaluation or gpus in ([2, 3], [4, 5]):
        env["NCCL_P2P_DISABLE"] = "1"
    else:
        env.pop("NCCL_P2P_DISABLE", None)
    return env


def validate_plan(plan):
    joint = plan.get("recipe") == "joint_fullft"
    if "recipe" in plan and not joint:
        raise ValueError("Only the explicit joint_fullft recipe or a legacy plan without recipe is admitted")
    pairs = ([0, 7], [4, 5]) if joint else ([0, 7], [2, 3])
    if (plan.get("version") != 1 or plan.get("gpus") not in pairs
            or not isinstance(plan.get("id"), str) or not plan["id"]):
        raise ValueError(f"Expected one version-1 {'joint' if joint else 'frozen'} full-training lane on {pairs}")
    gpus, uuids = plan["gpus"], plan["expected_gpu_uuids"]
    if (set(uuids) != {str(gpu) for gpu in gpus} or len(set(uuids.values())) != 2
            or any(not isinstance(value, str) or not value.startswith("GPU-") for value in uuids.values())):
        raise ValueError("Freeze one distinct physical GPU UUID for each lane GPU")
    budgets = [10, 25] if gpus == [0, 7] else [15, 20]
    if [job.get("expected_epochs") for job in plan["jobs"]] != budgets:
        raise ValueError(f"The approved sequential epoch budgets for {gpus} are {budgets}")
    if type(plan.get("poll_seconds")) is not int or not 1 <= plan["poll_seconds"] <= 60:
        raise ValueError("poll_seconds must be an integer from 1 to 60")
    if "predecessor" not in plan or (plan["predecessor"] is None and not joint):
        raise ValueError("A predecessor is required; only joint_fullft permits explicit predecessor: null")
    dep = plan["predecessor"]
    if dep is not None:
        local_path(dep["status_path"])
        if any(type(dep.get(key)) is not int or dep[key] <= 0 for key in ("parent_pid", "parent_start_ticks")):
            raise ValueError("Freeze the predecessor public queue PID/start_ticks")
        if (not dep.get("expected_job_keys") or len(set(dep["expected_job_keys"])) != len(dep["expected_job_keys"])):
            raise ValueError("Freeze every predecessor public-test job key exactly once")
    baseline_name = "fullft" if joint else "clip001"
    baseline = load_config(ROOT / f"configs/dnv3_va_m2f_vitl16_rrc_{baseline_name}.yaml")
    contents = {}
    for job in plan["jobs"]:
        epochs, name = job["expected_epochs"], job["name"]
        expected_name = f"dnv3_va_m2f_vitl16_fulltrain_{'joint' if joint else 'clip001'}_e{epochs}"
        source = local_path(job["config"])
        content, config = source.read_bytes(), load_config(source)
        expected = json.loads(json.dumps(baseline))
        expected["data"].update(train_manifest="data/manifests/dev_final_train.json", val_manifest=None,
            train_scenes=[1, 3, 5, 6, 7, 8], val_scenes=[], label_stats="data/manifests/label_stats_dev_final_train.json")
        expected["train"].update(epochs=epochs, output_dir=f"runs/{expected_name}", validate=False)
        if (name != expected_name or source.stem != name or config != expected
                or job.get("expected_updates") != epochs * 125):
            raise ValueError(f"{name}: config must preserve the approved independent "
                             f"{'joint fullft' if joint else 'frozen clip.01'} recipe/budget")
        output = local_path(config["train"]["output_dir"])
        if output.parent != local_path("runs") or output.name != name:
            raise ValueError("Training outputs must be matching direct children of runs")
        for key in ("root", "train_manifest", "label_stats"):
            local_path(config["data"][key])
        local_path(config["model"]["backbone_weights"])
        fulltrain_data_snapshot(config)
        contents[name] = content
    test = plan["public_test"]
    if test != {"data_root": "data/research_test/LuSNAR", "manifest": "data/research_test/manifest.json"}:
        raise ValueError("Use the fixed complete public scenes 2/4/9 research dataset")
    local_path(test["data_root"])
    local_path(test["manifest"])
    dataset = PublicTestResearchDataset(local_path(test["data_root"]), local_path(test["manifest"]), baseline["data"])
    for record in dataset.records:
        for field in ("image", "label"):
            local_path(dataset.root / record[field])
    local_path("build/ms_deform_attn")
    if not PYTHON.is_file() or not ROOT.is_relative_to(Path("/home/disk02")):
        raise ValueError("Run this scheduler from the physical disk02 code/environment")
    return contents


def predecessor_ready(plan):
    """Require the complete frozen queue, including both profiles of every selection."""
    if "predecessor" not in plan:
        raise ValueError("A predecessor must be declared explicitly")
    dep = plan["predecessor"]
    if dep is None:
        if plan.get("recipe") != "joint_fullft":
            raise ValueError("Only joint_fullft permits explicit predecessor: null")
        return True, "explicit joint_fullft lane without predecessor; shared locks and GPU idleness still required"
    state = read_json(local_path(dep["status_path"]))
    if (state.get("parent_pid"), state.get("parent_start_ticks")) != (dep["parent_pid"], dep["parent_start_ticks"]):
        raise ValueError("Predecessor public queue identity changed; explicitly review/freeze the replacement")
    config, jobs = state["config"], state["jobs"]
    validate_public_config(config)
    if config["gpus"] != plan["gpus"] or set(jobs) != set(dep["expected_job_keys"]):
        raise ValueError("Predecessor queue GPU assignment or complete job membership changed")
    if state["status"] in {"failed", "cancelled", "stopped", "blocked"} or any(
            job["status"] in {"failed", "cancelled", "blocked"} for job in jobs.values()):
        raise ValueError("Predecessor queue contains an unsuccessful evaluation; automatic continuation refused")
    if state["status"] != "completed" or any(job["status"] != "completed" for job in jobs.values()):
        if not same_process({"pid": dep["parent_pid"], "start_ticks": dep["parent_start_ticks"]}):
            raise ValueError("Predecessor parent exited before its complete public-test queue finished")
        return False, "waiting for the complete predecessor public-test queue"
    if any(job.get("pid") and same_process(job) for job in jobs.values()):
        return False, "waiting for predecessor evaluator processes to exit"
    prepared = {}
    selected = {f"{run['name']}/{Path(run['checkpoint']).stem}": run
                for original in config["runs"] for run in checkpoint_runs(original, config)}
    for key, job in jobs.items():
        if job.get("exit_code") not in (None, 0):
            raise ValueError(f"Predecessor evaluator failed: {key}")
        prefix = key.rsplit("/", 1)[0]
        run = selected[prefix]
        if prefix not in prepared:
            prepared[prefix] = prepare_checkpoint(run, run["checkpoint_policy"])[2]
        meta = prepared[prefix]
        if any(job.get(field) != meta[field] for field in ("epoch", "global_step", "file_identity")):
            raise ValueError(f"Predecessor score/checkpoint metadata mismatch: {key}")
        score_result(local_path(job["score_path"]), job, config)
    return True, "all predecessor public-test scores and checkpoint metadata verified"


def stage_worker(spec_path, fds):
    """Keep lock FDs and a durable exit code even when the lane parent disappears."""
    spec_path = local_path(spec_path)
    spec = read_json(spec_path)
    receipt_path = local_path(spec["receipt_path"])
    if receipt_path.exists():
        raise ValueError("Stage receipt already exists; stages are never executed twice")
    record = dict(status="running", pid=os.getpid(), **process_identity(os.getpid()),
                  spec_path=str(spec_path), started_at=now())
    atomic_json(receipt_path, record)
    try:
        with local_path(spec["log_path"]).open("x") as log:
            child = subprocess.Popen(spec["command"], cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, close_fds=True, pass_fds=tuple(fds))
            record.update(child_pid=child.pid, child_start_ticks=process_identity(child.pid)["start_ticks"])
            atomic_json(receipt_path, record)
            code = child.wait()
        record.update(status="completed" if code == 0 else "failed", exit_code=code, finished_at=now())
    except BaseException as error:
        record.update(status="failed", error=str(error), exit_code=1, finished_at=now())
        atomic_json(receipt_path, record)
        raise
    atomic_json(receipt_path, record)
    return code


class Lane:
    def __init__(self, plan_path, plan, contents):
        self.plan_path, self.plan, self.contents = plan_path, plan, contents
        self.directory = plan_path.with_name(plan_path.stem + ".state")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.parent_lock = (self.directory / "supervisor.lock").open("a+")
        try:
            fcntl.flock(self.parent_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("This lane already has an active supervisor; do not launch another") from None
        self.status_path = self.directory / "status.json"
        self.children = {}
        if self.status_path.exists():
            self.state = read_json(self.status_path)
            if self.state["plan"] != plan:
                raise ValueError("Saved lane plan differs; never modify/reuse an existing lane state directory")
            if self.state["status"] == "failed":
                raise ValueError("Lane previously failed. Inspect its stage receipt/log; automatic retry is disabled")
        else:
            for job in plan["jobs"]:
                output = local_path(f"runs/{job['name']}")
                if output.exists() and any(output.iterdir()):
                    raise ValueError(f"Refusing untracked existing training output: {output}")
            self.state = dict(version=1, plan=plan, plan_path=str(plan_path), status="starting",
                              created_at=now(), stages={}, jobs={})
        self.state.update(parent_pid=os.getpid(), parent_start_ticks=process_identity(os.getpid())["start_ticks"])
        snapshots = self.directory / "configs"
        snapshots.mkdir(exist_ok=True)
        for job in plan["jobs"]:
            snapshot = snapshots / Path(job["config"]).name
            content = contents[job["name"]]
            if snapshot.exists():
                if snapshot.read_bytes() != content:
                    raise ValueError(f"Configuration differs from its startup snapshot: {snapshot}")
            else:
                with snapshot.open("xb") as stream:
                    stream.write(content)
                    stream.flush()
                    os.fsync(stream.fileno())
        self.save()

    def save(self, **changes):
        self.state.update(**changes, updated_at=now())
        atomic_json(self.status_path, self.state)

    def wait_stage(self, key):
        stage = self.state["stages"][key]
        spec_path, receipt_path = Path(stage["spec_path"]), Path(stage["receipt_path"])
        spec = read_json(spec_path)
        if spec != {field: stage[field] for field in ("command", "receipt_path", "log_path")}:
            raise ValueError(f"Saved stage specification changed: {spec_path}")
        while True:
            receipt = read_json(receipt_path) if receipt_path.exists() else None
            alive = live_stage(spec_path)
            child = self.children.get(key)
            if child is not None:
                child.poll()  # Reap our own wrapper; adoption uses the durable receipt instead.
            if receipt and (receipt.get("spec_path") != str(spec_path)
                    or (stage.get("pid") is not None and
                        (receipt.get("pid"), receipt.get("start_ticks")) != (stage["pid"], stage["start_ticks"]))):
                raise ValueError(f"Stage worker receipt identity changed: {receipt_path}")
            if receipt and receipt["status"] in {"completed", "failed"}:
                if alive:
                    time.sleep(1)
                    continue
                if receipt["status"] != "completed" or receipt.get("exit_code") != 0:
                    raise ValueError(f"Stage {key} failed; inspect {receipt_path} and {stage['log_path']}")
                stage.update(status="completed", receipt=receipt, exit_code=0, finished_at=now())
                self.save()
                return
            if not alive:
                raise ValueError(f"Stage {key} has no live worker/completion receipt. Inspect its log and child "
                                 "PID in the receipt; this ambiguous attempt will never be launched again")
            if receipt:
                stage.update(status="running", pid=receipt["pid"], start_ticks=receipt["start_ticks"],
                             child_pid=receipt.get("child_pid"), child_start_ticks=receipt.get("child_start_ticks"))
                self.save()
            time.sleep(self.plan["poll_seconds"])

    def launch(self, key, command, log_path, env, locks):
        if key in self.state["stages"]:
            raise ValueError(f"Refusing a second execution of stage {key}")
        directory = self.directory / "stages" / key
        directory.mkdir(parents=True, exist_ok=False)
        spec_path, receipt_path = directory / "spec.json", directory / "receipt.json"
        spec = dict(command=command, receipt_path=str(receipt_path), log_path=str(log_path))
        atomic_json(spec_path, spec)
        stage = dict(status="launching", spec_path=str(spec_path), **spec, started_at=now(),
                     environment={key: env.get(key) for key in ("CUDA_VISIBLE_DEVICES", "NCCL_P2P_DISABLE",
                        "PYTHON", "VIRTUAL_ENV", "PYTHONPATH", "TORCH_EXTENSIONS_DIR", "XDG_CACHE_HOME",
                        "PYTORCH_CUDA_ALLOC_CONF", "OMP_NUM_THREADS")})
        self.state["stages"][key] = stage
        self.save()
        fds = tuple(lock.fileno() for lock in locks)
        child = subprocess.Popen([str(PYTHON), str(Path(__file__).resolve()), "--stage-spec", str(spec_path),
            "--inherited-fds", ",".join(map(str, fds))], cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
            start_new_session=True, close_fds=True, pass_fds=fds)
        self.children[key] = child
        stage.update(status="running", pid=child.pid, start_ticks=process_identity(child.pid)["start_ticks"])
        self.save()

    def checkpoint(self, job):
        original = load_config(self.directory / "configs" / Path(job["config"]).name)
        resolve_train_settings(original["train"])
        directory = local_path(f"runs/{job['name']}")
        saved = read_json(directory / "resolved_config.json")
        if {key: saved.get(key) for key in original} != original:
            raise ValueError(f"{job['name']}: resolved training configuration differs from startup snapshot")
        selected = {**job, "checkpoint": "last.pt"}
        config_path, path, metadata = prepare_checkpoint(selected, "final_epoch")
        if metadata["global_step"] != job["expected_updates"]:
            raise ValueError("Final checkpoint update budget differs from the approved plan")
        return config_path, path, metadata

    def evaluate_records(self, job, checkpoint, meta):
        result = {}
        for profile in PROFILES:
            directory = self.directory / "scores" / job["name"] / "last" / profile
            result[profile] = dict(profile=profile, checkpoint_path=str(checkpoint), **meta,
                                  checkpoint_policy="final_epoch", score_path=str(directory / "scores.json"),
                                  log_path=str(directory / "evaluation.log"))
        return result

    def verify_tests(self, job):
        _, checkpoint, meta = self.checkpoint(job)
        results = {}
        for profile, record in self.evaluate_records(job, checkpoint, meta).items():
            key = f"{job['name']}_{profile}"
            if self.state["stages"].get(key, {}).get("status") != "completed":
                raise ValueError(f"Missing complete evaluator stage receipt: {key}")
            results[profile] = score_result(Path(record["score_path"]), record, self.plan["public_test"])
        return results

    def run(self):
        try:
            # Orphan workers retain the GPU locks. Adopt/wait before trying to
            # reacquire those locks; otherwise a restart would deadlock itself.
            for key in list(self.state["stages"]):
                self.wait_stage(key)
            if self.state["status"] == "completed":
                for job in self.plan["jobs"]:
                    self.verify_tests(job)
                self.complete_marker()
                return 0
            self.save(status="waiting_dependency")
            while True:
                ready, message = predecessor_ready(self.plan)
                self.save(dependency=message)
                if ready:
                    break
                time.sleep(self.plan["poll_seconds"])
            with ExitStack() as stack:
                locks = []
                self.save(status="waiting_gpu_locks")
                for gpu in sorted(self.plan["gpus"]):
                    for filename in (f"runs/.training_gpu_{gpu}.lock", f"runs/evaluations_2026-10-23/gpu_{gpu}.lock"):
                        lock = stack.enter_context(local_path(filename).open("a+"))
                        while True:
                            try:
                                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                                break
                            except BlockingIOError:
                                time.sleep(self.plan["poll_seconds"])
                        locks.append(lock)
                if not predecessor_ready(self.plan)[0]:
                    raise ValueError("Predecessor changed while acquiring GPU locks")
                for job in self.plan["jobs"]:
                    self.execute_job(job, locks)
            self.save(status="completed", finished_at=now())
            self.complete_marker()
            return 0
        except Exception as error:
            self.save(status="failed", error=str(error), finished_at=now())
            raise

    def complete_marker(self):
        marker = self.directory / "completed.json"
        value = dict(status="completed", exit_code=0, plan_path=str(self.plan_path),
                     jobs=list(self.state["jobs"]), finished_at=self.state["finished_at"])
        if marker.exists():
            if read_json(marker) != value:
                raise ValueError("Lane completion marker differs from verified state")
        else:
            atomic_json(marker, value)

    def wait_idle(self):
        self.save(status="waiting_gpus")
        while True:
            snapshot = gpu_occupancy(self.plan["gpus"])
            if snapshot["gpu_uuids"] != self.plan["expected_gpu_uuids"]:
                raise ValueError("Physical GPU UUID mapping differs from the frozen lane")
            self.save(gpu_snapshot=snapshot)
            if not snapshot["occupied"]:
                return
            time.sleep(self.plan["poll_seconds"])

    def execute_job(self, job, locks):
        name, key = job["name"], job["name"] + "_train"
        if self.state["jobs"].get(name, {}).get("status") == "completed":
            self.verify_tests(job)
            return
        self.save(current_job=name)
        output = local_path(f"runs/{name}")
        if key not in self.state["stages"]:
            self.wait_idle()
            if output.exists() and any(output.iterdir()):
                raise ValueError(f"Refusing untracked/nonempty training output: {output}")
            source = local_path(job["config"])
            frozen = self.directory / "configs" / source.name
            if source.read_bytes() != frozen.read_bytes():
                raise ValueError("Source config changed after lane startup")
            output.mkdir(exist_ok=True)
            command = [str(PYTHON), "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
                       "-m", "moonseg.train", "--config", str(frozen)]
            self.save(status="training")
            self.launch(key, command, output / "train.log", environment(self.plan["gpus"],
                        self.plan["expected_gpu_uuids"]), locks)
            self.wait_stage(key)
        config_path, checkpoint, meta = self.checkpoint(job)
        marker = output / "queue_completed.json"
        completed = dict(status="completed", exit_code=0, name=name, run_dir=str(output),
            config=str(self.directory / "configs" / Path(job["config"]).name),
            expected_epochs=job["expected_epochs"], expected_updates=job["expected_updates"],
            **{field: self.state["stages"][key]["receipt"].get(field) for field in
               ("child_pid", "child_start_ticks")}, finished_at=now())
        completed.update(pid=completed.pop("child_pid"), start_ticks=completed.pop("child_start_ticks"))
        if not marker.exists():
            atomic_json(marker, completed)
        else:
            prior = read_json(marker)
            if any(prior.get(field) != completed[field] for field in completed if field != "finished_at"):
                raise ValueError("Training completion marker differs from the verified training receipt")
        records = self.evaluate_records(job, checkpoint, meta)
        pending = [profile for profile in PROFILES if f"{name}_{profile}" not in self.state["stages"]]
        if pending:
            self.wait_idle()
        self.save(status="public_test")
        # Both independent profiles launch before waiting: exactly one task/GPU.
        for gpu, profile in zip(self.plan["gpus"], PROFILES):
            if profile not in pending:
                continue
            record = records[profile]
            directory = Path(record["score_path"]).parent
            if directory.exists() and any(directory.iterdir()):
                raise ValueError(f"Refusing an untracked prior public-test attempt: {directory}")
            directory.mkdir(parents=True, exist_ok=True)
            command = [str(PYTHON), "-m", "moonseg.evaluate_public_test", "--config", str(config_path),
                "--checkpoint", str(checkpoint), "--weights", "model", "--checkpoint-policy", "final_epoch",
                "--data-root", str(local_path(self.plan["public_test"]["data_root"])),
                "--manifest", str(local_path(self.plan["public_test"]["manifest"])), "--profile", profile,
                "--output", record["score_path"]]
            self.launch(f"{name}_{profile}", command, Path(record["log_path"]),
                        environment([gpu], self.plan["expected_gpu_uuids"], evaluation=True), locks)
        for profile in PROFILES:
            self.wait_stage(f"{name}_{profile}")
        results = self.verify_tests(job)
        self.state["jobs"][name] = dict(status="completed", expected_epochs=job["expected_epochs"],
                                       expected_updates=job["expected_updates"], results=results, finished_at=now())
        self.save()


def main():
    os.chdir(ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stage-spec", help=argparse.SUPPRESS)
    parser.add_argument("--inherited-fds", default="", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.stage_spec:
        if args.plan or args.dry_run:
            raise ValueError("Internal worker mode cannot be combined with a lane launch")
        return stage_worker(args.stage_spec, [int(value) for value in args.inherited_fds.split(",") if value])
    if not args.plan:
        parser.error("--plan is required")
    plan_path = local_path(args.plan)
    if not plan_path.is_relative_to(local_path("runs/plans")):
        raise ValueError("Keep full-training plans and durable state under runs/plans")
    plan = read_json(plan_path)
    contents = validate_plan(plan)
    if args.dry_run:
        ready, message = predecessor_ready(plan)
        print(json.dumps(dict(plan=plan, ready=ready, dependency=message,
            state_dir=str(plan_path.with_name(plan_path.stem + ".state")), python=str(PYTHON), cwd=str(ROOT),
            training_environment={key: value for key, value in environment(plan["gpus"], plan["expected_gpu_uuids"]).items()
                if key in ("CUDA_VISIBLE_DEVICES", "NCCL_P2P_DISABLE", "PYTHON", "PYTHONPATH", "TORCH_EXTENSIONS_DIR")},
            test_profiles=PROFILES, checkpoint_policy="final_epoch", expected_public_samples=5004), indent=2))
        return 0
    return Lane(plan_path, plan, contents).run()


if __name__ == "__main__":
    raise SystemExit(main())
