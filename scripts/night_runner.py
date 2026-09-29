#!/usr/bin/env python
"""Night task runner: gate-driven overnight queue with monitoring.

Executes a declarative task plan (JSON) serially, in priority order, respecting
dependencies and decision gates. Built for unattended overnight runs on a
single-GPU (MPS) machine:

* task types: ``command`` (shell argv) and ``wait_file`` (poll for a file)
* gates: declarative conditions ``{file, key, op, value}`` evaluated against
  experiment-output JSON; a false gate resolves the task as ``skipped_gate``
  (that branch of the plan is simply not taken - never an error)
* both gate outcomes keep the queue moving: plan the pass branch and the fail
  branch as parallel task chains, each with its own gate
* monitoring: heartbeat file (30 s), full state file after every transition,
  ``--status`` for live inspection, and a closing markdown report with the
  completion rate
* dynamic adjustment: GPU-busy guard (defers GPU tasks while another training /
  evaluation process is alive), deadline budget (a task that cannot finish
  before the deadline is skipped, not started and killed), retries on failure,
  and restart-safe resume from the persisted state

Exit-code / status semantics (completion-rate accounting):

* ``done`` and ``skipped_gate`` count as resolved-successfully
* ``failed`` and ``skipped_missing_input`` count against the rate
* ``skipped_deadline`` is reported separately (planned deferral, not a failure)

Usage
-----
    python scripts/night_runner.py --plan configs/night_plan.json \
        --deadline "2026-09-28 06:00"          # run the night queue
    python scripts/night_runner.py --plan ... --dry-run
    python scripts/night_runner.py --plan ... --status
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NIGHT_DIR = REPO_ROOT / "outputs" / "night"
STATE_FILE = NIGHT_DIR / "runner_state.json"
HEARTBEAT_FILE = NIGHT_DIR / "heartbeat.json"
LOG_DIR = NIGHT_DIR / "logs"

RESOLVED_OK = {"done", "skipped_gate"}
RESOLVED_ALL = RESOLVED_OK | {"failed", "skipped_missing_input",
                              "skipped_deadline", "skipped_dep_failed"}

OPS = {
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}

GPU_PROCESS_PATTERN = re.compile(
    r"scripts/(train_segmentation|predict_test|evaluate_local)\.py"
)


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def dig(payload, dotted_key: str):
    """Walk a dotted key path into nested dicts/lists."""
    current = payload
    for part in dotted_key.split("."):
        if isinstance(current, list):
            try:
                current = current[int(part)]
                continue
            except (ValueError, IndexError):
                return None
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


class NightRunner:
    def __init__(self, plan: dict, deadline: datetime | None, dry_run: bool = False):
        self.tasks: dict[str, dict] = {t["id"]: dict(t) for t in plan["tasks"]}
        for task in self.tasks.values():
            task.setdefault("type", "command")
            task.setdefault("priority", 100)
            task.setdefault("depends_on", [])
            task.setdefault("gate", None)
            task.setdefault("gpu", False)
            task.setdefault("input_files", [])
            task.setdefault("est_seconds", 300)
            task.setdefault("timeout_s", max(600, task["est_seconds"] * 3))
            task.setdefault("retries", 1)
            task.setdefault("critical", False)
            task.setdefault("gate_timeout_s", 3600)
        self.plan_name = plan.get("name", "night")
        self.baseline_note = plan.get("baseline", "")
        self.deadline = deadline
        self.dry_run = dry_run
        self.state: dict[str, dict] = {}
        self.current_task: str | None = None
        self.stop_requested = False

    # ------------------------------------------------------------------ #
    # state persistence
    # ------------------------------------------------------------------ #
    def load_state(self) -> None:
        if STATE_FILE.is_file():
            saved = read_json(STATE_FILE) or {}
            for task_id, record in saved.get("tasks", {}).items():
                if task_id in self.tasks and record.get("status") in RESOLVED_ALL:
                    self.state[task_id] = record
            print(f"resumed state: {len(self.state)} task(s) already resolved")

    def save_state(self) -> None:
        NIGHT_DIR.mkdir(parents=True, exist_ok=True)
        target = (NIGHT_DIR / "runner_state_dryrun.json") if self.dry_run else STATE_FILE
        target.write_text(
            json.dumps(
                {
                    "plan": self.plan_name,
                    "updated_at": now_iso(),
                    "deadline": self.deadline.isoformat() if self.deadline else None,
                    "current_task": self.current_task,
                    "tasks": self.state,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def status_of(self, task_id: str) -> str:
        record = self.state.get(task_id)
        return record.get("status", "pending") if record else "pending"

    # ------------------------------------------------------------------ #
    # gates
    # ------------------------------------------------------------------ #
    def evaluate_gate(self, task: dict) -> tuple[bool, str]:
        """Return (passed, detail). Missing files -> not yet evaluable (False,
        with a 'pending-input' marker the scheduler understands)."""
        gate = task.get("gate")
        if not gate:
            return True, "no gate"
        details = []
        for condition in gate.get("all_of", []):
            path = REPO_ROOT / condition["file"]
            payload = read_json(path)
            if payload is None:
                return False, "pending-input"
            value = dig(payload, condition["key"])
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                return False, f"pending-input (key {condition['key']!r} not numeric)"
            op = OPS[condition["op"]]
            ok = op(float(value), float(condition["value"]))
            name = condition.get("name", f"{condition['file']}:{condition['key']}")
            details.append(f"{name} {value} {condition['op']} "
                           f"{condition['value']} -> {ok}")
            if not ok:
                return False, "; ".join(details)
        return True, "; ".join(details)

    # ------------------------------------------------------------------ #
    # resource & budget guards
    # ------------------------------------------------------------------ #
    def gpu_busy(self) -> bool:
        try:
            out = subprocess.run(
                ["ps", "-Ao", "command"], capture_output=True, text=True, timeout=15
            ).stdout
        except Exception:
            return False
        for line in out.splitlines():
            if GPU_PROCESS_PATTERN.search(line) and "night_runner" not in line:
                return True
        return False

    def fits_before_deadline(self, task: dict) -> tuple[bool, str]:
        if self.deadline is None or task["type"] == "wait_file":
            return True, ""
        remaining = (self.deadline - datetime.now()).total_seconds()
        needed = task["est_seconds"] + 600  # safety buffer
        if remaining < needed:
            return False, (f"needs ~{needed / 60:.0f} min, only "
                           f"{remaining / 60:.0f} min before deadline")
        return True, ""

    def inputs_ready(self, task: dict) -> bool:
        return all((REPO_ROOT / f).is_file() for f in task["input_files"])

    # ------------------------------------------------------------------ #
    # execution
    # ------------------------------------------------------------------ #
    def resolve(self, task_id: str, status: str, **extra) -> None:
        record = self.state.setdefault(task_id, {})
        record.update({"status": status, "ended_at": now_iso(), **extra})
        self.current_task = None
        self.save_state()

    def run_task(self, task: dict) -> None:
        task_id = task["id"]
        record = self.state.setdefault(task_id, {"status": "running",
                                                 "started_at": now_iso(),
                                                 "attempts": 0})
        record["status"] = "running"
        record["started_at"] = now_iso()
        self.current_task = task_id
        self.save_state()

        if task["type"] == "wait_file":
            self.run_wait_file(task, record)
            return

        env = {
            **{k: v for k, v in __import__("os").environ.items()},
            "PYTHONPATH": "src",
            "BERRY_CONFIG": "configs/default.yaml",
            "NIGHT_RUNNER_CHILD": "1",
        }
        log_path = LOG_DIR / f"{task_id}.log"
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        attempts = int(record.get("attempts", 0))
        max_attempts = 1 + max(0, int(task["retries"]))

        while attempts < max_attempts:
            attempts += 1
            record["attempts"] = attempts
            self.save_state()
            started = time.time()
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"\n===== attempt {attempts} start {now_iso()} =====\n")
                log.flush()
                try:
                    completed = subprocess.run(
                        task["command"],
                        cwd=REPO_ROOT,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=task["timeout_s"],
                    )
                    exit_code = completed.returncode
                except subprocess.TimeoutExpired:
                    log.write(f"\nTIMEOUT after {task['timeout_s']}s\n")
                    exit_code = 124
            record["duration_s"] = round(time.time() - started, 1)
            if exit_code == 0:
                self.resolve(task_id, "done", duration_s=record["duration_s"],
                             attempts=attempts)
                print(f"  [{task_id}] done in {record['duration_s']}s")
                return
            print(f"  [{task_id}] attempt {attempts} failed (exit {exit_code})")
        self.resolve(task_id, "failed", exit_code=exit_code,
                     duration_s=record.get("duration_s"), attempts=attempts,
                     log=str(log_path))

    def run_wait_file(self, task: dict, record: dict) -> None:
        target = REPO_ROOT / task["file"]
        started = time.time()
        while time.time() - started < task["timeout_s"]:
            if target.is_file():
                duration = round(time.time() - started, 1)
                self.resolve(task["id"], "done", duration_s=duration)
                print(f"  [{task['id']}] file appeared after {duration}s")
                return
            time.sleep(15)
        self.resolve(task["id"], "failed", reason=f"timeout waiting for {target}")

    # ------------------------------------------------------------------ #
    # scheduler
    # ------------------------------------------------------------------ #
    def next_task(self):
        """Pick the highest-priority runnable task, or None."""
        ready = []
        for task_id, task in self.tasks.items():
            if self.status_of(task_id) != "pending":
                continue
            deps = task["depends_on"]
            bad = [d for d in deps if self.status_of(d) in
                   {"failed", "skipped_missing_input"}]
            if bad:
                self.resolve(task_id, "skipped_dep_failed", reason=f"deps {bad}")
                continue
            deferred = [d for d in deps if self.status_of(d) == "skipped_deadline"]
            if deferred:
                self.resolve(task_id, "skipped_deadline", reason=f"deps deferred: {deferred}")
                continue
            gated_off = [d for d in deps if self.status_of(d) == "skipped_gate"]
            if gated_off:
                # An upstream gate decided its branch was not taken. Dependents
                # follow suit instead of blocking the queue forever: a
                # dependent whose own gate passes while its dependency was
                # skipped is a plan bug, so it resolves as failed.
                gate_ok, gate_detail = self.evaluate_gate(task)
                if task["gate"] and gate_ok:
                    self.resolve(task_id, "skipped_dep_failed",
                                 reason=f"deps skipped_gate but own gate true: {gated_off}")
                else:
                    self.resolve(task_id, "skipped_gate",
                                 reason=f"upstream gate not taken ({gated_off})")
                    print(f"  [{task_id}] skipped (upstream gate not taken)")
                continue
            if not all(self.status_of(d) in RESOLVED_OK for d in deps):
                continue
            if not self.inputs_ready(task):
                if self.dry_run:
                    pass  # dry-run: treat as ready so the plan fully unrolls
                else:
                    if (self.state.get(task_id, {}).get("first_seen")
                            and time.time() - self.state[task_id]["first_seen"]
                            > task["gate_timeout_s"]):
                        self.resolve(task_id, "skipped_missing_input",
                                     reason=f"inputs missing: {task['input_files']}")
                        continue
                    self.state.setdefault(task_id, {}).setdefault(
                        "first_seen", time.time())
                    continue
            gate_ok, gate_detail = self.evaluate_gate(task)
            if gate_detail == "pending-input" or gate_detail.startswith(
                    "pending-input"):
                if self.dry_run:
                    gate_ok, gate_detail = True, "dry-run: gate input absent"
                else:
                    self.state.setdefault(task_id, {}).setdefault(
                        "first_seen", time.time())
                    if (time.time() - self.state[task_id].setdefault(
                            "first_seen", time.time())) > task["gate_timeout_s"]:
                        self.resolve(task_id, "failed",
                                     reason=f"gate never became evaluable: {gate_detail}")
                    continue
            if not gate_ok:
                self.resolve(task_id, "skipped_gate", reason=gate_detail)
                print(f"  [{task_id}] skipped (gate): {gate_detail}")
                continue
            ready.append((task["priority"], task_id, task))
        if not ready:
            return None
        ready.sort(key=lambda item: (item[0], item[1]))
        return ready[0][2]

    def run(self) -> None:
        print(f"night runner start {now_iso()}  plan={self.plan_name}  "
              f"deadline={self.deadline}")
        self.save_state()
        while True:
            task = self.next_task()
            if task is None:
                if all(self.status_of(t) in RESOLVED_ALL for t in self.tasks):
                    break
                time.sleep(10)
                continue
            fits, why = self.fits_before_deadline(task)
            if not fits and not task["critical"]:
                self.resolve(task["id"], "skipped_deadline", reason=why)
                print(f"  [{task['id']}] skipped (deadline): {why}")
                continue
            if task["gpu"] and self.gpu_busy():
                print(f"  [{task['id']}] GPU busy, waiting 60s ...")
                time.sleep(60)
                continue
            if self.dry_run:
                print(f"  [dry-run] would run {task['id']} "
                      f"(priority {task['priority']}, est {task['est_seconds']}s)")
                self.resolve(task["id"], "done", dryrun=True)
                continue
            print(f"  [{task['id']}] start (priority {task['priority']})")
            self.run_task(task)
        self.write_report()

    # ------------------------------------------------------------------ #
    # reporting
    # ------------------------------------------------------------------ #
    def completion_stats(self) -> dict:
        statuses = {tid: self.status_of(tid) for tid in self.tasks}
        total = len(statuses)
        done = sum(1 for s in statuses.values() if s == "done")
        gated = sum(1 for s in statuses.values() if s == "skipped_gate")
        failed = sum(1 for s in statuses.values()
                     if s in {"failed", "skipped_missing_input",
                              "skipped_dep_failed"})
        deferred = sum(1 for s in statuses.values() if s == "skipped_deadline")
        denominator = total - deferred
        rate = (done + gated) / denominator if denominator else 1.0
        return {"total": total, "done": done, "skipped_gate": gated,
                "failed": failed, "skipped_deadline": deferred, "rate": round(rate, 3),
                "statuses": statuses}

    def write_report(self) -> None:
        stats = self.completion_stats()
        lines = [
            f"# 夜间任务报告 · {self.plan_name}",
            "",
            f"- 生成时间: {now_iso()}",
            f"- 基准: {self.baseline_note}",
            f"- 完成率: **{stats['rate'] * 100:.0f}%** "
            f"(done {stats['done']} + gate-not-taken {stats['skipped_gate']}"
            f" / 有效任务 {stats['total'] - stats['skipped_deadline']}"
            f", 失败 {stats['failed']}, 预算跳过 {stats['skipped_deadline']})",
            "",
            "| 任务 | 状态 | 耗时s | 说明 |",
            "|---|---|---|---|",
        ]
        for task_id, task in sorted(self.tasks.items(),
                                    key=lambda kv: kv[1]["priority"]):
            record = self.state.get(task_id, {})
            lines.append(
                f"| {task_id} | {record.get('status', 'pending')} "
                f"| {record.get('duration_s', '-')} "
                f"| {record.get('reason', record.get('gate_detail', ''))} |"
            )
        NIGHT_DIR.mkdir(parents=True, exist_ok=True)
        report_path = NIGHT_DIR / f"night_report_{datetime.now():%Y%m%d_%H%M}.md"
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"report written: {report_path}  completion rate "
              f"{stats['rate'] * 100:.0f}%")


def heartbeat_loop(runner: NightRunner) -> None:
    while not runner.stop_requested:
        stats = runner.completion_stats()
        HEARTBEAT_FILE.write_text(
            json.dumps(
                {
                    "timestamp": now_iso(),
                    "current_task": runner.current_task,
                    "resolved": stats["done"] + stats["skipped_gate"],
                    "failed": stats["failed"],
                    "remaining": stats["total"] - len(stats["statuses"]),
                    "deadline": runner.deadline.isoformat()
                    if runner.deadline else None,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        time.sleep(30)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=REPO_ROOT / "configs" / "night_plan.json")
    parser.add_argument("--deadline", default=None,
                        help='e.g. "2026-09-28 06:00"')
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true",
                        help="print current state and exit")
    parser.add_argument("--watch-until-deadline", action="store_true",
                        help="after the queue resolves, keep the heartbeat alive "
                             "until the deadline instead of exiting")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.status:
        state = read_json(STATE_FILE) or {}
        print(json.dumps(state, ensure_ascii=False, indent=2))
        heartbeat = read_json(HEARTBEAT_FILE)
        if heartbeat:
            print("\nheartbeat:", json.dumps(heartbeat, ensure_ascii=False))
        return 0

    plan = read_json(args.plan)
    if not plan:
        raise SystemExit(f"plan not found or invalid: {args.plan}")
    deadline = (datetime.strptime(args.deadline, "%Y-%m-%d %H:%M")
                if args.deadline else None)
    runner = NightRunner(plan, deadline, dry_run=args.dry_run)
    runner.load_state()
    threading.Thread(target=heartbeat_loop, args=(runner,), daemon=True).start()
    runner.run()
    if args.watch_until_deadline and deadline:
        while datetime.now() < deadline:
            time.sleep(30)
        runner.stop_requested = True
        print(f"deadline reached ({deadline}), runner exiting {now_iso()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
