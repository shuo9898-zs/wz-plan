"""Low-overhead, timestamped runtime monitoring for CARLA--SUMO workers.

The monitor deliberately has no dependency on CARLA or SUMO.  It can keep
writing a useful record while an RPC call is blocked or a simulator has died.
The environment supplies cached actor counts and lifecycle state; this module
adds OS/process/GPU state and persists one JSON object per line.
"""
from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import subprocess
import threading
import time
from typing import Any, Mapping

try:  # psutil is installed in the training environment, but keep diagnostics optional.
    import psutil  # type: ignore
except ImportError:  # pragma: no cover - fallback for a minimal runtime
    psutil = None


def _timestamp() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


def _json_default(value: Any) -> str:
    return str(value)


def configure_process_logging(log_path: str) -> None:
    """Attach one timestamped file handler to the current Python process.

    SubprocVecEnv workers do not reliably inherit the trainer's logging
    configuration on Windows, so this must be called inside each worker.
    """
    root = logging.getLogger()
    target = os.path.abspath(log_path)
    for handler in root.handlers:
        if isinstance(handler, logging.FileHandler) and os.path.abspath(handler.baseFilename) == target:
            return

    Path(target).parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(target, encoding="utf-8")
    handler.setFormatter(logging.Formatter(
        "%(asctime)s.%(msecs)03d %(process)d %(levelname)s %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    ))
    root.addHandler(handler)
    root.setLevel(logging.INFO)


def close_process_logging(log_path: str) -> None:
    """Close the file handler owned by this process/path (important on Windows)."""
    root = logging.getLogger()
    target = os.path.abspath(log_path)
    for handler in list(root.handlers):
        if not isinstance(handler, logging.FileHandler):
            continue
        if os.path.abspath(handler.baseFilename) != target:
            continue
        root.removeHandler(handler)
        handler.close()


def _process_snapshot(pid: int | None) -> dict[str, Any] | None:
    if pid is None:
        return None
    if psutil is None:
        return {"pid": pid, "metrics_available": False}
    try:
        proc = psutil.Process(pid)
        mem = proc.memory_info()
        snapshot: dict[str, Any] = {
            "pid": pid,
            "name": proc.name(),
            "status": proc.status(),
            "cpu_percent": proc.cpu_percent(interval=None),
            "rss_bytes": mem.rss,
            "vms_bytes": mem.vms,
            "memory_percent": proc.memory_percent(),
            "threads": proc.num_threads(),
        }
        if hasattr(proc, "num_handles"):
            try:
                snapshot["handles"] = proc.num_handles()
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                snapshot["handles"] = None
        return snapshot
    except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess) as exc:
        return {"pid": pid, "unavailable": type(exc).__name__}


def pid_listening_on_port(port: int) -> int | None:
    """Best-effort PID lookup for an externally managed local service."""
    if psutil is None:
        return None
    try:
        for connection in psutil.net_connections(kind="inet"):
            if not connection.laddr or connection.laddr.port != port:
                continue
            if connection.status == psutil.CONN_LISTEN and connection.pid is not None:
                return connection.pid
    except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
        return None
    return None


def command_for_pid(pid: int | None) -> list[str] | None:
    """Return a restartable command line for a process when access permits."""
    if pid is None or psutil is None:
        return None
    try:
        command = psutil.Process(pid).cmdline()
        return [str(part) for part in command] if command else None
    except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess, OSError):
        return None


def process_identity_for_pid(pid: int | None) -> dict[str, Any] | None:
    """Return a PID-reuse-safe process identity when access permits.

    A PID alone is not a stable identity on Windows: it can be reassigned after
    the original CARLA server exits.  Restart code therefore matches both the
    process creation time and the complete command line captured while the
    worker's CARLA port was known to be healthy.
    """
    if pid is None or psutil is None:
        return None
    try:
        proc = psutil.Process(pid)
        command = [str(part) for part in proc.cmdline()]
        if not command:
            return None
        return {
            "pid": pid,
            "create_time": proc.create_time(),
            "command": command,
        }
    except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess, OSError):
        return None


def pid_exists(pid: int | None) -> bool | None:
    """Return whether a PID is live, or ``None`` if it cannot be checked."""
    if pid is None or psutil is None:
        return None
    try:
        return bool(psutil.pid_exists(pid))
    except (psutil.AccessDenied, OSError):
        return None


def terminate_pid(pid: int, timeout_s: float = 20.0) -> dict[str, Any]:
    """Terminate one explicitly selected PID and report its final state."""
    if psutil is None:
        return {"pid": pid, "terminated": False, "reason": "psutil_unavailable"}
    try:
        proc = psutil.Process(pid)
        proc.terminate()
        try:
            returncode = proc.wait(timeout=timeout_s)
            return {"pid": pid, "terminated": True, "returncode": returncode}
        except psutil.TimeoutExpired:
            proc.kill()
            returncode = proc.wait(timeout=timeout_s)
            return {"pid": pid, "terminated": True, "killed": True, "returncode": returncode}
    except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess, OSError) as exc:
        return {"pid": pid, "terminated": False, "reason": type(exc).__name__}


def _system_snapshot(include_gpu: bool) -> dict[str, Any]:
    if psutil is None:
        return {"metrics_available": False}

    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()
    result: dict[str, Any] = {
        "per_core_cpu_percent": psutil.cpu_percent(interval=None, percpu=True),
        "available_ram_bytes": vm.available,
        "ram_total_bytes": vm.total,
        "ram_percent": vm.percent,
        "pagefile": {
            "total_bytes": swap.total,
            "used_bytes": swap.used,
            "free_bytes": swap.free,
            "percent": swap.percent,
        },
    }
    if include_gpu:
        result["gpu"] = _gpu_snapshot()
    return result


def _gpu_snapshot() -> dict[str, Any]:
    """Collect NVIDIA-wide and Windows 3D/compute engine data when available."""
    result: dict[str, Any] = {"available": False, "gpus": []}
    query = (
        "index,name,utilization.gpu,utilization.memory,memory.used,"
        "memory.total,temperature.gpu"
    )
    try:
        completed = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if completed.returncode == 0:
            for line in completed.stdout.splitlines():
                fields = [field.strip() for field in line.split(",")]
                if len(fields) != 7:
                    continue
                result["gpus"].append({
                    "index": fields[0],
                    "name": fields[1],
                    "gpu_util_percent": fields[2],
                    "memory_util_percent": fields[3],
                    "vram_used_mib": fields[4],
                    "vram_total_mib": fields[5],
                    "temperature_c": fields[6],
                })
            result["available"] = bool(result["gpus"])
        else:
            result["nvidia_smi_returncode"] = completed.returncode
            result["nvidia_smi_stderr"] = completed.stderr.strip()[-500:]
    except (FileNotFoundError, subprocess.SubprocessError, OSError) as exc:
        result["reason"] = type(exc).__name__

    if os.name == "nt":
        result["windows_engine_utilization"] = _windows_gpu_engine_utilization()
    return result


def _windows_gpu_engine_utilization() -> dict[str, Any]:
    """Return aggregate Windows GPU Engine 3D and Compute percentages.

    NVIDIA's global utilization does not separate graphics from compute.  The
    Windows performance counters do, when the driver exposes them.  Failure is
    reported as unavailable rather than making a training worker fail.
    """
    command = (
        "Get-Counter '\\GPU Engine(*)\\Utilization Percentage' | "
        "Select-Object -ExpandProperty CounterSamples | "
        "Select-Object InstanceName,CookedValue | ConvertTo-Json -Compress"
    )
    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True,
            text=True,
            timeout=4,
            check=False,
        )
        if completed.returncode != 0 or not completed.stdout.strip():
            return {"available": False, "returncode": completed.returncode}
        rows = json.loads(completed.stdout)
        if isinstance(rows, dict):
            rows = [rows]
        result = {"available": True, "3d_percent": 0.0, "compute_percent": 0.0}
        for row in rows:
            name = str(row.get("InstanceName", "")).lower()
            value = float(row.get("CookedValue", 0.0) or 0.0)
            if "engtype_3d" in name:
                result["3d_percent"] += value
            elif "engtype_compute" in name:
                result["compute_percent"] += value
        result["3d_percent"] = round(min(result["3d_percent"], 100.0), 2)
        result["compute_percent"] = round(min(result["compute_percent"], 100.0), 2)
        return result
    except (json.JSONDecodeError, FileNotFoundError, subprocess.SubprocessError, OSError) as exc:
        return {"available": False, "reason": type(exc).__name__}


class RuntimeMonitor:
    """Threaded JSONL monitor shared by a single worker or trainer process."""

    def __init__(
        self,
        *,
        log_path: str,
        worker: int | str,
        scenario: str,
        interval_s: float = 5.0,
        include_gpu: bool = False,
    ) -> None:
        self._path = Path(log_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._path.open("a", encoding="utf-8", buffering=1)
        self._worker = worker
        self._scenario = scenario
        self._interval_s = interval_s
        self._include_gpu = include_gpu
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._state: dict[str, Any] = {
            "worker": worker,
            "scenario": scenario,
            "episode": 0,
            "environment_step": 0,
            "processes": {"python_pid": os.getpid()},
            "ports": {},
            "actor_counts": {"by_role_name": {}, "by_type_id": {}},
        }
        self._latencies: dict[str, dict[str, float | int]] = {}
        self._counters: dict[str, int] = {}
        self._events: deque[dict[str, Any]] = deque(maxlen=32)
        self._last_exception: dict[str, Any] | None = None
        self._last_traci_command: dict[str, Any] | None = None
        self._thread = threading.Thread(
            target=self._run,
            name=f"runtime-monitor-{worker}-{scenario}",
            daemon=True,
        )
        self._thread.start()
        self.record_event("monitor_started", monitor_path=str(self._path))
        self.emit("start")

    @property
    def path(self) -> str:
        return str(self._path)

    def update_state(self, **values: Any) -> None:
        with self._lock:
            self._state.update(values)

    def record_timing(self, name: str, seconds: float) -> None:
        milliseconds = max(0.0, seconds * 1000.0)
        with self._lock:
            stat = self._latencies.setdefault(name, {"count": 0, "last_ms": 0.0, "max_ms": 0.0, "total_ms": 0.0})
            stat["count"] = int(stat["count"]) + 1
            stat["last_ms"] = round(milliseconds, 3)
            stat["max_ms"] = round(max(float(stat["max_ms"]), milliseconds), 3)
            stat["total_ms"] = round(float(stat["total_ms"]) + milliseconds, 3)

    def increment(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + amount

    def record_event(self, event: str, **details: Any) -> None:
        item = {"timestamp": _timestamp(), "event": event, **details}
        with self._lock:
            self._events.append(item)

    def record_exception(self, where: str, exc: BaseException, **details: Any) -> None:
        item = {
            "timestamp": _timestamp(),
            "where": where,
            "type": type(exc).__name__,
            "message": str(exc),
            **details,
        }
        with self._lock:
            self._last_exception = item
            self._events.append({"event": "exception", **item})
            self._counters["exceptions"] = self._counters.get("exceptions", 0) + 1

    def set_last_traci_command(self, command: Mapping[str, Any]) -> None:
        with self._lock:
            self._last_traci_command = dict(command)

    def emit(self, reason: str = "manual") -> None:
        with self._lock:
            state = dict(self._state)
            latencies = {name: dict(value) for name, value in self._latencies.items()}
            counters = dict(self._counters)
            events = list(self._events)
            last_exception = dict(self._last_exception) if self._last_exception else None
            last_traci = dict(self._last_traci_command) if self._last_traci_command else None

        pids = state.get("processes", {})
        process_metrics = {
            "python": _process_snapshot(pids.get("python_pid")),
            "carla": _process_snapshot(pids.get("carla_pid")),
            "sumo": _process_snapshot(pids.get("sumo_pid")),
        }
        record = {
            "timestamp": _timestamp(),
            "record_type": "runtime_monitor",
            "reason": reason,
            **state,
            "process_metrics": process_metrics,
            "system": _system_snapshot(include_gpu=self._include_gpu),
            "latencies": latencies,
            "counters": counters,
            "recent_events": events,
            "last_exception": last_exception,
            "last_traci_command": last_traci,
        }
        try:
            with self._lock:
                self._file.write(json.dumps(record, default=_json_default, sort_keys=True) + "\n")
                self._file.flush()
        except Exception:  # Monitoring must never terminate a simulation worker.
            logging.getLogger(__name__).exception("Could not write runtime monitor record")

    def close(self) -> None:
        if self._stop.is_set():
            return
        self.record_event("monitor_stopped")
        self.emit("stop")
        self._stop.set()
        self._thread.join(timeout=2.0)
        with self._lock:
            try:
                self._file.close()
            except Exception:
                pass

    def _run(self) -> None:
        while not self._stop.wait(self._interval_s):
            self.emit("interval")
