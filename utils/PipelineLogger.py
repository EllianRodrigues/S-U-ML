"""
Pipeline execution logger for capturing metrics and exporting to CSV.
"""

import csv
import json
import os
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from utils.SystemMonitor import SystemMonitor


class PipelineLogger:
    """Log pipeline execution metrics and export to CSV."""

    stage_order = [
        "use_case_extraction",
        "uml_generation",
        "uml_to_text",
        "semantic_comparison",
    ]

    def __init__(self, runs_dir: str = "./runs", logs_dir: str = "./logs", verbose: bool = False):
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(exist_ok=True)
        
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(exist_ok=True)
        
        self.verbose = verbose
        self._console_stream = sys.stdout
        self._progress_width = 28
        
        self.system_monitor = SystemMonitor()
        self.hardware_info = self.system_monitor.get_hardware_info()
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = self.runs_dir / f"run_{self.timestamp}.csv"
        self.log_path = self.logs_dir / f"run_{self.timestamp}.log"
        
        # Pipeline metrics
        self.metrics = {
            "timestamp": self.timestamp,
            "start_time": datetime.now().isoformat(),
        }
        
        # Stage timings (will be filled during execution)
        self.stages = {}
        
        self._log(f"Pipeline started at {self.timestamp}")

    class _LogStream:
        def __init__(self, logger: "PipelineLogger"):
            self.logger = logger
            self._buffer = ""

        def write(self, text: str) -> int:
            if not text:
                return 0

            self._buffer += text
            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                self.logger._log(line.rstrip("\r"))

            return len(text)

        def flush(self) -> None:
            if self._buffer:
                self.logger._log(self._buffer.rstrip("\r"))
                self._buffer = ""

        def isatty(self) -> bool:
            return False

    @contextmanager
    def capture_output(self):
        """Redirect stdout and stderr to the run log file."""
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        stream = self._LogStream(self)

        sys.stdout = stream
        sys.stderr = stream
        try:
            yield
        finally:
            stream.flush()
            sys.stdout = original_stdout
            sys.stderr = original_stderr
        
    def load_prompts(self, prompts_dir: str = "./prompts") -> None:
        """Load all prompts from JSON files."""
        prompts_path = Path(prompts_dir)
        
        self.metrics["prompt_use_case"] = self._load_json_field(prompts_path / "use_case.json", "extract_use_case_prompt")
        self.metrics["prompt_uml"] = self._load_json_field(prompts_path / "UML.json", "prompt")
        self.metrics["prompt_uml_to_text"] = self._load_json_field(prompts_path / "uml_to_text.json", "prompt")

    def _load_json_field(self, path: Path, field: str) -> Optional[str]:
        """Load a specific field from a JSON file."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get(field, "")
        except Exception as e:
            return f"ERROR: {str(e)}"

    def set_config(self, config: Dict[str, Any]) -> None:
        """Set pipeline configuration (input text, models, etc)."""
        self.metrics.update(config)

    def start_stage(self, stage_name: str) -> None:
        """Mark the start of a pipeline stage."""
        self.system_monitor.clear_gpu_cache()
        self.system_monitor.start_monitoring()

        self._render_progress(stage_name, completed_stages=len(self.stages), stage_state="running")
        
        self.stages[stage_name] = {
            "start_time": datetime.now(),
            "start_resources": self.system_monitor.get_resource_snapshot(),
        }

    def end_stage(self, stage_name: str, success: bool = True, output: Optional[str] = None, error: Optional[str] = None) -> None:
        """Mark the end of a pipeline stage."""
        if stage_name not in self.stages:
            return

        end_time = datetime.now()
        end_resources = self.system_monitor.get_resource_snapshot()
        cpu_time = self.system_monitor.stop_monitoring()
        
        start_time = self.stages[stage_name]["start_time"]
        duration = (end_time - start_time).total_seconds()
        
        # Compute peak GPU memory during this stage
        peak_gpu_mb = end_resources.get("gpu_allocated_mb", 0)
        
        # Store stage metrics
        self.stages[stage_name].update({
            "duration_seconds": round(duration, 2),
            "peak_gpu_mb": peak_gpu_mb,
            "end_cpu_percent": end_resources.get("cpu_percent", 0),
            "status": "success" if success else "failed",
            "output_preview": (output[:100] + "...") if output and len(output) > 100 else output,
            "error": error,
        })

        completed_stages = len([stage for stage in self.stages.values() if stage.get("status") in {"success", "failed"}])
        self._render_progress(stage_name, completed_stages=completed_stages, stage_state="done")

    def set_final_comparison(self, semantic_score: Optional[float] = None, comparison_data: Optional[Dict] = None) -> None:
        """Set final semantic comparison results."""
        self.metrics["semantic_similarity_score"] = semantic_score
        if comparison_data:
            self.metrics.update(comparison_data)

    def export_csv(self) -> Path:
        """Export all metrics to a CSV file."""
        # Flatten stage metrics into top-level keys
        flat_metrics = dict(self.metrics)
        
        for stage_name, stage_data in self.stages.items():
            for key, value in stage_data.items():
                flat_key = f"{stage_name}_{key}"
                if isinstance(value, datetime):
                    flat_metrics[flat_key] = value.isoformat()
                else:
                    flat_metrics[flat_key] = value
        
        # Add hardware info
        flat_metrics.update(self.hardware_info)
        
        # Add end time and total duration
        flat_metrics["end_time"] = datetime.now().isoformat()
        start_dt = datetime.fromisoformat(flat_metrics["start_time"])
        end_dt = datetime.fromisoformat(flat_metrics["end_time"])
        flat_metrics["total_duration_seconds"] = round((end_dt - start_dt).total_seconds(), 2)
        
        # Write CSV
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(flat_metrics.keys()))
            writer.writeheader()
            writer.writerow(flat_metrics)
        
        return self.csv_path

    def _log(self, message: str) -> None:
        """Write message to log file."""
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(f"[{datetime.now().isoformat()}] {message}\n")
        except Exception:
            pass

    def _render_progress(self, stage_name: str, completed_stages: int, stage_state: str) -> None:
        """Render a compact progress bar in the terminal only."""
        total_stages = len(self.stage_order)
        completed_stages = max(0, min(completed_stages, total_stages))

        filled = int(self._progress_width * completed_stages / total_stages)
        bar = "█" * filled + "░" * (self._progress_width - filled)
        percent = int((completed_stages / total_stages) * 100)
        current_index = min(completed_stages + (1 if stage_state == "running" else 0), total_stages)
        label = f"{current_index}/{total_stages} {stage_name}"

        if stage_state == "done" and completed_stages == total_stages:
            line = f"\r[{bar}] {percent:3d}% | {label} | done\n"
        else:
            line = f"\r[{bar}] {percent:3d}% | {label}"

        try:
            self._console_stream.write(line)
            self._console_stream.flush()
        except Exception:
            pass
