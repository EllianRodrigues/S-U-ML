"""
System monitoring utilities for GPU, CPU, and memory tracking.
"""

import os
import psutil
import torch
import platform
from typing import Dict, Any, Optional

try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False


class SystemMonitor:
    """Monitor and collect system hardware and performance metrics."""

    def __init__(self):
        self.start_cpu_time = None
        self.process = psutil.Process(os.getpid())
        self._collect_hardware_info()

    def _collect_hardware_info(self) -> None:
        """Collect static hardware information once."""
        self.cuda_available = torch.cuda.is_available()
        self.cuda_version = torch.version.cuda if self.cuda_available else None
        
        # GPU info
        if self.cuda_available:
            self.gpu_name = torch.cuda.get_device_name(0)
            self.gpu_compute_capability = torch.cuda.get_device_capability(0)
            try:
                self.total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            except Exception:
                self.total_vram_gb = None
        else:
            self.gpu_name = "None"
            self.gpu_compute_capability = None
            self.total_vram_gb = None

        # CPU info
        self.cpu_count = psutil.cpu_count(logical=False)
        self.cpu_count_logical = psutil.cpu_count(logical=True)
        self.cpu_freq = psutil.cpu_freq().current if psutil.cpu_freq() else None
        
        # RAM info
        ram = psutil.virtual_memory()
        self.system_ram_gb = ram.total / (1024 ** 3)
        
        # System info
        self.platform = platform.platform()
        self.processor = platform.processor()

    def get_hardware_info(self) -> Dict[str, Any]:
        """Return collected hardware information."""
        return {
            "cuda_available": self.cuda_available,
            "cuda_version": self.cuda_version,
            "gpu_name": self.gpu_name,
            "gpu_compute_capability": str(self.gpu_compute_capability) if self.gpu_compute_capability else None,
            "total_vram_gb": round(self.total_vram_gb, 2) if self.total_vram_gb else None,
            "cpu_cores": self.cpu_count,
            "cpu_cores_logical": self.cpu_count_logical,
            "cpu_freq_mhz": round(self.cpu_freq, 2) if self.cpu_freq else None,
            "system_ram_gb": round(self.system_ram_gb, 2),
            "platform": self.platform,
            "processor": self.processor,
        }

    def start_monitoring(self) -> None:
        """Start monitoring CPU time for this process."""
        self.start_cpu_time = self.process.cpu_times().user + self.process.cpu_times().system

    def get_resource_snapshot(self) -> Dict[str, Any]:
        """Get current GPU/CPU/memory usage snapshot."""
        snapshot = {}

        # GPU memory (MB)
        if self.cuda_available:
            try:
                snapshot["gpu_allocated_mb"] = round(torch.cuda.memory_allocated(0) / (1024 ** 2), 2)
                snapshot["gpu_reserved_mb"] = round(torch.cuda.memory_reserved(0) / (1024 ** 2), 2)
                snapshot["gpu_peak_allocated_mb"] = round(torch.cuda.max_memory_allocated(0) / (1024 ** 2), 2)
                snapshot["gpu_peak_reserved_mb"] = round(torch.cuda.max_memory_reserved(0) / (1024 ** 2), 2)
                snapshot["gpu_free_mb"] = round((self.total_vram_gb * 1024) - snapshot["gpu_allocated_mb"], 2)
            except Exception:
                snapshot["gpu_allocated_mb"] = None
                snapshot["gpu_reserved_mb"] = None
                snapshot["gpu_peak_allocated_mb"] = None
                snapshot["gpu_peak_reserved_mb"] = None
                snapshot["gpu_free_mb"] = None
        else:
            snapshot["gpu_allocated_mb"] = 0
            snapshot["gpu_reserved_mb"] = 0
            snapshot["gpu_peak_allocated_mb"] = 0
            snapshot["gpu_peak_reserved_mb"] = 0
            snapshot["gpu_free_mb"] = 0

        # CPU usage (%)
        try:
            snapshot["cpu_percent"] = round(self.process.cpu_percent(interval=0.1), 2)
        except Exception:
            snapshot["cpu_percent"] = None

        # Process memory (MB)
        try:
            mem_info = self.process.memory_info()
            snapshot["process_rss_mb"] = round(mem_info.rss / (1024 ** 2), 2)
            snapshot["process_vms_mb"] = round(mem_info.vms / (1024 ** 2), 2)
        except Exception:
            snapshot["process_rss_mb"] = None
            snapshot["process_vms_mb"] = None

        # System memory (%)
        try:
            snapshot["system_memory_percent"] = round(psutil.virtual_memory().percent, 2)
        except Exception:
            snapshot["system_memory_percent"] = None

        return snapshot

    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return total CPU time used."""
        if self.start_cpu_time is None:
            return {}

        end_cpu_time = self.process.cpu_times().user + self.process.cpu_times().system
        cpu_time_used = end_cpu_time - self.start_cpu_time

        return {
            "cpu_time_used_seconds": round(cpu_time_used, 2),
        }

    def clear_gpu_cache(self) -> None:
        """Clear GPU cache to get accurate memory readings."""
        if self.cuda_available:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
