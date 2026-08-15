"""GPU profiling and monitoring using NVML."""

import asyncio
import logging
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

import pynvml  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


class GPUStats:
    """Container for GPU statistics."""

    def __init__(self):
        self.device_id: int = 0
        self.memory_used_mb: float = 0.0
        self.memory_total_mb: float = 0.0
        self.memory_utilization: float = 0.0
        self.gpu_utilization: float = 0.0
        self.temperature_c: float = 0.0
        self.power_usage_w: float = 0.0
        self.power_limit_w: float = 0.0
        self.clock_sm_mhz: float = 0.0
        self.clock_memory_mhz: float = 0.0
        self.timestamp: datetime = datetime.now()

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "device_id": self.device_id,
            "memory_used_mb": self.memory_used_mb,
            "memory_total_mb": self.memory_total_mb,
            "memory_utilization": self.memory_utilization,
            "gpu_utilization": self.gpu_utilization,
            "temperature_c": self.temperature_c,
            "power_usage_w": self.power_usage_w,
            "power_limit_w": self.power_limit_w,
            "clock_sm_mhz": self.clock_sm_mhz,
            "clock_memory_mhz": self.clock_memory_mhz,
            "timestamp": self.timestamp.isoformat(),
        }


class GPUCollector:
    """Collect GPU metrics using NVML."""

    def __init__(self, device_ids: Optional[List[int]] = None):
        self.device_ids: List[int] = device_ids or []
        self.initialized: bool = False
        self.history: defaultdict[int, List[GPUStats]] = defaultdict(list)

    def initialize(self) -> None:
        """Initialize NVML."""
        if self.initialized:
            return

        try:
            pynvml.nvmlInit()
            self.initialized = True
            logger.info("NVML initialized successfully")

            device_count = pynvml.nvmlDeviceGetCount()
            logger.info(f"Available GPUs: {device_count}")

            if not self.device_ids:
                self.device_ids = list(range(device_count))
                logger.info(f"Monitoring all {device_count} GPUs")
            else:
                logger.info(f"Monitoring GPUs: {self.device_ids}")

            for device_id in self.device_ids:
                if device_id >= device_count:
                    logger.warning(f"GPU {device_id} not available (max: {device_count - 1})")

        except pynvml.NVMLError as e:
            logger.error(f"Failed to initialize NVML: {e}")
            raise

    def shutdown(self) -> None:
        """Shutdown NVML."""
        if self.initialized:
            pynvml.nvmlShutdown()
            self.initialized = False
            logger.info("NVML shutdown")

    def collect(self, device_id: int) -> GPUStats:
        """Collect metrics from a single GPU."""
        if not self.initialized:
            self.initialize()

        stats = GPUStats()
        stats.device_id = device_id

        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)

            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            stats.memory_used_mb = memory_info.used / (1024 * 1024)
            stats.memory_total_mb = memory_info.total / (1024 * 1024)
            stats.memory_utilization = memory_info.used / memory_info.total

            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            stats.gpu_utilization = utilization.gpu / 100.0

            stats.temperature_c = pynvml.nvmlDeviceGetTemperature(
                handle, pynvml.NVML_TEMPERATURE_GPU
            )

            try:
                power_usage = pynvml.nvmlDeviceGetPowerUsage(handle)
                stats.power_usage_w = power_usage / 1000.0

                power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
                stats.power_limit_w = power_limit / 1000.0
            except pynvml.NVMLError:
                pass

            try:
                stats.clock_sm_mhz = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
                stats.clock_memory_mhz = pynvml.nvmlDeviceGetClockInfo(
                    handle, pynvml.NVML_CLOCK_MEM
                )
            except pynvml.NVMLError:
                pass

            stats.timestamp = datetime.now()

        except pynvml.NVMLError as e:
            logger.error(f"Failed to collect metrics from GPU {device_id}: {e}")

        return stats

    def collect_all(self) -> List[GPUStats]:
        """Collect metrics from all monitored GPUs."""
        all_stats = []

        for device_id in self.device_ids:
            stats = self.collect(device_id)
            self.history[device_id].append(stats)
            all_stats.append(stats)

        return all_stats

    def get_aggregate_stats(self) -> Dict[str, float]:
        """Get aggregate metrics across all GPUs."""
        if not self.history:
            return {}

        total_memory_used = 0.0
        total_memory_total = 0.0
        total_gpu_util = 0.0
        max_temperature = 0.0
        total_power = 0.0

        device_count = len(self.device_ids)

        for device_id, history in self.history.items():
            if not history:
                continue

            latest = history[-1]
            total_memory_used += latest.memory_used_mb
            total_memory_total += latest.memory_total_mb
            total_gpu_util += latest.gpu_utilization
            max_temperature = max(max_temperature, latest.temperature_c)
            total_power += latest.power_usage_w

        if device_count > 0:
            return {
                "total_memory_used_mb": total_memory_used,
                "total_memory_total_mb": total_memory_total,
                "aggregate_memory_utilization": (
                    total_memory_used / total_memory_total if total_memory_total > 0 else 0.0
                ),
                "average_gpu_utilization": total_gpu_util / device_count,
                "max_temperature_c": max_temperature,
                "total_power_w": total_power,
                "device_count": device_count,
            }

        return {}

    def clear_history(self) -> None:
        """Clear historical metrics."""
        self.history.clear()


async def monitor_gpus(
    collector: GPUCollector,
    interval: float = 1.0,
    duration: Optional[float] = None,
) -> List[GPUStats]:
    """Monitor GPUs asynchronously."""
    stats_buffer = []
    start_time = asyncio.get_event_loop().time()

    try:
        while True:
            current_stats = collector.collect_all()
            stats_buffer.extend(current_stats)

            if duration is not None:
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed >= duration:
                    break

            await asyncio.sleep(interval)

    except asyncio.CancelledError:
        logger.info("GPU monitoring cancelled")

    return stats_buffer


def get_gpu_info(device_id: int = 0) -> Dict[str, str]:
    """Get GPU device information."""
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)

        name = pynvml.nvmlDeviceGetName(handle).decode("utf-8")
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        compute_capability = pynvml.nvmlDeviceGetCudaComputeCapability(handle)

        pynvml.nvmlShutdown()

        return {
            "name": name,
            "total_memory_gb": memory_info.total / (1024**3),
            "compute_capability": f"{compute_capability[0]}.{compute_capability[1]}",
        }

    except pynvml.NVMLError as e:
        logger.error(f"Failed to get GPU info: {e}")
        return {}
