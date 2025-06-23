"""
Performance Monitoring and Optimization
======================================

Advanced performance monitoring, profiling, and optimization
tools for ZoomCam system components.
"""

import functools
import gc
import logging
import threading
import time
import tracemalloc
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import cv2

    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False


@dataclass
class PerformanceSnapshot:
    """Performance metrics snapshot."""

    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    disk_io_read_mb: float = 0.0
    disk_io_write_mb: float = 0.0
    network_bytes_sent: float = 0.0
    network_bytes_recv: float = 0.0
    gpu_usage: float = 0.0
    gpu_memory_mb: float = 0.0
    process_count: int = 0
    thread_count: int = 0
    fps: float = 0.0
    frame_processing_time_ms: float = 0.0
    camera_count: int = 0
    active_cameras: int = 0


@dataclass
class ComponentMetrics:
    """Performance metrics for a specific component."""

    component_name: str
    operation_counts: Dict[str, int] = field(default_factory=dict)
    operation_times: Dict[str, deque] = field(default_factory=dict)
    error_counts: Dict[str, int] = field(default_factory=dict)
    memory_usage: deque = field(default_factory=lambda: deque(maxlen=100))
    cpu_usage: deque = field(default_factory=lambda: deque(maxlen=100))
    last_update: datetime = field(default_factory=datetime.now)


@dataclass
class OptimizationSuggestion:
    """Performance optimization suggestion."""

    component: str
    issue: str
    suggestion: str
    impact: str  # "low", "medium", "high"
    effort: str  # "low", "medium", "high"
    automated: bool = False
    priority: int = 1  # 1=high, 2=medium, 3=low


class PerformanceProfiler:
    """Advanced performance profiler for ZoomCam components."""

    def __init__(self, max_snapshots: int = 1000):
        self.max_snapshots = max_snapshots
        self.snapshots: deque = deque(maxlen=max_snapshots)
        self.component_metrics: Dict[str, ComponentMetrics] = {}
        self.profiling_active = False
        self.baseline_metrics: Optional[PerformanceSnapshot] = None

        # Memory tracking
        self.memory_tracking = False
        self.memory_snapshots: List[Any] = []

        # Threading
        self.lock = threading.RLock()
        self.background_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

        # Optimization tracking
        self.optimization_suggestions: List[OptimizationSuggestion] = []
        self.performance_alerts: List[Dict[str, Any]] = []

        # Initialize baseline
        if PSUTIL_AVAILABLE:
            self.baseline_metrics = self._capture_snapshot()

    def start_profiling(self, interval: float = 1.0):
        """Start continuous performance profiling."""
        if self.profiling_active:
            return

        self.profiling_active = True
        self.stop_event.clear()

        self.background_thread = threading.Thread(
            target=self._profiling_loop, args=(interval,), daemon=True
        )
        self.background_thread.start()

        logging.info(f"Performance profiling started (interval: {interval}s)")

    def stop_profiling(self):
        """Stop performance profiling."""
        if not self.profiling_active:
            return

        self.profiling_active = False
        self.stop_event.set()

        if self.background_thread:
            self.background_thread.join(timeout=5.0)

        logging.info("Performance profiling stopped")

    def _profiling_loop(self, interval: float):
        """Background profiling loop."""
        while not self.stop_event.is_set():
            try:
                snapshot = self._capture_snapshot()
                if snapshot:
                    with self.lock:
                        self.snapshots.append(snapshot)
                        self._analyze_performance(snapshot)

                self.stop_event.wait(interval)

            except Exception as e:
                logging.error(f"Profiling loop error: {e}")
                time.sleep(interval)

    def _capture_snapshot(self) -> Optional[PerformanceSnapshot]:
        """Capture current performance snapshot."""
        if not PSUTIL_AVAILABLE:
            return None

        try:
            # System metrics
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()

            # Process-specific metrics
            process = psutil.Process()
            process_memory = process.memory_info()

            # I/O metrics
            try:
                io_counters = psutil.disk_io_counters()
                disk_read = io_counters.read_bytes / (1024 * 1024) if io_counters else 0
                disk_write = (
                    io_counters.write_bytes / (1024 * 1024) if io_counters else 0
                )
            except Exception:
                disk_read = disk_write = 0

            # Network metrics
            try:
                net_io = psutil.net_io_counters()
                net_sent = net_io.bytes_sent / (1024 * 1024) if net_io else 0
                net_recv = net_io.bytes_recv / (1024 * 1024) if net_io else 0
            except Exception:
                net_sent = net_recv = 0

            # GPU metrics (if available)
            gpu_usage, gpu_memory = self._get_gpu_metrics()

            snapshot = PerformanceSnapshot(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_mb=process_memory.rss / (1024 * 1024),
                disk_io_read_mb=disk_read,
                disk_io_write_mb=disk_write,
                network_bytes_sent=net_sent,
                network_bytes_recv=net_recv,
                gpu_usage=gpu_usage,
                gpu_memory_mb=gpu_memory,
                process_count=len(psutil.pids()),
                thread_count=process.num_threads(),
            )

            return snapshot

        except Exception as e:
            logging.error(f"Failed to capture performance snapshot: {e}")
            return None

    def _get_gpu_metrics(self) -> Tuple[float, float]:
        """Get GPU usage and memory metrics."""
        try:
            import pynvml

            pynvml.nvmlInit()

            handle = pynvml.nvmlDeviceGetHandleByIndex(0)

            # GPU utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_usage = util.gpu

            # GPU memory
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_memory = mem_info.used / (1024 * 1024)  # Convert to MB

            return gpu_usage, gpu_memory

        except Exception:
            # Fallback to OpenCV if available
            try:
                if OPENCV_AVAILABLE and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                    # Basic CUDA device info (limited metrics)
                    return 0.0, 0.0  # Placeholder
            except Exception:
                pass

        return 0.0, 0.0

    def _analyze_performance(self, snapshot: PerformanceSnapshot):
        """Analyze performance and generate suggestions."""
        try:
            # Check for performance issues
            if snapshot.cpu_percent > 85:
                self._add_performance_alert(
                    "high_cpu",
                    f"High CPU usage: {snapshot.cpu_percent:.1f}%",
                    "warning",
                )

                if snapshot.cpu_percent > 95:
                    self._generate_cpu_optimization()

            if snapshot.memory_percent > 85:
                self._add_performance_alert(
                    "high_memory",
                    f"High memory usage: {snapshot.memory_percent:.1f}%",
                    "warning",
                )

                if snapshot.memory_percent > 95:
                    self._generate_memory_optimization()

            # Check for disk I/O issues
            if hasattr(self, "_last_snapshot") and self._last_snapshot:
                disk_write_rate = (
                    snapshot.disk_io_write_mb - self._last_snapshot.disk_io_write_mb
                )
                if disk_write_rate > 50:  # 50MB/s write rate
                    self._add_performance_alert(
                        "high_disk_io",
                        f"High disk write rate: {disk_write_rate:.1f} MB/s",
                        "info",
                    )

            self._last_snapshot = snapshot

        except Exception as e:
            logging.error(f"Performance analysis error: {e}")

    def _add_performance_alert(self, alert_type: str, message: str, severity: str):
        """Add performance alert."""
        alert = {
            "type": alert_type,
            "message": message,
            "severity": severity,
            "timestamp": datetime.now(),
            "count": 1,
        }

        # Check if we already have this alert recently
        for existing_alert in self.performance_alerts[-10:]:
            if (
                existing_alert["type"] == alert_type
                and (datetime.now() - existing_alert["timestamp"]).seconds < 60
            ):
                existing_alert["count"] += 1
                return

        self.performance_alerts.append(alert)

        # Keep only recent alerts
        if len(self.performance_alerts) > 100:
            self.performance_alerts = self.performance_alerts[-100:]

    def _generate_cpu_optimization(self):
        """Generate CPU optimization suggestions."""
        suggestion = OptimizationSuggestion(
            component="system",
            issue="High CPU usage detected",
            suggestion="Reduce camera resolution, lower frame rate, or disable non-essential features",
            impact="high",
            effort="medium",
            automated=True,
            priority=1,
        )
        self._add_optimization_suggestion(suggestion)

    def _generate_memory_optimization(self):
        """Generate memory optimization suggestions."""
        suggestion = OptimizationSuggestion(
            component="system",
            issue="High memory usage detected",
            suggestion="Clear caches, reduce buffer sizes, or restart components",
            impact="high",
            effort="low",
            automated=True,
            priority=1,
        )
        self._add_optimization_suggestion(suggestion)

    def _add_optimization_suggestion(self, suggestion: OptimizationSuggestion):
        """Add optimization suggestion."""
        # Avoid duplicate suggestions
        for existing in self.optimization_suggestions:
            if (
                existing.component == suggestion.component
                and existing.issue == suggestion.issue
            ):
                return

        self.optimization_suggestions.append(suggestion)

        # Keep only recent suggestions
        if len(self.optimization_suggestions) > 50:
            self.optimization_suggestions = self.optimization_suggestions[-50:]

    def record_component_operation(
        self, component: str, operation: str, duration: float, success: bool = True
    ):
        """Record component operation metrics."""
        with self.lock:
            if component not in self.component_metrics:
                self.component_metrics[component] = ComponentMetrics(component)

            metrics = self.component_metrics[component]

            # Update operation counts
            metrics.operation_counts[operation] = (
                metrics.operation_counts.get(operation, 0) + 1
            )

            # Update operation times
            if operation not in metrics.operation_times:
                metrics.operation_times[operation] = deque(maxlen=100)
            metrics.operation_times[operation].append(duration)

            # Update error counts
            if not success:
                error_key = f"{operation}_errors"
                metrics.error_counts[error_key] = (
                    metrics.error_counts.get(error_key, 0) + 1
                )

            metrics.last_update = datetime.now()

    def get_performance_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get performance summary for the last N hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        with self.lock:
            recent_snapshots = [s for s in self.snapshots if s.timestamp > cutoff_time]

        if not recent_snapshots:
            return {"error": "No performance data available"}

        # Calculate averages
        avg_cpu = sum(s.cpu_percent for s in recent_snapshots) / len(recent_snapshots)
        avg_memory = sum(s.memory_percent for s in recent_snapshots) / len(
            recent_snapshots
        )
        avg_memory_mb = sum(s.memory_mb for s in recent_snapshots) / len(
            recent_snapshots
        )

        # Get peak values
        peak_cpu = max(s.cpu_percent for s in recent_snapshots)
        peak_memory = max(s.memory_percent for s in recent_snapshots)

        # Recent alerts
        recent_alerts = [
            alert
            for alert in self.performance_alerts
            if alert["timestamp"] > cutoff_time
        ]

        return {
            "period_hours": hours,
            "snapshots_count": len(recent_snapshots),
            "averages": {
                "cpu_percent": avg_cpu,
                "memory_percent": avg_memory,
                "memory_mb": avg_memory_mb,
            },
            "peaks": {"cpu_percent": peak_cpu, "memory_percent": peak_memory},
            "current": recent_snapshots[-1] if recent_snapshots else None,
            "alerts_count": len(recent_alerts),
            "suggestions_count": len(self.optimization_suggestions),
            "components_monitored": len(self.component_metrics),
        }

    def get_component_metrics(self, component: str) -> Optional[Dict[str, Any]]:
        """Get metrics for specific component."""
        with self.lock:
            if component not in self.component_metrics:
                return None

            metrics = self.component_metrics[component]

            # Calculate operation statistics
            operation_stats = {}
            for operation, times in metrics.operation_times.items():
                if times:
                    operation_stats[operation] = {
                        "count": len(times),
                        "avg_time": sum(times) / len(times),
                        "min_time": min(times),
                        "max_time": max(times),
                        "total_calls": metrics.operation_counts.get(operation, 0),
                    }

            return {
                "component": component,
                "operation_stats": operation_stats,
                "error_counts": dict(metrics.error_counts),
                "last_update": metrics.last_update.isoformat(),
            }

    def get_optimization_suggestions(
        self, priority: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get optimization suggestions."""
        suggestions = self.optimization_suggestions

        if priority is not None:
            suggestions = [s for s in suggestions if s.priority <= priority]

        return [
            {
                "component": s.component,
                "issue": s.issue,
                "suggestion": s.suggestion,
                "impact": s.impact,
                "effort": s.effort,
                "automated": s.automated,
                "priority": s.priority,
            }
            for s in suggestions
        ]

    def get_recent_alerts(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent performance alerts."""
        return [
            {
                "type": alert["type"],
                "message": alert["message"],
                "severity": alert["severity"],
                "timestamp": alert["timestamp"].isoformat(),
                "count": alert["count"],
            }
            for alert in self.performance_alerts[-limit:]
        ]

    def start_memory_tracking(self):
        """Start detailed memory tracking."""
        if not self.memory_tracking:
            tracemalloc.start()
            self.memory_tracking = True
            logging.info("Memory tracking started")

    def stop_memory_tracking(self):
        """Stop memory tracking."""
        if self.memory_tracking:
            tracemalloc.stop()
            self.memory_tracking = False
            logging.info("Memory tracking stopped")

    def get_memory_snapshot(self) -> Optional[Dict[str, Any]]:
        """Get current memory snapshot."""
        if not self.memory_tracking:
            return None

        try:
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics("lineno")

            memory_info = {
                "total_traces": len(top_stats),
                "top_allocations": [],
                "total_memory_mb": sum(stat.size for stat in top_stats) / (1024 * 1024),
            }

            # Top 10 memory allocations
            for stat in top_stats[:10]:
                memory_info["top_allocations"].append(
                    {
                        "filename": stat.traceback.format()[0]
                        if stat.traceback.format()
                        else "unknown",
                        "size_mb": stat.size / (1024 * 1024),
                        "count": stat.count,
                    }
                )

            return memory_info

        except Exception as e:
            logging.error(f"Memory snapshot error: {e}")
            return None

    def clear_metrics(self, component: Optional[str] = None):
        """Clear metrics for component or all components."""
        with self.lock:
            if component:
                if component in self.component_metrics:
                    del self.component_metrics[component]
            else:
                self.component_metrics.clear()
                self.snapshots.clear()
                self.optimization_suggestions.clear()
                self.performance_alerts.clear()

    def export_metrics(self, format: str = "json") -> str:
        """Export metrics to string format."""
        data = {
            "export_timestamp": datetime.now().isoformat(),
            "snapshots": [
                {
                    "timestamp": s.timestamp.isoformat(),
                    "cpu_percent": s.cpu_percent,
                    "memory_percent": s.memory_percent,
                    "memory_mb": s.memory_mb,
                    "fps": s.fps,
                    "camera_count": s.camera_count,
                }
                for s in list(self.snapshots)
            ],
            "component_metrics": {
                name: {
                    "operation_counts": dict(metrics.operation_counts),
                    "error_counts": dict(metrics.error_counts),
                    "last_update": metrics.last_update.isoformat(),
                }
                for name, metrics in self.component_metrics.items()
            },
            "optimization_suggestions": self.get_optimization_suggestions(),
            "recent_alerts": self.get_recent_alerts(),
        }

        if format.lower() == "json":
            import json

            return json.dumps(data, indent=2)
        elif format.lower() == "yaml":
            import yaml

            return yaml.dump(data, default_flow_style=False)
        else:
            return str(data)


# Decorators for performance monitoring
def monitor_performance(
    component: str, operation: str, profiler: PerformanceProfiler = None
):
    """Decorator to monitor function performance."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            success = True

            try:
                result = func(*args, **kwargs)
                return result
            except Exception:
                success = False
                raise
            finally:
                duration = time.perf_counter() - start_time
                if profiler:
                    profiler.record_component_operation(
                        component, operation, duration, success
                    )

        return wrapper

    return decorator


def async_monitor_performance(
    component: str, operation: str, profiler: PerformanceProfiler = None
):
    """Async decorator to monitor function performance."""

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            success = True

            try:
                result = await func(*args, **kwargs)
                return result
            except Exception:
                success = False
                raise
            finally:
                duration = time.perf_counter() - start_time
                if profiler:
                    profiler.record_component_operation(
                        component, operation, duration, success
                    )

        return wrapper

    return decorator


# Context managers for performance monitoring
@contextmanager
def performance_context(
    component: str, operation: str, profiler: PerformanceProfiler = None
):
    """Context manager for performance monitoring."""
    start_time = time.perf_counter()
    success = True

    try:
        yield
    except Exception:
        success = False
        raise
    finally:
        duration = time.perf_counter() - start_time
        if profiler:
            profiler.record_component_operation(component, operation, duration, success)


# Resource usage monitoring
class ResourceMonitor:
    """Monitor system resource usage."""

    def __init__(self):
        self.baseline_memory = None
        self.peak_memory = 0
        self.start_time = time.time()

        if PSUTIL_AVAILABLE:
            process = psutil.Process()
            self.baseline_memory = process.memory_info().rss

    def get_memory_delta(self) -> float:
        """Get memory usage delta from baseline in MB."""
        if not PSUTIL_AVAILABLE or self.baseline_memory is None:
            return 0.0

        try:
            process = psutil.Process()
            current_memory = process.memory_info().rss
            delta = (current_memory - self.baseline_memory) / (1024 * 1024)
            self.peak_memory = max(self.peak_memory, current_memory)
            return delta
        except Exception:
            return 0.0

    def get_runtime_seconds(self) -> float:
        """Get runtime in seconds."""
        return time.time() - self.start_time

    def get_resource_summary(self) -> Dict[str, Any]:
        """Get resource usage summary."""
        return {
            "runtime_seconds": self.get_runtime_seconds(),
            "memory_delta_mb": self.get_memory_delta(),
            "peak_memory_mb": self.peak_memory / (1024 * 1024)
            if self.peak_memory
            else 0,
            "baseline_memory_mb": self.baseline_memory / (1024 * 1024)
            if self.baseline_memory
            else 0,
        }


# Performance optimization utilities
class PerformanceOptimizer:
    """Automatic performance optimization."""

    def __init__(self, profiler: PerformanceProfiler):
        self.profiler = profiler
        self.optimization_history: List[Dict[str, Any]] = []

    def auto_optimize(self, component_configs: Dict[str, Any]) -> Dict[str, Any]:
        """Automatically apply performance optimizations."""
        suggestions = self.profiler.get_optimization_suggestions(priority=1)
        applied_optimizations = []

        for suggestion in suggestions:
            if suggestion["automated"]:
                optimization = self._apply_optimization(suggestion, component_configs)
                if optimization:
                    applied_optimizations.append(optimization)

        return {
            "timestamp": datetime.now().isoformat(),
            "applied_optimizations": applied_optimizations,
            "suggestions_processed": len(suggestions),
        }

    def _apply_optimization(
        self, suggestion: Dict[str, Any], configs: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Apply single optimization."""
        try:
            component = suggestion["component"]

            if "reduce camera resolution" in suggestion["suggestion"].lower():
                return self._optimize_camera_resolution(configs)
            elif "lower frame rate" in suggestion["suggestion"].lower():
                return self._optimize_frame_rate(configs)
            elif "clear caches" in suggestion["suggestion"].lower():
                return self._clear_caches()

        except Exception as e:
            logging.error(f"Optimization application failed: {e}")

        return None

    def _optimize_camera_resolution(self, configs: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize camera resolution."""
        changes = {}

        for camera_id, camera_config in configs.get("cameras", {}).items():
            current_res = camera_config.get("resolution", "1920x1080")

            # Downgrade resolution
            if current_res == "1920x1080":
                new_res = "1280x720"
            elif current_res == "1280x720":
                new_res = "640x480"
            else:
                continue

            changes[f"cameras.{camera_id}.resolution"] = new_res

        if changes:
            return {
                "type": "camera_resolution",
                "changes": changes,
                "reason": "Reduce CPU load by lowering camera resolution",
            }

        return None

    def _optimize_frame_rate(self, configs: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize frame rate."""
        current_fps = configs.get("streaming", {}).get("target_fps", 30)

        if current_fps > 15:
            new_fps = max(15, current_fps - 5)
            return {
                "type": "frame_rate",
                "changes": {"streaming.target_fps": new_fps},
                "reason": f"Reduce frame rate from {current_fps} to {new_fps} FPS",
            }

        return None

    def _clear_caches(self) -> Dict[str, Any]:
        """Clear system caches."""
        try:
            # Force garbage collection
            gc.collect()

            # Clear OpenCV caches if available
            if OPENCV_AVAILABLE:
                cv2.setUseOptimized(True)

            return {
                "type": "cache_clear",
                "changes": {"system.caches_cleared": True},
                "reason": "Cleared memory caches and forced garbage collection",
            }

        except Exception as e:
            logging.error(f"Cache clearing failed: {e}")
            return None


# Global performance profiler instance
global_profiler = PerformanceProfiler()


# Utility functions
def start_global_profiling(interval: float = 1.0):
    """Start global performance profiling."""
    global_profiler.start_profiling(interval)


def stop_global_profiling():
    """Stop global performance profiling."""
    global_profiler.stop_profiling()


def get_performance_summary(hours: int = 1) -> Dict[str, Any]:
    """Get global performance summary."""
    return global_profiler.get_performance_summary(hours)


def record_operation(
    component: str, operation: str, duration: float, success: bool = True
):
    """Record operation in global profiler."""
    global_profiler.record_component_operation(component, operation, duration, success)


if __name__ == "__main__":
    # Example usage
    profiler = PerformanceProfiler()

    # Start profiling
    profiler.start_profiling(interval=0.5)

    # Simulate some operations
    for i in range(10):
        with performance_context("test_component", "test_operation", profiler):
            time.sleep(0.1)

        profiler.record_component_operation(
            "camera_manager", "capture_frame", 0.033, True
        )

    time.sleep(2)

    # Get results
    summary = profiler.get_performance_summary()
    print("Performance Summary:")
    print(f"CPU Average: {summary['averages']['cpu_percent']:.1f}%")
    print(f"Memory Average: {summary['averages']['memory_percent']:.1f}%")
    print(f"Snapshots: {summary['snapshots_count']}")

    component_metrics = profiler.get_component_metrics("test_component")
    if component_metrics:
        print("\nComponent Metrics:")
        print(f"Operations: {component_metrics['operation_stats']}")

    # Stop profiling
    profiler.stop_profiling()

    print("\nProfiling completed!")
