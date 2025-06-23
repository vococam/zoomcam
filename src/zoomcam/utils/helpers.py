"""
ZoomCam Helper Functions and Utilities
=====================================

Collection of utility functions, decorators, and helpers
used throughout the ZoomCam system.
"""

import asyncio
import functools
import time
import os
import sys
import subprocess
import platform
import socket
import threading
import uuid
import hashlib
import json
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union, Callable, TypeVar, Generic
from contextlib import contextmanager, asynccontextmanager
from dataclasses import dataclass, asdict
import re

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

T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])


# ============ SYSTEM INFORMATION ============

@dataclass
class SystemInfo:
    """System information container."""
    platform: str
    architecture: str
    cpu_count: int
    memory_total_gb: float
    python_version: str
    opencv_available: bool
    psutil_available: bool
    disk_space_gb: float
    network_interfaces: List[str]
    display_resolution: Optional[str] = None
    gpu_available: bool = False


def get_system_info() -> SystemInfo:
    """Get comprehensive system information."""
    info = SystemInfo(
        platform=platform.system(),
        architecture=platform.machine(),
        cpu_count=os.cpu_count() or 1,
        memory_total_gb=0.0,
        python_version=sys.version.split()[0],
        opencv_available=OPENCV_AVAILABLE,
        psutil_available=PSUTIL_AVAILABLE,
        disk_space_gb=0.0,
        network_interfaces=[]
    )

    # Memory information
    if PSUTIL_AVAILABLE:
        memory = psutil.virtual_memory()
        info.memory_total_gb = memory.total / (1024 ** 3)

        # Disk space
        disk = psutil.disk_usage('/')
        info.disk_space_gb = disk.free / (1024 ** 3)

        # Network interfaces
        info.network_interfaces = list(psutil.net_if_addrs().keys())

    # Display resolution detection
    info.display_resolution = detect_display_resolution()

    # GPU detection
    info.gpu_available = detect_gpu_support()

    return info


def detect_display_resolution() -> Optional[str]:
    """Detect display resolution."""
    try:
        if platform.system() == "Linux":
            # Try xrandr first
            result = subprocess.run(
                ["xrandr"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if '*' in line:  # Current resolution
                        match = re.search(r'(\d+x\d+)', line)
                        if match:
                            return match.group(1)

            # Try fbset as fallback
            result = subprocess.run(
                ["fbset", "-s"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'geometry' in line:
                        parts = line.split()
                        if len(parts) >= 3:
                            return f"{parts[1]}x{parts[2]}"

    except Exception:
        pass

    # Try Python tkinter as last resort
    try:
        import tkinter as tk
        root = tk.Tk()
        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()
        root.destroy()
        return f"{width}x{height}"
    except Exception:
        pass

    return None


def detect_gpu_support() -> bool:
    """Detect if GPU acceleration is available."""
    try:
        # Check for NVIDIA GPU
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            timeout=5
        )
        if result.returncode == 0:
            return True
    except Exception:
        pass

    try:
        # Check OpenCV GPU support
        if OPENCV_AVAILABLE:
            return cv2.cuda.getCudaEnabledDeviceCount() > 0
    except Exception:
        pass

    return False


def is_raspberry_pi() -> bool:
    """Check if running on Raspberry Pi."""
    try:
        with open('/proc/cpuinfo', 'r') as f:
            cpuinfo = f.read()
            return 'Raspberry Pi' in cpuinfo or 'BCM' in cpuinfo
    except Exception:
        return False


def get_cpu_temperature() -> Optional[float]:
    """Get CPU temperature (Raspberry Pi specific)."""
    try:
        if is_raspberry_pi():
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp = int(f.read().strip()) / 1000.0
                return temp
    except Exception:
        pass

    try:
        if PSUTIL_AVAILABLE:
            sensors = psutil.sensors_temperatures()
            if 'cpu_thermal' in sensors:
                return sensors['cpu_thermal'][0].current
            elif 'coretemp' in sensors:
                return sensors['coretemp'][0].current
    except Exception:
        pass

    return None


# ============ FILE AND PATH UTILITIES ============

def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if not."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_filename(filename: str) -> str:
    """Create safe filename by removing/replacing invalid characters."""
    # Remove or replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')

    # Remove leading/trailing spaces and dots
    filename = filename.strip('. ')

    # Limit length
    if len(filename) > 200:
        filename = filename[:200]

    return filename or 'unnamed'


def get_file_size_mb(file_path: Union[str, Path]) -> float:
    """Get file size in megabytes."""
    try:
        size_bytes = Path(file_path).stat().st_size
        return size_bytes / (1024 * 1024)
    except Exception:
        return 0.0


def cleanup_old_files(
        directory: Union[str, Path],
        max_age_days: int = 7,
        pattern: str = "*",
        dry_run: bool = False
) -> List[Path]:
    """Clean up old files in directory."""
    directory = Path(directory)
    if not directory.exists():
        return []

    cutoff_time = datetime.now() - timedelta(days=max_age_days)
    old_files = []

    for file_path in directory.glob(pattern):
        if file_path.is_file():
            mod_time = datetime.fromtimestamp(file_path.stat().st_mtime)
            if mod_time < cutoff_time:
                old_files.append(file_path)
                if not dry_run:
                    try:
                        file_path.unlink()
                    except Exception:
                        pass  # Skip files that can't be deleted

    return old_files


def backup_file(file_path: Union[str, Path], backup_suffix: str = None) -> Optional[Path]:
    """Create backup of file."""
    file_path = Path(file_path)
    if not file_path.exists():
        return None

    if backup_suffix is None:
        backup_suffix = datetime.now().strftime("_%Y%m%d_%H%M%S")

    backup_path = file_path.with_suffix(f"{backup_suffix}{file_path.suffix}")

    try:
        import shutil
        shutil.copy2(file_path, backup_path)
        return backup_path
    except Exception:
        return None


# ============ CONFIGURATION UTILITIES ============

def parse_resolution(resolution: str) -> Tuple[int, int]:
    """Parse resolution string to width, height tuple."""
    resolution = resolution.lower().strip()

    # Named resolutions
    named_resolutions = {
        '4k': (3840, 2160),
        'uhd': (3840, 2160),
        '1080p': (1920, 1080),
        'fhd': (1920, 1080),
        '720p': (1280, 720),
        'hd': (1280, 720),
        '480p': (854, 480),
        'vga': (640, 480),
        'qvga': (320, 240)
    }

    if resolution in named_resolutions:
        return named_resolutions[resolution]

    # Parse WxH format
    if 'x' in resolution:
        try:
            width, height = resolution.split('x')
            return (int(width), int(height))
        except ValueError:
            pass

    # Default fallback
    return (1920, 1080)


def validate_resolution(resolution: Tuple[int, int]) -> bool:
    """Validate resolution tuple."""
    width, height = resolution
    return (
            isinstance(width, int) and isinstance(height, int) and
            32 <= width <= 7680 and 32 <= height <= 4320 and
            width % 2 == 0 and height % 2 == 0  # Even numbers for video encoding
    )


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge configuration dictionaries."""
    result = base_config.copy()

    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result


def get_nested_value(data: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """Get nested value using dot notation (e.g., 'cameras.camera_1.zoom')."""
    keys = key_path.split('.')
    current = data

    try:
        for key in keys:
            current = current[key]
        return current
    except (KeyError, TypeError):
        return default


def set_nested_value(data: Dict[str, Any], key_path: str, value: Any) -> bool:
    """Set nested value using dot notation."""
    keys = key_path.split('.')
    current = data

    try:
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value
        return True
    except (KeyError, TypeError):
        return False


# ============ NETWORK UTILITIES ============

def get_local_ip() -> Optional[str]:
    """Get local IP address."""
    try:
        # Connect to a dummy address to determine local IP
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return None


def is_port_available(port: int, host: str = "localhost") -> bool:
    """Check if port is available."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            result = s.connect_ex((host, port))
            return result != 0
    except Exception:
        return False


def find_available_port(start_port: int = 8000, max_attempts: int = 100) -> Optional[int]:
    """Find next available port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        if is_port_available(port):
            return port
    return None


def validate_rtsp_url(url: str) -> bool:
    """Validate RTSP URL format."""
    rtsp_pattern = re.compile(
        r'^rtsp://'
        r'(?:(?P<user>[^:@]+)(?::(?P<password>[^@]+))?@)?'
        r'(?P<host>[^:/]+)'
        r'(?::(?P<port>\d+))?'
        r'(?P<path>/.*)?
    )
    return bool(rtsp_pattern.match(url))


def extract_rtsp_components(url: str) -> Dict[str, Optional[str]]:
    """Extract components from RTSP URL."""
    rtsp_pattern = re.compile(
        r'^rtsp://'
        r'(?:(?P<user>[^:@]+)(?::(?P<password>[^@]+))?@)?'
        r'(?P<host>[^:/]+)'
        r'(?::(?P<port>\d+))?'
        r'(?P<path>/.*)?
    )

    match = rtsp_pattern.match(url)
    if match:
        return match.groupdict()
    return {}


# ============ PERFORMANCE UTILITIES ============

class Timer:
    """Simple timer context manager."""

    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.duration = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time

    def __str__(self):
        if self.duration is not None:
            return f"{self.name}: {self.duration:.3f}s"
        return f"{self.name}: not measured"


class PerformanceMonitor:
    """Simple performance monitoring."""

    def __init__(self, max_samples: int = 1000):
        self.max_samples = max_samples
        self.measurements: Dict[str, List[float]] = {}
        self.lock = threading.Lock()

    def record(self, metric_name: str, value: float):
        """Record a measurement."""
        with self.lock:
            if metric_name not in self.measurements:
                self.measurements[metric_name] = []

            self.measurements[metric_name].append(value)

            # Keep only recent measurements
            if len(self.measurements[metric_name]) > self.max_samples:
                self.measurements[metric_name] = self.measurements[metric_name][-self.max_samples:]

    def get_stats(self, metric_name: str) -> Dict[str, float]:
        """Get statistics for a metric."""
        with self.lock:
            if metric_name not in self.measurements or not self.measurements[metric_name]:
                return {}

            values = self.measurements[metric_name]
            return {
                'count': len(values),
                'min': min(values),
                'max': max(values),
                'avg': sum(values) / len(values),
                'recent': values[-1] if values else 0
            }

    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all metrics."""
        return {name: self.get_stats(name) for name in self.measurements.keys()}


def measure_performance(metric_name: str, monitor: PerformanceMonitor = None):
    """Decorator to measure function performance."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.perf_counter() - start_time
                if monitor:
                    monitor.record(metric_name, duration)

        return wrapper

    return decorator


# ============ ASYNC UTILITIES ============

async def run_in_thread(func: Callable[..., T], *args, **kwargs) -> T:
    """Run synchronous function in thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, functools.partial(func, **kwargs), *args)


async def timeout_after(seconds: float, coro):
    """Run coroutine with timeout."""
    try:
        return await asyncio.wait_for(coro, timeout=seconds)
    except asyncio.TimeoutError:
        raise TimeoutError(f"Operation timed out after {seconds} seconds")


class AsyncCache:
    """Simple async cache with TTL."""

    def __init__(self, ttl_seconds: float = 300):
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.lock = asyncio.Lock()

    async def get(self, key: str, factory: Callable[[], Any] = None) -> Any:
        """Get value from cache or create using factory."""
        async with self.lock:
            now = time.time()

            # Check if we have valid cached value
            if key in self.cache:
                value, timestamp = self.cache[key]
                if now - timestamp < self.ttl_seconds:
                    return value
                else:
                    del self.cache[key]

            # Create new value if factory provided
            if factory:
                value = await factory() if asyncio.iscoroutinefunction(factory) else factory()
                self.cache[key] = (value, now)
                return value

            return None

    async def set(self, key: str, value: Any):
        """Set value in cache."""
        async with self.lock:
            self.cache[key] = (value, time.time())

    async def clear(self):
        """Clear all cached values."""
        async with self.lock:
            self.cache.clear()


# ============ RETRY UTILITIES ============

def retry_on_exception(
        max_attempts: int = 3,
        delay: float = 1.0,
        backoff: float = 2.0,
        exceptions: Tuple[Exception, ...] = (Exception,)
):
    """Decorator to retry function on specific exceptions."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts - 1:
                        raise

                    time.sleep(current_delay)
                    current_delay *= backoff

            return None

        return wrapper

    return decorator


async def async_retry(
        coro_func: Callable[..., Any],
        max_attempts: int = 3,
        delay: float = 1.0,
        backoff: float = 2.0,
        exceptions: Tuple[Exception, ...] = (Exception,),
        *args,
        **kwargs
):
    """Async retry function."""
    current_delay = delay

    for attempt in range(max_attempts):
        try:
            return await coro_func(*args, **kwargs)
        except exceptions as e:
            if attempt == max_attempts - 1:
                raise

            await asyncio.sleep(current_delay)
            current_delay *= backoff

    return None


# ============ VALIDATION UTILITIES ============

def validate_camera_id(camera_id: str) -> bool:
    """Validate camera ID format."""
    if not camera_id or not isinstance(camera_id, str):
        return False

    # Allow alphanumeric, underscore, hyphen
    pattern = re.compile(r'^[a-zA-Z0-9_-]+)
    return bool(pattern.match(camera_id)) and len(camera_id) <= 50


def validate_zoom_level(zoom: Union[int, float]) -> bool:
    """Validate zoom level."""
    try:
        zoom_float = float(zoom)
        return 1.0 <= zoom_float <= 10.0
    except (ValueError, TypeError):
        return False


def validate_fps(fps: Union[int, float]) -> bool:
    """Validate FPS value."""
    try:
        fps_float = float(fps)
        return 1.0 <= fps_float <= 120.0
    except (ValueError, TypeError):
        return False


def validate_percentage(value: Union[int, float]) -> bool:
    """Validate percentage value (0-100)."""
    try:
        percent = float(value)
        return 0.0 <= percent <= 100.0
    except (ValueError, TypeError):
        return False


# ============ STRING UTILITIES ============

def truncate_string(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate string to max length."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def format_bytes(bytes_value: int) -> str:
    """Format bytes as human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"


def format_duration(seconds: float) -> str:
    """Format duration in seconds as human readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def generate_session_id() -> str:
    """Generate unique session ID."""
    return str(uuid.uuid4())


def generate_hash(data: str) -> str:
    """Generate SHA-256 hash of string."""
    return hashlib.sha256(data.encode()).hexdigest()


# ============ COLOR UTILITIES ============

def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    """Convert RGB tuple to hex color."""
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def interpolate_color(color1: Tuple[int, int, int], color2: Tuple[int, int, int], factor: float) -> Tuple[
    int, int, int]:
    """Interpolate between two RGB colors."""
    factor = max(0.0, min(1.0, factor))
    return (
        int(color1[0] + (color2[0] - color1[0]) * factor),
        int(color1[1] + (color2[1] - color1[1]) * factor),
        int(color1[2] + (color2[2] - color1[2]) * factor)
    )


# ============ CONTEXT MANAGERS ============

@contextmanager
def temporary_file(suffix: str = "", content: bytes = b""):
    """Create temporary file context manager."""
    import tempfile

    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        if content:
            os.write(fd, content)
        os.close(fd)
        yield Path(path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


@asynccontextmanager
async def async_timer(name: str = "Operation"):
    """Async timer context manager."""
    start_time = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start_time
        print(f"{name}: {duration:.3f}s")


@contextmanager
def suppress_errors(*exceptions):
    """Context manager to suppress specific exceptions."""
    try:
        yield
    except exceptions:
        pass


# ============ CONVERSION UTILITIES ============

def any_to_bool(value: Any) -> bool:
    """Convert any value to boolean."""
    if isinstance(value, bool):
        return value
    elif isinstance(value, str):
        return value.lower() in ('true', '1', 'yes', 'on', 'enabled')
    elif isinstance(value, (int, float)):
        return value != 0
    else:
        return bool(value)


def any_to_int(value: Any, default: int = 0) -> int:
    """Convert any value to integer."""
    try:
        if isinstance(value, str):
            # Handle strings like "1920x1080" -> 1920
            if 'x' in value:
                value = value.split('x')[0]
        return int(float(value))
    except (ValueError, TypeError):
        return default


def any_to_float(value: Any, default: float = 0.0) -> float:
    """Convert any value to float."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


# ============ TESTING UTILITIES ============

class MockCamera:
    """Mock camera for testing."""

    def __init__(self, camera_id: str, resolution: Tuple[int, int] = (640, 480)):
        self.camera_id = camera_id
        self.resolution = resolution
        self.frame_count = 0
        self.is_open = True

    def read(self):
        """Simulate frame reading."""
        if not self.is_open:
            return False, None

        import numpy as np
        frame = np.random.randint(0, 255, (*self.resolution[::-1], 3), dtype=np.uint8)
        self.frame_count += 1
        return True, frame

    def release(self):
        """Release camera."""
        self.is_open = False


def create_test_config() -> Dict[str, Any]:
    """Create test configuration."""
    return {
        "system": {
            "display": {
                "target_resolution": "1920x1080",
                "refresh_rate": 30
            }
        },
        "cameras": {
            "test_camera": {
                "enabled": True,
                "source": "/dev/video0",
                "name": "Test Camera",
                "zoom": 2.0,
                "max_fragments": 2
            }
        },
        "streaming": {
            "bitrate": "2M",
            "target_fps": 30
        }
    }


# ============ HEALTH CHECK UTILITIES ============

class HealthChecker:
    """System health checker."""

    def __init__(self):
        self.checks: Dict[str, Callable[[], bool]] = {}

    def register_check(self, name: str, check_func: Callable[[], bool]):
        """Register health check function."""
        self.checks[name] = check_func

    def run_checks(self) -> Dict[str, Any]:
        """Run all health checks."""
        results = {
            "overall_health": "healthy",
            "timestamp": datetime.now().isoformat(),
            "checks": {}
        }

        unhealthy_count = 0

        for name, check_func in self.checks.items():
            try:
                is_healthy = check_func()
                results["checks"][name] = {
                    "status": "healthy" if is_healthy else "unhealthy",
                    "healthy": is_healthy
                }
                if not is_healthy:
                    unhealthy_count += 1
            except Exception as e:
                results["checks"][name] = {
                    "status": "error",
                    "healthy": False,
                    "error": str(e)
                }
                unhealthy_count += 1

        # Overall health assessment
        if unhealthy_count == 0:
            results["overall_health"] = "healthy"
        elif unhealthy_count <= len(self.checks) / 2:
            results["overall_health"] = "degraded"
        else:
            results["overall_health"] = "unhealthy"

        return results


def check_disk_space(path: str = "/", min_gb: float = 1.0) -> bool:
    """Check if enough disk space is available."""
    try:
        if PSUTIL_AVAILABLE:
            usage = psutil.disk_usage(path)
            free_gb = usage.free / (1024 ** 3)
            return free_gb >= min_gb
    except Exception:
        pass
    return True  # Assume healthy if can't check


def check_memory_usage(max_percent: float = 90.0) -> bool:
    """Check memory usage."""
    try:
        if PSUTIL_AVAILABLE:
            memory = psutil.virtual_memory()
            return memory.percent <= max_percent
    except Exception:
        pass
    return True


def check_cpu_usage(max_percent: float = 95.0) -> bool:
    """Check CPU usage."""
    try:
        if PSUTIL_AVAILABLE:
            cpu_percent = psutil.cpu_percent(interval=1)
            return cpu_percent <= max_percent
    except Exception:
        pass
    return True


# Global performance monitor instance
global_performance_monitor = PerformanceMonitor()

# Global health checker instance
global_health_checker = HealthChecker()

# Register default health checks
global_health_checker.register_check("disk_space", lambda: check_disk_space())
global_health_checker.register_check("memory_usage", lambda: check_memory_usage())
global_health_checker.register_check("cpu_usage", lambda: check_cpu_usage())

if __name__ == "__main__":
    # Example usage
    print("System Information:")
    sys_info = get_system_info()
    print(f"Platform: {sys_info.platform}")
    print(f"Memory: {sys_info.memory_total_gb:.1f} GB")
    print(f"Display: {sys_info.display_resolution}")
    print(f"GPU Available: {sys_info.gpu_available}")

    print("\nHealth Check:")
    health = global_health_checker.run_checks()
    print(f"Overall Health: {health['overall_health']}")

    print("\nTesting utilities...")
    with Timer("Test operation"):
        time.sleep(0.1)

    print("All tests completed!")