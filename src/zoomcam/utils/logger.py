"""
ZoomCam Logging System
=====================

Advanced logging configuration with structured logging,
performance monitoring, and multi-destination output.
"""

import logging
import logging.handlers
import json
import time
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
import threading
import queue
import traceback
from contextlib import contextmanager

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


@dataclass
class LogContext:
    """Logging context for structured logging."""
    component: str
    camera_id: Optional[str] = None
    session_id: Optional[str] = None
    operation: Optional[str] = None
    performance_data: Optional[Dict[str, Any]] = None


@dataclass
class PerformanceMetric:
    """Performance metric for logging."""
    name: str
    value: float
    unit: str
    timestamp: datetime
    context: Optional[Dict[str, Any]] = None


class ZoomCamFormatter(logging.Formatter):
    """Custom formatter for ZoomCam logs with structured output."""

    def __init__(self, include_performance: bool = True, json_format: bool = False):
        self.include_performance = include_performance
        self.json_format = json_format

        if json_format:
            super().__init__()
        else:
            format_str = (
                '%(asctime)s | %(levelname)-8s | %(name)-20s | '
                '%(camera_id)s | %(component)s | %(message)s'
            )
            super().__init__(format_str, datefmt='%Y-%m-%d %H:%M:%S')

    def format(self, record: logging.LogRecord) -> str:
        # Add default values for missing attributes
        if not hasattr(record, 'camera_id'):
            record.camera_id = getattr(record, 'camera_id', 'system')
        if not hasattr(record, 'component'):
            record.component = getattr(record, 'component', record.name)
        if not hasattr(record, 'operation'):
            record.operation = getattr(record, 'operation', 'general')

        # Add performance data if available
        if self.include_performance and PSUTIL_AVAILABLE:
            if not hasattr(record, 'cpu_percent'):
                record.cpu_percent = psutil.cpu_percent()
            if not hasattr(record, 'memory_percent'):
                record.memory_percent = psutil.virtual_memory().percent

        if self.json_format:
            return self._format_json(record)
        else:
            return super().format(record)

    def _format_json(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'component': getattr(record, 'component', record.name),
            'camera_id': getattr(record, 'camera_id', None),
            'operation': getattr(record, 'operation', None),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }

        # Add performance data
        if self.include_performance:
            log_data['performance'] = {
                'cpu_percent': getattr(record, 'cpu_percent', None),
                'memory_percent': getattr(record, 'memory_percent', None)
            }

        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }

        # Add any extra attributes
        extra_attrs = {}
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                           'filename', 'module', 'lineno', 'funcName', 'created',
                           'msecs', 'relativeCreated', 'thread', 'threadName',
                           'processName', 'process', 'exc_info', 'exc_text', 'stack_info',
                           'component', 'camera_id', 'operation', 'cpu_percent', 'memory_percent']:
                extra_attrs[key] = value

        if extra_attrs:
            log_data['extra'] = extra_attrs

        return json.dumps(log_data, default=str)


class PerformanceLogHandler(logging.Handler):
    """Handler for performance-specific logging."""

    def __init__(self, performance_file: str, max_entries: int = 10000):
        super().__init__()
        self.performance_file = Path(performance_file)
        self.max_entries = max_entries
        self.performance_data: List[Dict[str, Any]] = []
        self.lock = threading.Lock()

        # Ensure directory exists
        self.performance_file.parent.mkdir(parents=True, exist_ok=True)

        # Load existing data
        self._load_existing_data()

    def _load_existing_data(self):
        """Load existing performance data."""
        try:
            if self.performance_file.exists():
                with open(self.performance_file, 'r') as f:
                    self.performance_data = json.load(f)

                # Keep only recent entries
                if len(self.performance_data) > self.max_entries:
                    self.performance_data = self.performance_data[-self.max_entries:]
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to load performance data: {e}")
            self.performance_data = []

    def emit(self, record: logging.LogRecord):
        """Emit performance log record."""
        try:
            if hasattr(record, 'performance_metric'):
                metric = record.performance_metric

                with self.lock:
                    self.performance_data.append({
                        'timestamp': datetime.now().isoformat(),
                        'component': getattr(record, 'component', 'unknown'),
                        'camera_id': getattr(record, 'camera_id', None),
                        'metric': asdict(metric) if hasattr(metric, '__dataclass_fields__') else metric,
                        'level': record.levelname
                    })

                    # Trim if too many entries
                    if len(self.performance_data) > self.max_entries:
                        self.performance_data = self.performance_data[-self.max_entries:]

                    # Save periodically
                    if len(self.performance_data) % 100 == 0:
                        self._save_data()
        except Exception as e:
            # Don't let logging errors break the application
            print(f"Performance logging error: {e}", file=sys.stderr)

    def _save_data(self):
        """Save performance data to file."""
        try:
            with open(self.performance_file, 'w') as f:
                json.dump(self.performance_data, f, indent=2, default=str)
        except Exception as e:
            print(f"Failed to save performance data: {e}", file=sys.stderr)

    def close(self):
        """Close handler and save final data."""
        self._save_data()
        super().close()


class AsyncLogHandler(logging.Handler):
    """Asynchronous log handler to prevent blocking."""

    def __init__(self, target_handler: logging.Handler, queue_size: int = 1000):
        super().__init__()
        self.target_handler = target_handler
        self.queue = queue.Queue(maxsize=queue_size)
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.shutdown_event = threading.Event()
        self.worker_thread.start()

    def _worker(self):
        """Worker thread to process log records."""
        while not self.shutdown_event.is_set():
            try:
                record = self.queue.get(timeout=1.0)
                if record is None:  # Shutdown signal
                    break
                self.target_handler.emit(record)
                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Async log handler error: {e}", file=sys.stderr)

    def emit(self, record: logging.LogRecord):
        """Emit log record asynchronously."""
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            # Drop the record if queue is full
            pass

    def close(self):
        """Close the handler and worker thread."""
        self.shutdown_event.set()
        try:
            self.queue.put_nowait(None)  # Shutdown signal
        except queue.Full:
            pass

        self.worker_thread.join(timeout=5.0)
        self.target_handler.close()
        super().close()


class ZoomCamLogger:
    """Main logger class for ZoomCam with context management."""

    def __init__(self, name: str, context: Optional[LogContext] = None):
        self.logger = logging.getLogger(name)
        self.context = context or LogContext(component=name)
        self.performance_metrics: List[PerformanceMetric] = []

    def _log_with_context(self, level: int, message: str, **kwargs):
        """Log with context information."""
        extra = {
            'component': self.context.component,
            'camera_id': self.context.camera_id,
            'operation': self.context.operation,
            **kwargs
        }

        # Add performance data if available
        if self.context.performance_data:
            extra.update(self.context.performance_data)

        self.logger.log(level, message, extra=extra)

    def debug(self, message: str, **kwargs):
        """Debug level logging."""
        self._log_with_context(logging.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs):
        """Info level logging."""
        self._log_with_context(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Warning level logging."""
        self._log_with_context(logging.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs):
        """Error level logging."""
        self._log_with_context(logging.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs):
        """Critical level logging."""
        self._log_with_context(logging.CRITICAL, message, **kwargs)

    def performance(self, metric: PerformanceMetric, level: int = logging.INFO):
        """Log performance metric."""
        self.performance_metrics.append(metric)

        extra = {
            'component': self.context.component,
            'camera_id': self.context.camera_id,
            'performance_metric': metric
        }

        message = f"Performance: {metric.name} = {metric.value} {metric.unit}"
        self.logger.log(level, message, extra=extra)

    def with_context(self, **context_updates) -> 'ZoomCamLogger':
        """Create new logger with updated context."""
        new_context = LogContext(
            component=context_updates.get('component', self.context.component),
            camera_id=context_updates.get('camera_id', self.context.camera_id),
            session_id=context_updates.get('session_id', self.context.session_id),
            operation=context_updates.get('operation', self.context.operation),
            performance_data=context_updates.get('performance_data', self.context.performance_data)
        )

        return ZoomCamLogger(self.logger.name, new_context)

    @contextmanager
    def operation(self, operation_name: str):
        """Context manager for operation logging."""
        start_time = time.perf_counter()
        operation_logger = self.with_context(operation=operation_name)

        operation_logger.debug(f"Starting operation: {operation_name}")

        try:
            yield operation_logger
        except Exception as e:
            operation_logger.error(f"Operation failed: {operation_name}", exc_info=True)
            raise
        finally:
            duration = time.perf_counter() - start_time
            operation_logger.performance(
                PerformanceMetric(
                    name=f"{operation_name}_duration",
                    value=duration,
                    unit="seconds",
                    timestamp=datetime.now()
                )
            )
            operation_logger.debug(f"Completed operation: {operation_name} in {duration:.3f}s")

    @contextmanager
    def camera_context(self, camera_id: str):
        """Context manager for camera-specific logging."""
        camera_logger = self.with_context(camera_id=camera_id)
        yield camera_logger


class LogManager:
    """Global log manager for ZoomCam."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized'):
            return

        self._initialized = True
        self.handlers: List[logging.Handler] = []
        self.performance_handler: Optional[PerformanceLogHandler] = None
        self.log_directory = Path("logs")
        self.config = {}

    def setup_logging(
            self,
            level: Union[str, int] = logging.INFO,
            log_file: Optional[str] = None,
            performance_file: Optional[str] = None,
            json_format: bool = False,
            async_logging: bool = True,
            max_file_size: int = 10 * 1024 * 1024,  # 10MB
            backup_count: int = 5
    ):
        """Setup logging configuration."""

        # Convert string level to int
        if isinstance(level, str):
            level = getattr(logging, level.upper())

        # Clear existing handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        self.handlers.clear()

        # Create formatters
        console_formatter = ZoomCamFormatter(
            include_performance=True,
            json_format=False
        )

        file_formatter = ZoomCamFormatter(
            include_performance=True,
            json_format=json_format
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(level)

        if async_logging:
            console_handler = AsyncLogHandler(console_handler)

        self.handlers.append(console_handler)

        # File handler
        if log_file:
            self.log_directory.mkdir(parents=True, exist_ok=True)
            log_path = self.log_directory / log_file

            file_handler = logging.handlers.RotatingFileHandler(
                log_path,
                maxBytes=max_file_size,
                backupCount=backup_count
            )
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(logging.DEBUG)  # File gets all levels

            if async_logging:
                file_handler = AsyncLogHandler(file_handler)

            self.handlers.append(file_handler)

        # Performance handler
        if performance_file:
            perf_path = self.log_directory / performance_file
            self.performance_handler = PerformanceLogHandler(str(perf_path))
            self.performance_handler.setLevel(logging.DEBUG)
            self.handlers.append(self.performance_handler)

        # Error file handler (separate file for errors)
        error_file = self.log_directory / "error.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_file,
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        error_handler.setFormatter(file_formatter)
        error_handler.setLevel(logging.ERROR)

        if async_logging:
            error_handler = AsyncLogHandler(error_handler)

        self.handlers.append(error_handler)

        # Add all handlers to root logger
        root_logger.setLevel(logging.DEBUG)
        for handler in self.handlers:
            root_logger.addHandler(handler)

        # Store configuration
        self.config = {
            'level': level,
            'log_file': log_file,
            'performance_file': performance_file,
            'json_format': json_format,
            'async_logging': async_logging
        }

        # Log setup completion
        setup_logger = self.get_logger('logging_setup')
        setup_logger.info("Logging system initialized",
                          log_level=logging.getLevelName(level),
                          handlers_count=len(self.handlers))

    def get_logger(self, name: str, context: Optional[LogContext] = None) -> ZoomCamLogger:
        """Get a logger with optional context."""
        return ZoomCamLogger(name, context)

    def get_camera_logger(self, camera_id: str, component: str = "camera") -> ZoomCamLogger:
        """Get a camera-specific logger."""
        context = LogContext(component=component, camera_id=camera_id)
        return ZoomCamLogger(f"camera.{camera_id}", context)

    def get_performance_metrics(self, hours: int = 1) -> List[Dict[str, Any]]:
        """Get performance metrics from the last N hours."""
        if not self.performance_handler:
            return []

        cutoff_time = datetime.now() - timedelta(hours=hours)

        return [
            metric for metric in self.performance_handler.performance_data
            if datetime.fromisoformat(metric['timestamp']) > cutoff_time
        ]

    def flush_logs(self):
        """Flush all log handlers."""
        for handler in self.handlers:
            try:
                handler.flush()
            except Exception as e:
                print(f"Error flushing handler: {e}", file=sys.stderr)

    def shutdown(self):
        """Shutdown logging system."""
        self.flush_logs()

        for handler in self.handlers:
            try:
                handler.close()
            except Exception as e:
                print(f"Error closing handler: {e}", file=sys.stderr)

        self.handlers.clear()

        # Remove handlers from root logger
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)


# Global functions for easy access
_log_manager = LogManager()


def setup_logging(**kwargs):
    """Setup logging with the given configuration."""
    _log_manager.setup_logging(**kwargs)


def get_logger(name: str, context: Optional[LogContext] = None) -> ZoomCamLogger:
    """Get a logger instance."""
    return _log_manager.get_logger(name, context)


def get_camera_logger(camera_id: str, component: str = "camera") -> ZoomCamLogger:
    """Get a camera-specific logger."""
    return _log_manager.get_camera_logger(camera_id, component)


def shutdown_logging():
    """Shutdown the logging system."""
    _log_manager.shutdown()


# Performance logging utilities
def log_performance(logger: ZoomCamLogger, name: str, value: float, unit: str, **context):
    """Log a performance metric."""
    metric = PerformanceMetric(
        name=name,
        value=value,
        unit=unit,
        timestamp=datetime.now(),
        context=context
    )
    logger.performance(metric)


@contextmanager
def timed_operation(logger: ZoomCamLogger, operation_name: str, log_level: int = logging.INFO):
    """Context manager for timing operations."""
    start_time = time.perf_counter()

    try:
        yield
    finally:
        duration = time.perf_counter() - start_time
        metric = PerformanceMetric(
            name=f"{operation_name}_duration",
            value=duration,
            unit="seconds",
            timestamp=datetime.now()
        )
        logger.performance(metric, log_level)


# Exception logging utilities
def log_exception(logger: ZoomCamLogger, operation: str, exception: Exception):
    """Log an exception with context."""
    logger.error(
        f"Exception in {operation}: {type(exception).__name__}: {exception}",
        exc_info=True,
        operation=operation,
        exception_type=type(exception).__name__
    )


# Memory usage logging (if psutil available)
def log_memory_usage(logger: ZoomCamLogger):
    """Log current memory usage."""
    if PSUTIL_AVAILABLE:
        memory = psutil.virtual_memory()
        logger.performance(
            PerformanceMetric(
                name="memory_usage",
                value=memory.percent,
                unit="percent",
                timestamp=datetime.now(),
                context={
                    "available_mb": memory.available / (1024 * 1024),
                    "used_mb": memory.used / (1024 * 1024)
                }
            )
        )


def log_cpu_usage(logger: ZoomCamLogger):
    """Log current CPU usage."""
    if PSUTIL_AVAILABLE:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        logger.performance(
            PerformanceMetric(
                name="cpu_usage",
                value=cpu_percent,
                unit="percent",
                timestamp=datetime.now()
            )
        )


if __name__ == "__main__":
    # Example usage
    setup_logging(
        level="DEBUG",
        log_file="zoomcam.log",
        performance_file="performance.json",
        json_format=False
    )

    # Test logging
    logger = get_logger("test_component")

    logger.info("Starting test")

    with logger.operation("test_operation"):
        time.sleep(0.1)
        logger.debug("Inside operation")

    # Camera-specific logging
    cam_logger = get_camera_logger("camera_1", "motion_detector")
    cam_logger.info("Motion detected", zones=3, activity_level=0.75)

    # Performance logging
    log_performance(logger, "frame_processing", 33.5, "ms")

    logger.info("Test completed")

    # Shutdown
    shutdown_logging()