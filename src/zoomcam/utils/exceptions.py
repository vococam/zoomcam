"""
ZoomCam Custom Exceptions
========================

Comprehensive exception hierarchy for ZoomCam system with
detailed error information, recovery suggestions, and logging integration.
"""

import traceback
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""
    CAMERA = "camera"
    STREAMING = "streaming"
    LAYOUT = "layout"
    CONFIGURATION = "configuration"
    PERFORMANCE = "performance"
    NETWORK = "network"
    HARDWARE = "hardware"
    SOFTWARE = "software"
    USER_INPUT = "user_input"
    SYSTEM = "system"


@dataclass
class ErrorContext:
    """Additional context information for errors."""
    component: str
    operation: Optional[str] = None
    camera_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    system_state: Optional[Dict[str, Any]] = None
    user_action: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class RecoverySuggestion:
    """Recovery suggestion for error handling."""
    action: str
    description: str
    automated: bool = False
    priority: int = 1  # 1=high, 2=medium, 3=low
    risk_level: str = "low"  # low, medium, high


class ZoomCamError(Exception):
    """Base exception class for all ZoomCam errors."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        context: Optional[ErrorContext] = None,
        recovery_suggestions: Optional[List[RecoverySuggestion]] = None,
        technical_details: Optional[Dict[str, Any]] = None,
        user_message: Optional[str] = None,
        original_exception: Optional[Exception] = None
    ):
        super().__init__(message)

        self.message = message
        self.error_code = error_code or self._generate_error_code()
        self.severity = severity
        self.category = category
        self.context = context or ErrorContext(component="unknown")
        self.recovery_suggestions = recovery_suggestions or []
        self.technical_details = technical_details or {}
        self.user_message = user_message or self._generate_user_message()
        self.original_exception = original_exception
        self.traceback_info = traceback.format_exc()

        # Add automatic recovery suggestions
        self._add_automatic_suggestions()

    def _generate_error_code(self) -> str:
        """Generate unique error code."""
        category_code = self.category.value[:3].upper()
        severity_code = self.severity.value[0].upper()
        timestamp_code = datetime.now().strftime("%H%M%S")
        return f"ZC-{category_code}-{severity_code}{timestamp_code}"

    def _generate_user_message(self) -> str:
        """Generate user-friendly error message."""
        if self.severity == ErrorSeverity.CRITICAL:
            return f"Critical system error occurred. Please contact support. (Error: {self.error_code})"
        elif self.severity == ErrorSeverity.HIGH:
            return f"An important component has encountered an error. System may be unstable."
        elif self.severity == ErrorSeverity.MEDIUM:
            return f"A component error occurred. Some features may not work properly."
        else:
            return f"Minor issue detected. System should continue working normally."

    def _add_automatic_suggestions(self):
        """Add automatic recovery suggestions based on error type."""
        if self.category == ErrorCategory.CAMERA:
            self.recovery_suggestions.append(
                RecoverySuggestion(
                    action="restart_camera",
                    description="Restart the affected camera",
                    automated=True,
                    priority=1
                )
            )

        if self.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            self.recovery_suggestions.append(
                RecoverySuggestion(
                    action="system_restart",
                    description="Restart the ZoomCam system",
                    automated=False,
                    priority=2,
                    risk_level="medium"
                )
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for serialization."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "user_message": self.user_message,
            "severity": self.severity.value,
            "category": self.category.value,
            "context": {
                "component": self.context.component,
                "operation": self.context.operation,
                "camera_id": self.context.camera_id,
                "timestamp": self.context.timestamp.isoformat() if self.context.timestamp else None,
                "user_action": self.context.user_action
            },
            "recovery_suggestions": [
                {
                    "action": s.action,
                    "description": s.description,
                    "automated": s.automated,
                    "priority": s.priority,
                    "risk_level": s.risk_level
                }
                for s in self.recovery_suggestions
            ],
            "technical_details": self.technical_details,
            "original_exception": str(self.original_exception) if self.original_exception else None,
            "traceback": self.traceback_info
        }

    def get_user_friendly_message(self) -> str:
        """Get user-friendly error message with suggestions."""
        message = self.user_message

        if self.recovery_suggestions:
            suggestions = [s.description for s in self.recovery_suggestions if s.priority == 1]
            if suggestions:
                message += f"\n\nSuggested actions:\n" + "\n".join(f"• {s}" for s in suggestions[:3])

        return message


# ============ CAMERA ERRORS ============

class CameraError(ZoomCamError):
    """Base class for camera-related errors."""

    def __init__(self, message: str, camera_id: Optional[str] = None, **kwargs):
        context = kwargs.get('context') or ErrorContext(
            component="camera_manager",
            camera_id=camera_id
        )
        kwargs['context'] = context
        kwargs['category'] = ErrorCategory.CAMERA
        super().__init__(message, **kwargs)


class CameraConnectionError(CameraError):
    """Camera connection failed."""

    def __init__(self, camera_id: str, source: str, **kwargs):
        message = f"Failed to connect to camera {camera_id} at {source}"
        kwargs['technical_details'] = kwargs.get('technical_details', {})
        kwargs['technical_details'].update({
            "camera_id": camera_id,
            "source": source
        })
        kwargs['recovery_suggestions'] = [
            RecoverySuggestion(
                action="check_connection",
                description="Check camera connection and power",
                priority=1
            ),
            RecoverySuggestion(
                action="test_different_source",
                description="Try a different camera source or port",
                priority=2
            ),
            RecoverySuggestion(
                action="restart_camera_service",
                description="Restart camera service or driver",
                priority=3,
                risk_level="medium"
            )
        ]
        super().__init__(message, camera_id=camera_id, **kwargs)


class CameraNotFoundError(CameraError):
    """Camera not found or not detected."""

    def __init__(self, camera_id: str, **kwargs):
        message = f"Camera {camera_id} not found or not detected"
        kwargs['severity'] = ErrorSeverity.HIGH
        kwargs['recovery_suggestions'] = [
            RecoverySuggestion(
                action="rescan_cameras",
                description="Rescan for available cameras",
                automated=True,
                priority=1
            ),
            RecoverySuggestion(
                action="check_camera_setup",
                description="Verify camera setup and configuration",
                priority=2
            )
        ]
        super().__init__(message, camera_id=camera_id, **kwargs)


class CameraTimeoutError(CameraError):
    """Camera operation timed out."""

    def __init__(self, camera_id: str, operation: str, timeout: float, **kwargs):
        message = f"Camera {camera_id} operation '{operation}' timed out after {timeout}s"
        kwargs['technical_details'] = kwargs.get('technical_details', {})
        kwargs['technical_details'].update({
            "operation": operation,
            "timeout_seconds": timeout
        })
        super().__init__(message, camera_id=camera_id, **kwargs)


class CameraFrameError(CameraError):
    """Error reading frame from camera."""

    def __init__(self, camera_id: str, frame_number: Optional[int] = None, **kwargs):
        message = f"Failed to read frame from camera {camera_id}"
        if frame_number:
            message += f" (frame #{frame_number})"

        kwargs['recovery_suggestions'] = [
            RecoverySuggestion(
                action="retry_frame_read",
                description="Retry reading frame",
                automated=True,
                priority=1
            ),
            RecoverySuggestion(
                action="reset_camera_buffer",
                description="Reset camera buffer",
                automated=True,
                priority=2
            )
        ]
        super().__init__(message, camera_id=camera_id, **kwargs)


class RTSPError(CameraError):
    """RTSP stream specific errors."""

    def __init__(self, url: str, **kwargs):
        message = f"RTSP stream error for URL: {url}"
        kwargs['technical_details'] = kwargs.get('technical_details', {})
        kwargs['technical_details']['rtsp_url'] = url
        kwargs['recovery_suggestions'] = [
            RecoverySuggestion(
                action="check_rtsp_credentials",
                description="Verify RTSP username and password",
                priority=1
            ),
            RecoverySuggestion(
                action="test_rtsp_connectivity",
                description="Test network connectivity to RTSP source",
                priority=2
            ),
            RecoverySuggestion(
                action="try_different_rtsp_transport",
                description="Try different RTSP transport protocol (TCP/UDP)",
                priority=3
            )
        ]
        super().__init__(message, **kwargs)


# ============ STREAMING ERRORS ============

class StreamProcessingError(ZoomCamError):
    """Base class for stream processing errors."""

    def __init__(self, message: str, **kwargs):
        kwargs['category'] = ErrorCategory.STREAMING
        context = kwargs.get('context') or ErrorContext(component="stream_processor")
        kwargs['context'] = context
        super().__init__(message, **kwargs)


class HLSGenerationError(StreamProcessingError):
    """HLS stream generation failed."""

    def __init__(self, reason: str, **kwargs):
        message = f"HLS stream generation failed: {reason}"
        kwargs['severity'] = ErrorSeverity.HIGH
        kwargs['recovery_suggestions'] = [
            RecoverySuggestion(
                action="restart_ffmpeg",
                description="Restart FFmpeg process",
                automated=True,
                priority=1
            ),
            RecoverySuggestion(
                action="check_output_directory",
                description="Verify HLS output directory permissions",
                priority=2
            ),
            RecoverySuggestion(
                action="reduce_stream_quality",
                description="Reduce stream quality to lower processing load",
                automated=True,
                priority=3
            )
        ]
        super().__init__(message, **kwargs)


class FrameCompositionError(StreamProcessingError):
    """Error composing multiple camera frames."""

    def __init__(self, camera_count: int, **kwargs):
        message = f"Failed to compose frame from {camera_count} cameras"
        kwargs['technical_details'] = kwargs.get('technical_details', {})
        kwargs['technical_details']['camera_count'] = camera_count
        super().__init__(message, **kwargs)


class StreamingPerformanceError(StreamProcessingError):
    """Streaming performance degraded."""

    def __init__(self, fps_target: float, fps_actual: float, **kwargs):
        message = f"Streaming performance degraded: {fps_actual:.1f}/{fps_target:.1f} FPS"
        kwargs['severity'] = ErrorSeverity.MEDIUM
        kwargs['technical_details'] = kwargs.get('technical_details', {})
        kwargs['technical_details'].update({
            "fps_target": fps_target,
            "fps_actual": fps_actual,
            "performance_ratio": fps_actual / fps_target
        })
        kwargs['recovery_suggestions'] = [
            RecoverySuggestion(
                action="reduce_resolution",
                description="Reduce camera resolution to improve performance",
                automated=True,
                priority=1
            ),
            RecoverySuggestion(
                action="reduce_frame_rate",
                description="Lower target frame rate",
                automated=True,
                priority=2
            ),
            RecoverySuggestion(
                action="disable_inactive_cameras",
                description="Temporarily disable inactive cameras",
                automated=True,
                priority=3
            )
        ]
        super().__init__(message, **kwargs)


# ============ LAYOUT ERRORS ============

class LayoutError(ZoomCamError):
    """Base class for layout engine errors."""

    def __init__(self, message: str, **kwargs):
        kwargs['category'] = ErrorCategory.LAYOUT
        context = kwargs.get('context') or ErrorContext(component="layout_engine")
        kwargs['context'] = context
        super().__init__(message, **kwargs)


class LayoutCalculationError(LayoutError):
    """Layout calculation failed."""

    def __init__(self, fragment_count: int, **kwargs):
        message = f"Failed to calculate layout for {fragment_count} fragments"
        kwargs['technical_details'] = kwargs.get('technical_details', {})
        kwargs['technical_details']['fragment_count'] = fragment_count
        kwargs['recovery_suggestions'] = [
            RecoverySuggestion(
                action="use_fallback_layout",
                description="Use simple grid layout as fallback",
                automated=True,
                priority=1
            ),
            RecoverySuggestion(
                action="reduce_fragments",
                description="Reduce number of camera fragments",
                automated=True,
                priority=2
            )
        ]
        super().__init__(message, **kwargs)


class InvalidResolutionError(LayoutError):
    """Invalid resolution specified."""

    def __init__(self, resolution: str, **kwargs):
        message = f"Invalid resolution specified: {resolution}"
        kwargs['severity'] = ErrorSeverity.MEDIUM
        kwargs['technical_details'] = kwargs.get('technical_details', {})
        kwargs['technical_details']['invalid_resolution'] = resolution
        kwargs['recovery_suggestions'] = [
            RecoverySuggestion(
                action="use_default_resolution",
                description="Use default resolution (1920x1080)",
                automated=True,
                priority=1
            ),
            RecoverySuggestion(
                action="auto_detect_resolution",
                description="Auto-detect display resolution",
                automated=True,
                priority=2
            )
        ]
        super().__init__(message, **kwargs)


# ============ CONFIGURATION ERRORS ============

class ConfigurationError(ZoomCamError):
    """Base class for configuration errors."""

    def __init__(self, message: str, **kwargs):
        kwargs['category'] = ErrorCategory.CONFIGURATION
        context = kwargs.get('context') or ErrorContext(component="config_manager")
        kwargs['context'] = context
        super().__init__(message, **kwargs)


class ConfigFileError(ConfigurationError):
    """Configuration file error."""

    def __init__(self, config_file: str, reason: str, **kwargs):
        message = f"Configuration file error in {config_file}: {reason}"
        kwargs['technical_details'] = kwargs.get('technical_details', {})
        kwargs['technical_details'].update({
            "config_file": config_file,
            "error_reason": reason
        })
        kwargs['recovery_suggestions'] = [
            RecoverySuggestion(
                action="create_default_config",
                description="Create default configuration file",
                automated=True,
                priority=1
            ),
            RecoverySuggestion(
                action="validate_config_syntax",
                description="Check configuration file syntax",
                priority=2
            ),
            RecoverySuggestion(
                action="restore_backup_config",
                description="Restore from backup configuration",
                priority=3
            )
        ]
        super().__init__(message, **kwargs)


class ConfigValidationError(ConfigurationError):
    """Configuration validation failed."""

    def __init__(self, field: str, value: Any, expected: str, **kwargs):
        message = f"Invalid configuration value for '{field}': {value} (expected: {expected})"
        kwargs['technical_details'] = kwargs.get('technical_details', {})
        kwargs['technical_details'].update({
            "field": field,
            "invalid_value": str(value),
            "expected_format": expected
        })
        super().__init__(message, **kwargs)


# ============ PERFORMANCE ERRORS ============

class PerformanceError(ZoomCamError):
    """Base class for performance-related errors."""

    def __init__(self, message: str, **kwargs):
        kwargs['category'] = ErrorCategory.PERFORMANCE
        context = kwargs.get('context') or ErrorContext(component="performance_monitor")
        kwargs['context'] = context
        super().__init__(message, **kwargs)


class HighCPUUsageError(PerformanceError):
    """CPU usage too high."""

    def __init__(self, cpu_percent: float, threshold: float, **kwargs):
        message = f"High CPU usage detected: {cpu_percent:.1f}% (threshold: {threshold:.1f}%)"
        kwargs['severity'] = ErrorSeverity.HIGH if cpu_percent > 90 else ErrorSeverity.MEDIUM
        kwargs['technical_details'] = kwargs.get('technical_details', {})
        kwargs['technical_details'].update({
            "cpu_percent": cpu_percent,
            "threshold": threshold
        })
        kwargs['recovery_suggestions'] = [
            RecoverySuggestion(
                action="reduce_camera_quality",
                description="Reduce camera resolution and frame rate",
                automated=True,
                priority=1
            ),
            RecoverySuggestion(
                action="disable_non_essential_features",
                description="Disable recording and advanced processing",
                automated=True,
                priority=2
            ),
            RecoverySuggestion(
                action="reduce_active_cameras",
                description="Temporarily disable some cameras",
                automated=False,
                priority=3
            )
        ]
        super().__init__(message, **kwargs)


class MemoryError(PerformanceError):
    """Memory usage too high."""

    def __init__(self, memory_percent: float, available_mb: float, **kwargs):
        message = f"High memory usage: {memory_percent:.1f}% (available: {available_mb:.0f}MB)"
        kwargs['severity'] = ErrorSeverity.HIGH if memory_percent > 90 else ErrorSeverity.MEDIUM
        kwargs['technical_details'] = kwargs.get('technical_details', {})
        kwargs['technical_details'].update({
            "memory_percent": memory_percent,
            "available_mb": available_mb
        })
        kwargs['recovery_suggestions'] = [
            RecoverySuggestion(
                action="clear_buffers",
                description="Clear video buffers and caches",
                automated=True,
                priority=1
            ),
            RecoverySuggestion(
                action="reduce_buffer_sizes",
                description="Reduce buffer sizes for streams",
                automated=True,
                priority=2
            )
        ]
        super().__init__(message, **kwargs)


# ============ NETWORK ERRORS ============

class NetworkError(ZoomCamError):
    """Base class for network-related errors."""

    def __init__(self, message: str, **kwargs):
        kwargs['category'] = ErrorCategory.NETWORK
        context = kwargs.get('context') or ErrorContext(component="network")
        kwargs['context'] = context
        super().__init__(message, **kwargs)


class ConnectionTimeoutError(NetworkError):
    """Network connection timed out."""

    def __init__(self, host: str, port: int, timeout: float, **kwargs):
        message = f"Connection timeout to {host}:{port} after {timeout}s"
        kwargs['technical_details'] = kwargs.get('technical_details', {})
        kwargs['technical_details'].update({
            "host": host,
            "port": port,
            "timeout": timeout
        })
        super().__init__(message, **kwargs)


# ============ MOTION DETECTION ERRORS ============

class MotionDetectionError(ZoomCamError):
    """Motion detection processing error."""

    def __init__(self, message: str, camera_id: Optional[str] = None, **kwargs):
        kwargs['category'] = ErrorCategory.CAMERA
        context = kwargs.get('context') or ErrorContext(
            component="motion_detector",
            camera_id=camera_id
        )
        kwargs['context'] = context
        super().__init__(message, **kwargs)


# ============ INTERPOLATION ERRORS ============

class InterpolationError(ZoomCamError):
    """Image interpolation error."""

    def __init__(self, message: str, **kwargs):
        kwargs['category'] = ErrorCategory.SOFTWARE
        context = kwargs.get('context') or ErrorContext(component="interpolation_engine")
        kwargs['context'] = context
        kwargs['recovery_suggestions'] = [
            RecoverySuggestion(
                action="use_fallback_algorithm",
                description="Use simpler interpolation algorithm",
                automated=True,
                priority=1
            ),
            RecoverySuggestion(
                action="skip_interpolation",
                description="Skip interpolation for this frame",
                automated=True,
                priority=2
            )
        ]
        super().__init__(message, **kwargs)


# ============ GIT LOGGER ERRORS ============

class GitLoggerError(ZoomCamError):
    """Git logging system error."""

    def __init__(self, message: str, **kwargs):
        kwargs['category'] = ErrorCategory.SOFTWARE
        context = kwargs.get('context') or ErrorContext(component="git_logger")
        kwargs['context'] = context
        kwargs['severity'] = ErrorSeverity.LOW  # Git logging errors shouldn't break the system
        super().__init__(message, **kwargs)


# ============ AUTO CONFIG ERRORS ============

class AutoConfigError(ZoomCamError):
    """Auto configuration management error."""

    def __init__(self, message: str, **kwargs):
        kwargs['category'] = ErrorCategory.CONFIGURATION
        context = kwargs.get('context') or ErrorContext(component="auto_config_manager")
        kwargs['context'] = context
        super().__init__(message, **kwargs)


# ============ UTILITY FUNCTIONS ============

def handle_exception(
    operation: str,
    exception: Exception,
    component: str,
    camera_id: Optional[str] = None,
    severity: Optional[ErrorSeverity] = None,
    additional_context: Optional[Dict[str, Any]] = None
) -> ZoomCamError:
    """Convert generic exception to ZoomCamError with context."""

    # Determine appropriate ZoomCam exception type
    if isinstance(exception, ZoomCamError):
        return exception

    # Map common exceptions to ZoomCam exceptions
    if isinstance(exception, ConnectionError):
        if camera_id:
            return CameraConnectionError(
                camera_id=camera_id,
                source="unknown",
                original_exception=exception,
                context=ErrorContext(
                    component=component,
                    operation=operation,
                    camera_id=camera_id,
                    system_state=additional_context
                )
            )
        else:
            return NetworkError(
                message=f"Connection error during {operation}: {exception}",
                original_exception=exception,
                context=ErrorContext(
                    component=component,
                    operation=operation,
                    system_state=additional_context
                )
            )

    elif isinstance(exception, TimeoutError):
        return CameraTimeoutError(
            camera_id=camera_id or "unknown",
            operation=operation,
            timeout=30.0,  # Default timeout
            original_exception=exception,
            context=ErrorContext(
                component=component,
                operation=operation,
                camera_id=camera_id,
                system_state=additional_context
            )
        )

    elif isinstance(exception, ValueError):
        return ConfigValidationError(
            field="unknown",
            value=str(exception),
            expected="valid value",
            original_exception=exception,
            context=ErrorContext(
                component=component,
                operation=operation,
                system_state=additional_context
            )
        )

    elif isinstance(exception, FileNotFoundError):
        return ConfigFileError(
            config_file=str(exception.filename) if exception.filename else "unknown",
            reason="File not found",
            original_exception=exception,
            context=ErrorContext(
                component=component,
                operation=operation,
                system_state=additional_context
            )
        )

    else:
        # Generic ZoomCam error for unknown exceptions
        return ZoomCamError(
            message=f"Unexpected error in {operation}: {type(exception).__name__}: {exception}",
            severity=severity or ErrorSeverity.MEDIUM,
            category=ErrorCategory.SOFTWARE,
            original_exception=exception,
            context=ErrorContext(
                component=component,
                operation=operation,
                camera_id=camera_id,
                system_state=additional_context
            )
        )


def create_error_response(error: ZoomCamError) -> Dict[str, Any]:
    """Create standardized error response for API."""
    return {
        "success": False,
        "error": {
            "code": error.error_code,
            "message": error.user_message,
            "technical_message": error.message,
            "severity": error.severity.value,
            "category": error.category.value,
            "recovery_suggestions": [
                {
                    "action": s.action,
                    "description": s.description,
                    "automated": s.automated,
                    "priority": s.priority
                }
                for s in error.recovery_suggestions
                if s.priority <= 2  # Only show high and medium priority suggestions
            ],
            "timestamp": error.context.timestamp.isoformat() if error.context.timestamp else None,
            "component": error.context.component
        }
    }


def log_error(error: ZoomCamError, logger=None):
    """Log ZoomCamError with appropriate level."""
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)

    # Choose log level based on severity
    if error.severity == ErrorSeverity.CRITICAL:
        log_level = logging.CRITICAL
    elif error.severity == ErrorSeverity.HIGH:
        log_level = logging.ERROR
    elif error.severity == ErrorSeverity.MEDIUM:
        log_level = logging.WARNING
    else:
        log_level = logging.INFO

    # Log with context
    logger.log(
        log_level,
        f"{error.error_code}: {error.message}",
        extra={
            'error_code': error.error_code,
            'component': error.context.component,
            'camera_id': error.context.camera_id,
            'operation': error.context.operation,
            'severity': error.severity.value,
            'category': error.category.value,
            'recovery_suggestions_count': len(error.recovery_suggestions)
        },
        exc_info=error.original_exception is not None
    )


# Exception handling decorators
def handle_camera_exceptions(camera_id: str):
    """Decorator to handle camera-related exceptions."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error = handle_exception(
                    operation=func.__name__,
                    exception=e,
                    component="camera_manager",
                    camera_id=camera_id
                )
                log_error(error)
                raise error
        return wrapper
    return decorator


def handle_stream_exceptions(func):
    """Decorator to handle streaming-related exceptions."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error = handle_exception(
                operation=func.__name__,
                exception=e,
                component="stream_processor"
            )
            log_error(error)
            raise error
    return wrapper


def handle_config_exceptions(func):
    """Decorator to handle configuration-related exceptions."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error = handle_exception(
                operation=func.__name__,
                exception=e,
                component="config_manager"
            )
            log_error(error)
            raise error
    return wrapper


if __name__ == "__main__":
    # Example usage
    try:
        # Simulate a camera error
        raise CameraConnectionError(
            camera_id="camera_1",
            source="/dev/video0",
            severity=ErrorSeverity.HIGH,
            technical_details={"attempts": 3, "last_error": "Device busy"}
        )
    except ZoomCamError as e:
        print("Error occurred:")
        print(f"Code: {e.error_code}")
        print(f"Message: {e.user_message}")
        print(f"Severity: {e.severity.value}")
        print(f"Recovery suggestions:")
        for suggestion in e.recovery_suggestions:
            print(f"  - {suggestion.description}")

        # Convert to API response
        response = create_error_response(e)
        print("\nAPI Response:")
        print(json.dumps(response, indent=2))