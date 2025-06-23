"""
Configuration Validator
=======================

Validates ZoomCam configuration files against schema
with detailed error reporting and suggestions.
"""

import re
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from pathlib import Path
import ipaddress


@dataclass
class ValidationError:
    """Configuration validation error."""

    field: str
    message: str
    value: Any
    suggestion: Optional[str] = None
    severity: str = "error"  # error, warning, info


class ConfigValidator:
    """
    Configuration validator with comprehensive rules.

    Features:
    - Schema-based validation
    - Type checking
    - Range validation
    - Format validation (URLs, resolutions, etc.)
    - Cross-field validation
    - Custom validation rules
    - Detailed error messages with suggestions
    """

    def __init__(self):
        self.errors: List[ValidationError] = []
        self.warnings: List[ValidationError] = []

        # Validation rules
        self.rules = {
            # System validation
            "system.display.target_resolution": self._validate_resolution,
            "system.display.refresh_rate": lambda v: self._validate_range(
                v, 1, 120, int
            ),
            "system.performance.max_cpu_usage": lambda v: self._validate_percentage(v),
            "system.performance.memory_limit": lambda v: self._validate_range(
                v, 64, 8192, int
            ),
            # Camera validation
            "cameras.*.enabled": self._validate_boolean,
            "cameras.*.source": self._validate_camera_source,
            "cameras.*.resolution": self._validate_resolution_or_auto,
            "cameras.*.zoom": lambda v: self._validate_range(v, 1.0, 10.0, float),
            "cameras.*.max_fragments": lambda v: self._validate_range(v, 1, 5, int),
            # Recording validation
            "cameras.*.recording.enabled": self._validate_boolean,
            "cameras.*.recording.quality": lambda v: self._validate_choice(
                v, ["low", "medium", "high"]
            ),
            "cameras.*.recording.reaction_time": lambda v: self._validate_range(
                v, 0.1, 10.0, float
            ),
            "cameras.*.recording.max_duration": lambda v: self._validate_range(
                v, 10, 3600, int
            ),
            # Motion detection validation
            "cameras.*.motion_detection.sensitivity": lambda v: self._validate_range(
                v, 0.0, 1.0, float
            ),
            "cameras.*.motion_detection.min_area": lambda v: self._validate_range(
                v, 10, 10000, int
            ),
            "cameras.*.motion_detection.max_zones": lambda v: self._validate_range(
                v, 1, 10, int
            ),
            # Layout validation
            "layout.algorithm": lambda v: self._validate_choice(
                v, ["adaptive_grid", "priority_based", "equal_grid"]
            ),
            "layout.gap_size": lambda v: self._validate_range(v, 0, 50, int),
            "layout.border_width": lambda v: self._validate_range(v, 0, 10, int),
            "layout.inactive_timeout": lambda v: self._validate_range(v, 5, 300, int),
            # Streaming validation
            "streaming.hls_output_dir": self._validate_directory_path,
            "streaming.segment_duration": lambda v: self._validate_range(v, 1, 10, int),
            "streaming.segment_count": lambda v: self._validate_range(v, 3, 20, int),
            "streaming.bitrate": self._validate_bitrate,
            # Recording validation
            "recording.output_dir": self._validate_directory_path,
            "recording.cleanup_days": lambda v: self._validate_range(v, 1, 365, int),
            "recording.compression": lambda v: self._validate_choice(
                v, ["h264_fast", "h264_medium", "h264_slow"]
            ),
            # Git logging validation
            "logging.git.enabled": self._validate_boolean,
            "logging.git.commit_interval": lambda v: self._validate_range(
                v, 10, 3600, int
            ),
            "logging.git.screenshot_interval": lambda v: self._validate_range(
                v, 1, 300, int
            ),
            "logging.git.max_history_days": lambda v: self._validate_range(
                v, 1, 365, int
            ),
        }

    def validate(self, config: Dict[str, Any]) -> List[ValidationError]:
        """Validate entire configuration."""
        self.errors = []
        self.warnings = []

        try:
            # Basic structure validation
            self._validate_structure(config)

            # Field-specific validation
            self._validate_fields(config)

            # Cross-field validation
            self._validate_cross_fields(config)

            # Custom business logic validation
            self._validate_business_logic(config)

        except Exception as e:
            self.errors.append(
                ValidationError(
                    field="validation",
                    message=f"Validation error: {e}",
                    value=None,
                    severity="error",
                )
            )

        return self.errors + self.warnings

    def _validate_structure(self, config: Dict[str, Any]):
        """Validate basic configuration structure."""
        required_sections = ["system", "cameras", "layout", "streaming"]

        for section in required_sections:
            if section not in config:
                self.errors.append(
                    ValidationError(
                        field=section,
                        message=f"Required section '{section}' is missing",
                        value=None,
                        suggestion=f"Add '{section}:' section to configuration",
                    )
                )

        # Validate cameras section has at least one camera
        if "cameras" in config:
            if not config["cameras"] or not isinstance(config["cameras"], dict):
                self.errors.append(
                    ValidationError(
                        field="cameras",
                        message="At least one camera must be configured",
                        value=config.get("cameras"),
                        suggestion="Add at least one camera configuration",
                    )
                )

    def _validate_fields(self, config: Dict[str, Any], prefix: str = ""):
        """Validate individual fields against rules."""
        for key, value in config.items():
            current_path = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                # Handle wildcard patterns (e.g., cameras.*)
                if prefix == "cameras":
                    # This is a camera configuration
                    for camera_key, camera_value in value.items():
                        camera_path = f"cameras.{camera_key}"
                        if isinstance(camera_value, dict):
                            self._validate_fields(camera_value, camera_path)
                else:
                    # Recursively validate nested dictionaries
                    self._validate_fields(value, current_path)
            else:
                # Find matching validation rule
                matching_rule = self._find_matching_rule(current_path)
                if matching_rule:
                    try:
                        error = matching_rule(value)
                        if error:
                            self.errors.append(
                                ValidationError(
                                    field=current_path, message=error, value=value
                                )
                            )
                    except Exception as e:
                        self.errors.append(
                            ValidationError(
                                field=current_path,
                                message=f"Validation failed: {e}",
                                value=value,
                            )
                        )

    def _find_matching_rule(self, field_path: str) -> Optional[callable]:
        """Find matching validation rule for field path."""
        # Direct match
        if field_path in self.rules:
            return self.rules[field_path]

        # Wildcard match
        for pattern, rule in self.rules.items():
            if "*" in pattern:
                # Convert pattern to regex
                regex_pattern = pattern.replace("*", "[^.]+").replace(".", r"\.")
                if re.match(f"^{regex_pattern}$", field_path):
                    return rule

        return None

    def _validate_cross_fields(self, config: Dict[str, Any]):
        """Validate relationships between fields."""
        try:
            # Validate display resolution vs camera resolutions
            system_res = (
                config.get("system", {}).get("display", {}).get("target_resolution")
            )
            if system_res:
                system_width, system_height = self._parse_resolution(system_res)

                for camera_id, camera_config in config.get("cameras", {}).items():
                    camera_res = camera_config.get("resolution", "auto")
                    if camera_res != "auto":
                        try:
                            cam_width, cam_height = self._parse_resolution(camera_res)
                            if (
                                cam_width > system_width * 2
                                or cam_height > system_height * 2
                            ):
                                self.warnings.append(
                                    ValidationError(
                                        field=f"cameras.{camera_id}.resolution",
                                        message=f"Camera resolution ({camera_res}) is much higher than display resolution ({system_res})",
                                        value=camera_res,
                                        suggestion="Consider reducing camera resolution for better performance",
                                        severity="warning",
                                    )
                                )
                        except:
                            pass

            # Validate HLS settings
            segment_duration = config.get("streaming", {}).get("segment_duration", 2)
            segment_count = config.get("streaming", {}).get("segment_count", 10)
            total_buffer_time = segment_duration * segment_count

            if total_buffer_time < 10:
                self.warnings.append(
                    ValidationError(
                        field="streaming",
                        message=f"Total HLS buffer time ({total_buffer_time}s) is very low",
                        value=f"{segment_duration}s x {segment_count}",
                        suggestion="Increase segment_duration or segment_count for better stability",
                        severity="warning",
                    )
                )

            # Validate recording settings vs storage
            recording_enabled_count = 0
            for camera_config in config.get("cameras", {}).values():
                if camera_config.get("recording", {}).get("enabled", True):
                    recording_enabled_count += 1

            if recording_enabled_count > 4:
                self.warnings.append(
                    ValidationError(
                        field="recording",
                        message=f"Recording enabled for {recording_enabled_count} cameras may require significant storage",
                        value=recording_enabled_count,
                        suggestion="Monitor disk space usage or reduce recording duration",
                        severity="warning",
                    )
                )

        except Exception as e:
            self.errors.append(
                ValidationError(
                    field="cross_validation",
                    message=f"Cross-field validation error: {e}",
                    value=None,
                )
            )

    def _validate_business_logic(self, config: Dict[str, Any]):
        """Validate business logic rules."""
        try:
            # Check for camera source conflicts
            used_sources = set()
            for camera_id, camera_config in config.get("cameras", {}).items():
                source = camera_config.get("source")
                if source:
                    if source in used_sources:
                        self.errors.append(
                            ValidationError(
                                field=f"cameras.{camera_id}.source",
                                message=f"Camera source '{source}' is used by multiple cameras",
                                value=source,
                                suggestion="Each camera must have a unique source",
                            )
                        )
                    used_sources.add(source)

            # Check total fragment count
            total_fragments = 0
            for camera_config in config.get("cameras", {}).values():
                if camera_config.get("enabled", True):
                    total_fragments += camera_config.get("max_fragments", 2)

            if total_fragments > 16:
                self.warnings.append(
                    ValidationError(
                        field="layout",
                        message=f"Total fragments ({total_fragments}) may create crowded layout",
                        value=total_fragments,
                        suggestion="Consider reducing max_fragments for some cameras",
                        severity="warning",
                    )
                )

            # Check performance implications
            active_cameras = len(
                [
                    c
                    for c in config.get("cameras", {}).values()
                    if c.get("enabled", True)
                ]
            )
            fps = config.get("streaming", {}).get("target_fps", 30)
            performance_score = active_cameras * fps
            if performance_score > 120:  # 4 cameras at 30fps
                self.warnings.append(
                    ValidationError(
                        field="performance",
                        message=f"High performance requirements: {active_cameras} cameras at {fps} FPS",
                        value=f"{active_cameras}x{fps}",
                        suggestion="Consider reducing FPS or disabling some cameras",
                        severity="warning",
                    )
                )

        except Exception as e:
            self.errors.append(
                ValidationError(
                    field="business_logic",
                    message=f"Business logic validation error: {e}",
                    value=None,
                )
            )

    def _parse_resolution(self, resolution: str) -> Tuple[int, int]:
        """Parse resolution string into (width, height) tuple."""
        if resolution.lower() == "auto":
            return 1920, 1080  # Default resolution

        try:
            width, height = map(int, resolution.lower().split("x"))
            if width <= 0 or height <= 0:
                raise ValueError("Dimensions must be positive")
            return width, height
        except (ValueError, AttributeError) as e:
            raise ValueError(
                f"Invalid resolution format: {resolution}. Expected WxH (e.g., 1920x1080)"
            )

    def _validate_resolution(self, value: Any) -> Optional[str]:
        """Validate resolution format (e.g., 1920x1080)."""
        if not isinstance(value, str):
            return "Resolution must be a string"

        try:
            self._parse_resolution(value)
            return None
        except ValueError as e:
            return str(e)

    def _validate_resolution_or_auto(self, value: Any) -> Optional[str]:
        """Validate resolution or 'auto'."""
        if value == "auto":
            return None
        return self._validate_resolution(value)

    def _validate_range(
        self, value: Any, min_val: float, max_val: float, value_type: type
    ) -> Optional[str]:
        """Validate that a value is within the specified range."""
        try:
            typed_value = value_type(value)
            if not (min_val <= typed_value <= max_val):
                return f"Value must be between {min_val} and {max_val}"
            return None
        except (ValueError, TypeError):
            return (
                f"Value must be a {value_type.__name__} between {min_val} and {max_val}"
            )

    def _validate_boolean(self, value: Any) -> Optional[str]:
        """Validate boolean value."""
        if not isinstance(value, bool):
            return "Value must be a boolean (true/false)"
        return None

    def _validate_percentage(self, value: Any) -> Optional[str]:
        """Validate percentage value (0-100)."""
        # First try to validate as float, then as int if that fails
        error = self._validate_range(value, 0, 100, float)
        if error and isinstance(value, int):
            error = self._validate_range(value, 0, 100, int)
        return error

    def _validate_choice(self, value: Any, choices: list) -> Optional[str]:
        """Validate that value is one of the specified choices."""
        if value not in choices:
            return f"Value must be one of: {', '.join(map(str, choices))}"
        return None

    def _validate_camera_source(self, value: Any) -> Optional[str]:
        """Validate camera source (RTSP, USB, CSI, etc.)."""
        if not isinstance(value, str):
            return "Camera source must be a string"

        if value.startswith("rtsp://"):
            # Basic RTSP URL validation
            if not re.match(r"^rtsp://[^\s/$.?#].[^\s]*$", value):
                return "Invalid RTSP URL format"
        elif value.startswith("/dev/video"):
            # Basic video device validation
            if not re.match(r"^/dev/video\d+$", value):
                return "Invalid video device path"
        elif value != "auto":
            return "Source must be an RTSP URL, video device path, or 'auto'"

        return None

    def _validate_directory_path(self, value: Any) -> Optional[str]:
        """Validate directory path."""
        if not isinstance(value, str):
            return "Path must be a string"

        if not value.strip():
            return "Path cannot be empty"

        # Check for invalid characters (basic check)
        if any(c in value for c in '<>:"|?*'):
            return "Path contains invalid characters"

        return None

    def _validate_bitrate(self, value: Any) -> Optional[str]:
        """Validate bitrate value (e.g., '2M' for 2 Mbps)."""
        if not isinstance(value, str):
            return "Bitrate must be a string (e.g., '2M' for 2 Mbps)"

        try:
            if value.endswith("K"):
                bitrate = int(value[:-1]) * 1000
            elif value.endswith("M"):
                bitrate = int(value[:-1]) * 1000000
            else:
                bitrate = int(value)

            if bitrate <= 0:
                return "Bitrate must be positive"

            return None

        except (ValueError, TypeError):
            return "Invalid bitrate format. Use format like '2M' for 2 Mbps"
