"""
Auto Config Manager - Automatic Configuration Management
=======================================================

Monitors system performance and camera activity to automatically
adjust configuration parameters for optimal performance.
"""

import asyncio
import logging
import yaml
import psutil
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import statistics

from zoomcam.config.config_manager import ConfigManager
from zoomcam.core.git_logger import GitLogger
from zoomcam.utils.exceptions import AutoConfigError


@dataclass
class PerformanceMetrics:
    """System performance metrics."""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    memory_available_mb: int
    fps_actual: float
    fps_target: float
    processing_time_ms: float
    camera_count: int
    active_cameras: int


@dataclass
class CameraMetrics:
    """Camera-specific metrics."""
    camera_id: str
    resolution: Tuple[int, int]
    fps: float
    activity_level: float
    motion_zones: int
    error_count: int
    last_activity: datetime


@dataclass
class OptimizationSuggestion:
    """Configuration optimization suggestion."""
    parameter: str
    current_value: Any
    suggested_value: Any
    reason: str
    priority: int  # 1=low, 2=medium, 3=high
    impact: str


class AutoConfigManager:
    """
    Automatically manages configuration based on system performance.

    Features:
    - Real-time performance monitoring
    - Adaptive quality adjustment
    - Camera resolution optimization
    - Layout algorithm tuning
    - Performance-based recommendations
    - Automatic configuration updates
    """

    def __init__(self, config_manager: ConfigManager, git_logger: Optional[GitLogger] = None):
        self.config_manager = config_manager
        self.git_logger = git_logger

        # Configuration
        self.monitoring_enabled = True
        self.auto_adjust_enabled = True
        self.monitoring_interval = 5.0  # seconds
        self.optimization_interval = 60.0  # seconds

        # Performance tracking
        self.performance_history: List[PerformanceMetrics] = []
        self.camera_metrics: Dict[str, CameraMetrics] = {}
        self.max_history_size = 720  # 1 hour at 5s intervals

        # Thresholds
        self.cpu_warning_threshold = 80.0
        self.cpu_critical_threshold = 90.0
        self.memory_warning_threshold = 80.0
        self.fps_drop_threshold = 0.8  # 80% of target FPS

        # Auto-configuration state
        self.auto_config_path = Path("config/auto-config.yaml")
        self.auto_config: Dict[str, Any] = {}
        self.last_optimization = datetime.now()
        self.optimization_suggestions: List[OptimizationSuggestion] = []

        # Load existing auto-config
        self._load_auto_config()

        logging.info("Auto Config Manager initialized")

    def _load_auto_config(self) -> None:
        """Load existing auto-configuration."""
        try:
            if self.auto_config_path.exists():
                with open(self.auto_config_path, 'r') as f:
                    self.auto_config = yaml.safe_load(f) or {}
            else:
                self.auto_config = self._get_default_auto_config()
                self._save_auto_config()
        except Exception as e:
            logging.error(f"Failed to load auto-config: {e}")
            self.auto_config = self._get_default_auto_config()

    def _get_default_auto_config(self) -> Dict[str, Any]:
        """Get default auto-configuration."""
        return {
            "system": {
                "detected_display": {
                    "actual_resolution": "1920x1080",
                    "refresh_rate": 60,
                    "color_depth": 24,
                    "aspect_ratio": "16:9"
                },
                "performance": {
                    "current_cpu_usage": 0.0,
                    "memory_usage": 0,
                    "gpu_acceleration": False,
                    "last_optimization": None
                }
            },
            "cameras": {},
            "layout": {
                "current_grid": "1x1",
                "active_fragments": 0,
                "last_recalculation": None,
                "css_grid_state": "",
                "efficiency_score": 0.0
            },
            "optimization": {
                "suggestions": [],
                "last_applied": [],
                "performance_trend": "stable"
            }
        }

    def _save_auto_config(self) -> None:
        """Save auto-configuration to file."""
        try:
            self.auto_config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.auto_config_path, 'w') as f:
                yaml.dump(self.auto_config, f, default_flow_style=False, indent=2)
        except Exception as e:
            logging.error(f"Failed to save auto-config: {e}")

    async def start_monitoring(self) -> None:
        """Start performance monitoring and optimization."""
        if not self.monitoring_enabled:
            return

        logging.info("Starting auto-config monitoring")

        # Start monitoring tasks
        monitor_task = asyncio.create_task(self._performance_monitoring_loop())
        optimize_task = asyncio.create_task(self._optimization_loop())

        try:
            await asyncio.gather(monitor_task, optimize_task)
        except asyncio.CancelledError:
            logging.info("Auto-config monitoring cancelled")
        except Exception as e:
            logging.error(f"Auto-config monitoring error: {e}")

    async def _performance_monitoring_loop(self) -> None:
        """Main performance monitoring loop."""
        while self.monitoring_enabled:
            try:
                # Collect performance metrics
                metrics = self._collect_performance_metrics()

                # Add to history
                self.performance_history.append(metrics)
                if len(self.performance_history) > self.max_history_size:
                    self.performance_history.pop(0)

                # Update auto-config
                self._update_auto_config_performance(metrics)

                # Check for immediate issues
                await self._check_performance_alerts(metrics)

                await asyncio.sleep(self.monitoring_interval)

            except Exception as e:
                logging.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(self.monitoring_interval)

    async def _optimization_loop(self) -> None:
        """Optimization analysis loop."""
        while self.monitoring_enabled:
            try:
                await asyncio.sleep(self.optimization_interval)

                # Perform optimization analysis
                if len(self.performance_history) >= 10:  # Need some history
                    await self._analyze_and_optimize()

            except Exception as e:
                logging.error(f"Optimization loop error: {e}")

    def _collect_performance_metrics(self) -> PerformanceMetrics:
        """Collect current system performance metrics."""
        try:
            # System metrics
            cpu_usage = psutil.cpu_percent(interval=1.0)
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            memory_available_mb = memory.available / (1024 * 1024)

            # Get FPS and processing metrics from components
            fps_actual = self._get_current_fps()
            fps_target = self._get_target_fps()
            processing_time_ms = self._get_processing_time()

            # Camera metrics
            camera_count = len(self.camera_metrics)
            active_cameras = len([c for c in self.camera_metrics.values()
                                  if (datetime.now() - c.last_activity).total_seconds() < 30])

            return PerformanceMetrics(
                timestamp=datetime.now(),
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                memory_available_mb=memory_available_mb,
                fps_actual=fps_actual,
                fps_target=fps_target,
                processing_time_ms=processing_time_ms,
                camera_count=camera_count,
                active_cameras=active_cameras
            )

        except Exception as e:
            logging.error(f"Failed to collect performance metrics: {e}")
            return PerformanceMetrics(
                timestamp=datetime.now(),
                cpu_usage=0.0, memory_usage=0.0, memory_available_mb=0,
                fps_actual=0.0, fps_target=30.0, processing_time_ms=0.0,
                camera_count=0, active_cameras=0
            )

    def _get_current_fps(self) -> float:
        """Get current FPS from stream processor."""
        # This would be injected from the stream processor
        return 30.0  # Placeholder

    def _get_target_fps(self) -> float:
        """Get target FPS from configuration."""
        try:
            config = self.config_manager.get_config()
            return config.get("streaming", {}).get("target_fps", 30.0)
        except:
            return 30.0

    def _get_processing_time(self) -> float:
        """Get average processing time."""
        # This would be injected from components
        return 33.0  # Placeholder (33ms for 30fps)

    def _update_auto_config_performance(self, metrics: PerformanceMetrics) -> None:
        """Update auto-config with performance metrics."""
        self.auto_config["system"]["performance"].update({
            "current_cpu_usage": metrics.cpu_usage,
            "memory_usage": metrics.memory_usage,
            "memory_available_mb": metrics.memory_available_mb,
            "fps_actual": metrics.fps_actual,
            "processing_time_ms": metrics.processing_time_ms,
            "last_update": metrics.timestamp.isoformat()
        })

        # Save periodically
        if len(self.performance_history) % 12 == 0:  # Every minute
            self._save_auto_config()

    async def _check_performance_alerts(self, metrics: PerformanceMetrics) -> None:
        """Check for performance issues and trigger alerts."""
        alerts = []

        # CPU usage alerts
        if metrics.cpu_usage > self.cpu_critical_threshold:
            alerts.append({
                "type": "cpu_critical",
                "message": f"Critical CPU usage: {metrics.cpu_usage:.1f}%",
                "severity": "critical",
                "suggestions": ["Reduce camera count", "Lower resolution", "Disable recording"]
            })
        elif metrics.cpu_usage > self.cpu_warning_threshold:
            alerts.append({
                "type": "cpu_warning",
                "message": f"High CPU usage: {metrics.cpu_usage:.1f}%",
                "severity": "warning",
                "suggestions": ["Consider reducing quality", "Check camera efficiency"]
            })

        # Memory usage alerts
        if metrics.memory_usage > self.memory_warning_threshold:
            alerts.append({
                "type": "memory_warning",
                "message": f"High memory usage: {metrics.memory_usage:.1f}%",
                "severity": "warning",
                "suggestions": ["Reduce buffer sizes", "Lower camera resolution"]
            })

        # FPS drop alerts
        fps_ratio = metrics.fps_actual / max(metrics.fps_target, 1)
        if fps_ratio < self.fps_drop_threshold:
            alerts.append({
                "type": "fps_drop",
                "message": f"FPS below target: {metrics.fps_actual:.1f}/{metrics.fps_target:.1f}",
                "severity": "warning",
                "suggestions": ["Reduce processing load", "Optimize layout"]
            })

        # Log alerts
        for alert in alerts:
            if self.git_logger:
                await self.git_logger.log_performance_alert({
                    "alert": alert,
                    "metrics": asdict(metrics)
                })

            # Auto-apply critical fixes if enabled
            if alert["severity"] == "critical" and self.auto_adjust_enabled:
                await self._apply_emergency_optimizations(alert["type"])

    async def _apply_emergency_optimizations(self, alert_type: str) -> None:
        """Apply emergency optimizations for critical issues."""
        try:
            config = self.config_manager.get_config()
            changes = {}

            if alert_type == "cpu_critical":
                # Reduce quality settings
                changes["streaming.bitrate"] = "1M"  # Reduce from 2M
                changes["system.display.interpolation.algorithm"] = "linear"  # Fastest

                # Disable features for critical cameras
                for camera_id in config.get("cameras", {}):
                    changes[f"cameras.{camera_id}.recording.enabled"] = False
                    changes[f"cameras.{camera_id}.max_fragments"] = 1

            elif alert_type == "memory_warning":
                # Reduce buffer sizes
                changes["streaming.segment_count"] = 5  # Reduce from 10

            if changes:
                await self._apply_config_changes(changes, "Emergency optimization")
                logging.warning(f"Applied emergency optimizations for {alert_type}")

        except Exception as e:
            logging.error(f"Failed to apply emergency optimizations: {e}")

    async def _analyze_and_optimize(self) -> None:
        """Analyze performance history and generate optimizations."""
        try:
            # Get recent metrics
            recent_metrics = self.performance_history[-60:]  # Last 5 minutes
            if len(recent_metrics) < 10:
                return

            # Calculate trends
            cpu_trend = self._calculate_trend([m.cpu_usage for m in recent_metrics])
            memory_trend = self._calculate_trend([m.memory_usage for m in recent_metrics])
            fps_trend = self._calculate_trend([m.fps_actual for m in recent_metrics])

            # Generate suggestions
            suggestions = []

            # CPU optimization
            avg_cpu = statistics.mean([m.cpu_usage for m in recent_metrics])
            if avg_cpu > 70:
                suggestions.extend(self._generate_cpu_optimizations(avg_cpu, cpu_trend))

            # Memory optimization
            avg_memory = statistics.mean([m.memory_usage for m in recent_metrics])
            if avg_memory > 70:
                suggestions.extend(self._generate_memory_optimizations(avg_memory, memory_trend))

            # FPS optimization
            avg_fps = statistics.mean([m.fps_actual for m in recent_metrics])
            target_fps = recent_metrics[-1].fps_target
            if avg_fps < target_fps * 0.9:
                suggestions.extend(self._generate_fps_optimizations(avg_fps, target_fps))

            # Camera-specific optimizations
            suggestions.extend(self._generate_camera_optimizations())

            # Update suggestions
            self.optimization_suggestions = suggestions
            self._update_auto_config_suggestions(suggestions)

            # Auto-apply high-priority suggestions if enabled
            if self.auto_adjust_enabled:
                await self._apply_high_priority_suggestions(suggestions)

            self.last_optimization = datetime.now()

        except Exception as e:
            logging.error(f"Optimization analysis failed: {e}")

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from values."""
        if len(values) < 5:
            return "stable"

        # Simple linear regression slope
        x = list(range(len(values)))
        y = values

        n = len(values)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)

        if slope > 1:
            return "increasing"
        elif slope < -1:
            return "decreasing"
        else:
            return "stable"

    def _generate_cpu_optimizations(self, avg_cpu: float, trend: str) -> List[OptimizationSuggestion]:
        """Generate CPU optimization suggestions."""
        suggestions = []

        if avg_cpu > 85:
            suggestions.append(OptimizationSuggestion(
                parameter="streaming.bitrate",
                current_value="2M",
                suggested_value="1M",
                reason=f"High CPU usage ({avg_cpu:.1f}%) - reduce bitrate",
                priority=3,
                impact="Reduced video quality but lower CPU load"
            ))

            suggestions.append(OptimizationSuggestion(
                parameter="system.display.interpolation.algorithm",
                current_value="lanczos",
                suggested_value="linear",
                reason="CPU overload - use faster interpolation",
                priority=3,
                impact="Faster processing but slightly lower image quality"
            ))

        elif avg_cpu > 75 and trend == "increasing":
            suggestions.append(OptimizationSuggestion(
                parameter="layout.algorithm",
                current_value="adaptive_grid",
                suggested_value="equal_grid",
                reason="Rising CPU usage - simplify layout calculations",
                priority=2,
                impact="Less adaptive layout but better performance"
            ))

        return suggestions

    def _generate_memory_optimizations(self, avg_memory: float, trend: str) -> List[OptimizationSuggestion]:
        """Generate memory optimization suggestions."""
        suggestions = []

        if avg_memory > 80:
            suggestions.append(OptimizationSuggestion(
                parameter="streaming.segment_count",
                current_value=10,
                suggested_value=5,
                reason=f"High memory usage ({avg_memory:.1f}%) - reduce buffer",
                priority=3,
                impact="Less buffering but reduced stream stability"
            ))

        return suggestions

    def _generate_fps_optimizations(self, avg_fps: float, target_fps: float) -> List[OptimizationSuggestion]:
        """Generate FPS optimization suggestions."""
        suggestions = []

        fps_deficit = target_fps - avg_fps
        if fps_deficit > 5:
            suggestions.append(OptimizationSuggestion(
                parameter="streaming.target_fps",
                current_value=target_fps,
                suggested_value=max(15, target_fps - 5),
                reason=f"FPS dropping ({avg_fps:.1f}/{target_fps:.1f}) - reduce target",
                priority=2,
                impact="Lower frame rate but more stable performance"
            ))

        return suggestions

    def _generate_camera_optimizations(self) -> List[OptimizationSuggestion]:
        """Generate camera-specific optimizations."""
        suggestions = []

        for camera_id, metrics in self.camera_metrics.items():
            # Inactive camera optimization
            time_since_activity = (datetime.now() - metrics.last_activity).total_seconds()
            if time_since_activity > 300 and metrics.activity_level < 0.1:  # 5 minutes inactive
                suggestions.append(OptimizationSuggestion(
                    parameter=f"cameras.{camera_id}.recording.enabled",
                    current_value=True,
                    suggested_value=False,
                    reason=f"Camera {camera_id} inactive for {time_since_activity / 60:.1f} minutes",
                    priority=1,
                    impact="Reduced processing load for inactive camera"
                ))

            # High error rate optimization
            if metrics.error_count > 10:
                suggestions.append(OptimizationSuggestion(
                    parameter=f"cameras.{camera_id}.resolution",
                    current_value=f"{metrics.resolution[0]}x{metrics.resolution[1]}",
                    suggested_value="auto",
                    reason=f"Camera {camera_id} has {metrics.error_count} errors",
                    priority=2,
                    impact="Auto-detect resolution to fix connection issues"
                ))

        return suggestions

    def _update_auto_config_suggestions(self, suggestions: List[OptimizationSuggestion]) -> None:
        """Update auto-config with optimization suggestions."""
        self.auto_config["optimization"]["suggestions"] = [
            {
                "parameter": s.parameter,
                "current_value": s.current_value,
                "suggested_value": s.suggested_value,
                "reason": s.reason,
                "priority": s.priority,
                "impact": s.impact,
                "timestamp": datetime.now().isoformat()
            }
            for s in suggestions
        ]
        self._save_auto_config()

    async def _apply_high_priority_suggestions(self, suggestions: List[OptimizationSuggestion]) -> None:
        """Auto-apply high priority suggestions."""
        high_priority = [s for s in suggestions if s.priority >= 3]

        if not high_priority:
            return

        changes = {}
        applied_suggestions = []

        for suggestion in high_priority:
            changes[suggestion.parameter] = suggestion.suggested_value
            applied_suggestions.append({
                "parameter": suggestion.parameter,
                "reason": suggestion.reason,
                "impact": suggestion.impact
            })

        if changes:
            await self._apply_config_changes(changes, "Auto-optimization")

            # Log applied suggestions
            self.auto_config["optimization"]["last_applied"] = applied_suggestions
            self._save_auto_config()

            logging.info(f"Auto-applied {len(changes)} optimization changes")

    async def _apply_config_changes(self, changes: Dict[str, Any], reason: str) -> None:
        """Apply configuration changes."""
        try:
            config = self.config_manager.get_config()
            original_config = config.copy()

            # Apply changes
            for key, value in changes.items():
                self._set_nested_value(config, key, value)

            # Save configuration
            await self.config_manager.save_config(config)

            # Log to Git if available
            if self.git_logger:
                await self.git_logger.log_config_change(
                    {
                        "reason": reason,
                        "changes": changes,
                        "timestamp": datetime.now().isoformat()
                    },
                    config_snapshot=config
                )

            logging.info(f"Applied config changes: {reason}")

        except Exception as e:
            logging.error(f"Failed to apply config changes: {e}")
            raise AutoConfigError(f"Config update failed: {e}")

    def _set_nested_value(self, config: Dict, key: str, value: Any) -> None:
        """Set nested configuration value using dot notation."""
        keys = key.split('.')
        current = config

        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        current[keys[-1]] = value

    async def update_camera_detection(self, detected_cameras: Dict[str, Any]) -> None:
        """Update auto-config with detected cameras."""
        self.auto_config["cameras"] = {
            camera_id: {
                "detected_resolution": camera_info["resolution"],
                "source_type": camera_info["type"],
                "last_detected": datetime.now().isoformat(),
                "auto_configured": True
            }
            for camera_id, camera_info in detected_cameras.items()
        }
        self._save_auto_config()

    async def update_motion_data(self, camera_id: str, motion_zones: List[Dict]) -> None:
        """Update auto-config with motion detection data."""
        if camera_id not in self.camera_metrics:
            self.camera_metrics[camera_id] = CameraMetrics(
                camera_id=camera_id,
                resolution=(640, 480),  # Default
                fps=30.0,
                activity_level=0.0,
                motion_zones=0,
                error_count=0,
                last_activity=datetime.now()
            )

        metrics = self.camera_metrics[camera_id]

        if motion_zones:
            # Calculate activity level
            total_activity = sum(zone.get("activity_level", 0) for zone in motion_zones)
            metrics.activity_level = min(1.0, total_activity / len(motion_zones))
            metrics.motion_zones = len(motion_zones)
            metrics.last_activity = datetime.now()
        else:
            metrics.activity_level = 0.0
            metrics.motion_zones = 0

        # Update auto-config
        self.auto_config["cameras"][camera_id] = {
            **self.auto_config["cameras"].get(camera_id, {}),
            "activity_level": metrics.activity_level,
            "motion_zones": metrics.motion_zones,
            "last_activity": metrics.last_activity.isoformat()
        }

    async def update_layout_state(self, layout_result) -> None:
        """Update auto-config with layout state."""
        self.auto_config["layout"].update({
            "active_fragments": len(layout_result.fragments),
            "efficiency_score": layout_result.layout_efficiency,
            "last_recalculation": datetime.now().isoformat(),
            "css_grid_state": layout_result.css_template[:200] + "..." if len(
                layout_result.css_template) > 200 else layout_result.css_template
        })

    async def log_camera_config_change(self, camera_id: str, changes: Dict[str, Any]) -> None:
        """Log camera configuration change."""
        if self.git_logger:
            await self.git_logger.log_config_change(
                {
                    "camera_id": camera_id,
                    "changes": changes,
                    "timestamp": datetime.now().isoformat()
                },
                self.config_manager.get_config()
            )

    async def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status."""
        recent_metrics = self.performance_history[-10:] if self.performance_history else []

        if recent_metrics:
            avg_cpu = statistics.mean([m.cpu_usage for m in recent_metrics])
            avg_memory = statistics.mean([m.memory_usage for m in recent_metrics])
            avg_fps = statistics.mean([m.fps_actual for m in recent_metrics])
        else:
            avg_cpu = avg_memory = avg_fps = 0.0

        return {
            "monitoring_enabled": self.monitoring_enabled,
            "auto_adjust_enabled": self.auto_adjust_enabled,
            "last_optimization": self.last_optimization.isoformat(),
            "suggestions_count": len(self.optimization_suggestions),
            "high_priority_suggestions": len([s for s in self.optimization_suggestions if s.priority >= 3]),
            "performance_summary": {
                "avg_cpu_usage": avg_cpu,
                "avg_memory_usage": avg_memory,
                "avg_fps": avg_fps,
                "camera_count": len(self.camera_metrics),
                "active_cameras": len([c for c in self.camera_metrics.values()
                                       if (datetime.now() - c.last_activity).total_seconds() < 60])
            },
            "recent_suggestions": [
                {
                    "parameter": s.parameter,
                    "reason": s.reason,
                    "priority": s.priority
                }
                for s in self.optimization_suggestions[-5:]  # Last 5 suggestions
            ]
        }

    async def get_performance_history(self, hours: int = 1) -> List[Dict[str, Any]]:
        """Get performance history for specified hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        filtered_metrics = [
            {
                "timestamp": m.timestamp.isoformat(),
                "cpu_usage": m.cpu_usage,
                "memory_usage": m.memory_usage,
                "fps_actual": m.fps_actual,
                "fps_target": m.fps_target,
                "processing_time_ms": m.processing_time_ms,
                "active_cameras": m.active_cameras
            }
            for m in self.performance_history
            if m.timestamp > cutoff_time
        ]

        return filtered_metrics

    def get_auto_config(self) -> Dict[str, Any]:
        """Get current auto-configuration."""
        return self.auto_config.copy()

    async def shutdown(self) -> None:
        """Shutdown auto-config manager."""
        logging.info("Shutting down Auto Config Manager...")

        self.monitoring_enabled = False

        # Save final state
        self._save_auto_config()

        # Log shutdown
        if self.git_logger:
            await self.git_logger.log_event(
                "system_shutdown",
                {
                    "final_performance": asdict(self.performance_history[-1]) if self.performance_history else {},
                    "optimization_count": len(self.optimization_suggestions),
                    "monitoring_duration": (datetime.now() - (self.performance_history[
                                                                  0].timestamp if self.performance_history else datetime.now())).total_seconds()
                },
                summary="Auto Config Manager shutdown"
            )

        logging.info("Auto Config Manager shutdown complete")