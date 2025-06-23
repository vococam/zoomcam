"""
Interpolation Engine - Advanced Image Scaling and Enhancement
============================================================

Provides high-quality image interpolation with adaptive algorithms,
performance optimization, and quality enhancement features.
"""

import logging
import cv2
import numpy as np
import time
from typing import Dict, Any, Tuple, Optional, List
from enum import Enum
from dataclasses import dataclass
import threading
from functools import lru_cache

from zoomcam.utils.exceptions import InterpolationError


class InterpolationAlgorithm(Enum):
    """Available interpolation algorithms."""
    NEAREST = (cv2.INTER_NEAREST, "nearest", 1, 10)
    LINEAR = (cv2.INTER_LINEAR, "linear", 4, 8)
    CUBIC = (cv2.INTER_CUBIC, "cubic", 7, 5)
    LANCZOS = (cv2.INTER_LANCZOS4, "lanczos", 9, 3)

    def __init__(self, cv2_flag, name, quality, performance):
        self.cv2_flag = cv2_flag
        self.algorithm_name = name
        self.quality_score = quality  # 1-10 scale
        self.performance_score = performance  # 1-10 scale (higher = faster)


@dataclass
class InterpolationRequest:
    """Interpolation request with metadata."""
    source_frame: np.ndarray
    target_resolution: Tuple[int, int]
    camera_id: str
    algorithm: Optional[InterpolationAlgorithm] = None
    quality_priority: str = "balanced"  # "performance", "balanced", "quality"
    timestamp: float = 0.0


@dataclass
class InterpolationResult:
    """Interpolation result with performance metrics."""
    interpolated_frame: np.ndarray
    algorithm_used: InterpolationAlgorithm
    processing_time_ms: float
    quality_score: float
    source_resolution: Tuple[int, int]
    target_resolution: Tuple[int, int]
    scale_factor: float


@dataclass
class PerformanceMetrics:
    """Performance tracking for interpolation."""
    camera_id: str
    algorithm: InterpolationAlgorithm
    avg_processing_time: float
    total_frames: int
    quality_score: float
    last_update: float


class InterpolationEngine:
    """
    Advanced image interpolation engine with adaptive algorithms.

    Features:
    - Multiple interpolation algorithms (Nearest, Linear, Cubic, Lanczos)
    - Adaptive algorithm selection based on performance
    - Quality enhancement with sharpening and noise reduction
    - Performance monitoring and optimization
    - Camera-specific algorithm preferences
    - CPU load-based fallback algorithms
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # Default configuration
        self.default_algorithm = InterpolationAlgorithm.LINEAR
        self.quality_vs_performance = self.config.get("quality_vs_performance", 0.7)  # 0=performance, 1=quality
        self.adaptive_enabled = self.config.get("adaptive_enabled", True)
        self.cpu_threshold = self.config.get("cpu_threshold", 80)  # CPU % threshold for fallback
        self.sharpening_enabled = self.config.get("sharpening_enabled", True)
        self.noise_reduction_enabled = self.config.get("noise_reduction_enabled", False)

        # Performance tracking
        self.performance_metrics: Dict[str, PerformanceMetrics] = {}
        self.recent_processing_times: List[float] = []
        self.max_metrics_history = 1000

        # Algorithm preferences per camera
        self.camera_algorithms: Dict[str, InterpolationAlgorithm] = {}

        # Caching for resolution mappings
        self.resolution_cache_size = 100
        self._resolution_cache: Dict[Tuple, InterpolationAlgorithm] = {}

        # Threading for performance monitoring
        self.performance_lock = threading.Lock()

        logging.info("Interpolation Engine initialized")

    def interpolate(
            self,
            frame: np.ndarray,
            target_resolution: Tuple[int, int],
            camera_id: str,
            algorithm: Optional[InterpolationAlgorithm] = None,
            quality_priority: str = "balanced"
    ) -> np.ndarray:
        """
        Interpolate frame to target resolution.

        Args:
            frame: Input frame to interpolate
            target_resolution: Target (width, height)
            camera_id: Camera identifier for performance tracking
            algorithm: Specific algorithm to use (None for auto-selection)
            quality_priority: "performance", "balanced", or "quality"

        Returns:
            Interpolated frame
        """
        start_time = time.perf_counter()

        try:
            # Validate inputs
            if frame is None or frame.size == 0:
                raise InterpolationError("Invalid input frame")

            source_resolution = (frame.shape[1], frame.shape[0])

            # Skip interpolation if already correct size
            if source_resolution == target_resolution:
                return frame.copy()

            # Select algorithm
            if algorithm is None:
                algorithm = self._select_optimal_algorithm(
                    source_resolution, target_resolution, camera_id, quality_priority
                )

            # Perform interpolation
            result = self._perform_interpolation(frame, target_resolution, algorithm)

            # Apply post-processing enhancements
            if self.sharpening_enabled or self.noise_reduction_enabled:
                result = self._apply_enhancements(result, algorithm, target_resolution)

            # Track performance
            processing_time = (time.perf_counter() - start_time) * 1000
            self._update_performance_metrics(camera_id, algorithm, processing_time, source_resolution,
                                             target_resolution)

            return result

        except Exception as e:
            logging.error(f"Interpolation failed for camera {camera_id}: {e}")
            # Fallback to nearest neighbor
            try:
                return cv2.resize(frame, target_resolution, interpolation=cv2.INTER_NEAREST)
            except Exception as fallback_error:
                logging.error(f"Fallback interpolation failed: {fallback_error}")
                raise InterpolationError(f"Complete interpolation failure: {e}")

    def _select_optimal_algorithm(
            self,
            source_resolution: Tuple[int, int],
            target_resolution: Tuple[int, int],
            camera_id: str,
            quality_priority: str
    ) -> InterpolationAlgorithm:
        """Select optimal interpolation algorithm."""

        # Calculate scale factor
        scale_factor = (target_resolution[0] * target_resolution[1]) / (source_resolution[0] * source_resolution[1])

        # Check cache first
        cache_key = (source_resolution, target_resolution, quality_priority)
        if cache_key in self._resolution_cache:
            return self._resolution_cache[cache_key]

        # Check camera-specific preference
        if camera_id in self.camera_algorithms and not self.adaptive_enabled:
            return self.camera_algorithms[camera_id]

        # Algorithm selection based on priority and scale factor
        if quality_priority == "performance":
            algorithm = self._select_performance_algorithm(scale_factor)
        elif quality_priority == "quality":
            algorithm = self._select_quality_algorithm(scale_factor)
        else:  # balanced
            algorithm = self._select_balanced_algorithm(scale_factor, camera_id)

        # Check CPU load for potential fallback
        if self.adaptive_enabled:
            algorithm = self._check_cpu_fallback(algorithm)

        # Cache the selection
        if len(self._resolution_cache) < self.resolution_cache_size:
            self._resolution_cache[cache_key] = algorithm

        return algorithm

    def _select_performance_algorithm(self, scale_factor: float) -> InterpolationAlgorithm:
        """Select algorithm prioritizing performance."""
        if scale_factor > 4.0:  # Large upscaling
            return InterpolationAlgorithm.LINEAR
        elif scale_factor < 0.25:  # Large downscaling
            return InterpolationAlgorithm.LINEAR
        else:
            return InterpolationAlgorithm.NEAREST if scale_factor > 2.0 else InterpolationAlgorithm.LINEAR

    def _select_quality_algorithm(self, scale_factor: float) -> InterpolationAlgorithm:
        """Select algorithm prioritizing quality."""
        if scale_factor > 2.0:  # Upscaling
            return InterpolationAlgorithm.LANCZOS
        elif scale_factor < 0.5:  # Downscaling
            return InterpolationAlgorithm.LANCZOS
        else:
            return InterpolationAlgorithm.CUBIC

    def _select_balanced_algorithm(self, scale_factor: float, camera_id: str) -> InterpolationAlgorithm:
        """Select balanced algorithm considering performance history."""

        # Check performance history for this camera
        if camera_id in self.performance_metrics:
            metrics = self.performance_metrics[camera_id]
            if metrics.avg_processing_time > 50:  # If slow, prefer performance
                return self._select_performance_algorithm(scale_factor)

        # Default balanced selection
        if scale_factor > 3.0:  # Large upscaling
            return InterpolationAlgorithm.CUBIC
        elif scale_factor < 0.3:  # Large downscaling
            return InterpolationAlgorithm.CUBIC
        elif scale_factor > 1.5:  # Medium upscaling
            return InterpolationAlgorithm.LINEAR
        else:  # Minor scaling
            return InterpolationAlgorithm.LINEAR

    def _check_cpu_fallback(self, algorithm: InterpolationAlgorithm) -> InterpolationAlgorithm:
        """Check if CPU fallback is needed."""
        try:
            import psutil
            cpu_usage = psutil.cpu_percent(interval=0.1)

            if cpu_usage > self.cpu_threshold:
                # CPU is high, fallback to faster algorithm
                if algorithm == InterpolationAlgorithm.LANCZOS:
                    return InterpolationAlgorithm.CUBIC
                elif algorithm == InterpolationAlgorithm.CUBIC:
                    return InterpolationAlgorithm.LINEAR
                elif algorithm == InterpolationAlgorithm.LINEAR:
                    return InterpolationAlgorithm.NEAREST

        except ImportError:
            pass  # psutil not available
        except Exception as e:
            logging.warning(f"CPU check failed: {e}")

        return algorithm

    def _perform_interpolation(
            self,
            frame: np.ndarray,
            target_resolution: Tuple[int, int],
            algorithm: InterpolationAlgorithm
    ) -> np.ndarray:
        """Perform the actual interpolation."""
        try:
            return cv2.resize(frame, target_resolution, interpolation=algorithm.cv2_flag)
        except Exception as e:
            logging.error(f"OpenCV resize failed with {algorithm.algorithm_name}: {e}")
            # Fallback to linear interpolation
            return cv2.resize(frame, target_resolution, interpolation=cv2.INTER_LINEAR)

    def _apply_enhancements(
            self,
            frame: np.ndarray,
            algorithm: InterpolationAlgorithm,
            target_resolution: Tuple[int, int]
    ) -> np.ndarray:
        """Apply post-processing enhancements."""
        try:
            enhanced_frame = frame.copy()

            # Apply sharpening for upscaled images
            if self.sharpening_enabled and algorithm in [InterpolationAlgorithm.CUBIC, InterpolationAlgorithm.LANCZOS]:
                enhanced_frame = self._apply_sharpening(enhanced_frame)

            # Apply noise reduction for heavily processed images
            if self.noise_reduction_enabled and target_resolution[0] * target_resolution[1] > 1920 * 1080:
                enhanced_frame = self._apply_noise_reduction(enhanced_frame)

            return enhanced_frame

        except Exception as e:
            logging.warning(f"Enhancement failed: {e}")
            return frame

    def _apply_sharpening(self, frame: np.ndarray, strength: float = 0.2) -> np.ndarray:
        """Apply unsharp mask sharpening."""
        try:
            # Create Gaussian blur
            blurred = cv2.GaussianBlur(frame, (0, 0), 1.0)

            # Create sharpening kernel
            sharpened = cv2.addWeighted(frame, 1 + strength, blurred, -strength, 0)

            return sharpened
        except Exception as e:
            logging.warning(f"Sharpening failed: {e}")
            return frame

    def _apply_noise_reduction(self, frame: np.ndarray) -> np.ndarray:
        """Apply noise reduction."""
        try:
            # Use bilateral filter for noise reduction while preserving edges
            if len(frame.shape) == 3:
                # Color image
                return cv2.bilateralFilter(frame, 5, 50, 50)
            else:
                # Grayscale image
                return cv2.bilateralFilter(frame, 5, 50, 50)
        except Exception as e:
            logging.warning(f"Noise reduction failed: {e}")
            return frame

    def _update_performance_metrics(
            self,
            camera_id: str,
            algorithm: InterpolationAlgorithm,
            processing_time: float,
            source_resolution: Tuple[int, int],
            target_resolution: Tuple[int, int]
    ) -> None:
        """Update performance metrics for camera and algorithm."""
        with self.performance_lock:
            # Update camera-specific metrics
            if camera_id not in self.performance_metrics:
                self.performance_metrics[camera_id] = PerformanceMetrics(
                    camera_id=camera_id,
                    algorithm=algorithm,
                    avg_processing_time=processing_time,
                    total_frames=1,
                    quality_score=algorithm.quality_score,
                    last_update=time.time()
                )
            else:
                metrics = self.performance_metrics[camera_id]
                # Exponential moving average
                alpha = 0.1
                metrics.avg_processing_time = (
                        alpha * processing_time + (1 - alpha) * metrics.avg_processing_time
                )
                metrics.total_frames += 1
                metrics.algorithm = algorithm
                metrics.quality_score = algorithm.quality_score
                metrics.last_update = time.time()

            # Update global processing times
            self.recent_processing_times.append(processing_time)
            if len(self.recent_processing_times) > self.max_metrics_history:
                self.recent_processing_times.pop(0)

            # Adaptive algorithm learning
            if self.adaptive_enabled:
                self._update_camera_algorithm_preference(camera_id, algorithm, processing_time)

    def _update_camera_algorithm_preference(
            self,
            camera_id: str,
            algorithm: InterpolationAlgorithm,
            processing_time: float
    ) -> None:
        """Update camera's preferred algorithm based on performance."""
        # If processing time is consistently good, stick with this algorithm
        if processing_time < 30:  # Less than 30ms is good
            self.camera_algorithms[camera_id] = algorithm
        elif processing_time > 100:  # More than 100ms is too slow
            # Try a faster algorithm next time
            if algorithm == InterpolationAlgorithm.LANCZOS:
                self.camera_algorithms[camera_id] = InterpolationAlgorithm.CUBIC
            elif algorithm == InterpolationAlgorithm.CUBIC:
                self.camera_algorithms[camera_id] = InterpolationAlgorithm.LINEAR
            elif algorithm == InterpolationAlgorithm.LINEAR:
                self.camera_algorithms[camera_id] = InterpolationAlgorithm.NEAREST

    def get_optimal_algorithm_for_resolution(
            self,
            source_resolution: Tuple[int, int],
            target_resolution: Tuple[int, int],
            quality_priority: str = "balanced"
    ) -> InterpolationAlgorithm:
        """Get optimal algorithm for resolution transformation."""
        return self._select_optimal_algorithm(
            source_resolution, target_resolution, "preview", quality_priority
        )

    def benchmark_algorithms(
            self,
            test_frame: np.ndarray,
            target_resolution: Tuple[int, int],
            iterations: int = 10
    ) -> Dict[str, Dict[str, float]]:
        """Benchmark all algorithms on a test frame."""
        results = {}

        for algorithm in InterpolationAlgorithm:
            times = []

            for _ in range(iterations):
                start_time = time.perf_counter()
                try:
                    cv2.resize(test_frame, target_resolution, interpolation=algorithm.cv2_flag)
                    processing_time = (time.perf_counter() - start_time) * 1000
                    times.append(processing_time)
                except Exception as e:
                    logging.warning(f"Benchmark failed for {algorithm.algorithm_name}: {e}")
                    times.append(float('inf'))

            results[algorithm.algorithm_name] = {
                "avg_time_ms": sum(times) / len(times),
                "min_time_ms": min(times),
                "max_time_ms": max(times),
                "quality_score": algorithm.quality_score,
                "performance_score": algorithm.performance_score
            }

        return results

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        with self.performance_lock:
            stats = {
                "global_stats": {
                    "total_interpolations": len(self.recent_processing_times),
                    "avg_processing_time_ms": (
                        sum(self.recent_processing_times) / len(self.recent_processing_times)
                        if self.recent_processing_times else 0
                    ),
                    "adaptive_enabled": self.adaptive_enabled,
                    "cpu_threshold": self.cpu_threshold
                },
                "camera_stats": {},
                "algorithm_distribution": {}
            }

            # Camera-specific stats
            for camera_id, metrics in self.performance_metrics.items():
                stats["camera_stats"][camera_id] = {
                    "avg_processing_time_ms": metrics.avg_processing_time,
                    "total_frames": metrics.total_frames,
                    "preferred_algorithm": metrics.algorithm.algorithm_name,
                    "quality_score": metrics.quality_score,
                    "last_update": metrics.last_update
                }

            # Algorithm usage distribution
            algorithm_counts = {}
            for metrics in self.performance_metrics.values():
                algo_name = metrics.algorithm.algorithm_name
                algorithm_counts[algo_name] = algorithm_counts.get(algo_name, 0) + 1

            total_cameras = len(self.performance_metrics)
            if total_cameras > 0:
                for algo_name, count in algorithm_counts.items():
                    stats["algorithm_distribution"][algo_name] = {
                        "usage_count": count,
                        "usage_percentage": (count / total_cameras) * 100
                    }

            return stats

    def get_camera_algorithm_recommendation(
            self,
            camera_id: str,
            source_resolution: Tuple[int, int],
            target_resolution: Tuple[int, int]
    ) -> Dict[str, Any]:
        """Get algorithm recommendation for specific camera."""

        # Calculate scale factor and direction
        scale_factor = (target_resolution[0] * target_resolution[1]) / (source_resolution[0] * source_resolution[1])
        scale_direction = "upscale" if scale_factor > 1.0 else "downscale"

        # Get current performance if available
        current_performance = None
        if camera_id in self.performance_metrics:
            metrics = self.performance_metrics[camera_id]
            current_performance = {
                "current_algorithm": metrics.algorithm.algorithm_name,
                "avg_processing_time_ms": metrics.avg_processing_time,
                "total_frames": metrics.total_frames
            }

        # Get recommendations for different priorities
        recommendations = {
            "performance": self._select_performance_algorithm(scale_factor),
            "balanced": self._select_balanced_algorithm(scale_factor, camera_id),
            "quality": self._select_quality_algorithm(scale_factor)
        }

        return {
            "camera_id": camera_id,
            "source_resolution": source_resolution,
            "target_resolution": target_resolution,
            "scale_factor": scale_factor,
            "scale_direction": scale_direction,
            "current_performance": current_performance,
            "recommendations": {
                priority: {
                    "algorithm": algo.algorithm_name,
                    "quality_score": algo.quality_score,
                    "performance_score": algo.performance_score,
                    "description": f"{algo.algorithm_name.title()} interpolation"
                }
                for priority, algo in recommendations.items()
            }
        }

    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update interpolation engine configuration."""
        self.config.update(new_config)

        # Update parameters
        self.quality_vs_performance = new_config.get("quality_vs_performance", self.quality_vs_performance)
        self.adaptive_enabled = new_config.get("adaptive_enabled", self.adaptive_enabled)
        self.cpu_threshold = new_config.get("cpu_threshold", self.cpu_threshold)
        self.sharpening_enabled = new_config.get("sharpening_enabled", self.sharpening_enabled)
        self.noise_reduction_enabled = new_config.get("noise_reduction_enabled", self.noise_reduction_enabled)

        # Clear caches to apply new settings
        self._resolution_cache.clear()

        logging.info("Interpolation engine configuration updated")

    def set_camera_algorithm(self, camera_id: str, algorithm: InterpolationAlgorithm) -> None:
        """Manually set preferred algorithm for camera."""
        self.camera_algorithms[camera_id] = algorithm
        logging.info(f"Set camera {camera_id} algorithm to {algorithm.algorithm_name}")

    def reset_camera_metrics(self, camera_id: str) -> None:
        """Reset performance metrics for specific camera."""
        if camera_id in self.performance_metrics:
            del self.performance_metrics[camera_id]
        if camera_id in self.camera_algorithms:
            del self.camera_algorithms[camera_id]
        logging.info(f"Reset metrics for camera {camera_id}")

    def get_algorithm_by_name(self, name: str) -> Optional[InterpolationAlgorithm]:
        """Get algorithm enum by name."""
        for algorithm in InterpolationAlgorithm:
            if algorithm.algorithm_name.lower() == name.lower():
                return algorithm
        return None

    def create_test_frame(self, resolution: Tuple[int, int] = (640, 480)) -> np.ndarray:
        """Create test frame for benchmarking."""
        # Create test pattern with various features
        frame = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)

        # Add gradients
        for y in range(resolution[1]):
            for x in range(resolution[0]):
                frame[y, x] = [
                    int((x / resolution[0]) * 255),  # Red gradient
                    int((y / resolution[1]) * 255),  # Green gradient
                    int(((x + y) / (resolution[0] + resolution[1])) * 255)  # Blue gradient
                ]

        # Add some geometric patterns
        center_x, center_y = resolution[0] // 2, resolution[1] // 2

        # Draw circles
        for radius in range(20, min(center_x, center_y), 40):
            cv2.circle(frame, (center_x, center_y), radius, (255, 255, 255), 2)

        # Draw lines
        for angle in range(0, 360, 30):
            end_x = int(center_x + 100 * np.cos(np.radians(angle)))
            end_y = int(center_y + 100 * np.sin(np.radians(angle)))
            cv2.line(frame, (center_x, center_y), (end_x, end_y), (200, 200, 200), 1)

        # Add text
        cv2.putText(frame, "TEST PATTERN", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return frame

    @lru_cache(maxsize=128)
    def get_cached_resize_params(
            self,
            source_res: Tuple[int, int],
            target_res: Tuple[int, int],
            algorithm_name: str
    ) -> Tuple[int, Tuple[int, int]]:
        """Get cached resize parameters for common transformations."""
        algorithm = self.get_algorithm_by_name(algorithm_name)
        if algorithm:
            return algorithm.cv2_flag, target_res
        return cv2.INTER_LINEAR, target_res

    def optimize_for_raspberry_pi(self) -> None:
        """Optimize settings for Raspberry Pi performance."""
        self.config.update({
            "quality_vs_performance": 0.3,  # Favor performance
            "cpu_threshold": 70,  # Lower threshold
            "adaptive_enabled": True,
            "sharpening_enabled": False,  # Disable for performance
            "noise_reduction_enabled": False
        })

        # Set all cameras to performance algorithms
        for camera_id in self.camera_algorithms:
            self.camera_algorithms[camera_id] = InterpolationAlgorithm.LINEAR

        self._resolution_cache.clear()
        logging.info("Optimized interpolation engine for Raspberry Pi")

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        import sys

        cache_size = len(self._resolution_cache)
        metrics_size = len(self.performance_metrics)
        history_size = len(self.recent_processing_times)

        # Estimate memory usage
        estimated_cache_mb = cache_size * 0.001  # Rough estimate
        estimated_metrics_mb = metrics_size * 0.001
        estimated_history_mb = history_size * 0.00001

        return {
            "cache_entries": cache_size,
            "performance_metrics_count": metrics_size,
            "processing_history_count": history_size,
            "estimated_memory_mb": estimated_cache_mb + estimated_metrics_mb + estimated_history_mb,
            "cache_hit_info": f"{cache_size}/{self.resolution_cache_size} entries"
        }

    def cleanup_old_metrics(self, max_age_hours: int = 24) -> None:
        """Clean up old performance metrics."""
        cutoff_time = time.time() - (max_age_hours * 3600)

        with self.performance_lock:
            # Remove old camera metrics
            cameras_to_remove = [
                camera_id for camera_id, metrics in self.performance_metrics.items()
                if metrics.last_update < cutoff_time
            ]

            for camera_id in cameras_to_remove:
                del self.performance_metrics[camera_id]
                if camera_id in self.camera_algorithms:
                    del self.camera_algorithms[camera_id]

            # Keep only recent processing times (last 1000 entries)
            if len(self.recent_processing_times) > 1000:
                self.recent_processing_times = self.recent_processing_times[-1000:]

        if cameras_to_remove:
            logging.info(f"Cleaned up metrics for {len(cameras_to_remove)} inactive cameras")