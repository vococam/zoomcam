"""
Motion Detector - OpenCV-based motion detection and analysis
===========================================================

Detects motion in camera frames, segments active areas,
and provides activity metrics for layout optimization.
"""

import logging
import cv2
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import threading

from zoomcam.utils.exceptions import MotionDetectionError


@dataclass
class MotionZone:
    """Represents a detected motion zone."""
    id: str
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    area: int
    activity_level: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    centroid: Tuple[int, int]
    contour: np.ndarray
    first_detected: datetime
    last_updated: datetime


@dataclass
class MotionFrame:
    """Motion detection result for a frame."""
    frame_number: int
    timestamp: datetime
    zones: List[MotionZone]
    total_motion_area: int
    motion_percentage: float
    background_stability: float


class MotionDetector:
    """
    OpenCV-based motion detection with advanced features.

    Features:
    - Background subtraction using MOG2
    - Motion zone segmentation and tracking
    - Noise filtering and morphological operations
    - Activity level calculation
    - Zone merging for nearby motion areas
    - Configurable sensitivity and thresholds
    """

    def __init__(self, camera_id: str, config: Dict[str, Any]):
        self.camera_id = camera_id
        self.config = config

        # Detection parameters
        self.sensitivity = config.get("sensitivity", 0.3)
        self.min_area = config.get("min_area", 500)
        self.max_zones = config.get("max_zones", 5)
        self.noise_filter_size = config.get("noise_filter_size", 5)
        self.dilate_iterations = config.get("dilate_iterations", 2)
        self.erode_iterations = config.get("erode_iterations", 1)
        self.zone_merge_distance = config.get("zone_merge_distance", 50)

        # Background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=16,
            detectShadows=True
        )

        # Morphological kernels
        self.kernel_noise = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.noise_filter_size, self.noise_filter_size)
        )
        self.kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        self.kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        # State tracking
        self.frame_count = 0
        self.motion_history: List[MotionFrame] = []
        self.active_zones: Dict[str, MotionZone] = {}
        self.zone_counter = 0

        # Ignore zones (areas to exclude from detection)
        self.ignore_zones = config.get("ignore_zones", [])

        # Performance tracking
        self.processing_times: List[float] = []
        self.lock = threading.Lock()

        logging.info(f"Motion detector initialized for camera {camera_id}")

    def detect_motion(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect motion in frame and return motion zones.

        Args:
            frame: Input frame from camera

        Returns:
            List of motion zone dictionaries with bbox, activity_level, etc.
        """
        start_time = datetime.now()

        try:
            self.frame_count += 1

            # Preprocess frame
            processed_frame = self._preprocess_frame(frame)

            # Background subtraction
            fg_mask = self.bg_subtractor.apply(processed_frame)

            # Apply ignore zones
            if self.ignore_zones:
                fg_mask = self._apply_ignore_zones(fg_mask)

            # Noise reduction and morphological operations
            fg_mask = self._clean_mask(fg_mask)

            # Find contours and create motion zones
            motion_zones = self._find_motion_zones(fg_mask, frame.shape[:2])

            # Merge nearby zones
            merged_zones = self._merge_nearby_zones(motion_zones)

            # Limit number of zones
            if len(merged_zones) > self.max_zones:
                merged_zones = sorted(merged_zones, key=lambda z: z.area, reverse=True)[:self.max_zones]

            # Calculate frame-level metrics
            total_motion_area = sum(zone.area for zone in merged_zones)
            frame_area = frame.shape[0] * frame.shape[1]
            motion_percentage = (total_motion_area / frame_area) * 100

            # Create motion frame result
            motion_frame = MotionFrame(
                frame_number=self.frame_count,
                timestamp=datetime.now(),
                zones=merged_zones,
                total_motion_area=total_motion_area,
                motion_percentage=motion_percentage,
                background_stability=self._calculate_background_stability()
            )

            # Update history
            self._update_motion_history(motion_frame)

            # Convert to API format
            zone_dicts = []
            for zone in merged_zones:
                zone_dict = {
                    "id": zone.id,
                    "bbox": zone.bbox,
                    "area": zone.area,
                    "activity_level": zone.activity_level,
                    "confidence": zone.confidence,
                    "centroid": zone.centroid,
                    "priority": self._calculate_zone_priority(zone),
                    "timestamp": zone.last_updated.isoformat()
                }
                zone_dicts.append(zone_dict)

            # Track processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            self.processing_times.append(processing_time)
            if len(self.processing_times) > 100:
                self.processing_times.pop(0)

            return zone_dicts

        except Exception as e:
            logging.error(f"Motion detection error for camera {self.camera_id}: {e}")
            raise MotionDetectionError(f"Motion detection failed: {e}")

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for motion detection."""
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        return blurred

    def _apply_ignore_zones(self, mask: np.ndarray) -> np.ndarray:
        """Apply ignore zones to mask."""
        for zone in self.ignore_zones:
            x, y, w, h = zone
            mask[y:y + h, x:x + w] = 0
        return mask

    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """Clean mask using morphological operations."""
        # Remove noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_noise)

        # Fill holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_noise)

        # Dilate to connect nearby regions
        mask = cv2.dilate(mask, self.kernel_dilate, iterations=self.dilate_iterations)

        # Erode to restore original size
        mask = cv2.erode(mask, self.kernel_erode, iterations=self.erode_iterations)

        return mask

    def _find_motion_zones(self, mask: np.ndarray, frame_shape: Tuple[int, int]) -> List[MotionZone]:
        """Find motion zones from cleaned mask."""
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_zones = []
        current_time = datetime.now()

        for contour in contours:
            area = cv2.contourArea(contour)

            # Filter by minimum area
            if area < self.min_area:
                continue

            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)

            # Calculate centroid
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x + w // 2, y + h // 2

            # Calculate activity level based on area and motion intensity
            frame_area = frame_shape[0] * frame_shape[1]
            area_ratio = area / frame_area
            activity_level = min(1.0, area_ratio * 10)  # Scale factor

            # Calculate confidence based on contour properties
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                confidence = min(1.0, circularity + area_ratio)
            else:
                confidence = area_ratio

            # Generate zone ID
            self.zone_counter += 1
            zone_id = f"{self.camera_id}_zone_{self.zone_counter}"

            # Create motion zone
            zone = MotionZone(
                id=zone_id,
                bbox=(x, y, w, h),
                area=int(area),
                activity_level=activity_level,
                confidence=confidence,
                centroid=(cx, cy),
                contour=contour,
                first_detected=current_time,
                last_updated=current_time
            )

            motion_zones.append(zone)

        return motion_zones

    def _merge_nearby_zones(self, zones: List[MotionZone]) -> List[MotionZone]:
        """Merge zones that are close to each other."""
        if len(zones) <= 1:
            return zones

        merged_zones = []
        used_indices = set()

        for i, zone1 in enumerate(zones):
            if i in used_indices:
                continue

            # Find nearby zones to merge
            merge_group = [zone1]
            used_indices.add(i)

            for j, zone2 in enumerate(zones):
                if j in used_indices or i == j:
                    continue

                # Calculate distance between centroids
                dist = np.sqrt(
                    (zone1.centroid[0] - zone2.centroid[0]) ** 2 +
                    (zone1.centroid[1] - zone2.centroid[1]) ** 2
                )

                if dist < self.zone_merge_distance:
                    merge_group.append(zone2)
                    used_indices.add(j)

            # Merge zones in group
            if len(merge_group) == 1:
                merged_zones.append(merge_group[0])
            else:
                merged_zone = self._merge_zone_group(merge_group)
                merged_zones.append(merged_zone)

        return merged_zones

    def _merge_zone_group(self, zones: List[MotionZone]) -> MotionZone:
        """Merge a group of zones into a single zone."""
        # Calculate combined bounding box
        min_x = min(zone.bbox[0] for zone in zones)
        min_y = min(zone.bbox[1] for zone in zones)
        max_x = max(zone.bbox[0] + zone.bbox[2] for zone in zones)
        max_y = max(zone.bbox[1] + zone.bbox[3] for zone in zones)

        combined_bbox = (min_x, min_y, max_x - min_x, max_y - min_y)

        # Calculate combined metrics
        total_area = sum(zone.area for zone in zones)
        avg_activity = sum(zone.activity_level for zone in zones) / len(zones)
        avg_confidence = sum(zone.confidence for zone in zones) / len(zones)

        # Calculate weighted centroid
        total_weight = sum(zone.area for zone in zones)
        if total_weight > 0:
            weighted_cx = sum(zone.centroid[0] * zone.area for zone in zones) / total_weight
            weighted_cy = sum(zone.centroid[1] * zone.area for zone in zones) / total_weight
            combined_centroid = (int(weighted_cx), int(weighted_cy))
        else:
            combined_centroid = zones[0].centroid

        # Use earliest detection time
        earliest_time = min(zone.first_detected for zone in zones)

        self.zone_counter += 1
        merged_id = f"{self.camera_id}_merged_{self.zone_counter}"

        return MotionZone(
            id=merged_id,
            bbox=combined_bbox,
            area=total_area,
            activity_level=min(1.0, avg_activity * 1.2),  # Slight boost for merged zones
            confidence=avg_confidence,
            centroid=combined_centroid,
            contour=zones[0].contour,  # Use first contour as representative
            first_detected=earliest_time,
            last_updated=datetime.now()
        )

    def _calculate_zone_priority(self, zone: MotionZone) -> int:
        """Calculate priority for a motion zone."""
        # Priority based on activity level, area, and recency
        base_priority = int(zone.activity_level * 10)
        area_bonus = min(5, int(zone.area / 1000))  # Bonus for larger areas

        # Recency bonus (newer detections get higher priority)
        time_since_detection = (datetime.now() - zone.last_updated).total_seconds()
        recency_bonus = max(0, 5 - int(time_since_detection / 10))

        return base_priority + area_bonus + recency_bonus

    def _calculate_background_stability(self) -> float:
        """Calculate background stability metric."""
        # This is a simplified metric - in practice you might want more sophisticated analysis
        if len(self.motion_history) < 10:
            return 1.0

        recent_motion = [frame.motion_percentage for frame in self.motion_history[-10:]]
        variance = np.var(recent_motion)

        # Stability is inverse of variance (normalized)
        stability = max(0.0, 1.0 - min(1.0, variance / 100))
        return stability

    def _update_motion_history(self, motion_frame: MotionFrame) -> None:
        """Update motion history with frame result."""
        with self.lock:
            self.motion_history.append(motion_frame)

            # Keep only recent history (last 5 minutes at 30fps = ~9000 frames)
            max_history = 9000
            if len(self.motion_history) > max_history:
                self.motion_history = self.motion_history[-max_history:]

    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update motion detection configuration."""
        self.config.update(new_config)

        # Update parameters
        self.sensitivity = self.config.get("sensitivity", self.sensitivity)
        self.min_area = self.config.get("min_area", self.min_area)
        self.max_zones = self.config.get("max_zones", self.max_zones)
        self.ignore_zones = self.config.get("ignore_zones", self.ignore_zones)

        # Update background subtractor if needed
        if "history" in new_config or "varThreshold" in new_config:
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=new_config.get("history", 500),
                varThreshold=new_config.get("varThreshold", 16),
                detectShadows=True
            )

        logging.info(f"Updated motion detection config for camera {self.camera_id}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get motion detection statistics."""
        with self.lock:
            if not self.motion_history:
                return {"status": "no_data"}

            recent_frames = self.motion_history[-100:]  # Last 100 frames

            avg_motion = np.mean([frame.motion_percentage for frame in recent_frames])
            avg_zones = np.mean([len(frame.zones) for frame in recent_frames])
            avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0

            return {
                "camera_id": self.camera_id,
                "total_frames_processed": self.frame_count,
                "average_motion_percentage": float(avg_motion),
                "average_zones_per_frame": float(avg_zones),
                "average_processing_time_ms": float(avg_processing_time * 1000),
                "background_stability": self._calculate_background_stability(),
                "config": {
                    "sensitivity": self.sensitivity,
                    "min_area": self.min_area,
                    "max_zones": self.max_zones,
                    "ignore_zones_count": len(self.ignore_zones)
                }
            }

    def reset_background_model(self) -> None:
        """Reset the background subtraction model."""
        logging.info(f"Resetting background model for camera {self.camera_id}")
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=16,
            detectShadows=True
        )

    def add_ignore_zone(self, bbox: Tuple[int, int, int, int]) -> None:
        """Add new ignore zone."""
        self.ignore_zones.append(bbox)
        logging.info(f"Added ignore zone to camera {self.camera_id}: {bbox}")

    def remove_ignore_zone(self, index: int) -> None:
        """Remove ignore zone by index."""
        if 0 <= index < len(self.ignore_zones):
            removed = self.ignore_zones.pop(index)
            logging.info(f"Removed ignore zone from camera {self.camera_id}: {removed}")

    def get_recent_motion_zones(self, seconds: int = 30) -> List[MotionZone]:
        """Get motion zones from recent frames."""
        cutoff_time = datetime.now() - timedelta(seconds=seconds)

        recent_zones = []
        with self.lock:
            for frame in reversed(self.motion_history):
                if frame.timestamp < cutoff_time:
                    break
                recent_zones.extend(frame.zones)

        return recent_zones