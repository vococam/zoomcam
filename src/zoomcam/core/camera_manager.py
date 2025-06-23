"""
Camera Manager - Core camera handling and management
==================================================

Manages multiple cameras, auto-detection, and frame processing.
Handles USB cameras, RTSP streams, and resolution detection.
"""

import asyncio
import logging
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor

from zoomcam.core.motion_detector import MotionDetector
from zoomcam.core.interpolation_engine import InterpolationEngine
from zoomcam.utils.exceptions import CameraError, ZoomCamError


@dataclass
class CameraInfo:
    """Camera information and status."""
    id: str
    name: str
    source: str
    resolution: Tuple[int, int]
    fps: float
    status: str  # "active", "inactive", "error", "no_signal"
    last_frame_time: Optional[datetime] = None
    zoom: float = 3.0
    max_fragments: int = 2
    recording_enabled: bool = True


@dataclass
class CameraFrame:
    """Camera frame with metadata."""
    camera_id: str
    frame: np.ndarray
    timestamp: datetime
    frame_number: int
    motion_zones: List[Dict] = None


class CameraManager:
    """
    Manages multiple cameras and their processing.

    Features:
    - Auto-detection of USB cameras and RTSP streams
    - Resolution auto-detection and scaling
    - Motion detection integration
    - Frame buffering and processing
    - Error handling and reconnection
    """

    def __init__(self, config: Dict[str, Any], auto_config=None):
        self.config = config
        self.auto_config = auto_config
        self.cameras: Dict[str, CameraInfo] = {}
        self.captures: Dict[str, cv2.VideoCapture] = {}
        self.motion_detectors: Dict[str, MotionDetector] = {}
        self.interpolation_engine = InterpolationEngine()

        # Frame processing
        self.frame_queues: Dict[str, asyncio.Queue] = {}
        self.latest_frames: Dict[str, CameraFrame] = {}
        self.frame_processors: Dict[str, threading.Thread] = {}
        self.running = False

        # Performance monitoring
        self.frame_counts: Dict[str, int] = {}
        self.error_counts: Dict[str, int] = {}

        self.executor = ThreadPoolExecutor(max_workers=4)

        logging.info("Camera Manager initialized")

    async def initialize(self) -> None:
        """Initialize camera manager and detect cameras."""
        try:
            await self._detect_cameras()
            await self._setup_cameras()
            logging.info(f"Initialized {len(self.cameras)} cameras")
        except Exception as e:
            raise CameraError(f"Failed to initialize cameras: {e}")

    async def _detect_cameras(self) -> None:
        """Detect available cameras automatically."""
        detected_cameras = {}

        # Detect USB cameras
        for i in range(10):  # Check first 10 USB ports
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Test if camera actually works
                ret, frame = cap.read()
                if ret and frame is not None:
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)

                    camera_id = f"usb_camera_{i}"
                    detected_cameras[camera_id] = {
                        "source": f"/dev/video{i}",
                        "resolution": (width, height),
                        "fps": fps,
                        "type": "usb"
                    }
                    logging.info(f"Detected USB camera {i}: {width}x{height} @ {fps}fps")
                cap.release()

        # Add configured RTSP cameras
        for camera_id, camera_config in self.config.items():
            if camera_config.get("enabled", True):
                source = camera_config["source"]
                if source.startswith("rtsp://"):
                    # Test RTSP connection
                    resolution = await self._test_rtsp_camera(source)
                    if resolution:
                        detected_cameras[camera_id] = {
                            "source": source,
                            "resolution": resolution,
                            "fps": 30.0,  # Default for RTSP
                            "type": "rtsp"
                        }
                        logging.info(f"Detected RTSP camera {camera_id}: {resolution}")

        # Update auto-config with detected cameras
        if self.auto_config:
            await self.auto_config.update_camera_detection(detected_cameras)

        self.detected_cameras = detected_cameras

    async def _test_rtsp_camera(self, url: str) -> Optional[Tuple[int, int]]:
        """Test RTSP camera connection and get resolution."""
        try:
            cap = cv2.VideoCapture(url)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    height, width = frame.shape[:2]
                    cap.release()
                    return (width, height)
            cap.release()
        except Exception as e:
            logging.warning(f"Failed to test RTSP camera {url}: {e}")
        return None

    async def _setup_cameras(self) -> None:
        """Setup cameras based on configuration and detection."""
        for camera_id, camera_config in self.config.items():
            if not camera_config.get("enabled", True):
                continue

            try:
                # Get camera info from detection or config
                if camera_id in self.detected_cameras:
                    detected = self.detected_cameras[camera_id]
                    source = detected["source"]
                    resolution = detected["resolution"]
                else:
                    source = camera_config["source"]
                    resolution = self._parse_resolution(
                        camera_config.get("resolution", "auto")
                    )

                # Create camera info
                camera_info = CameraInfo(
                    id=camera_id,
                    name=camera_config.get("name", f"Camera {camera_id}"),
                    source=source,
                    resolution=resolution,
                    fps=30.0,  # Will be updated during processing
                    status="inactive",
                    zoom=camera_config.get("zoom", 3.0),
                    max_fragments=camera_config.get("max_fragments", 2),
                    recording_enabled=camera_config.get("recording", {}).get("enabled", True)
                )

                self.cameras[camera_id] = camera_info

                # Setup motion detector
                motion_config = camera_config.get("motion_detection", {})
                self.motion_detectors[camera_id] = MotionDetector(
                    camera_id=camera_id,
                    config=motion_config
                )

                # Setup frame queue
                self.frame_queues[camera_id] = asyncio.Queue(maxsize=10)
                self.frame_counts[camera_id] = 0
                self.error_counts[camera_id] = 0

                logging.info(f"Setup camera {camera_id}: {camera_info.name}")

            except Exception as e:
                logging.error(f"Failed to setup camera {camera_id}: {e}")
                self.error_counts[camera_id] = self.error_counts.get(camera_id, 0) + 1

    def _parse_resolution(self, resolution_str: str) -> Tuple[int, int]:
        """Parse resolution string to tuple."""
        if resolution_str == "auto":
            return (1920, 1080)  # Default
        elif resolution_str == "4K":
            return (3840, 2160)
        elif resolution_str == "1080p":
            return (1920, 1080)
        elif resolution_str == "720p":
            return (1280, 720)
        elif "x" in resolution_str:
            width, height = resolution_str.split("x")
            return (int(width), int(height))
        else:
            return (1920, 1080)  # Default fallback

    async def start_processing(self) -> None:
        """Start camera processing loops."""
        self.running = True

        # Start frame capture threads for each camera
        for camera_id in self.cameras:
            thread = threading.Thread(
                target=self._capture_frames,
                args=(camera_id,),
                daemon=True
            )
            thread.start()
            self.frame_processors[camera_id] = thread

        # Start frame processing loop
        await self._process_frames()

    def _capture_frames(self, camera_id: str) -> None:
        """Capture frames from camera in separate thread."""
        camera = self.cameras[camera_id]
        cap = None
        frame_number = 0

        try:
            # Open camera
            if camera.source.startswith("rtsp://"):
                cap = cv2.VideoCapture(camera.source, cv2.CAP_FFMPEG)
            elif camera.source.startswith("/dev/video"):
                cap = cv2.VideoCapture(int(camera.source.split("video")[1]))
            else:
                cap = cv2.VideoCapture(camera.source)

            if not cap.isOpened():
                raise CameraError(f"Failed to open camera {camera_id}")

            # Set camera properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera.resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera.resolution[1])
            cap.set(cv2.CAP_PROP_FPS, 30)

            self.captures[camera_id] = cap
            camera.status = "active"

            logging.info(f"Started capturing from camera {camera_id}")

            while self.running:
                ret, frame = cap.read()

                if not ret or frame is None:
                    camera.status = "no_signal"
                    await asyncio.sleep(1)
                    continue

                # Process frame
                frame_number += 1
                camera_frame = CameraFrame(
                    camera_id=camera_id,
                    frame=frame,
                    timestamp=datetime.now(),
                    frame_number=frame_number
                )

                # Apply zoom and interpolation if needed
                processed_frame = await self._process_camera_frame(camera_frame)

                # Motion detection
                motion_zones = self.motion_detectors[camera_id].detect_motion(
                    processed_frame.frame
                )
                processed_frame.motion_zones = motion_zones

                # Update latest frame
                self.latest_frames[camera_id] = processed_frame
                camera.last_frame_time = processed_frame.timestamp
                camera.status = "active"

                # Add to queue (non-blocking)
                try:
                    self.frame_queues[camera_id].put_nowait(processed_frame)
                except asyncio.QueueFull:
                    # Remove oldest frame and add new one
                    try:
                        self.frame_queues[camera_id].get_nowait()
                        self.frame_queues[camera_id].put_nowait(processed_frame)
                    except asyncio.QueueEmpty:
                        pass

                self.frame_counts[camera_id] += 1

                # Small delay to prevent excessive CPU usage
                await asyncio.sleep(0.033)  # ~30 FPS

        except Exception as e:
            logging.error(f"Error capturing from camera {camera_id}: {e}")
            camera.status = "error"
            self.error_counts[camera_id] += 1
        finally:
            if cap:
                cap.release()

    async def _process_camera_frame(self, camera_frame: CameraFrame) -> CameraFrame:
        """Process individual camera frame (zoom, interpolation, etc)."""
        camera = self.cameras[camera_frame.camera_id]
        frame = camera_frame.frame

        # Apply zoom if needed
        if camera.zoom != 1.0:
            frame = self._apply_zoom(frame, camera.zoom)

        # Apply interpolation/scaling if needed
        target_resolution = self._get_target_resolution(camera_frame.camera_id)
        if frame.shape[:2][::-1] != target_resolution:
            frame = self.interpolation_engine.interpolate(
                frame, target_resolution, camera_frame.camera_id
            )

        camera_frame.frame = frame
        return camera_frame

    def _apply_zoom(self, frame: np.ndarray, zoom: float) -> np.ndarray:
        """Apply zoom to frame."""
        height, width = frame.shape[:2]

        # Calculate crop area for zoom
        crop_width = int(width / zoom)
        crop_height = int(height / zoom)

        # Center crop
        start_x = (width - crop_width) // 2
        start_y = (height - crop_height) // 2

        cropped = frame[start_y:start_y + crop_height, start_x:start_x + crop_width]

        # Resize back to original size
        return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LANCZOS4)

    def _get_target_resolution(self, camera_id: str) -> Tuple[int, int]:
        """Get target resolution for camera based on display and layout."""
        # This will be updated by layout engine
        # For now, return camera's native resolution
        return self.cameras[camera_id].resolution

    async def _process_frames(self) -> None:
        """Main frame processing loop."""
        while self.running:
            try:
                # Process motion detection and layout updates
                for camera_id, camera in self.cameras.items():
                    if camera.status == "active" and camera_id in self.latest_frames:
                        latest_frame = self.latest_frames[camera_id]

                        # Update auto-config with motion data
                        if (self.auto_config and
                                latest_frame.motion_zones):
                            await self.auto_config.update_motion_data(
                                camera_id, latest_frame.motion_zones
                            )

                await asyncio.sleep(0.1)  # 10 Hz processing

            except Exception as e:
                logging.error(f"Error in frame processing loop: {e}")
                await asyncio.sleep(1)

    async def get_latest_frame(self, camera_id: str) -> Optional[CameraFrame]:
        """Get latest frame from camera."""
        return self.latest_frames.get(camera_id)

    async def get_camera_status(self) -> Dict[str, Any]:
        """Get status of all cameras."""
        status = {}
        for camera_id, camera in self.cameras.items():
            status[camera_id] = {
                "id": camera.id,
                "name": camera.name,
                "status": camera.status,
                "resolution": camera.resolution,
                "fps": camera.fps,
                "frame_count": self.frame_counts.get(camera_id, 0),
                "error_count": self.error_counts.get(camera_id, 0),
                "last_frame_time": camera.last_frame_time.isoformat() if camera.last_frame_time else None,
                "zoom": camera.zoom,
                "max_fragments": camera.max_fragments,
                "recording_enabled": camera.recording_enabled
            }
        return status

    async def update_camera_config(self, camera_id: str, updates: Dict[str, Any]) -> None:
        """Update camera configuration dynamically."""
        if camera_id not in self.cameras:
            raise CameraError(f"Camera {camera_id} not found")

        camera = self.cameras[camera_id]

        # Update allowed properties
        if "zoom" in updates:
            camera.zoom = max(1.0, min(10.0, float(updates["zoom"])))

        if "max_fragments" in updates:
            camera.max_fragments = max(1, min(5, int(updates["max_fragments"])))

        if "recording_enabled" in updates:
            camera.recording_enabled = bool(updates["recording_enabled"])

        # Update motion detector config if provided
        if "motion_detection" in updates:
            self.motion_detectors[camera_id].update_config(updates["motion_detection"])

        # Log configuration change
        if self.auto_config:
            await self.auto_config.log_camera_config_change(camera_id, updates)

        logging.info(f"Updated camera {camera_id} configuration: {updates}")

    async def restart_camera(self, camera_id: str) -> None:
        """Restart specific camera."""
        if camera_id not in self.cameras:
            raise CameraError(f"Camera {camera_id} not found")

        logging.info(f"Restarting camera {camera_id}")

        # Stop current capture
        if camera_id in self.captures:
            self.captures[camera_id].release()
            del self.captures[camera_id]

        # Reset camera status
        self.cameras[camera_id].status = "inactive"
        self.error_counts[camera_id] = 0

        # Restart capture thread
        if camera_id in self.frame_processors:
            # The thread will restart automatically in the next capture cycle
            pass

    async def add_camera(self, camera_config: Dict[str, Any]) -> str:
        """Add new camera dynamically."""
        camera_id = camera_config.get("id", f"camera_{len(self.cameras) + 1}")

        if camera_id in self.cameras:
            raise CameraError(f"Camera {camera_id} already exists")

        # Test camera connection
        source = camera_config["source"]
        if source.startswith("rtsp://"):
            resolution = await self._test_rtsp_camera(source)
            if not resolution:
                raise CameraError(f"Cannot connect to RTSP camera: {source}")
        else:
            resolution = self._parse_resolution(camera_config.get("resolution", "auto"))

        # Create camera info
        camera_info = CameraInfo(
            id=camera_id,
            name=camera_config.get("name", f"Camera {camera_id}"),
            source=source,
            resolution=resolution,
            fps=30.0,
            status="inactive",
            zoom=camera_config.get("zoom", 3.0),
            max_fragments=camera_config.get("max_fragments", 2),
            recording_enabled=camera_config.get("recording", {}).get("enabled", True)
        )

        self.cameras[camera_id] = camera_info

        # Setup motion detector
        motion_config = camera_config.get("motion_detection", {})
        self.motion_detectors[camera_id] = MotionDetector(
            camera_id=camera_id,
            config=motion_config
        )

        # Setup frame queue
        self.frame_queues[camera_id] = asyncio.Queue(maxsize=10)
        self.frame_counts[camera_id] = 0
        self.error_counts[camera_id] = 0

        # Start capture thread if processing is running
        if self.running:
            thread = threading.Thread(
                target=self._capture_frames,
                args=(camera_id,),
                daemon=True
            )
            thread.start()
            self.frame_processors[camera_id] = thread

        logging.info(f"Added new camera {camera_id}: {camera_info.name}")
        return camera_id

    async def remove_camera(self, camera_id: str) -> None:
        """Remove camera dynamically."""
        if camera_id not in self.cameras:
            raise CameraError(f"Camera {camera_id} not found")

        logging.info(f"Removing camera {camera_id}")

        # Stop capture
        if camera_id in self.captures:
            self.captures[camera_id].release()
            del self.captures[camera_id]

        # Clean up resources
        if camera_id in self.motion_detectors:
            del self.motion_detectors[camera_id]

        if camera_id in self.frame_queues:
            del self.frame_queues[camera_id]

        if camera_id in self.latest_frames:
            del self.latest_frames[camera_id]

        if camera_id in self.frame_processors:
            del self.frame_processors[camera_id]

        if camera_id in self.frame_counts:
            del self.frame_counts[camera_id]

        if camera_id in self.error_counts:
            del self.error_counts[camera_id]

        # Remove from cameras dict
        del self.cameras[camera_id]

        logging.info(f"Camera {camera_id} removed successfully")

    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        total_frames = sum(self.frame_counts.values())
        total_errors = sum(self.error_counts.values())

        active_cameras = len([c for c in self.cameras.values() if c.status == "active"])

        return {
            "total_cameras": len(self.cameras),
            "active_cameras": active_cameras,
            "total_frames_processed": total_frames,
            "total_errors": total_errors,
            "error_rate": total_errors / max(total_frames, 1),
            "cameras": {
                camera_id: {
                    "frames": self.frame_counts.get(camera_id, 0),
                    "errors": self.error_counts.get(camera_id, 0),
                    "status": camera.status
                }
                for camera_id, camera in self.cameras.items()
            }
        }

    async def shutdown(self) -> None:
        """Shutdown camera manager gracefully."""
        logging.info("Shutting down Camera Manager...")

        self.running = False

        # Release all captures
        for camera_id, cap in self.captures.items():
            try:
                cap.release()
                logging.info(f"Released camera {camera_id}")
            except Exception as e:
                logging.error(f"Error releasing camera {camera_id}: {e}")

        # Wait for threads to finish
        for camera_id, thread in self.frame_processors.items():
            try:
                thread.join(timeout=5.0)
            except Exception as e:
                logging.error(f"Error joining thread for camera {camera_id}: {e}")

        # Shutdown executor
        self.executor.shutdown(wait=True)

        logging.info("Camera Manager shutdown complete")