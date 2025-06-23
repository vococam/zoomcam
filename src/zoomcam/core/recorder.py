"""
Recording Manager - Advanced Video Recording System
==================================================

Manages video recording from multiple cameras with motion-triggered recording,
automatic file management, compression, and storage optimization.
"""

import asyncio
import logging
import cv2
import threading
import time
import subprocess
import shutil
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import queue
import json
import concurrent.futures

from zoomcam.utils.exceptions import ZoomCamError, handle_exception, ErrorSeverity, ErrorCategory
from zoomcam.utils.logger import get_logger, performance_context
from zoomcam.utils.helpers import safe_filename, format_bytes, ensure_directory, cleanup_old_files
from zoomcam.utils.performance import monitor_performance, global_profiler


class RecordingState(Enum):
    """Recording states."""
    IDLE = "idle"
    WAITING = "waiting"  # Waiting for motion to start recording
    RECORDING = "recording"
    STOPPING = "stopping"
    ERROR = "error"


class RecordingQuality(Enum):
    """Recording quality presets."""
    LOW = ("libx264", "ultrafast", "23", "1M")
    MEDIUM = ("libx264", "fast", "21", "2M")
    HIGH = ("libx264", "medium", "19", "4M")
    ULTRA = ("libx264", "slow", "17", "8M")

    def __init__(self, codec, preset, crf, bitrate):
        self.codec = codec
        self.preset = preset
        self.crf = crf
        self.bitrate = bitrate


@dataclass
class RecordingSession:
    """Recording session information."""
    session_id: str
    camera_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    file_path: Optional[Path] = None
    file_size_bytes: int = 0
    frame_count: int = 0
    duration_seconds: float = 0.0
    trigger_reason: str = "manual"
    motion_events: List[Dict[str, Any]] = field(default_factory=list)
    compression_ratio: float = 0.0
    quality_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class RecorderConfig:
    """Recorder configuration for a camera."""
    camera_id: str
    enabled: bool = True
    quality: RecordingQuality = RecordingQuality.MEDIUM
    reaction_time: float = 0.5  # Time to wait before starting recording
    max_duration: int = 300  # Maximum recording duration in seconds
    post_motion_duration: int = 5  # Continue recording after motion stops
    min_duration: int = 3  # Minimum recording duration
    motion_threshold: float = 0.1  # Motion activity threshold to trigger recording
    output_format: str = "mp4"
    fps: int = 30
    resolution: Optional[Tuple[int, int]] = None  # None = use camera native
    audio_enabled: bool = False  # Audio recording (if supported)


class CameraRecorder:
    """
    Individual camera recorder with motion-triggered recording.

    Features:
    - Motion-triggered automatic recording
    - Configurable pre/post motion buffers
    - Multiple quality presets
    - Real-time compression
    - Frame dropping on performance issues
    - Metadata tracking
    """

    def __init__(self, config: RecorderConfig, output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        self.logger = get_logger("recorder",
                                context={"camera_id": config.camera_id, "component": "recorder"})

        # Recording state
        self.state = RecordingState.IDLE
        self.current_session: Optional[RecordingSession] = None
        self.last_motion_time: Optional[datetime] = None
        self.recording_start_scheduled: Optional[datetime] = None

        # Frame buffer for pre-motion recording
        self.frame_buffer: queue.Queue = queue.Queue(maxsize=150)  # 5 seconds at 30fps
        self.buffer_enabled = True

        # Video writer and encoding
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.encoder_process: Optional[subprocess.Popen] = None
        self.temp_file_path: Optional[Path] = None

        # Threading
        self.recording_thread: Optional[threading.Thread] = None
        self.buffer_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.frame_queue: queue.Queue = queue.Queue(maxsize=60)

        # Performance tracking
        self.frames_recorded = 0
        self.frames_dropped = 0
        self.recording_sessions: List[RecordingSession] = []

        # Callbacks
        self.session_callbacks: List[Callable[[RecordingSession], None]] = []

        self.logger.info(f"Camera recorder initialized for {config.camera_id}")

    async def start_recorder(self):
        """Start the recorder (ready to respond to motion)."""
        if self.state != RecordingState.IDLE:
            return

        self.stop_event.clear()

        # Start buffer thread
        self.buffer_thread = threading.Thread(
            target=self._buffer_worker,
            daemon=True
        )
        self.buffer_thread.start()

        self.state = RecordingState.WAITING
        self.logger.info("Camera recorder started and waiting for motion")

    async def stop_recorder(self):
        """Stop the recorder and any active recording."""
        self.logger.info("Stopping camera recorder")

        # Stop current recording if active
        if self.state == RecordingState.RECORDING:
            await self._stop_recording()

        # Signal threads to stop
        self.stop_event.set()
        self.state = RecordingState.IDLE

        # Wait for threads to finish
        if self.buffer_thread:
            self.buffer_thread.join(timeout=5.0)

        if self.recording_thread:
            self.recording_thread.join(timeout=10.0)

        self.logger.info("Camera recorder stopped")

    async def process_frame(self, frame: np.ndarray, motion_data: Optional[Dict[str, Any]] = None):
        """Process incoming frame and handle recording logic."""
        try:
            current_time = datetime.now()

            # Add frame to buffer if enabled
            if self.buffer_enabled:
                try:
                    # Remove old frame if buffer is full
                    if self.frame_buffer.full():
                        try:
                            self.frame_buffer.get_nowait()
                        except queue.Empty:
                            pass

                    self.frame_buffer.put_nowait((frame.copy(), current_time, motion_data))
                except queue.Full:
                    pass  # Skip frame if buffer is full

            # Handle motion detection
            if motion_data:
                await self._handle_motion_detected(motion_data, current_time)

            # Handle recording state machine
            await self._handle_recording_state(frame, current_time)

        except Exception as e:
            self.logger.error(f"Error processing frame: {e}", exc_info=True)

    async def _handle_motion_detected(self, motion_data: Dict[str, Any], current_time: datetime):
        """Handle motion detection for recording triggers."""
        activity_level = motion_data.get('activity_level', 0.0)

        if activity_level >= self.config.motion_threshold:
            self.last_motion_time = current_time

            # Schedule recording start if not already recording
            if self.state == RecordingState.WAITING and not self.recording_start_scheduled:
                self.recording_start_scheduled = current_time + timedelta(seconds=self.config.reaction_time)
                self.logger.debug(f"Motion detected, recording scheduled in {self.config.reaction_time}s")

    async def _handle_recording_state(self, frame: np.ndarray, current_time: datetime):
        """Handle recording state machine."""

        # Check if recording should start
        if (self.state == RecordingState.WAITING and
            self.recording_start_scheduled and
            current_time >= self.recording_start_scheduled):

            await self._start_recording(current_time)
            self.recording_start_scheduled = None

        # Continue recording if active
        if self.state == RecordingState.RECORDING:
            await self._write_frame(frame, current_time)

            # Check stop conditions
            should_stop = False

            # Maximum duration exceeded
            if (self.current_session and
                (current_time - self.current_session.start_time).total_seconds() >= self.config.max_duration):
                should_stop = True
                self.logger.info("Recording stopped: maximum duration reached")

            # No motion for post_motion_duration
            elif (self.last_motion_time and
                  (current_time - self.last_motion_time).total_seconds() >= self.config.post_motion_duration):
                should_stop = True
                self.logger.info("Recording stopped: no motion detected")

            if should_stop:
                await self._stop_recording()

    async def _start_recording(self, start_time: datetime):
        """Start recording session."""
        try:
            with performance_context("recorder", "start_recording", global_profiler):
                self.state = RecordingState.RECORDING

                # Create recording session
                session_id = f"{self.config.camera_id}_{start_time.strftime('%Y%m%d_%H%M%S')}"
                self.current_session = RecordingSession(
                    session_id=session_id,
                    camera_id=self.config.camera_id,
                    start_time=start_time,
                    trigger_reason="motion_detected"
                )

                # Create output file path
                filename = f"{session_id}.{self.config.output_format}"
                self.current_session.file_path = self.output_dir / filename
                self.temp_file_path = self.output_dir / f"temp_{filename}"

                # Setup video writer or encoder
                if self.config.quality in [RecordingQuality.HIGH, RecordingQuality.ULTRA]:
                    await self._setup_ffmpeg_encoder()
                else:
                    await self._setup_opencv_writer()

                # Write buffered frames (pre-motion)
                await self._write_buffered_frames()

                self.logger.info(f"Recording started: {self.current_session.session_id}")

        except Exception as e:
            self.state = RecordingState.ERROR
            self.logger.error(f"Failed to start recording: {e}", exc_info=True)
            raise

    async def _setup_opencv_writer(self):
        """Setup OpenCV VideoWriter for basic recording."""
        try:
            # Determine codec
            fourcc_map = {
                'mp4': cv2.VideoWriter_fourcc(*'mp4v'),
                'avi': cv2.VideoWriter_fourcc(*'XVID'),
                'mov': cv2.VideoWriter_fourcc(*'mp4v')
            }

            fourcc = fourcc_map.get(self.config.output_format, cv2.VideoWriter_fourcc(*'mp4v'))

            # Get frame dimensions
            resolution = self.config.resolution or (1920, 1080)  # Default resolution

            # Create video writer
            self.video_writer = cv2.VideoWriter(
                str(self.temp_file_path),
                fourcc,
                self.config.fps,
                resolution
            )

            if not self.video_writer.isOpened():
                raise RuntimeError("Failed to open video writer")

        except Exception as e:
            raise RuntimeError(f"Failed to setup OpenCV writer: {e}")

    async def _setup_ffmpeg_encoder(self):
        """Setup FFmpeg encoder for high-quality recording."""
        try:
            resolution = self.config.resolution or (1920, 1080)
            quality = self.config.quality

            # FFmpeg command for high-quality encoding
            ffmpeg_cmd = [
                'ffmpeg',
                '-y',  # Overwrite output file
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-pix_fmt', 'bgr24',
                '-s', f'{resolution[0]}x{resolution[1]}',
                '-r', str(self.config.fps),
                '-i', '-',  # Input from stdin
                '-c:v', quality.codec,
                '-preset', quality.preset,
                '-crf', quality.crf,
                '-b:v', quality.bitrate,
                '-movflags', '+faststart',  # Optimize for streaming
                str(self.temp_file_path)
            ]

            self.encoder_process = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0
            )

        except Exception as e:
            raise RuntimeError(f"Failed to setup FFmpeg encoder: {e}")

    async def _write_buffered_frames(self):
        """Write pre-motion buffered frames to recording."""
        buffered_frames = []

        # Extract all buffered frames
        while not self.frame_buffer.empty():
            try:
                buffered_frames.append(self.frame_buffer.get_nowait())
            except queue.Empty:
                break

        # Write buffered frames
        for frame, timestamp, motion_data in buffered_frames:
            await self._write_frame(frame, timestamp)

            # Add motion event to session
            if motion_data and self.current_session:
                self.current_session.motion_events.append({
                    'timestamp': timestamp.isoformat(),
                    'activity_level': motion_data.get('activity_level', 0.0),
                    'zones': len(motion_data.get('zones', []))
                })

    async def _write_frame(self, frame: np.ndarray, timestamp: datetime):
        """Write frame to recording."""
        try:
            if not self.current_session:
                return

            # Resize frame if needed
            if self.config.resolution:
                frame = cv2.resize(frame, self.config.resolution)

            # Write frame based on encoder type
            if self.video_writer:
                # OpenCV writer
                success = self.video_writer.write(frame)
                if not success:
                    self.frames_dropped += 1
                    self.logger.warning("Failed to write frame with OpenCV")
                else:
                    self.frames_recorded += 1
                    self.current_session.frame_count += 1

            elif self.encoder_process:
                # FFmpeg encoder
                try:
                    self.encoder_process.stdin.write(frame.tobytes())
                    self.encoder_process.stdin.flush()
                    self.frames_recorded += 1
                    self.current_session.frame_count += 1
                except BrokenPipeError:
                    self.logger.error("FFmpeg encoder pipe broken")
                    self.frames_dropped += 1
                except Exception as e:
                    self.logger.error(f"Error writing to FFmpeg: {e}")
                    self.frames_dropped += 1

        except Exception as e:
            self.frames_dropped += 1
            self.logger.error(f"Error writing frame: {e}")

    async def _stop_recording(self):
        """Stop current recording session."""
        if not self.current_session:
            return

        try:
            with performance_context("recorder", "stop_recording", global_profiler):
                self.state = RecordingState.STOPPING

                # Close video writer/encoder
                if self.video_writer:
                    self.video_writer.release()
                    self.video_writer = None

                if self.encoder_process:
                    try:
                        self.encoder_process.stdin.close()
                        self.encoder_process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        self.encoder_process.kill()
                        self.encoder_process.wait()
                    self.encoder_process = None

                # Finalize session
                end_time = datetime.now()
                self.current_session.end_time = end_time
                self.current_session.duration_seconds = (
                    end_time - self.current_session.start_time
                ).total_seconds()

                # Check minimum duration
                if self.current_session.duration_seconds < self.config.min_duration:
                    self.logger.info(f"Recording too short ({self.current_session.duration_seconds:.1f}s), deleting")
                    if self.temp_file_path and self.temp_file_path.exists():
                        self.temp_file_path.unlink()
                else:
                    # Move temp file to final location
                    await self._finalize_recording()

                # Add to session history
                self.recording_sessions.append(self.current_session)

                # Notify callbacks
                for callback in self.session_callbacks:
                    try:
                        callback(self.current_session)
                    except Exception as e:
                        self.logger.error(f"Error in session callback: {e}")

                self.logger.info(f"Recording stopped: {self.current_session.session_id}")
                self.current_session = None
                self.state = RecordingState.WAITING

        except Exception as e:
            self.state = RecordingState.ERROR
            self.logger.error(f"Error stopping recording: {e}", exc_info=True)

    async def _finalize_recording(self):
        """Finalize recording file and calculate metrics."""
        if not self.current_session or not self.temp_file_path:
            return

        try:
            # Move temp file to final location
            if self.temp_file_path.exists():
                shutil.move(str(self.temp_file_path), str(self.current_session.file_path))

                # Get file size
                self.current_session.file_size_bytes = self.current_session.file_path.stat().st_size

                # Calculate quality metrics
                await self._calculate_quality_metrics()

                self.logger.info(
                    f"Recording finalized: {self.current_session.file_path.name} "
                    f"({format_bytes(self.current_session.file_size_bytes)}, "
                    f"{self.current_session.duration_seconds:.1f}s)"
                )

        except Exception as e:
            self.logger.error(f"Error finalizing recording: {e}")

    async def _calculate_quality_metrics(self):
        """Calculate recording quality metrics."""
        if not self.current_session or not self.current_session.file_path:
            return

        try:
            # Basic metrics
            duration = self.current_session.duration_seconds
            file_size_mb = self.current_session.file_size_bytes / (1024 * 1024)

            if duration > 0:
                # Average bitrate
                avg_bitrate_mbps = (file_size_mb * 8) / duration

                # Compression ratio (frames vs file size)
                expected_raw_size_mb = (
                    self.current_session.frame_count *
                    (1920 * 1080 * 3) / (1024 * 1024)  # Assume 1080p RGB
                )
                compression_ratio = expected_raw_size_mb / file_size_mb if file_size_mb > 0 else 0

                self.current_session.quality_metrics = {
                    'avg_bitrate_mbps': avg_bitrate_mbps,
                    'compression_ratio': compression_ratio,
                    'fps_actual': self.current_session.frame_count / duration,
                    'file_size_mb': file_size_mb
                }

        except Exception as e:
            self.logger.error(f"Error calculating quality metrics: {e}")

    def _buffer_worker(self):
        """Background worker for frame buffer management."""
        while not self.stop_event.is_set():
            try:
                # Clean old frames from buffer (keep only last 5 seconds)
                current_time = datetime.now()
                cutoff_time = current_time - timedelta(seconds=5)

                # This is simplified - in practice, you'd want a more efficient buffer
                self.stop_event.wait(1.0)  # Check every second

            except Exception as e:
                self.logger.error(f"Buffer worker error: {e}")
                time.sleep(1.0)

    def add_session_callback(self, callback: Callable[[RecordingSession], None]):
        """Add callback for recording session events."""
        self.session_callbacks.append(callback)

    def get_statistics(self) -> Dict[str, Any]:
        """Get recorder statistics."""
        return {
            'camera_id': self.config.camera_id,
            'state': self.state.value,
            'frames_recorded': self.frames_recorded,
            'frames_dropped': self.frames_dropped,
            'drop_rate': self.frames_dropped / max(self.frames_recorded + self.frames_dropped, 1),
            'total_sessions': len(self.recording_sessions),
            'current_session': {
                'session_id': self.current_session.session_id if self.current_session else None,
                'duration': (
                    (datetime.now() - self.current_session.start_time).total_seconds()
                    if self.current_session else 0
                ),
                'frames': self.current_session.frame_count if self.current_session else 0
            } if self.current_session else None
        }


class RecordingManager:
    """
    Main recording manager coordinating multiple camera recorders.

    Features:
    - Multi-camera recording coordination
    - Storage management and cleanup
    - Recording scheduling and triggers
    - Performance monitoring
    - Metadata database
    - Export and archiving
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger("recording_manager")

        # Storage configuration
        self.output_dir = Path(config.get('output_dir', '/var/lib/zoomcam/recordings'))
        self.cleanup_days = config.get('cleanup_days', 7)
        self.max_storage_gb = config.get('max_storage_gb', 100)

        # Camera recorders
        self.recorders: Dict[str, CameraRecorder] = {}

        # Global recording state
        self.global_recording_enabled = config.get('enabled', True)
        self.storage_monitoring_enabled = True

        # Background tasks
        self.cleanup_task: Optional[asyncio.Task] = None
        self.storage_monitor_task: Optional[asyncio.Task] = None

        # Statistics
        self.total_recordings = 0
        self.total_storage_used = 0

        # Setup
        self._setup_storage()

        self.logger.info("Recording Manager initialized")

    def _setup_storage(self):
        """Setup recording storage directories."""
        ensure_directory(self.output_dir)

        # Create subdirectories by date
        today = datetime.now()
        for days_offset in range(7):  # Pre-create directories for next week
            date = today + timedelta(days=days_offset)
            date_dir = self.output_dir / date.strftime('%Y-%m-%d')
            ensure_directory(date_dir)

    async def start_recording_manager(self):
        """Start recording manager and background tasks."""
        self.logger.info("Starting recording manager")

        # Start background tasks
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        self.storage_monitor_task = asyncio.create_task(self._storage_monitor_loop())

        self.logger.info("Recording manager started")

    async def stop_recording_manager(self):
        """Stop recording manager and all recorders."""
        self.logger.info("Stopping recording manager")

        # Stop all recorders
        for recorder in self.recorders.values():
            await recorder.stop_recorder()

        # Cancel background tasks
        if self.cleanup_task:
            self.cleanup_task.cancel()
        if self.storage_monitor_task:
            self.storage_monitor_task.cancel()

        self.logger.info("Recording manager stopped")

    async def setup_camera_recorder(self, camera_id: str, camera_config: Dict[str, Any]):
        """Setup recorder for a camera."""
        try:
            # Create recorder config
            recording_config = camera_config.get('recording', {})

            recorder_config = RecorderConfig(
                camera_id=camera_id,
                enabled=recording_config.get('enabled', True),
                quality=RecordingQuality[recording_config.get('quality', 'MEDIUM').upper()],
                reaction_time=recording_config.get('reaction_time', 0.5),
                max_duration=recording_config.get('max_duration', 300),
                post_motion_duration=recording_config.get('post_motion_duration', 5),
                min_duration=recording_config.get('min_duration', 3),
                motion_threshold=recording_config.get('motion_threshold', 0.1),
                fps=camera_config.get('fps', 30),
                resolution=self._parse_resolution(camera_config.get('resolution', 'auto'))
            )

            # Create output directory for camera
            camera_output_dir = self.output_dir / camera_id
            ensure_directory(camera_output_dir)

            # Create recorder
            recorder = CameraRecorder(recorder_config, camera_output_dir)

            # Add session callback
            recorder.add_session_callback(self._on_recording_session_complete)

            self.recorders[camera_id] = recorder

            # Start recorder if enabled
            if recorder_config.enabled and self.global_recording_enabled:
                await recorder.start_recorder()

            self.logger.info(f"Setup camera recorder for {camera_id}")

        except Exception as e:
            self.logger.error(f"Failed to setup recorder for {camera_id}: {e}")
            raise

    def _parse_resolution(self, resolution_str: str) -> Optional[Tuple[int, int]]:
        """Parse resolution string."""
        if resolution_str == 'auto' or not resolution_str:
            return None

        try:
            if 'x' in resolution_str:
                width, height = resolution_str.split('x')
                return (int(width), int(height))
        except Exception:
            pass

        return None

    async def process_camera_frame(self, camera_id: str, frame: np.ndarray, motion_data: Optional[Dict[str, Any]] = None):
        """Process frame from camera for recording."""
        if camera_id in self.recorders:
            await self.recorders[camera_id].process_frame(frame, motion_data)

    async def start_manual_recording(self, camera_id: str, duration: Optional[int] = None) -> str:
        """Start manual recording for camera."""
        if camera_id not in self.recorders:
            raise ValueError(f"No recorder found for camera {camera_id}")

        recorder = self.recorders[camera_id]

        # Force start recording
        session_id = f"{camera_id}_manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # This is simplified - you'd need to implement manual recording mode
        self.logger.info(f"Manual recording started for {camera_id}: {session_id}")

        return session_id

    async def stop_manual_recording(self, camera_id: str):
        """Stop manual recording for camera."""
        if camera_id in self.recorders:
            # This would stop any active recording
            await self.recorders[camera_id]._stop_recording()

    def _on_recording_session_complete(self, session: RecordingSession):
        """Handle completed recording session."""
        try:
            self.total_recordings += 1

            if session.file_path and session.file_path.exists():
                self.total_storage_used += session.file_size_bytes

            self.logger.info(
                f"Recording session completed: {session.session_id} "
                f"({session.duration_seconds:.1f}s, {format_bytes(session.file_size_bytes)})"
            )

        except Exception as e:
            self.logger.error(f"Error handling session completion: {e}")

    async def _cleanup_loop(self):
        """Background cleanup loop."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run cleanup every hour
                await self._cleanup_old_recordings()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error

    async def _cleanup_old_recordings(self):
        """Clean up old recordings based on retention policy."""
        try:
            cutoff_time = datetime.now() - timedelta(days=self.cleanup_days)

            deleted_files = cleanup_old_files(
                self.output_dir,
                max_age_days=self.cleanup_days,
                pattern="*.mp4"
            )

            if deleted_files:
                total_size = sum(f.stat().st_size for f in deleted_files if f.exists())
                self.logger.info(
                    f"Cleaned up {len(deleted_files)} old recordings "
                    f"({format_bytes(total_size)} freed)"
                )

        except Exception as e:
            self.logger.error(f"Error cleaning up recordings: {e}")

    async def _storage_monitor_loop(self):
        """Monitor storage usage."""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                await self._check_storage_usage()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Storage monitor error: {e}")
                await asyncio.sleep(60)

    async def _check_storage_usage(self):
        """Check storage usage and take action if needed."""
        try:
            # Calculate total usage
            total_size = 0
            for file_path in self.output_dir.rglob("*.mp4"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size

            usage_gb = total_size / (1024**3)

            if usage_gb > self.max_storage_gb:
                self.logger.warning(
                    f"Storage usage ({usage_gb:.