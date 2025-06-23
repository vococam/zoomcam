"""
Stream Processor - HLS Stream Processing and Composition
=======================================================

Processes camera frames, composes layout, and generates HLS streams
for web browser consumption with real-time layout adaptation.
"""

import asyncio
import logging
import cv2
import numpy as np
import subprocess
import os
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import threading
import queue
import tempfile

from zoomcam.core.layout_engine import LayoutEngine, LayoutResult
from zoomcam.utils.exceptions import StreamProcessingError


@dataclass
class StreamSegment:
    """HLS stream segment information."""
    filename: str
    duration: float
    sequence: int
    timestamp: datetime


@dataclass
class CompositeFrame:
    """Composed frame with layout information."""
    frame: np.ndarray
    layout: LayoutResult
    timestamp: datetime
    frame_number: int


class StreamProcessor:
    """
    Processes multiple camera streams and generates HLS output.

    Features:
    - Real-time frame composition based on layout
    - HLS stream generation with FFmpeg
    - Adaptive bitrate and quality
    - CSS-synchronized layout updates
    - Performance monitoring and optimization
    """

    def __init__(self, config: Dict[str, Any], layout_engine: LayoutEngine, auto_config=None):
        self.config = config
        self.layout_engine = layout_engine
        self.auto_config = auto_config

        # Stream configuration
        self.output_dir = Path(config.get("hls_output_dir", "/tmp/zoomcam_hls"))
        self.segment_duration = config.get("segment_duration", 2)
        self.segment_count = config.get("segment_count", 10)
        self.bitrate = config.get("bitrate", "2M")
        self.target_fps = config.get("target_fps", 30)

        # Processing state
        self.running = False
        self.frame_queue = asyncio.Queue(maxsize=60)  # 2 seconds buffer at 30fps
        self.output_resolution = (1920, 1080)  # Will be set from config
        self.current_layout: Optional[LayoutResult] = None

        # FFmpeg process
        self.ffmpeg_process: Optional[subprocess.Popen] = None
        self.ffmpeg_input_pipe = None

        # Performance tracking
        self.frames_processed = 0
        self.frames_dropped = 0
        self.processing_times: List[float] = []

        # Threading
        self.compositor_thread: Optional[threading.Thread] = None
        self.compositor_queue = queue.Queue(maxsize=30)

        # Initialize output directory
        self._setup_output_directory()

        logging.info("Stream Processor initialized")

    def _setup_output_directory(self) -> None:
        """Setup HLS output directory."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Clean old segments
        for file in self.output_dir.glob("*.ts"):
            try:
                file.unlink()
            except OSError:
                pass

        for file in self.output_dir.glob("*.m3u8"):
            try:
                file.unlink()
            except OSError:
                pass

    async def start_processing(self) -> None:
        """Start stream processing pipeline."""
        try:
            self.running = True

            # Start FFmpeg process for HLS generation
            self._start_ffmpeg_process()

            # Start compositor thread
            self.compositor_thread = threading.Thread(
                target=self._compositor_worker,
                daemon=True
            )
            self.compositor_thread.start()

            # Main processing loop
            await self._processing_loop()

        except Exception as e:
            logging.error(f"Stream processing error: {e}")
            raise StreamProcessingError(f"Failed to start stream processing: {e}")

    def _start_ffmpeg_process(self) -> None:
        """Start FFmpeg process for HLS encoding."""
        playlist_path = self.output_dir / "stream.m3u8"

        # FFmpeg command for HLS generation
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",  # Overwrite output files
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self.output_resolution[0]}x{self.output_resolution[1]}",
            "-r", str(self.target_fps),
            "-i", "-",  # Input from stdin
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-b:v", self.bitrate,
            "-maxrate", self.bitrate,
            "-bufsize", f"{int(self.bitrate[:-1]) * 2}M",
            "-g", str(self.target_fps * 2),  # Keyframe interval
            "-hls_time", str(self.segment_duration),
            "-hls_list_size", str(self.segment_count),
            "-hls_flags", "delete_segments",
            "-hls_allow_cache", "0",
            "-hls_segment_filename", str(self.output_dir / "segment_%03d.ts"),
            str(playlist_path)
        ]

        try:
            self.ffmpeg_process = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0
            )

            self.ffmpeg_input_pipe = self.ffmpeg_process.stdin
            logging.info("FFmpeg HLS process started")

        except Exception as e:
            raise StreamProcessingError(f"Failed to start FFmpeg: {e}")

    async def _processing_loop(self) -> None:
        """Main processing loop."""
        frame_interval = 1.0 / self.target_fps
        next_frame_time = datetime.now()

        while self.running:
            try:
                current_time = datetime.now()

                # Wait for next frame time
                if current_time < next_frame_time:
                    sleep_time = (next_frame_time - current_time).total_seconds()
                    await asyncio.sleep(max(0, sleep_time))

                # Get current layout
                layout = await self.layout_engine.calculate_layout()
                self.current_layout = layout

                # Request frame composition
                composition_request = {
                    'layout': layout,
                    'timestamp': datetime.now(),
                    'frame_number': self.frames_processed + 1
                }

                try:
                    self.compositor_queue.put_nowait(composition_request)
                except queue.Full:
                    self.frames_dropped += 1
                    logging.warning("Compositor queue full, dropping frame")

                # Update auto-config with layout changes
                if self.auto_config and layout:
                    await self.auto_config.update_layout_state(layout)

                next_frame_time += timedelta(seconds=frame_interval)

            except Exception as e:
                logging.error(f"Processing loop error: {e}")
                await asyncio.sleep(1)

    def _compositor_worker(self) -> None:
        """Worker thread for frame composition."""
        while self.running:
            try:
                # Get composition request
                try:
                    request = self.compositor_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                start_time = datetime.now()

                # Compose frame
                composite_frame = self._compose_frame(request['layout'])

                if composite_frame is not None:
                    # Send to FFmpeg
                    self._send_frame_to_ffmpeg(composite_frame)
                    self.frames_processed += 1

                    # Track processing time
                    processing_time = (datetime.now() - start_time).total_seconds()
                    self.processing_times.append(processing_time)
                    if len(self.processing_times) > 100:
                        self.processing_times.pop(0)

                self.compositor_queue.task_done()

            except Exception as e:
                logging.error(f"Compositor worker error: {e}")

    def _compose_frame(self, layout: LayoutResult) -> Optional[np.ndarray]:
        """Compose frame based on layout."""
        try:
            # Create blank canvas
            canvas = np.zeros(
                (self.output_resolution[1], self.output_resolution[0], 3),
                dtype=np.uint8
            )

            if not layout or not layout.cells:
                # No layout - show blank screen or logo
                self._draw_no_signal_screen(canvas)
                return canvas

            # Calculate cell dimensions
            canvas_width, canvas_height = self.output_resolution

            # Parse grid structure
            grid_cols = len(layout.grid_columns.split())
            grid_rows = len(layout.grid_rows.split())

            cell_width = canvas_width // grid_cols
            cell_height = canvas_height // grid_rows

            # Compose each cell
            for cell in layout.cells:
                if not cell.camera_fragment:
                    continue

                # Get camera frame
                camera_frame = self._get_camera_frame(cell.camera_fragment.camera_id)
                if camera_frame is None:
                    # Draw no signal for this cell
                    self._draw_cell_no_signal(
                        canvas, cell, cell_width, cell_height,
                        cell.camera_fragment.camera_id
                    )
                    continue

                # Calculate cell position
                cell_x = cell.x * cell_width
                cell_y = cell.y * cell_height
                cell_w = cell.width * cell_width
                cell_h = cell.height * cell_height

                # Resize camera frame to fit cell
                resized_frame = cv2.resize(camera_frame, (cell_w, cell_h))

                # Apply fragment cropping if needed
                if hasattr(cell.camera_fragment, 'bbox') and cell.camera_fragment.bbox:
                    bbox = cell.camera_fragment.bbox
                    # Crop to specific region (for motion zones)
                    x, y, w, h = bbox
                    if x >= 0 and y >= 0 and x + w <= resized_frame.shape[1] and y + h <= resized_frame.shape[0]:
                        resized_frame = resized_frame[y:y + h, x:x + w]
                        resized_frame = cv2.resize(resized_frame, (cell_w, cell_h))

                # Apply activity-based effects
                if cell.camera_fragment.activity_level > 0.5:
                    # Enhance active areas
                    resized_frame = self._enhance_active_frame(resized_frame)
                elif cell.camera_fragment.activity_level < 0.1:
                    # Dim inactive areas
                    resized_frame = self._dim_inactive_frame(resized_frame)

                # Draw border based on activity
                border_color = self._get_border_color(cell.camera_fragment.activity_level)
                border_thickness = 3 if cell.camera_fragment.activity_level > 0.3 else 1

                cv2.rectangle(
                    resized_frame,
                    (0, 0),
                    (cell_w - 1, cell_h - 1),
                    border_color,
                    border_thickness
                )

                # Overlay frame info
                self._draw_frame_info(resized_frame, cell.camera_fragment)

                # Place frame on canvas
                canvas[cell_y:cell_y + cell_h, cell_x:cell_x + cell_w] = resized_frame

            # Draw grid lines
            self._draw_grid_lines(canvas, grid_cols, grid_rows)

            # Draw global overlay info
            self._draw_global_overlay(canvas, layout)

            return canvas

        except Exception as e:
            logging.error(f"Frame composition error: {e}")
            return None

    def _get_camera_frame(self, camera_id: str) -> Optional[np.ndarray]:
        """Get latest frame from camera manager."""
        # This would be injected or accessed via camera manager
        # For now, return a placeholder
        if hasattr(self, 'camera_manager'):
            frame_data = asyncio.run_coroutine_threadsafe(
                self.camera_manager.get_latest_frame(camera_id),
                asyncio.get_event_loop()
            ).result(timeout=0.1)

            if frame_data and frame_data.frame is not None:
                return frame_data.frame

        # Return test pattern if no frame available
        return self._generate_test_pattern(camera_id)

    def _generate_test_pattern(self, camera_id: str) -> np.ndarray:
        """Generate test pattern for missing camera."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Draw checkerboard pattern
        for y in range(0, 480, 40):
            for x in range(0, 640, 40):
                if (x // 40 + y // 40) % 2 == 0:
                    frame[y:y + 40, x:x + 40] = (64, 64, 64)

        # Add camera ID text
        cv2.putText(
            frame,
            f"Camera: {camera_id}",
            (50, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "TEST PATTERN",
            (200, 280),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        return frame

    def _draw_no_signal_screen(self, canvas: np.ndarray) -> None:
        """Draw no signal screen."""
        height, width = canvas.shape[:2]

        # Dark background
        canvas.fill(20)

        # Draw "No Signal" text
        text = "ZoomCam - No Active Cameras"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)[0]
        text_x = (width - text_size[0]) // 2
        text_y = (height + text_size[1]) // 2

        cv2.putText(
            canvas,
            text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (100, 100, 100),
            3
        )

        # Draw timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(
            canvas,
            timestamp,
            (50, height - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (150, 150, 150),
            2
        )

    def _draw_cell_no_signal(
            self,
            canvas: np.ndarray,
            cell,
            cell_width: int,
            cell_height: int,
            camera_id: str
    ) -> None:
        """Draw no signal for specific cell."""
        cell_x = cell.x * cell_width
        cell_y = cell.y * cell_height
        cell_w = cell.width * cell_width
        cell_h = cell.height * cell_height

        # Dark cell background
        canvas[cell_y:cell_y + cell_h, cell_x:cell_x + cell_w] = (30, 30, 30)

        # Red dashed border
        cv2.rectangle(
            canvas,
            (cell_x, cell_y),
            (cell_x + cell_w - 1, cell_y + cell_h - 1),
            (0, 0, 255),
            2
        )

        # "No Signal" text
        text = "No Signal"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = cell_x + (cell_w - text_size[0]) // 2
        text_y = cell_y + (cell_h + text_size[1]) // 2

        cv2.putText(
            canvas,
            text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (100, 100, 255),
            2
        )

        # Camera ID
        cv2.putText(
            canvas,
            camera_id,
            (cell_x + 10, cell_y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (150, 150, 150),
            1
        )

    def _enhance_active_frame(self, frame: np.ndarray) -> np.ndarray:
        """Enhance frame for active areas."""
        # Slight brightness and contrast boost
        enhanced = cv2.convertScaleAbs(frame, alpha=1.1, beta=10)
        return enhanced

    def _dim_inactive_frame(self, frame: np.ndarray) -> np.ndarray:
        """Dim frame for inactive areas."""
        # Reduce brightness
        dimmed = cv2.convertScaleAbs(frame, alpha=0.7, beta=-20)
        return dimmed

    def _get_border_color(self, activity_level: float) -> Tuple[int, int, int]:
        """Get border color based on activity level."""
        if activity_level > 0.7:
            return (0, 255, 0)  # Green - high activity
        elif activity_level > 0.3:
            return (0, 255, 255)  # Yellow - medium activity
        elif activity_level > 0.1:
            return (255, 100, 0)  # Orange - low activity
        else:
            return (100, 100, 100)  # Gray - no activity

    def _draw_frame_info(self, frame: np.ndarray, fragment) -> None:
        """Draw frame information overlay."""
        height, width = frame.shape[:2]

        # Activity indicator
        activity_text = f"Act: {fragment.activity_level:.1%}"
        cv2.putText(
            frame,
            activity_text,
            (10, height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

    def _draw_grid_lines(self, canvas: np.ndarray, grid_cols: int, grid_rows: int) -> None:
        """Draw grid lines on canvas."""
        height, width = canvas.shape[:2]

        # Vertical lines
        for i in range(1, grid_cols):
            x = (width * i) // grid_cols
            cv2.line(canvas, (x, 0), (x, height), (80, 80, 80), 1)

        # Horizontal lines
        for i in range(1, grid_rows):
            y = (height * i) // grid_rows
            cv2.line(canvas, (0, y), (width, y), (80, 80, 80), 1)

    def _draw_global_overlay(self, canvas: np.ndarray, layout: LayoutResult) -> None:
        """Draw global overlay information."""
        height, width = canvas.shape[:2]

        # System info
        timestamp = datetime.now().strftime("%H:%M:%S")
        fps_text = f"FPS: {self.target_fps} | {timestamp}"

        cv2.putText(
            canvas,
            fps_text,
            (width - 250, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 200, 200),
            1
        )

        # Layout info
        layout_info = f"Layout: {len(layout.cells)} cells, {layout.layout_efficiency:.1%} eff"
        cv2.putText(
            canvas,
            layout_info,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1
        )

    def _send_frame_to_ffmpeg(self, frame: np.ndarray) -> None:
        """Send frame to FFmpeg process."""
        try:
            if self.ffmpeg_process and self.ffmpeg_input_pipe:
                # Ensure frame is correct size and format
                if frame.shape[:2] != (self.output_resolution[1], self.output_resolution[0]):
                    frame = cv2.resize(frame, self.output_resolution)

                # Write frame data
                self.ffmpeg_input_pipe.write(frame.tobytes())
                self.ffmpeg_input_pipe.flush()

        except BrokenPipeError:
            logging.error("FFmpeg pipe broken, restarting...")
            self._restart_ffmpeg()
        except Exception as e:
            logging.error(f"Error sending frame to FFmpeg: {e}")

    def _restart_ffmpeg(self) -> None:
        """Restart FFmpeg process."""
        try:
            if self.ffmpeg_process:
                self.ffmpeg_process.terminate()
                self.ffmpeg_process.wait(timeout=5)
        except:
            pass

        self._start_ffmpeg_process()

    async def get_stream_url(self) -> str:
        """Get HLS stream URL."""
        return f"/hls/stream.m3u8"

    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get stream processing performance statistics."""
        avg_processing_time = (
            sum(self.processing_times) / len(self.processing_times)
            if self.processing_times else 0
        )

        return {
            "frames_processed": self.frames_processed,
            "frames_dropped": self.frames_dropped,
            "drop_rate": self.frames_dropped / max(self.frames_processed, 1),
            "avg_processing_time_ms": avg_processing_time * 1000,
            "target_fps": self.target_fps,
            "current_layout_efficiency": (
                self.current_layout.layout_efficiency if self.current_layout else 0
            ),
            "output_resolution": f"{self.output_resolution[0]}x{self.output_resolution[1]}",
            "bitrate": self.bitrate
        }

    async def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update stream processing configuration."""
        restart_needed = False

        # Check if restart is needed
        if (new_config.get("bitrate") != self.bitrate or
                new_config.get("target_fps") != self.target_fps):
            restart_needed = True

        # Update configuration
        self.config.update(new_config)
        self.bitrate = new_config.get("bitrate", self.bitrate)
        self.target_fps = new_config.get("target_fps", self.target_fps)

        if restart_needed and self.running:
            logging.info("Restarting FFmpeg due to configuration change")
            self._restart_ffmpeg()

    def set_camera_manager(self, camera_manager) -> None:
        """Set camera manager reference."""
        self.camera_manager = camera_manager

    async def shutdown(self) -> None:
        """Shutdown stream processor gracefully."""
        logging.info("Shutting down Stream Processor...")

        self.running = False

        # Stop FFmpeg
        if self.ffmpeg_process:
            try:
                self.ffmpeg_input_pipe.close()
                self.ffmpeg_process.terminate()
                self.ffmpeg_process.wait(timeout=10)
            except Exception as e:
                logging.error(f"Error stopping FFmpeg: {e}")
                try:
                    self.ffmpeg_process.kill()
                except:
                    pass

        # Wait for compositor thread
        if self.compositor_thread:
            self.compositor_thread.join(timeout=5)

        logging.info("Stream Processor shutdown complete")