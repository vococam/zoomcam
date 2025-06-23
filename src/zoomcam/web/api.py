"""
FastAPI Web Interface - REST API and Setup Panel
===============================================

Provides web-based configuration, camera setup wizard,
and real-time monitoring interface for ZoomCam system.
"""

import asyncio
import logging
import json
import yaml
import cv2
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, Depends
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from zoomcam.core.camera_manager import CameraManager
from zoomcam.core.layout_engine import LayoutEngine
from zoomcam.core.stream_processor import StreamProcessor
from zoomcam.core.git_logger import GitLogger
from zoomcam.core.auto_config_manager import AutoConfigManager
from zoomcam.config.config_manager import ConfigManager
from zoomcam.utils.exceptions import ZoomCamError


# Pydantic models for API
class CameraDetectionRequest(BaseModel):
    """Request model for camera detection."""
    scan_usb: bool = True
    scan_rtsp: bool = False
    rtsp_urls: List[str] = Field(default_factory=list)
    timeout_seconds: int = 10


class CameraTestRequest(BaseModel):
    """Request model for camera testing."""
    source: str
    timeout_seconds: int = 5


class CameraConfigRequest(BaseModel):
    """Request model for camera configuration."""
    name: str
    source: str
    resolution: str = "auto"
    zoom: float = Field(ge=1.0, le=10.0, default=3.0)
    max_fragments: int = Field(ge=1, le=5, default=2)
    recording_enabled: bool = True
    motion_sensitivity: float = Field(ge=0.0, le=1.0, default=0.3)


class SystemConfigRequest(BaseModel):
    """Request model for system configuration."""
    display_resolution: str
    target_fps: int = Field(ge=5, le=60, default=30)
    quality_priority: str = Field(regex="^(performance|balanced|quality)$", default="balanced")
    recording_enabled: bool = True
    git_logging_enabled: bool = True


class RTSPTestResult(BaseModel):
    """RTSP test result model."""
    url: str
    success: bool
    resolution: Optional[str] = None
    fps: Optional[float] = None
    error_message: Optional[str] = None


# Global app state
app_state = {
    "camera_manager": None,
    "layout_engine": None,
    "stream_processor": None,
    "config_manager": None,
    "git_logger": None,
    "auto_config": None,
    "setup_mode": True,  # Start in setup mode
    "setup_completed": False
}


def create_app(
        camera_manager: CameraManager,
        layout_engine: LayoutEngine,
        stream_processor: StreamProcessor,
        config_manager: ConfigManager,
        git_logger: Optional[GitLogger] = None,
        auto_config: Optional[AutoConfigManager] = None
) -> FastAPI:
    """Create FastAPI application with all components."""

    # Update global state
    app_state.update({
        "camera_manager": camera_manager,
        "layout_engine": layout_engine,
        "stream_processor": stream_processor,
        "config_manager": config_manager,
        "git_logger": git_logger,
        "auto_config": auto_config
    })

    app = FastAPI(
        title="ZoomCam",
        description="Intelligent Adaptive Camera Monitoring System",
        version="0.1.0"
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Templates and static files
    templates = Jinja2Templates(directory="src/zoomcam/web/templates")

    # Mount static files
    app.mount("/static", StaticFiles(directory="src/zoomcam/web/static"), name="static")
    app.mount("/hls", StaticFiles(directory="/tmp/zoomcam_hls"), name="hls")
    app.mount("/screenshots", StaticFiles(directory="logs/screenshots"), name="screenshots")

    # Store components in app state
    app.state.camera_manager = camera_manager
    app.state.layout_engine = layout_engine
    app.state.stream_processor = stream_processor
    app.state.config_manager = config_manager
    app.state.git_logger = git_logger
    app.state.auto_config = auto_config

    # Setup routes
    setup_routes(app, templates)

    return app


def setup_routes(app: FastAPI, templates: Jinja2Templates):
    """Setup all application routes."""

    # ============ MAIN ROUTES ============

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        """Main application page or setup wizard."""
        if app_state["setup_mode"] and not app_state["setup_completed"]:
            return templates.TemplateResponse("setup_wizard.html", {
                "request": request,
                "step": "welcome"
            })
        else:
            return templates.TemplateResponse("index.html", {
                "request": request,
                "stream_url": "/hls/stream.m3u8"
            })

    @app.get("/setup", response_class=HTMLResponse)
    async def setup_wizard(request: Request, step: str = "welcome"):
        """Setup wizard interface."""
        return templates.TemplateResponse("setup_wizard.html", {
            "request": request,
            "step": step
        })

    @app.get("/config", response_class=HTMLResponse)
    async def config_panel(request: Request):
        """Configuration panel."""
        return templates.TemplateResponse("config.html", {"request": request})

    @app.get("/timeline", response_class=HTMLResponse)
    async def timeline_view(request: Request):
        """Timeline view for event history."""
        return templates.TemplateResponse("timeline.html", {"request": request})

    @app.get("/monitor", response_class=HTMLResponse)
    async def admin_monitor(request: Request):
        """Admin monitoring panel."""
        return templates.TemplateResponse("admin_monitor.html", {"request": request})

    # ============ SETUP API ROUTES ============

    @app.post("/api/setup/detect-cameras")
    async def detect_cameras(request: CameraDetectionRequest):
        """Detect available cameras for setup."""
        try:
            detected_cameras = []

            if request.scan_usb:
                # Scan USB cameras
                for i in range(10):
                    cap = cv2.VideoCapture(i)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            fps = cap.get(cv2.CAP_PROP_FPS)

                            detected_cameras.append({
                                "id": f"usb_camera_{i}",
                                "name": f"USB Camera {i}",
                                "source": f"/dev/video{i}",
                                "type": "usb",
                                "resolution": f"{width}x{height}",
                                "fps": fps,
                                "status": "detected"
                            })
                        cap.release()

            if request.scan_rtsp:
                # Test RTSP URLs
                for url in request.rtsp_urls:
                    result = await test_rtsp_camera(url, request.timeout_seconds)
                    if result.success:
                        detected_cameras.append({
                            "id": f"rtsp_camera_{len(detected_cameras)}",
                            "name": f"RTSP Camera ({url})",
                            "source": url,
                            "type": "rtsp",
                            "resolution": result.resolution,
                            "fps": result.fps,
                            "status": "detected"
                        })

            return {"cameras": detected_cameras, "total_found": len(detected_cameras)}

        except Exception as e:
            logging.error(f"Camera detection failed: {e}")
            raise HTTPException(status_code=500, detail=f"Detection failed: {e}")

    @app.post("/api/setup/test-camera")
    async def test_camera_endpoint(request: CameraTestRequest):
        """Test camera connection."""
        try:
            if request.source.startswith("rtsp://"):
                result = await test_rtsp_camera(request.source, request.timeout_seconds)
                return {
                    "success": result.success,
                    "resolution": result.resolution,
                    "fps": result.fps,
                    "error": result.error_message
                }
            else:
                # Test USB camera
                success, message = test_usb_camera(request.source)
                return {
                    "success": success,
                    "message": message
                }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    @app.post("/api/setup/configure-cameras")
    async def configure_cameras(cameras: List[CameraConfigRequest]):
        """Configure detected cameras."""
        try:
            config = app_state["config_manager"].get_config()

            # Clear existing cameras
            config["cameras"] = {}

            # Add new cameras
            for i, camera in enumerate(cameras):
                camera_id = f"camera_{i + 1}"
                config["cameras"][camera_id] = {
                    "enabled": True,
                    "name": camera.name,
                    "source": camera.source,
                    "resolution": camera.resolution,
                    "zoom": camera.zoom,
                    "max_fragments": camera.max_fragments,
                    "recording": {
                        "enabled": camera.recording_enabled,
                        "quality": "medium",
                        "reaction_time": 0.5,
                        "max_duration": 300
                    },
                    "motion_detection": {
                        "sensitivity": camera.motion_sensitivity,
                        "min_area": 500,
                        "max_zones": 5,
                        "ignore_zones": []
                    },
                    "display": {
                        "min_size_percent": 10,
                        "max_size_percent": 70,
                        "transition_speed": 0.8
                    }
                }

            # Save configuration
            await app_state["config_manager"].save_config(config)

            # Log configuration change
            if app_state["git_logger"]:
                await app_state["git_logger"].log_config_change(
                    {"cameras_configured": len(cameras)},
                    config
                )

            return {"success": True, "cameras_configured": len(cameras)}

        except Exception as e:
            logging.error(f"Camera configuration failed: {e}")
            raise HTTPException(status_code=500, detail=f"Configuration failed: {e}")

    @app.post("/api/setup/configure-system")
    async def configure_system(request: SystemConfigRequest):
        """Configure system settings."""
        try:
            config = app_state["config_manager"].get_config()

            # Update system configuration
            config["system"]["display"]["target_resolution"] = request.display_resolution
            config["system"]["display"]["interpolation"]["quality_priority"] = request.quality_priority
            config["streaming"]["target_fps"] = request.target_fps
            config["recording"]["enabled"] = request.recording_enabled
            config["logging"]["git"]["enabled"] = request.git_logging_enabled

            # Save configuration
            await app_state["config_manager"].save_config(config)

            return {"success": True, "message": "System configured successfully"}

        except Exception as e:
            logging.error(f"System configuration failed: {e}")
            raise HTTPException(status_code=500, detail=f"Configuration failed: {e}")

    @app.post("/api/setup/complete")
    async def complete_setup():
        """Complete setup process and start normal operation."""
        try:
            app_state["setup_completed"] = True
            app_state["setup_mode"] = False

            # Initialize components with new configuration
            # This would restart the camera manager, etc.

            # Log setup completion
            if app_state["git_logger"]:
                await app_state["git_logger"].log_event(
                    "setup_completed",
                    {"timestamp": datetime.now().isoformat()},
                    summary="Initial system setup completed"
                )

            return {"success": True, "message": "Setup completed successfully"}

        except Exception as e:
            logging.error(f"Setup completion failed: {e}")
            raise HTTPException(status_code=500, detail=f"Setup completion failed: {e}")

    # ============ CAMERA API ROUTES ============

    @app.get("/api/cameras")
    async def get_cameras():
        """Get camera status and configuration."""
        try:
            if app_state["camera_manager"]:
                return await app_state["camera_manager"].get_camera_status()
            return {"cameras": {}}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/cameras/{camera_id}/config")
    async def update_camera_config(camera_id: str, updates: Dict[str, Any]):
        """Update camera configuration."""
        try:
            if app_state["camera_manager"]:
                await app_state["camera_manager"].update_camera_config(camera_id, updates)
                return {"success": True, "message": f"Camera {camera_id} updated"}
            raise HTTPException(status_code=503, detail="Camera manager not available")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/cameras/{camera_id}/restart")
    async def restart_camera(camera_id: str):
        """Restart specific camera."""
        try:
            if app_state["camera_manager"]:
                await app_state["camera_manager"].restart_camera(camera_id)
                return {"success": True, "message": f"Camera {camera_id} restarted"}
            raise HTTPException(status_code=503, detail="Camera manager not available")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/cameras/add")
    async def add_camera(camera_config: CameraConfigRequest):
        """Add new camera dynamically."""
        try:
            if app_state["camera_manager"]:
                camera_dict = {
                    "name": camera_config.name,
                    "source": camera_config.source,
                    "resolution": camera_config.resolution,
                    "zoom": camera_config.zoom,
                    "max_fragments": camera_config.max_fragments,
                    "recording": {"enabled": camera_config.recording_enabled},
                    "motion_detection": {"sensitivity": camera_config.motion_sensitivity}
                }
                camera_id = await app_state["camera_manager"].add_camera(camera_dict)
                return {"success": True, "camera_id": camera_id}
            raise HTTPException(status_code=503, detail="Camera manager not available")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/api/cameras/{camera_id}")
    async def remove_camera(camera_id: str):
        """Remove camera."""
        try:
            if app_state["camera_manager"]:
                await app_state["camera_manager"].remove_camera(camera_id)
                return {"success": True, "message": f"Camera {camera_id} removed"}
            raise HTTPException(status_code=503, detail="Camera manager not available")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # ============ STREAMING API ROUTES ============

    @app.get("/api/stream/url")
    async def get_stream_url():
        """Get HLS stream URL."""
        try:
            if app_state["stream_processor"]:
                stream_url = await app_state["stream_processor"].get_stream_url()
                return {"stream_url": stream_url}
            return {"stream_url": "/hls/stream.m3u8"}  # Default
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/stream/stats")
    async def get_stream_stats():
        """Get streaming performance statistics."""
        try:
            if app_state["stream_processor"]:
                return await app_state["stream_processor"].get_performance_stats()
            return {"error": "Stream processor not available"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # ============ LAYOUT API ROUTES ============

    @app.get("/api/layout/current")
    async def get_current_layout():
        """Get current layout configuration."""
        try:
            if app_state["layout_engine"]:
                layout = app_state["layout_engine"].get_current_layout()
                if layout:
                    return {
                        "grid_columns": layout.grid_columns,
                        "grid_rows": layout.grid_rows,
                        "grid_areas": layout.grid_areas,
                        "css_template": layout.css_template,
                        "efficiency": layout.layout_efficiency,
                        "fragments": len(layout.fragments),
                        "timestamp": layout.timestamp.isoformat()
                    }
            return {"error": "No layout available"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/layout/recalculate")
    async def recalculate_layout():
        """Force layout recalculation."""
        try:
            if app_state["layout_engine"]:
                layout = await app_state["layout_engine"].force_layout_recalculation()
                return {"success": True, "efficiency": layout.layout_efficiency}
            raise HTTPException(status_code=503, detail="Layout engine not available")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/layout/stats")
    async def get_layout_stats():
        """Get layout statistics."""
        try:
            if app_state["layout_engine"]:
                return await app_state["layout_engine"].get_layout_stats()
            return {"error": "Layout engine not available"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # ============ CONFIGURATION API ROUTES ============

    @app.get("/api/config")
    async def get_config():
        """Get current configuration."""
        try:
            if app_state["config_manager"]:
                return app_state["config_manager"].get_config()
            return {"error": "Config manager not available"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/config")
    async def update_config(config_updates: Dict[str, Any]):
        """Update configuration."""
        try:
            if app_state["config_manager"]:
                config = app_state["config_manager"].get_config()

                # Apply updates
                for key, value in config_updates.items():
                    _set_nested_value(config, key, value)

                await app_state["config_manager"].save_config(config)

                # Log configuration change
                if app_state["git_logger"]:
                    await app_state["git_logger"].log_config_change(
                        {"updates": config_updates},
                        config
                    )

                return {"success": True, "updated_keys": list(config_updates.keys())}
            raise HTTPException(status_code=503, detail="Config manager not available")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/config/auto")
    async def get_auto_config():
        """Get auto-generated configuration."""
        try:
            if app_state["auto_config"]:
                return app_state["auto_config"].get_auto_config()
            return {"error": "Auto config not available"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # ============ MONITORING API ROUTES ============

    @app.get("/api/system/status")
    async def get_system_status():
        """Get overall system status."""
        try:
            status = {
                "setup_mode": app_state["setup_mode"],
                "setup_completed": app_state["setup_completed"],
                "timestamp": datetime.now().isoformat(),
                "components": {}
            }

            # Check component status
            if app_state["camera_manager"]:
                camera_stats = await app_state["camera_manager"].get_performance_stats()
                status["components"]["cameras"] = {
                    "status": "active",
                    "total_cameras": camera_stats["total_cameras"],
                    "active_cameras": camera_stats["active_cameras"]
                }

            if app_state["stream_processor"]:
                stream_stats = await app_state["stream_processor"].get_performance_stats()
                status["components"]["streaming"] = {
                    "status": "active",
                    "fps": stream_stats["target_fps"],
                    "frames_processed": stream_stats["frames_processed"]
                }

            if app_state["auto_config"]:
                auto_stats = await app_state["auto_config"].get_optimization_status()
                status["components"]["optimization"] = {
                    "status": "active",
                    "suggestions": auto_stats["suggestions_count"],
                    "monitoring": auto_stats["monitoring_enabled"]
                }

            return status

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/system/performance")
    async def get_performance_metrics():
        """Get system performance metrics."""
        try:
            metrics = {}

            if app_state["camera_manager"]:
                metrics["cameras"] = await app_state["camera_manager"].get_performance_stats()

            if app_state["stream_processor"]:
                metrics["streaming"] = await app_state["stream_processor"].get_performance_stats()

            if app_state["auto_config"]:
                metrics["optimization"] = await app_state["auto_config"].get_optimization_status()

            if app_state["git_logger"]:
                metrics["logging"] = await app_state["git_logger"].get_statistics()

            return metrics

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # ============ TIMELINE API ROUTES ============

    @app.get("/api/timeline/events")
    async def get_timeline_events(limit: int = 50, filter_type: Optional[str] = None):
        """Get timeline events."""
        try:
            if app_state["git_logger"]:
                events = await app_state["git_logger"].get_timeline_events(limit, filter_type)
                return {"events": events, "total": len(events)}
            return {"events": [], "total": 0}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/git/commit/{commit_hash}")
    async def get_commit_diff(commit_hash: str):
        """Get Git commit diff."""
        try:
            if app_state["git_logger"]:
                diff = await app_state["git_logger"].get_commit_diff(commit_hash)
                return {"commit_hash": commit_hash, "diff": diff}
            return {"error": "Git logger not available"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # ============ SERVER-SENT EVENTS ============

    @app.get("/api/events")
    async def event_stream(request: Request):
        """Server-sent events for real-time updates."""

        async def generate():
            try:
                while True:
                    if await request.is_disconnected():
                        break

                    # Get current status
                    data = {
                        "timestamp": datetime.now().isoformat(),
                        "type": "status_update"
                    }

                    # Add camera status
                    if app_state["camera_manager"]:
                        data["cameras"] = await app_state["camera_manager"].get_camera_status()

                    # Add layout status
                    if app_state["layout_engine"]:
                        layout_stats = await app_state["layout_engine"].get_layout_stats()
                        data["layout"] = layout_stats

                    # Add performance data
                    if app_state["auto_config"]:
                        perf_data = await app_state["auto_config"].get_optimization_status()
                        data["performance"] = perf_data

                    yield f"data: {json.dumps(data)}\n\n"

                    await asyncio.sleep(2)  # Update every 2 seconds

            except Exception as e:
                logging.error(f"Event stream error: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream"
            }
        )

    # ============ UTILITY FUNCTIONS ============

    async def test_rtsp_camera(url: str, timeout: int = 5) -> RTSPTestResult:
        """Test RTSP camera connection."""
        try:
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not cap.isOpened():
                return RTSPTestResult(
                    url=url,
                    success=False,
                    error_message="Failed to open RTSP stream"
                )

            # Try to read a frame with timeout
            start_time = datetime.now()
            ret, frame = cap.read()

            if not ret or frame is None:
                cap.release()
                return RTSPTestResult(
                    url=url,
                    success=False,
                    error_message="Failed to read frame from RTSP stream"
                )

            # Get stream properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            cap.release()

            return RTSPTestResult(
                url=url,
                success=True,
                resolution=f"{width}x{height}",
                fps=fps if fps > 0 else None
            )

        except Exception as e:
            return RTSPTestResult(
                url=url,
                success=False,
                error_message=str(e)
            )

    def test_usb_camera(source: str) -> tuple[bool, str]:
        """Test USB camera connection."""
        try:
            # Extract device number
            if "/dev/video" in source:
                device_num = int(source.split("video")[1])
            else:
                device_num = int(source)

            cap = cv2.VideoCapture(device_num)

            if not cap.isOpened():
                return False, f"Failed to open USB camera {source}"

            ret, frame = cap.read()
            if not ret or frame is None:
                cap.release()
                return False, f"Failed to read from USB camera {source}"

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            return True, f"USB camera working: {width}x{height}"

        except Exception as e:
            return False, f"USB camera test failed: {e}"

    def _set_nested_value(config: Dict, key: str, value: Any) -> None:
        """Set nested configuration value using dot notation."""
        keys = key.split('.')
        current = config

        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        current[keys[-1]] = value


# Helper function to create templates directory if it doesn't exist
def ensure_templates_directory():
    """Ensure templates directory exists."""
    templates_dir = Path("src/zoomcam/web/templates")
    templates_dir.mkdir(parents=True, exist_ok=True)

    # Create basic template if it doesn't exist
    setup_template = templates_dir / "setup_wizard.html"
    if not setup_template.exists():
        create_basic_setup_template(setup_template)


def create_basic_setup_template(template_path: Path):
    """Create basic setup template."""
    template_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ZoomCam Setup Wizard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .step { display: none; }
        .step.active { display: block; }
        .btn { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; margin: 10px 5px; }
        .btn:hover { background: #0056b3; }
        .btn-secondary { background: #6c757d; }
        .camera-list { margin: 20px 0; }
        .camera-item { padding: 15px; border: 1px solid #ddd; margin: 10px 0; border-radius: 5px; }
        .form-group { margin: 15px 0; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        input, select { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; box-sizing: border-box; }
        .progress { background: #e9ecef; height: 20px; border-radius: 10px; margin: 20px 0; }
        .progress-bar { background: #007bff; height: 100%; border-radius: 10px; transition: width 0.3s; }
        .alert { padding: 15px; margin: 15px 0; border-radius: 5px; }
        .alert-success { background: #d4edda; border: 1px solid #c3e6cb; color: #155724; }
        .alert-error { background: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; }
        .loading { text-align: center; padding: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎥 ZoomCam Setup Wizard</h1>
        <div class="progress">
            <div class="progress-bar" id="progressBar" style="width: 25%"></div>
        </div>

        <!-- Step 1: Welcome -->
        <div class="step active" id="step-welcome">
            <h2>Welcome to ZoomCam</h2>
            <p>This wizard will help you set up your intelligent camera monitoring system.</p>
            <p><strong>What we'll do:</strong></p>
            <ul>
                <li>Detect your cameras</li>
                <li>Configure camera settings</li>
                <li>Set up your display preferences</li>
                <li>Test your configuration</li>
            </ul>
            <button class="btn" onclick="nextStep()">Get Started</button>
        </div>

        <!-- Step 2: Camera Detection -->
        <div class="step" id="step-cameras">
            <h2>Camera Detection</h2>
            <p>Let's find your cameras. This may take a moment...</p>

            <div class="form-group">
                <label>
                    <input type="checkbox" id="scanUsb" checked> Scan USB cameras
                </label>
            </div>

            <div class="form-group">
                <label>
                    <input type="checkbox" id="scanRtsp"> Scan RTSP cameras
                </label>
            </div>

            <div id="rtspUrls" style="display: none;">
                <div class="form-group">
                    <label>RTSP URLs (one per line):</label>
                    <textarea id="rtspUrlList" rows="4" placeholder="rtsp://192.168.1.100:554/stream1&#10;rtsp://192.168.1.101:554/stream1"></textarea>
                </div>
            </div>

            <button class="btn" onclick="detectCameras()">Detect Cameras</button>
            <button class="btn btn-secondary" onclick="previousStep()">Back</button>

            <div id="cameraResults" class="camera-list"></div>
        </div>

        <!-- Step 3: Camera Configuration -->
        <div class="step" id="step-config">
            <h2>Camera Configuration</h2>
            <p>Configure your detected cameras:</p>
            <div id="cameraConfig"></div>
            <button class="btn" onclick="configureSystem()">Continue</button>
            <button class="btn btn-secondary" onclick="previousStep()">Back</button>
        </div>

        <!-- Step 4: System Settings -->
        <div class="step" id="step-system">
            <h2>System Settings</h2>
            <div class="form-group">
                <label>Display Resolution:</label>
                <select id="displayResolution">
                    <option value="1920x1080">1920x1080 (Full HD)</option>
                    <option value="1280x720">1280x720 (HD)</option>
                    <option value="3840x2160">3840x2160 (4K)</option>
                </select>
            </div>

            <div class="form-group">
                <label>Target Frame Rate:</label>
                <select id="targetFps">
                    <option value="30">30 FPS</option>
                    <option value="25">25 FPS</option>
                    <option value="15">15 FPS</option>
                </select>
            </div>

            <div class="form-group">
                <label>Quality Priority:</label>
                <select id="qualityPriority">
                    <option value="balanced">Balanced</option>
                    <option value="performance">Performance</option>
                    <option value="quality">Quality</option>
                </select>
            </div>

            <div class="form-group">
                <label>
                    <input type="checkbox" id="recordingEnabled" checked> Enable recording
                </label>
            </div>

            <div class="form-group">
                <label>
                    <input type="checkbox" id="gitLogging" checked> Enable Git logging
                </label>
            </div>

            <button class="btn" onclick="completeSetup()">Complete Setup</button>
            <button class="btn btn-secondary" onclick="previousStep()">Back</button>
        </div>

        <!-- Step 5: Complete -->
        <div class="step" id="step-complete">
            <h2>Setup Complete! 🎉</h2>
            <div class="alert alert-success">
                <p><strong>Congratulations!</strong> Your ZoomCam system is now configured and ready to use.</p>
            </div>
            <p>Your system is starting up. You'll be redirected to the main interface in a moment.</p>
            <button class="btn" onclick="goToMainInterface()">Go to ZoomCam</button>
        </div>
    </div>

    <script src="/static/js/setup-wizard.js"></script>
</body>
</html>'''

    with open(template_path, 'w') as f:
        f.write(template_content)


if __name__ == "__main__":
    # Standalone server for testing
    ensure_templates_directory()

    app = FastAPI(title="ZoomCam Setup")


    @app.get("/")
    async def index():
        return {"message": "ZoomCam API Server", "setup_mode": True}


    uvicorn.run(app, host="0.0.0.0", port=8000)