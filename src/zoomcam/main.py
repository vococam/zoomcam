"""
ZoomCam - Main Application Entry Point
=====================================

Intelligent adaptive camera monitoring system for Raspberry Pi.
Automatically adjusts camera layout based on motion detection.
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI

from zoomcam.config.config_manager import ConfigManager
from zoomcam.core.camera_manager import CameraManager
from zoomcam.core.layout_engine import LayoutEngine
from zoomcam.core.stream_processor import StreamProcessor
from zoomcam.core.git_logger import GitLogger
from zoomcam.core.auto_config_manager import AutoConfigManager
from zoomcam.utils.logger import setup_logging
from zoomcam.utils.exceptions import ZoomCamError
from zoomcam.web.api import create_app


class ZoomCamApplication:
    """Main ZoomCam application class."""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config/user-config.yaml"
        self.app: Optional[FastAPI] = None
        self.config_manager: Optional[ConfigManager] = None
        self.camera_manager: Optional[CameraManager] = None
        self.layout_engine: Optional[LayoutEngine] = None
        self.stream_processor: Optional[StreamProcessor] = None
        self.git_logger: Optional[GitLogger] = None
        self.auto_config: Optional[AutoConfigManager] = None
        self.running = False

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logging.info(f"Received signal {signum}, shutting down...")
        self.running = False

    async def initialize(self) -> None:
        """Initialize all application components."""
        try:
            # Setup logging
            setup_logging()
            logging.info("Starting ZoomCam application...")

            # Load configuration
            self.config_manager = ConfigManager(self.config_path)
            await self.config_manager.load_config()
            config = self.config_manager.get_config()

            # Initialize Git logger
            if config.get("logging", {}).get("git", {}).get("enabled", True):
                self.git_logger = GitLogger(config["logging"]["git"])
                logging.info("Git logging initialized")

            # Initialize auto config manager
            self.auto_config = AutoConfigManager(
                config_manager=self.config_manager,
                git_logger=self.git_logger
            )

            # Initialize camera manager
            self.camera_manager = CameraManager(
                config=config["cameras"],
                auto_config=self.auto_config
            )
            await self.camera_manager.initialize()

            # Initialize layout engine
            self.layout_engine = LayoutEngine(
                config=config["layout"],
                screen_resolution=config["system"]["display"]["target_resolution"]
            )

            # Initialize stream processor
            self.stream_processor = StreamProcessor(
                config=config["streaming"],
                layout_engine=self.layout_engine,
                auto_config=self.auto_config
            )

            # Create FastAPI app
            self.app = create_app(
                camera_manager=self.camera_manager,
                layout_engine=self.layout_engine,
                stream_processor=self.stream_processor,
                config_manager=self.config_manager,
                git_logger=self.git_logger,
                auto_config=self.auto_config
            )

            logging.info("ZoomCam application initialized successfully")

        except Exception as e:
            logging.error(f"Failed to initialize ZoomCam: {e}")
            raise ZoomCamError(f"Initialization failed: {e}")

    async def start_processing(self) -> None:
        """Start the main processing loop."""
        self.running = True

        try:
            # Start camera manager
            camera_task = asyncio.create_task(
                self.camera_manager.start_processing()
            )

            # Start stream processor
            stream_task = asyncio.create_task(
                self.stream_processor.start_processing()
            )

            # Start auto config manager
            auto_config_task = asyncio.create_task(
                self.auto_config.start_monitoring()
            )

            # Wait for shutdown signal
            while self.running:
                await asyncio.sleep(1)

            # Cancel all tasks
            camera_task.cancel()
            stream_task.cancel()
            auto_config_task.cancel()

            # Wait for tasks to complete
            await asyncio.gather(
                camera_task, stream_task, auto_config_task,
                return_exceptions=True
            )

        except Exception as e:
            logging.error(f"Error in processing loop: {e}")
            raise

    async def shutdown(self) -> None:
        """Shutdown application gracefully."""
        logging.info("Shutting down ZoomCam application...")

        self.running = False

        if self.stream_processor:
            await self.stream_processor.shutdown()

        if self.camera_manager:
            await self.camera_manager.shutdown()

        if self.git_logger:
            await self.git_logger.shutdown()

        logging.info("ZoomCam application shutdown complete")


async def run_server(
        app: ZoomCamApplication,
        host: str = "0.0.0.0",
        port: int = 8000,
        debug: bool = False
) -> None:
    """Run the web server."""
    config = uvicorn.Config(
        app.app,
        host=host,
        port=port,
        log_level="debug" if debug else "info",
        access_log=True,
        reload=debug
    )

    server = uvicorn.Server(config)

    # Start processing in background
    processing_task = asyncio.create_task(app.start_processing())

    try:
        # Start web server
        await server.serve()
    finally:
        # Ensure processing task is cancelled
        processing_task.cancel()
        try:
            await processing_task
        except asyncio.CancelledError:
            pass


async def async_main(
        config_path: Optional[str] = None,
        host: str = "0.0.0.0",
        port: int = 8000,
        debug: bool = False
) -> None:
    """Main async entry point."""
    app = ZoomCamApplication(config_path)

    try:
        await app.initialize()
        await run_server(app, host, port, debug)
    except KeyboardInterrupt:
        logging.info("Received keyboard interrupt")
    except Exception as e:
        logging.error(f"Application error: {e}")
        sys.exit(1)
    finally:
        await app.shutdown()


def main() -> None:
    """Main entry point for the application."""
    import argparse

    parser = argparse.ArgumentParser(
        description="ZoomCam - Intelligent Adaptive Camera Monitoring"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config/user-config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8000,
        help="Port to bind to"
    )
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Enable debug mode"
    )
    parser.add_argument(
        "--test-config",
        action="store_true",
        help="Test configuration and exit"
    )

    args = parser.parse_args()

    if args.test_config:
        # Test configuration only
        try:
            import yaml
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
            print("✅ Configuration file is valid")
            sys.exit(0)
        except Exception as e:
            print(f"❌ Configuration error: {e}")
            sys.exit(1)

    # Ensure config directory exists
    config_path = Path(args.config)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Create default config if it doesn't exist
    if not config_path.exists():
        from zoomcam.config.defaults import create_default_config
        create_default_config(config_path)
        print(f"Created default configuration at {config_path}")

    # Run the application
    try:
        asyncio.run(async_main(
            config_path=args.config,
            host=args.host,
            port=args.port,
            debug=args.debug
        ))
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()