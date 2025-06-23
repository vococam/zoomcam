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
import traceback
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI

from zoomcam.config.config_manager import ConfigManager
from zoomcam.core.auto_config_manager import AutoConfigManager
from zoomcam.core.camera_manager import CameraManager
from zoomcam.core.git_logger import GitLogger
from zoomcam.core.layout_engine import LayoutEngine
from zoomcam.core.stream_processor import StreamProcessor
from zoomcam.utils.exceptions import ErrorCategory, ErrorSeverity, ZoomCamError
from zoomcam.utils.logger import setup_logging
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
            # Setup logging with basic error handling in case logging setup fails
            print(
                "Setting up logging..."
            )  # Use print as logging may not be initialized yet
            try:
                setup_logging()
                logging.info("Logging system initialized")
            except Exception as e:
                # If logging setup fails, print error and wrap in a proper ZoomCamError
                error_msg = f"Failed to initialize logging: {str(e)}"
                print(f"ERROR: {error_msg}")
                print(f"Traceback: {traceback.format_exc()}")
                raise ZoomCamError(
                    error_msg,
                    category=ErrorCategory.SYSTEM,
                    severity=ErrorSeverity.CRITICAL,
                    original_exception=e,
                ) from e

            logging.info("Starting ZoomCam application...")

            try:
                # Initialize configuration manager
                logging.info("Initializing configuration manager...")
                self.config_manager = ConfigManager(self.config_path)
                logging.info(f"Loading configuration from {self.config_path}...")
                await self.config_manager.load_config()
                config = self.config_manager.get_config()
                logging.info("Configuration loaded successfully")
            except Exception as e:
                logging.error(f"Failed to load configuration: {str(e)}", exc_info=True)
                raise ZoomCamError(
                    f"Configuration loading failed: {str(e)}",
                    category=ErrorCategory.CONFIGURATION,
                    severity=ErrorSeverity.CRITICAL,
                ) from e

            try:
                # Initialize Git logger if enabled
                if config.get("logging", {}).get("git", {}).get("enabled", True):
                    logging.info("Initializing Git logger...")
                    self.git_logger = GitLogger(config["logging"]["git"])
                    logging.info("Git logging initialized")
                else:
                    logging.info("Git logging is disabled in configuration")
            except Exception as e:
                logging.warning(f"Failed to initialize Git logger: {str(e)}")
                self.git_logger = None

            try:
                # Initialize auto config manager
                logging.info("Initializing auto config manager...")
                self.auto_config = AutoConfigManager(
                    config_manager=self.config_manager, git_logger=self.git_logger
                )
                logging.info("Auto config manager initialized")
            except Exception as e:
                logging.error(
                    f"Failed to initialize auto config manager: {str(e)}", exc_info=True
                )
                raise ZoomCamError(
                    f"Auto config manager initialization failed: {str(e)}",
                    category=ErrorCategory.CONFIGURATION,
                    severity=ErrorSeverity.HIGH,
                ) from e

            try:
                # Initialize camera manager
                logging.info("Initializing camera manager...")
                self.camera_manager = CameraManager(
                    config=config["cameras"], auto_config=self.auto_config
                )
                await self.camera_manager.initialize()
                logging.info("Camera manager initialized successfully")
            except Exception as e:
                logging.error(
                    f"Failed to initialize camera manager: {str(e)}", exc_info=True
                )
                raise ZoomCamError(
                    f"Camera manager initialization failed: {str(e)}",
                    category=ErrorCategory.CAMERA,
                    severity=ErrorSeverity.CRITICAL,
                ) from e

            try:
                # Initialize layout engine
                logging.info("Initializing layout engine...")
                self.layout_engine = LayoutEngine(
                    config=config["layout"],
                    screen_resolution=config["system"]["display"]["target_resolution"],
                )
                logging.info("Layout engine initialized successfully")
            except Exception as e:
                logging.error(
                    f"Failed to initialize layout engine: {str(e)}", exc_info=True
                )
                raise ZoomCamError(
                    f"Layout engine initialization failed: {str(e)}",
                    category=ErrorCategory.LAYOUT,
                    severity=ErrorSeverity.HIGH,
                ) from e

            try:
                # Initialize stream processor
                logging.info("Initializing stream processor...")
                self.stream_processor = StreamProcessor(
                    config=config["streaming"],
                    layout_engine=self.layout_engine,
                    auto_config=self.auto_config,
                )
                logging.info("Stream processor initialized successfully")
            except Exception as e:
                logging.error(
                    f"Failed to initialize stream processor: {str(e)}", exc_info=True
                )
                raise ZoomCamError(
                    f"Stream processor initialization failed: {str(e)}",
                    category=ErrorCategory.STREAMING,
                    severity=ErrorSeverity.HIGH,
                ) from e

            try:
                # Create FastAPI app
                logging.info("Creating FastAPI application...")
                self.app = create_app(
                    camera_manager=self.camera_manager,
                    layout_engine=self.layout_engine,
                    stream_processor=self.stream_processor,
                    config_manager=self.config_manager,
                    git_logger=self.git_logger,
                    auto_config=self.auto_config,
                )
                logging.info("FastAPI application created successfully")
                logging.info("ZoomCam application initialized successfully")
            except Exception as e:
                logging.error(
                    f"Failed to create FastAPI application: {str(e)}", exc_info=True
                )
                raise ZoomCamError(
                    f"Failed to create FastAPI application: {str(e)}",
                    category=ErrorCategory.SOFTWARE,
                    severity=ErrorSeverity.CRITICAL,
                ) from e

        except ZoomCamError as ze:
            # Re-raise ZoomCamError with additional context if needed
            logging.error(f"ZoomCam initialization failed: {str(ze)}", exc_info=True)
            raise
        except Exception as e:
            # Catch any other exceptions and wrap them in a ZoomCamError
            logging.error(
                f"Unexpected error during initialization: {str(e)}", exc_info=True
            )
            raise ZoomCamError(
                f"Initialization failed: {str(e)}",
                category=ErrorCategory.SYSTEM,
                severity=ErrorSeverity.CRITICAL,
                original_exception=e,
            ) from e

    async def start_processing(self) -> None:
        """Start the main processing loop."""
        self.running = True

        try:
            # Start camera manager
            camera_task = asyncio.create_task(self.camera_manager.start_processing())

            # Start stream processor
            stream_task = asyncio.create_task(self.stream_processor.start_processing())

            # Start auto config manager
            auto_config_task = asyncio.create_task(self.auto_config.start_monitoring())

            # Wait for shutdown signal
            while self.running:
                await asyncio.sleep(1)

            # Cancel all tasks
            camera_task.cancel()
            stream_task.cancel()
            auto_config_task.cancel()

            # Wait for tasks to complete
            await asyncio.gather(
                camera_task, stream_task, auto_config_task, return_exceptions=True
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
    port: int = 5000,  # Changed from 8080 to 5000 to avoid conflicts
    debug: bool = False,
) -> None:
    """Run the web server."""
    config = uvicorn.Config(
        app.app,
        host=host,
        port=port,
        log_level="debug" if debug else "info",
        access_log=True,
        reload=debug,
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
    port: int = 5000,  # Changed from 8080 to 5000 to avoid conflicts
    debug: bool = False,
) -> None:
    """Main async entry point."""
    logger = logging.getLogger(__name__)
    logger.info(f"Starting async_main with config: {config_path}")

    try:
        # Create application instance
        app = ZoomCamApplication(config_path=config_path)

        try:
            logger.info("Initializing application...")
            try:
                await app.initialize()
                logger.info("Application initialized, starting server...")
                await run_server(app, host=host, port=port, debug=debug)
            except ZoomCamError as ze:
                # Log the ZoomCamError with all its attributes
                logger.error(
                    f"ZoomCamError during initialization: {str(ze)}\n"
                    f"Category: {getattr(ze, 'category', 'NOT SET')}\n"
                    f"Severity: {getattr(ze, 'severity', 'NOT SET')}\n"
                    f"Error Code: {getattr(ze, 'error_code', 'NOT SET')}",
                    exc_info=True,
                )
                # Re-raise the error
                raise
            except Exception as e:
                # Log any other exceptions with full traceback
                logger.error(
                    f"Unexpected error during initialization: {str(e)}", exc_info=True
                )
                # Wrap in a proper ZoomCamError with category
                raise ZoomCamError(
                    f"Unexpected error during initialization: {str(e)}",
                    category=ErrorCategory.SYSTEM,
                    severity=ErrorSeverity.CRITICAL,
                    original_exception=e,
                ) from e
        except Exception as e:
            logger.error(f"Error in async_main: {e}", exc_info=True)
            raise
        finally:
            logger.info("Shutting down application...")
            try:
                await app.shutdown()
            except Exception as e:
                logger.error(f"Error during shutdown: {e}", exc_info=True)
    except Exception as e:
        # Final catch-all to ensure we log any unhandled exceptions
        logger.critical(
            f"Unhandled exception in async_main: {str(e)}\n"
            f"Type: {type(e).__name__}\n"
            f"Attributes: {', '.join(f'{k}={v}' for k, v in vars(e).items())}",
            exc_info=True,
        )
        # Re-raise to ensure the application exits with an error code
        raise


def main() -> None:
    """Main entry point for the application."""
    import argparse
    import logging

    # Set up basic logging first
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting ZoomCam application")

    parser = argparse.ArgumentParser(
        description="ZoomCam - Intelligent Adaptive Camera Monitoring"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="config/user-config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument(
        "--port", "-p", type=int, default=5000, help="Port to bind to (default: 5000)"
    )
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--test-config", action="store_true", help="Test configuration and exit"
    )

    args = parser.parse_args()

    if args.test_config:
        # Test configuration only
        try:
            import yaml

            with open(args.config, "r") as f:
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
        asyncio.run(
            async_main(
                config_path=args.config,
                host=args.host,
                port=args.port,
                debug=args.debug,
            )
        )
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
