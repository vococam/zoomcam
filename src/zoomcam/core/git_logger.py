"""
Git Logger - Version Control Based Event Logging
===============================================

Logs system events, configuration changes, and screenshots to Git repository
for detailed history tracking and debugging capabilities.
"""

import asyncio
import logging
import json
import yaml
import cv2
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
import threading
import queue
import subprocess

try:
    import git
    GIT_AVAILABLE = True
except ImportError:
    GIT_AVAILABLE = False
    logging.warning("GitPython not available, using fallback Git commands")

from zoomcam.utils.exceptions import GitLoggerError


@dataclass
class LogEvent:
    """Event to be logged to Git."""
    event_type: str
    timestamp: datetime
    data: Dict[str, Any]
    config_snapshot: Optional[Dict[str, Any]] = None
    screenshot: Optional[np.ndarray] = None
    summary: Optional[str] = None
    camera_id: Optional[str] = None


@dataclass
class CommitInfo:
    """Git commit information."""
    sha: str
    timestamp: datetime
    message: str
    files: List[str]
    event_type: str


class GitLogger:
    """
    Git-based logging system for ZoomCam events.

    Features:
    - Automatic Git repository management
    - Event-based commits with metadata
    - Screenshot capture and storage
    - Configuration change tracking
    - History browsing and analysis
    - Automatic cleanup and compression
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get("enabled", True)
        self.repo_path = Path(config.get("repository_path", "/var/lib/zoomcam/history"))
        self.auto_commit = config.get("auto_commit", True)
        self.commit_interval = config.get("commit_interval", 30)
        self.screenshot_interval = config.get("screenshot_interval", 5)
        self.max_history_days = config.get("max_history_days", 30)

        # Git repository
        self.repo: Optional[Union[git.Repo, object]] = None
        self.last_commit_time: Optional[datetime] = None
        self.last_screenshot_time: Optional[datetime] = None

        # Event queue for async processing
        self.event_queue = asyncio.Queue()
        self.processing_task: Optional[asyncio.Task] = None
        self.running = False

        # Screenshot configuration
        self.screenshot_config = config.get("screenshots", {})
        self.screenshot_enabled = self.screenshot_config.get("enabled", True)
        self.screenshot_quality = self.screenshot_config.get("quality", 80)
        self.screenshot_format = self.screenshot_config.get("format", "jpg")

        # Performance tracking
        self.commits_count = 0
        self.events_logged = 0

        if self.enabled:
            self._initialize_repository()

        logging.info(f"Git Logger initialized: {'enabled' if self.enabled else 'disabled'}")

    def _initialize_repository(self) -> None:
        """Initialize Git repository."""
        try:
            self.repo_path.mkdir(parents=True, exist_ok=True)

            if GIT_AVAILABLE:
                # Use GitPython
                if (self.repo_path / ".git").exists():
                    self.repo = git.Repo(self.repo_path)
                    logging.info("Opened existing Git repository")
                else:
                    self.repo = git.Repo.init(self.repo_path)
                    self._create_initial_commit()
                    logging.info("Created new Git repository")
            else:
                # Use subprocess fallback
                if not (self.repo_path / ".git").exists():
                    subprocess.run(
                        ["git", "init"],
                        cwd=self.repo_path,
                        check=True,
                        capture_output=True
                    )
                    self._create_initial_commit_subprocess()
                    logging.info("Created new Git repository (subprocess)")
                else:
                    logging.info("Using existing Git repository (subprocess)")

        except Exception as e:
            logging.error(f"Failed to initialize Git repository: {e}")
            self.enabled = False
            raise GitLoggerError(f"Git initialization failed: {e}")

    def _create_initial_commit(self) -> None:
        """Create initial commit with README."""
        readme_content = f"""# ZoomCam Event History

This repository contains the complete event history for ZoomCam system.

- **Created**: {datetime.now().isoformat()}
- **System**: ZoomCam Adaptive Camera Monitoring
- **Purpose**: Event logging, configuration tracking, and debugging

## Structure

- `events/` - Event metadata files
- `configs/` - Configuration snapshots
- `screenshots/` - System screenshots
- `performance/` - Performance metrics

## Event Types

- `motion_detected` - Motion detection events
- `layout_change` - Layout adaptation events
- `config_change` - Configuration modifications
- `camera_connected` - Camera connection events
- `camera_disconnected` - Camera disconnection events
- `performance_alert` - Performance threshold alerts
- `system_startup` - System startup events
- `system_shutdown` - System shutdown events

Generated by ZoomCam Git Logger
"""

        readme_path = self.repo_path / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)

        # Create directory structure
        for dir_name in ["events", "configs", "screenshots", "performance"]:
            (self.repo_path / dir_name).mkdir(exist_ok=True)
            gitkeep_path = self.repo_path / dir_name / ".gitkeep"
            gitkeep_path.touch()

        if GIT_AVAILABLE:
            self.repo.index.add([str(readme_path)] + [str(self.repo_path / d / ".gitkeep") for d in ["events", "configs", "screenshots", "performance"]])
            self.repo.index.commit("Initial commit: ZoomCam Git Logger setup")
        else:
            self._create_initial_commit_subprocess()

    def _create_initial_commit_subprocess(self) -> None:
        """Create initial commit using subprocess."""
        try:
            subprocess.run(["git", "add", "."], cwd=self.repo_path, check=True)
            subprocess.run(
                ["git", "commit", "-m", "Initial commit: ZoomCam Git Logger setup"],
                cwd=self.repo_path,
                check=True
            )
        except subprocess.CalledProcessError as e:
            logging.error(f"Git subprocess commit failed: {e}")

    async def start_processing(self) -> None:
        """Start async event processing."""
        if not self.enabled:
            return

        self.running = True
        self.processing_task = asyncio.create_task(self._process_events())
        logging.info("Git Logger processing started")

    async def _process_events(self) -> None:
        """Main event processing loop."""
        while self.running:
            try:
                # Get event from queue (with timeout)
                try:
                    event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                # Process the event
                await self._process_single_event(event)
                self.event_queue.task_done()

            except Exception as e:
                logging.error(f"Event processing error: {e}")
                await asyncio.sleep(1)

    async def _process_single_event(self, event: LogEvent) -> None:
        """Process a single log event."""
        try:
            timestamp_str = event.timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]

            # Create event directory
            event_dir = self.repo_path / "events" / f"{timestamp_str}_{event.event_type}"
            event_dir.mkdir(parents=True, exist_ok=True)

            files_to_add = []

            # Save event metadata
            event_data = {
                "event_type": event.event_type,
                "timestamp": event.timestamp.isoformat(),
                "data": event.data,
                "summary": event.summary,
                "camera_id": event.camera_id
            }

            event_file = event_dir / "event.json"
            with open(event_file, 'w') as f:
                json.dump(event_data, f, indent=2, default=str)
            files_to_add.append(str(event_file.relative_to(self.repo_path)))

            # Save configuration snapshot
            if event.config_snapshot:
                config_file = event_dir / "config_snapshot.yaml"
                with open(config_file, 'w') as f:
                    yaml.dump(event.config_snapshot, f, default_flow_style=False, indent=2)
                files_to_add.append(str(config_file.relative_to(self.repo_path)))

            # Save screenshot
            if event.screenshot is not None and self.screenshot_enabled:
                screenshot_file = event_dir / f"screenshot.{self.screenshot_format}"
                self._save_screenshot(event.screenshot, screenshot_file)
                files_to_add.append(str(screenshot_file.relative_to(self.repo_path)))

            # Commit to Git
            if self.auto_commit:
                await self._commit_files(files_to_add, event)

            self.events_logged += 1

        except Exception as e:
            logging.error(f"Failed to process event {event.event_type}: {e}")

    def _save_screenshot(self, screenshot: np.ndarray, file_path: Path) -> None:
        """Save screenshot to file."""
        try:
            if self.screenshot_format.lower() == 'jpg':
                cv2.imwrite(
                    str(file_path),
                    screenshot,
                    [cv2.IMWRITE_JPEG_QUALITY, self.screenshot_quality]
                )
            elif self.screenshot_format.lower() == 'png':
                cv2.imwrite(str(file_path), screenshot)
            else:
                # Default to JPEG
                cv2.imwrite(
                    str(file_path),
                    screenshot,
                    [cv2.IMWRITE_JPEG_QUALITY, self.screenshot_quality]
                )
        except Exception as e:
            logging.error(f"Failed to save screenshot: {e}")

    async def _commit_files(self, files: List[str], event: LogEvent) -> None:
        """Commit files to Git repository."""
        try:
            # Check commit interval
            current_time = datetime.now()
            if (self.last_commit_time and
                (current_time - self.last_commit_time).total_seconds() < self.commit_interval):
                return

            commit_message = self._generate_commit_message(event)

            if GIT_AVAILABLE and self.repo:
                # Use GitPython
                self.repo.index.add(files)
                self.repo.index.commit(commit_message)
            else:
                # Use subprocess
                subprocess.run(["git", "add"] + files, cwd=self.repo_path, check=True)
                subprocess.run(
                    ["git", "commit", "-m", commit_message],
                    cwd=self.repo_path,
                    check=True
                )

            self.last_commit_time = current_time
            self.commits_count += 1

            logging.debug(f"Committed event: {event.event_type}")

            # Cleanup old commits if needed
            await self._cleanup_old_history()

        except Exception as e:
            logging.error(f"Git commit failed: {e}")

    def _generate_commit_message(self, event: LogEvent) -> str:
        """Generate Git commit message for event."""
        timestamp = event.timestamp.strftime("%Y-%m-%d %H:%M:%S")

        if event.summary:
            message = f"{event.event_type}: {event.summary}"
        else:
            message = f"{event.event_type}: {timestamp}"

        # Add details
        details = []
        if event.camera_id:
            details.append(f"Camera: {event.camera_id}")

        if event.data:
            # Add key data points
            if "activity_level" in event.data:
                details.append(f"Activity: {event.data['activity_level']:.1%}")
            if "layout_efficiency" in event.data:
                details.append(f"Layout: {event.data['layout_efficiency']:.1%}")
            if "cpu_usage" in event.data:
                details.append(f"CPU: {event.data['cpu_usage']:.1f}%")

        if details:
            message += "\n\n" + "\n".join(f"- {detail}" for detail in details)

        return message

    async def _cleanup_old_history(self) -> None:
        """Cleanup old history based on retention policy."""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.max_history_days)

            # Find old event directories
            events_dir = self.repo_path / "events"
            if not events_dir.exists():
                return

            old_dirs = []
            for event_dir in events_dir.iterdir():
                if event_dir.is_dir():
                    # Parse timestamp from directory name
                    try:
                        timestamp_part = event_dir.name.split("_")[0]  # YYYYMMDD
                        if len(timestamp_part) >= 8:
                            event_date = datetime.strptime(timestamp_part[:8], "%Y%m%d")
                            if event_date < cutoff_date:
                                old_dirs.append(event_dir)
                    except (ValueError, IndexError):
                        continue

            # Remove old directories
            for old_dir in old_dirs:
                try:
                    shutil.rmtree(old_dir)
                    logging.debug(f"Cleaned up old event directory: {old_dir.name}")
                except Exception as e:
                    logging.error(f"Failed to cleanup {old_dir}: {e}")

            # Git add removals
            if old_dirs:
                if GIT_AVAILABLE and self.repo:
                    self.repo.index.add([str(events_dir)])
                    self.repo.index.commit(f"Cleanup: Removed {len(old_dirs)} old event directories")
                else:
                    subprocess.run(["git", "add", "events/"], cwd=self.repo_path, check=True)
                    subprocess.run(
                        ["git", "commit", "-m", f"Cleanup: Removed {len(old_dirs)} old event directories"],
                        cwd=self.repo_path,
                        check=True
                    )

        except Exception as e:
            logging.error(f"History cleanup failed: {e}")

    async def log_event(
        self,
        event_type: str,
        data: Dict[str, Any],
        config_snapshot: Optional[Dict[str, Any]] = None,
        screenshot: Optional[np.ndarray] = None,
        summary: Optional[str] = None,
        camera_id: Optional[str] = None
    ) -> None:
        """Log an event asynchronously."""
        if not self.enabled:
            return

        event = LogEvent(
            event_type=event_type,
            timestamp=datetime.now(),
            data=data,
            config_snapshot=config_snapshot,
            screenshot=screenshot,
            summary=summary,
            camera_id=camera_id
        )

        try:
            await self.event_queue.put(event)
        except Exception as e:
            logging.error(f"Failed to queue event {event_type}: {e}")

    async def log_motion_detected(self, camera_id: str, motion_data: Dict[str, Any], screenshot: Optional[np.ndarray] = None) -> None:
        """Log motion detection event."""
        summary = f"Motion detected in {len(motion_data.get('zones', []))} zones"
        await self.log_event(
            event_type="motion_detected",
            data=motion_data,
            screenshot=screenshot,
            summary=summary,
            camera_id=camera_id
        )

    async def log_layout_change(self, layout_data: Dict[str, Any], screenshot: Optional[np.ndarray] = None) -> None:
        """Log layout change event."""
        summary = f"Layout changed: {layout_data.get('total_fragments', 0)} fragments"
        await self.log_event(
            event_type="layout_change",
            data=layout_data,
            screenshot=screenshot,
            summary=summary
        )

    async def log_config_change(self, config_data: Dict[str, Any], config_snapshot: Dict[str, Any]) -> None:
        """Log configuration change event."""
        changes = config_data.get('changes', {})
        summary = f"Config changed: {len(changes)} parameters"
        await self.log_event(
            event_type="config_change",
            data=config_data,
            config_snapshot=config_snapshot,
            summary=summary
        )

    async def log_camera_event(self, camera_id: str, event_type: str, data: Dict[str, Any]) -> None:
        """Log camera-related event."""
        summary = f"{event_type.replace('_', ' ').title()}: {camera_id}"
        await self.log_event(
            event_type=event_type,
            data=data,
            summary=summary,
            camera_id=camera_id
        )

    async def log_performance_alert(self, performance_data: Dict[str, Any]) -> None:
        """Log performance alert event."""
        cpu_usage = performance_data.get('cpu_usage', 0)
        memory_usage = performance_data.get('memory_usage', 0)
        summary = f"Performance alert: CPU {cpu_usage:.1f}%, Memory {memory_usage:.1f}%"
        await self.log_event(
            event_type="performance_alert",
            data=performance_data,
            summary=summary
        )

    async def get_timeline_events(self, limit: int = 100, filter_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get timeline events for web interface."""
        try:
            events = []
            events_dir = self.repo_path / "events"

            if not events_dir.exists():
                return events

            # Get event directories sorted by timestamp (newest first)
            event_dirs = sorted(
                [d for d in events_dir.iterdir() if d.is_dir()],
                key=lambda x: x.name,
                reverse=True
            )

            count = 0
            for event_dir in event_dirs:
                if count >= limit:
                    break

                try:
                    # Load event metadata
                    event_file = event_dir / "event.json"
                    if not event_file.exists():
                        continue

                    with open(event_file, 'r') as f:
                        event_data = json.load(f)

                    # Apply filter
                    if filter_type and event_data.get('event_type') != filter_type:
                        continue

                    # Add file paths
                    event_data['screenshot'] = None
                    screenshot_file = event_dir / f"screenshot.{self.screenshot_format}"
                    if screenshot_file.exists():
                        event_data['screenshot'] = f"/screenshots/{screenshot_file.name}"

                    config_file = event_dir / "config_snapshot.yaml"
                    if config_file.exists():
                        event_data['config_snapshot_available'] = True

                    # Add Git commit info
                    commit_info = await self._get_commit_for_event(event_dir.name)
                    if commit_info:
                        event_data['commit_hash'] = commit_info['sha']
                        event_data['commit_message'] = commit_info['message']

                    events.append(event_data)
                    count += 1

                except Exception as e:
                    logging.error(f"Error loading event {event_dir.name}: {e}")
                    continue

            return events

        except Exception as e:
            logging.error(f"Failed to get timeline events: {e}")
            return []

    async def _get_commit_for_event(self, event_name: str) -> Optional[Dict[str, Any]]:
        """Get Git commit information for event."""
        try:
            if GIT_AVAILABLE and self.repo:
                # Search for commits containing the event
                commits = list(self.repo.iter_commits(max_count=1000))
                for commit in commits:
                    if event_name in commit.message or any(event_name in f for f in commit.stats.files):
                        return {
                            'sha': commit.hexsha[:8],
                            'message': commit.message.strip(),
                            'timestamp': commit.committed_datetime.isoformat()
                        }
            else:
                # Use subprocess to get commit info
                result = subprocess.run(
                    ["git", "log", "--oneline", "--grep", event_name, "-1"],
                    cwd=self.repo_path,
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0 and result.stdout.strip():
                    line = result.stdout.strip()
                    sha = line.split()[0]
                    message = " ".join(line.split()[1:])
                    return {
                        'sha': sha,
                        'message': message,
                        'timestamp': datetime.now().isoformat()  # Simplified
                    }
        except Exception as e:
            logging.error(f"Failed to get commit for event {event_name}: {e}")

        return None

    async def get_commit_diff(self, commit_hash: str) -> str:
        """Get diff for specific commit."""
        try:
            if GIT_AVAILABLE and self.repo:
                commit = self.repo.commit(commit_hash)
                return commit.diff(commit.parents[0] if commit.parents else None, create_patch=True)
            else:
                result = subprocess.run(
                    ["git", "show", commit_hash],
                    cwd=self.repo_path,
                    capture_output=True,
                    text=True
                )
                return result.stdout if result.returncode == 0 else "Diff not available"
        except Exception as e:
            logging.error(f"Failed to get diff for commit {commit_hash}: {e}")
            return f"Error getting diff: {e}"

    async def get_statistics(self) -> Dict[str, Any]:
        """Get Git logger statistics."""
        try:
            events_dir = self.repo_path / "events"
            total_events = len(list(events_dir.iterdir())) if events_dir.exists() else 0

            # Get repository size
            repo_size = sum(f.stat().st_size for f in self.repo_path.rglob('*') if f.is_file())

            return {
                "enabled": self.enabled,
                "total_events": total_events,
                "commits_count": self.commits_count,
                "events_logged": self.events_logged,
                "repository_size_mb": repo_size / (1024 * 1024),
                "repository_path": str(self.repo_path),
                "last_commit_time": self.last_commit_time.isoformat() if self.last_commit_time else None,
                "config": {
                    "auto_commit": self.auto_commit,
                    "commit_interval": self.commit_interval,
                    "screenshot_enabled": self.screenshot_enabled,
                    "max_history_days": self.max_history_days
                }
            }
        except Exception as e:
            logging.error(f"Failed to get Git logger statistics: {e}")
            return {"enabled": self.enabled, "error": str(e)}

    async def shutdown(self) -> None:
        """Shutdown Git logger gracefully."""
        if not self.enabled:
            return

        logging.info("Shutting down Git Logger...")

        self.running = False

        # Wait for processing task to complete
        if self.processing_task:
            try:
                await asyncio.wait_for(self.processing_task, timeout=10.0)
            except asyncio.TimeoutError:
                logging.warning("Git logger processing task timeout during shutdown")
                self.processing_task.cancel()

        # Process remaining events in queue
        remaining_events = 0
        while not self.event_queue.empty():
            try:
                event = self.event_queue.get_nowait()
                await self._process_single_event(event)
                remaining_events += 1
            except:
                break

        if remaining_events > 0:
            logging.info(f"Processed {remaining_events} remaining events during shutdown")

        # Final commit if needed
        try:
            if self.auto_commit:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                if GIT_AVAILABLE and self.repo:
                    # Check if there are uncommitted changes
                    if self.repo.is_dirty():
                        self.repo.index.add(["."])
                        self.repo.index.commit(f"System shutdown: {timestamp}")
                else:
                    subprocess.run(["git", "add", "."], cwd=self.repo_path, check=True)
                    subprocess.run(
                        ["git", "commit", "-m", f"System shutdown: {timestamp}"],
                        cwd=self.repo_path,
                        check=True
                    )
        except Exception as e:
            logging.error(f"Failed to create shutdown commit: {e}")

        logging.info("Git Logger shutdown complete")