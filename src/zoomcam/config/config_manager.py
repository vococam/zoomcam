"""
Configuration Manager - Core Configuration Handling
==================================================

Manages configuration loading, validation, saving, and hot-reloading
with support for multiple configuration sources and formats.
"""

import asyncio
import logging
import yaml
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
import threading
import hashlib
import copy

from zoomcam.config.validator import ConfigValidator
from zoomcam.config.defaults import get_default_user_config, merge_configs
from zoomcam.utils.exceptions import ConfigurationError, ConfigFileError, ConfigValidationError
from zoomcam.utils.helpers import backup_file, get_nested_value, set_nested_value
from zoomcam.utils.logger import get_logger


@dataclass
class ConfigChange:
    """Configuration change record."""
    timestamp: datetime
    key: str
    old_value: Any
    new_value: Any
    source: str = "manual"
    user: Optional[str] = None


@dataclass
class ConfigSource:
    """Configuration source definition."""
    name: str
    path: Path
    format: str = "yaml"  # yaml, json, env
    priority: int = 0  # Higher priority overrides lower
    watch: bool = True
    required: bool = True


class ConfigManager:
    """
    Comprehensive configuration manager for ZoomCam.

    Features:
    - Multiple configuration sources (YAML, JSON, environment)
    - Hot-reloading with file watching
    - Configuration validation
    - Change tracking and history
    - Backup and restore
    - Environment variable substitution
    - Configuration merging with priorities
    """

    def __init__(self, primary_config_path: Union[str, Path]):
        self.primary_config_path = Path(primary_config_path)
        self.logger = get_logger("config_manager")

        # Configuration state
        self.config: Dict[str, Any] = {}
        self.config_sources: List[ConfigSource] = []
        self.file_hashes: Dict[str, str] = {}
        self.change_history: List[ConfigChange] = []

        # Validation
        self.validator = ConfigValidator()
        self.validation_enabled = True

        # Hot reloading
        self.watch_enabled = False
        self.watch_thread: Optional[threading.Thread] = None
        self.stop_watching = threading.Event()

        # Change callbacks
        self.change_callbacks: List[Callable[[str, Any, Any], None]] = []

        # Threading
        self.lock = threading.RLock()

        # Setup configuration sources
        self._setup_config_sources()

        self.logger.info(f"Configuration manager initialized with primary config: {self.primary_config_path}")

    def _setup_config_sources(self):
        """Setup configuration sources in priority order."""
        config_dir = self.primary_config_path.parent

        # 1. Default configuration (lowest priority)
        default_source = ConfigSource(
            name="defaults",
            path=config_dir / "defaults.yaml",
            priority=0,
            watch=False,
            required=False
        )
        self.config_sources.append(default_source)

        # 2. Main user configuration
        user_source = ConfigSource(
            name="user",
            path=self.primary_config_path,
            priority=10,
            watch=True,
            required=True
        )
        self.config_sources.append(user_source)

        # 3. Auto-generated configuration
        auto_source = ConfigSource(
            name="auto",
            path=config_dir / "auto-config.yaml",
            priority=5,
            watch=True,
            required=False
        )
        self.config_sources.append(auto_source)

        # 4. Environment overrides (highest priority)
        env_source = ConfigSource(
            name="environment",
            path=config_dir / "env-config.yaml",
            priority=20,
            watch=False,
            required=False
        )
        self.config_sources.append(env_source)

        # Sort by priority
        self.config_sources.sort(key=lambda x: x.priority)

    async def load_config(self) -> Dict[str, Any]:
        """Load configuration from all sources."""
        with self.lock:
            try:
                self.logger.info("Loading configuration from all sources")

                # Start with default configuration
                merged_config = get_default_user_config()

                # Load and merge each source
                for source in self.config_sources:
                    try:
                        source_config = await self._load_config_source(source)
                        if source_config:
                            merged_config = merge_configs(merged_config, source_config)
                            self.logger.debug(f"Merged configuration from {source.name}")
                    except Exception as e:
                        if source.required:
                            raise ConfigFileError(
                                config_file=str(source.path),
                                reason=f"Failed to load required config source: {e}"
                            )
                        else:
                            self.logger.warning(f"Failed to load optional config source {source.name}: {e}")

                # Apply environment variable substitution
                merged_config = self._substitute_environment_variables(merged_config)

                # Validate configuration
                if self.validation_enabled:
                    validation_errors = self.validator.validate(merged_config)
                    if validation_errors:
                        error_messages = [f"{error.field}: {error.message}" for error in validation_errors]
                        raise ConfigValidationError(
                            field="multiple",
                            value="configuration",
                            expected="valid configuration according to schema"
                        )

                # Store the merged configuration
                old_config = self.config.copy()
                self.config = merged_config

                # Track changes
                if old_config:
                    self._track_config_changes(old_config, merged_config, "reload")

                self.logger.info("Configuration loaded successfully")
                return self.config

            except Exception as e:
                self.logger.error(f"Failed to load configuration: {e}")
                raise

    async def _load_config_source(self, source: ConfigSource) -> Optional[Dict[str, Any]]:
        """Load configuration from a single source."""
        if not source.path.exists():
            if source.required:
                # Create default config file
                if source.name == "user":
                    self.logger.info(f"Creating default configuration file: {source.path}")
                    await self._create_default_config_file(source.path)
                else:
                    raise ConfigFileError(
                        config_file=str(source.path),
                        reason="Required configuration file not found"
                    )
            else:
                return None

        try:
            # Check if file has changed
            current_hash = self._calculate_file_hash(source.path)
            if source.path.name in self.file_hashes and self.file_hashes[source.path.name] == current_hash:
                return None  # File unchanged, skip loading

            # Load the configuration file
            with open(source.path, 'r', encoding='utf-8') as f:
                if source.format == "yaml":
                    config_data = yaml.safe_load(f)
                elif source.format == "json":
                    config_data = json.load(f)
                else:
                    raise ConfigFileError(
                        config_file=str(source.path),
                        reason=f"Unsupported format: {source.format}"
                    )

            # Store file hash
            self.file_hashes[source.path.name] = current_hash

            self.logger.debug(f"Loaded configuration from {source.name}: {source.path}")
            return config_data or {}

        except yaml.YAMLError as e:
            raise ConfigFileError(
                config_file=str(source.path),
                reason=f"YAML parsing error: {e}"
            )
        except json.JSONDecodeError as e:
            raise ConfigFileError(
                config_file=str(source.path),
                reason=f"JSON parsing error: {e}"
            )
        except Exception as e:
            raise ConfigFileError(
                config_file=str(source.path),
                reason=f"File reading error: {e}"
            )

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file content."""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception:
            return ""

    def _substitute_environment_variables(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Substitute environment variables in configuration values."""
        def substitute_value(value):
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_var = value[2:-1]
                default_value = None

                # Handle default values: ${VAR:default}
                if ":" in env_var:
                    env_var, default_value = env_var.split(":", 1)

                return os.getenv(env_var, default_value)
            elif isinstance(value, dict):
                return {k: substitute_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [substitute_value(item) for item in value]
            else:
                return value

        return substitute_value(config)

    async def _create_default_config_file(self, config_path: Path):
        """Create default configuration file."""
        config_path.parent.mkdir(parents=True, exist_ok=True)

        default_config = get_default_user_config()

        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(default_config, f, default_flow_style=False, indent=2)

        self.logger.info(f"Created default configuration file: {config_path}")

    def _track_config_changes(self, old_config: Dict[str, Any], new_config: Dict[str, Any], source: str):
        """Track configuration changes."""
        def find_changes(old_dict, new_dict, prefix=""):
            changes = []

            # Check for modified or new values
            for key, new_value in new_dict.items():
                full_key = f"{prefix}.{key}" if prefix else key

                if key not in old_dict:
                    # New key
                    changes.append(ConfigChange(
                        timestamp=datetime.now(),
                        key=full_key,
                        old_value=None,
                        new_value=new_value,
                        source=source
                    ))
                elif isinstance(new_value, dict) and isinstance(old_dict[key], dict):
                    # Recursive check for nested dicts
                    changes.extend(find_changes(old_dict[key], new_value, full_key))
                elif old_dict[key] != new_value:
                    # Modified value
                    changes.append(ConfigChange(
                        timestamp=datetime.now(),
                        key=full_key,
                        old_value=old_dict[key],
                        new_value=new_value,
                        source=source
                    ))

            # Check for deleted values
            for key, old_value in old_dict.items():
                full_key = f"{prefix}.{key}" if prefix else key
                if key not in new_dict:
                    changes.append(ConfigChange(
                        timestamp=datetime.now(),
                        key=full_key,
                        old_value=old_value,
                        new_value=None,
                        source=source
                    ))

            return changes

        changes = find_changes(old_config, new_config)
        self.change_history.extend(changes)

        # Keep only recent changes (last 1000)
        if len(self.change_history) > 1000:
            self.change_history = self.change_history[-1000:]

        # Notify callbacks
        for change in changes:
            self._notify_change_callbacks(change)

    def _notify_change_callbacks(self, change: ConfigChange):
        """Notify registered change callbacks."""
        for callback in self.change_callbacks:
            try:
                callback(change.key, change.old_value, change.new_value)
            except Exception as e:
                self.logger.error(f"Error in config change callback: {e}")

    async def save_config(self, config: Optional[Dict[str, Any]] = None, backup: bool = True) -> bool:
        """Save configuration to primary config file."""
        with self.lock:
            try:
                config_to_save = config or self.config

                # Create backup if requested
                if backup and self.primary_config_path.exists():
                    backup_path = backup_file(self.primary_config_path)
                    if backup_path:
                        self.logger.info(f"Created configuration backup: {backup_path}")

                # Validate before saving
                if self.validation_enabled:
                    validation_errors = self.validator.validate(config_to_save)
                    if validation_errors:
                        raise ConfigValidationError(
                            field="configuration",
                            value="invalid",
                            expected="valid configuration"
                        )

                # Save to file
                self.primary_config_path.parent.mkdir(parents=True, exist_ok=True)

                with open(self.primary_config_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config_to_save, f, default_flow_style=False, indent=2)

                # Update internal state
                old_config = self.config.copy()
                self.config = config_to_save

                # Track changes
                self._track_config_changes(old_config, config_to_save, "save")

                # Update file hash
                self.file_hashes[self.primary_config_path.name] = self._calculate_file_hash(self.primary_config_path)

                self.logger.info(f"Configuration saved to {self.primary_config_path}")
                return True

            except Exception as e:
                self.logger.error(f"Failed to save configuration: {e}")
                raise ConfigurationError(f"Configuration save failed: {e}")

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        with self.lock:
            return copy.deepcopy(self.config)

    def get_value(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        with self.lock:
            return get_nested_value(self.config, key, default)

    async def set_value(self, key: str, value: Any, save: bool = True) -> bool:
        """Set configuration value using dot notation."""
        with self.lock:
            try:
                old_config = self.config.copy()

                if set_nested_value(self.config, key, value):
                    # Track change
                    change = ConfigChange(
                        timestamp=datetime.now(),
                        key=key,
                        old_value=get_nested_value(old_config, key),
                        new_value=value,
                        source="api"
                    )
                    self.change_history.append(change)
                    self._notify_change_callbacks(change)

                    # Save if requested
                    if save:
                        await self.save_config()

                    self.logger.info(f"Configuration value updated: {key} = {value}")
                    return True
                else:
                    raise ConfigValidationError(
                        field=key,
                        value=value,
                        expected="valid key path"
                    )

            except Exception as e:
                self.logger.error(f"Failed to set configuration value {key}: {e}")
                raise

    async def update_config(self, updates: Dict[str, Any], save: bool = True) -> bool:
        """Update multiple configuration values."""
        with self.lock:
            try:
                old_config = self.config.copy()

                # Apply all updates
                for key, value in updates.items():
                    if not set_nested_value(self.config, key, value):
                        raise ConfigValidationError(
                            field=key,
                            value=value,
                            expected="valid key path"
                        )

                # Track changes
                self._track_config_changes(old_config, self.config, "bulk_update")

                # Save if requested
                if save:
                    await self.save_config()

                self.logger.info(f"Configuration updated with {len(updates)} changes")
                return True

            except Exception as e:
                # Rollback changes
                self.config = old_config
                self.logger.error(f"Failed to update configuration: {e}")
                raise

    def start_watching(self):
        """Start watching configuration files for changes."""
        if self.watch_enabled:
            return

        self.watch_enabled = True
        self.stop_watching.clear()

        self.watch_thread = threading.Thread(
            target=self._watch_files,
            daemon=True
        )
        self.watch_thread.start()

        self.logger.info("Started configuration file watching")

    def stop_watching(self):
        """Stop watching configuration files."""
        if not self.watch_enabled:
            return

        self.watch_enabled = False
        self.stop_watching.set()

        if self.watch_thread:
            self.watch_thread.join(timeout=5.0)

        self.logger.info("Stopped configuration file watching")

    def _watch_files(self):
        """Watch configuration files for changes."""
        import time

        while not self.stop_watching.is_set():
            try:
                # Check each watched source
                for source in self.config_sources:
                    if not source.watch or not source.path.exists():
                        continue

                    current_hash = self._calculate_file_hash(source.path)
                    stored_hash = self.file_hashes.get(source.path.name)

                    if stored_hash and current_hash != stored_hash:
                        self.logger.info(f"Configuration file changed: {source.path}")

                        # Reload configuration
                        try:
                            asyncio.run_coroutine_threadsafe(
                                self.load_config(),
                                asyncio.get_event_loop()
                            ).result(timeout=10.0)
                        except Exception as e:
                            self.logger.error(f"Failed to reload configuration: {e}")

                # Sleep before next check
                self.stop_watching.wait(2.0)

            except Exception as e:
                self.logger.error(f"Error in file watching: {e}")
                time.sleep(5.0)

    def add_change_callback(self, callback: Callable[[str, Any, Any], None]):
        """Add configuration change callback."""
        self.change_callbacks.append(callback)

    def remove_change_callback(self, callback: Callable[[str, Any, Any], None]):
        """Remove configuration change callback."""
        if callback in self.change_callbacks:
            self.change_callbacks.remove(callback)

    def get_change_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get configuration change history."""
        with self.lock:
            recent_changes = self.change_history[-limit:] if limit else self.change_history

            return [
                {
                    "timestamp": change.timestamp.isoformat(),
                    "key": change.key,
                    "old_value": change.old_value,
                    "new_value": change.new_value,
                    "source": change.source,
                    "user": change.user
                }
                for change in recent_changes
            ]

    def validate_config(self, config: Optional[Dict[str, Any]] = None) -> List[str]:
        """Validate configuration and return list of errors."""
        config_to_validate = config or self.config

        if not self.validation_enabled:
            return []

        validation_errors = self.validator.validate(config_to_validate)
        return [f"{error.field}: {error.message}" for error in validation_errors]

    def export_config(self, format: str = "yaml", include_defaults: bool = False) -> str:
        """Export configuration to string."""
        with self.lock:
            config_to_export = self.config

            if not include_defaults:
                # Remove default values to make export cleaner
                default_config = get_default_user_config()
                config_to_export = self._remove_default_values(self.config, default_config)

            if format.lower() == "yaml":
                return yaml.dump(config_to_export, default_flow_style=False, indent=2)
            elif format.lower() == "json":
                return json.dumps(config_to_export, indent=2)
            else:
                raise ConfigurationError(f"Unsupported export format: {format}")

    def _remove_default_values(self, config: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
        """Remove values that match defaults."""
        result = {}

        for key, value in config.items():
            if key not in defaults:
                result[key] = value
            elif isinstance(value, dict) and isinstance(defaults[key], dict):
                nested_result = self._remove_default_values(value, defaults[key])
                if nested_result:  # Only include if there are non-default values
                    result[key] = nested_result
            elif value != defaults[key]:
                result[key] = value

        return result

    async def import_config(self, config_data: str, format: str = "yaml", merge: bool = True) -> bool:
        """Import configuration from string."""
        try:
            if format.lower() == "yaml":
                imported_config = yaml.safe_load(config_data)
            elif format.lower() == "json":
                imported_config = json.loads(config_data)
            else:
                raise ConfigurationError(f"Unsupported import format: {format}")

            if merge:
                # Merge with existing configuration
                merged_config = merge_configs(self.config, imported_config)
                await self.save_config(merged_config)
            else:
                # Replace entire configuration
                await self.save_config(imported_config)

            self.logger.info(f"Configuration imported from {format}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to import configuration: {e}")
            raise ConfigurationError(f"Configuration import failed: {e}")

    def get_config_info(self) -> Dict[str, Any]:
        """Get information about configuration sources and state."""
        with self.lock:
            return {
                "primary_config_path": str(self.primary_config_path),
                "sources": [
                    {
                        "name": source.name,
                        "path": str(source.path),
                        "exists": source.path.exists(),
                        "priority": source.priority,
                        "watch": source.watch,
                        "required": source.required,
                        "last_hash": self.file_hashes.get(source.path.name)
                    }
                    for source in self.config_sources
                ],
                "validation_enabled": self.validation_enabled,
                "watch_enabled": self.watch_enabled,
                "change_count": len(self.change_history),
                "callbacks_count": len(self.change_callbacks),
                "last_loaded": datetime.now().isoformat()
            }

    async def reset_to_defaults(self, backup: bool = True) -> bool:
        """Reset configuration to defaults."""
        try:
            if backup:
                backup_path = backup_file(self.primary_config_path, "_reset_backup")
                if backup_path:
                    self.logger.info(f"Created reset backup: {backup_path}")

            default_config = get_default_user_config()
            await self.save_config(default_config, backup=False)

            self.logger.info("Configuration reset to defaults")
            return True

        except Exception as e:
            self.logger.error(f"Failed to reset configuration: {e}")
            raise ConfigurationError(f"Configuration reset failed: {e}")

    async def shutdown(self):
        """Shutdown configuration manager."""
        self.logger.info("Shutting down configuration manager")

        # Stop file watching
        self.stop_watching()

        # Save any pending changes
        try:
            if self.config:
                await self.save_config()
        except Exception as e:
            self.logger.error(f"Failed to save configuration during shutdown: {e}")

        self.logger.info("Configuration manager shutdown complete")


# Utility functions for external use
async def load_config_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from a single file."""
    file_path = Path(file_path)

    if not file_path.exists():
        raise ConfigFileError(str(file_path), "File not found")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.suffix.lower() in ['.yaml', '.yml']:
                return yaml.safe_load(f) or {}
            elif file_path.suffix.lower() == '.json':
                return json.load(f) or {}
            else:
                raise ConfigFileError(str(file_path), "Unsupported file format")
    except Exception as e:
        raise ConfigFileError(str(file_path), f"Failed to load file: {e}")


async def save_config_file(file_path: Union[str, Path], config: Dict[str, Any], backup: bool = True):
    """Save configuration to a file."""
    file_path = Path(file_path)

    if backup and file_path.exists():
        backup_file(file_path)

    file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            if file_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            elif file_path.suffix.lower() == '.json':
                json.dump(config, f, indent=2)
            else:
                raise ConfigFileError(str(file_path), "Unsupported file format")
    except Exception as e:
        raise ConfigFileError(str(file_path), f"Failed to save file: {e}")


if __name__ == "__main__":
    # Example usage
    import asyncio

    async def main():
        # Create config manager
        config_manager = ConfigManager("config/test-config.yaml")

        # Load configuration
        config = await config_manager.load_config()
        print("Loaded configuration:")
        print(yaml.dump(config, default_flow_style=False, indent=2))

        # Update a value
        await config_manager.set_value("system.display.target_resolution", "1280x720")

        # Get change history
        changes = config_manager.get_change_history(5)
        print(f"\nRecent changes: {len(changes)}")
        for change in changes:
            print(f"  {change['key']}: {change['old_value']} -> {change['new_value']}")

        # Export configuration
        yaml_export = config_manager.export_config("yaml")
        print(f"\nExported configuration:\n{yaml_export}")

    asyncio.run(main())