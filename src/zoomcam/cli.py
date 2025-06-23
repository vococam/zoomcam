"""
ZoomCam CLI - Command Line Interface
===================================

Provides command-line tools for setup, configuration, and management.
"""

import asyncio
import sys
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import click
import cv2
import psutil

from zoomcam.config.defaults import create_all_default_configs, get_raspberry_pi_optimized_config
from zoomcam.utils.logger import setup_logging


@click.group()
@click.option('--config', '-c', default='config/user-config.yaml', help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.pass_context
def cli(ctx, config, verbose):
    """ZoomCam - Intelligent Adaptive Camera Monitoring System"""
    ctx.ensure_object(dict)
    ctx.obj['config_path'] = config
    ctx.obj['verbose'] = verbose

    if verbose:
        setup_logging(level='DEBUG')


@cli.command()
@click.option('--output-dir', '-o', default='config', help='Output directory for configs')
@click.option('--raspberry-pi', is_flag=True, help='Generate Raspberry Pi optimized config')
@click.option('--force', is_flag=True, help='Overwrite existing files')
def init(output_dir, raspberry_pi, force):
    """Initialize ZoomCam configuration files."""
    output_path = Path(output_dir)

    if output_path.exists() and not force:
        if any(output_path.iterdir()):
            click.echo(f"❌ Directory {output_path} already contains files. Use --force to overwrite.")
            return

    click.echo(f"🔧 Initializing ZoomCam configuration in {output_path}")

    try:
        create_all_default_configs(output_path)

        if raspberry_pi:
            # Create RPi-optimized config
            rpi_config = get_raspberry_pi_optimized_config()
            config_file = output_path / "user-config.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(rpi_config, f, default_flow_style=False, indent=2)
            click.echo("🥧 Created Raspberry Pi optimized configuration")

        click.echo("✅ Configuration files created successfully!")
        click.echo(f"📝 Edit {output_path}/user-config.yaml to customize your setup")

    except Exception as e:
        click.echo(f"❌ Failed to create configuration: {e}")
        sys.exit(1)


@cli.command()
@click.pass_context
def setup(ctx):
    """Interactive setup wizard."""
    click.echo("🚀 ZoomCam Setup Wizard")
    click.echo("=" * 50)

    config_path = Path(ctx.obj['config_path'])

    # Detect system capabilities
    click.echo("🔍 Detecting system capabilities...")
    system_info = detect_system_capabilities()

    click.echo(f"📊 System: {system_info['platform']}")
    click.echo(f"💾 RAM: {system_info['memory_gb']:.1f} GB")
    click.echo(f"🖥️  Display: {system_info['display_resolution']}")
    click.echo()

    # Detect cameras
    click.echo("📹 Detecting cameras...")
    cameras = detect_cameras()

    if cameras:
        click.echo(f"✅ Found {len(cameras)} camera(s):")
        for i, camera in enumerate(cameras):
            click.echo(f"  {i + 1}. {camera['name']} ({camera['resolution']})")
    else:
        click.echo("⚠️  No cameras detected. You can add them manually later.")

    click.echo()

    # Interactive configuration
    config = create_interactive_config(system_info, cameras)

    # Save configuration
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

    click.echo(f"✅ Configuration saved to {config_path}")
    click.echo("🎉 Setup complete! Run 'zoomcam' to start the system.")


@cli.command()
@click.pass_context
def detect_cameras(ctx):
    """Detect available cameras."""
    click.echo("🔍 Scanning for cameras...")

    cameras = detect_cameras()

    if cameras:
        click.echo(f"✅ Found {len(cameras)} camera(s):")
        click.echo()

        for camera in cameras:
            click.echo(f"📹 {camera['name']}")
            click.echo(f"   Source: {camera['source']}")
            click.echo(f"   Resolution: {camera['resolution']}")
            click.echo(f"   Type: {camera['type']}")
            click.echo()
    else:
        click.echo("❌ No cameras found")
        click.echo("💡 Make sure cameras are connected and not in use by other applications")


@cli.command()
@click.option('--camera-id', help='Test specific camera ID')
@click.pass_context
def test_cameras(ctx, camera_id):
    """Test camera connections."""
    config_path = Path(ctx.obj['config_path'])

    if not config_path.exists():
        click.echo(f"❌ Configuration file not found: {config_path}")
        click.echo("💡 Run 'zoomcam-setup init' first")
        return

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    cameras_config = config.get('cameras', {})

    if camera_id:
        if camera_id not in cameras_config:
            click.echo(f"❌ Camera {camera_id} not found in configuration")
            return
        cameras_to_test = {camera_id: cameras_config[camera_id]}
    else:
        cameras_to_test = cameras_config

    click.echo("🧪 Testing camera connections...")

    for cam_id, cam_config in cameras_to_test.items():
        if not cam_config.get('enabled', True):
            click.echo(f"⏭️  Skipping disabled camera {cam_id}")
            continue

        click.echo(f"📹 Testing {cam_id} ({cam_config.get('name', 'Unknown')})...")

        success, message = test_camera_connection(cam_config['source'])

        if success:
            click.echo(f"✅ {message}")
        else:
            click.echo(f"❌ {message}")


@cli.command()
@click.pass_context
def validate_config(ctx):
    """Validate configuration file."""
    config_path = Path(ctx.obj['config_path'])

    if not config_path.exists():
        click.echo(f"❌ Configuration file not found: {config_path}")
        return

    click.echo(f"🔍 Validating configuration: {config_path}")

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Basic validation
        errors = validate_configuration(config)

        if errors:
            click.echo("❌ Configuration validation failed:")
            for error in errors:
                click.echo(f"   • {error}")
        else:
            click.echo("✅ Configuration is valid!")

    except yaml.YAMLError as e:
        click.echo(f"❌ YAML syntax error: {e}")
    except Exception as e:
        click.echo(f"❌ Validation error: {e}")


@cli.command()
@click.option('--duration', default=10, help='Test duration in seconds')
@click.pass_context
def benchmark(ctx, duration):
    """Run system performance benchmark."""
    click.echo(f"⚡ Running {duration}s performance benchmark...")

    results = run_performance_benchmark(duration)

    click.echo("📊 Benchmark Results:")
    click.echo(f"   CPU Usage: {results['avg_cpu']:.1f}%")
    click.echo(f"   Memory Usage: {results['avg_memory']:.1f}%")
    click.echo(f"   Frame Processing: {results['frames_per_second']:.1f} FPS")
    click.echo(f"   Motion Detection: {results['motion_detection_fps']:.1f} FPS")

    # Recommendations
    if results['avg_cpu'] > 80:
        click.echo("⚠️  High CPU usage detected. Consider reducing camera count or quality.")

    if results['avg_memory'] > 80:
        click.echo("⚠️  High memory usage detected. Consider reducing buffer sizes.")

    click.echo(f"💾 Benchmark results saved to benchmark_results.json")

    with open("benchmark_results.json", 'w') as f:
        json.dump(results, f, indent=2)


@cli.command()
@click.option('--key', help='Configuration key to get/set (e.g., cameras.camera_1.zoom)')
@click.option('--value', help='Value to set (omit to get current value)')
@click.option('--type', 'value_type', type=click.Choice(['str', 'int', 'float', 'bool']), default='str',
              help='Value type')
@click.pass_context
def config(ctx, key, value, value_type):
    """Get or set configuration values."""
    config_path = Path(ctx.obj['config_path'])

    if not config_path.exists():
        click.echo(f"❌ Configuration file not found: {config_path}")
        return

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if value is None:
        # Get value
        if key:
            result = get_nested_config_value(config, key)
            if result is not None:
                click.echo(f"{key}: {result}")
            else:
                click.echo(f"❌ Key not found: {key}")
        else:
            # Show all config
            click.echo(yaml.dump(config, default_flow_style=False, indent=2))
    else:
        # Set value
        if not key:
            click.echo("❌ Key required when setting value")
            return

        # Convert value to appropriate type
        try:
            if value_type == 'int':
                value = int(value)
            elif value_type == 'float':
                value = float(value)
            elif value_type == 'bool':
                value = value.lower() in ('true', '1', 'yes', 'on')
        except ValueError:
            click.echo(f"❌ Invalid {value_type} value: {value}")
            return

        # Set value
        if set_nested_config_value(config, key, value):
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            click.echo(f"✅ Set {key} = {value}")
        else:
            click.echo(f"❌ Failed to set {key}")


# Helper functions

def detect_system_capabilities() -> Dict[str, Any]:
    """Detect system capabilities."""
    import platform

    # Get display resolution
    try:
        import tkinter as tk
        root = tk.Tk()
        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()
        root.destroy()
        display_resolution = f"{width}x{height}"
    except:
        display_resolution = "1920x1080"  # Default

    return {
        'platform': platform.system(),
        'architecture': platform.machine(),
        'memory_gb': psutil.virtual_memory().total / (1024 ** 3),
        'cpu_count': psutil.cpu_count(),
        'display_resolution': display_resolution
    }


def detect_cameras() -> List[Dict[str, Any]]:
    """Detect available cameras."""
    cameras = []

    # Detect USB cameras
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                cameras.append({
                    'name': f'USB Camera {i}',
                    'source': f'/dev/video{i}',
                    'resolution': f'{width}x{height}',
                    'type': 'usb'
                })
            cap.release()

    return cameras


def test_camera_connection(source: str) -> tuple[bool, str]:
    """Test camera connection."""
    try:
        if source.startswith('rtsp://'):
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        else:
            # Extract device number for USB cameras
            if '/dev/video' in source:
                device_num = int(source.split('video')[1])
                cap = cv2.VideoCapture(device_num)
            else:
                cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            return False, f"Failed to open camera: {source}"

        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            return False, f"Failed to read from camera: {source}"

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        return True, f"Camera working: {width}x{height}"

    except Exception as e:
        return False, f"Error testing camera: {e}"


def create_interactive_config(system_info: Dict, cameras: List[Dict]) -> Dict[str, Any]:
    """Create configuration through interactive prompts."""
    from zoomcam.config.defaults import get_default_user_config

    config = get_default_user_config()

    # System configuration
    if system_info['memory_gb'] < 1.0:
        # Low memory system (like Raspberry Pi Zero)
        config['system']['performance']['memory_limit'] = 128
        config['system']['display']['interpolation']['algorithm'] = 'nearest'
    elif system_info['memory_gb'] < 2.0:
        # Raspberry Pi 3/4
        config['system']['performance']['memory_limit'] = 256
        config['system']['display']['interpolation']['algorithm'] = 'linear'

    # Set detected display resolution
    config['system']['display']['target_resolution'] = system_info['display_resolution']

    # Configure detected cameras
    if cameras:
        config['cameras'] = {}
        for i, camera in enumerate(cameras):
            camera_id = f"camera_{i + 1}"
            config['cameras'][camera_id] = {
                'enabled': True,
                'source': camera['source'],
                'name': camera['name'],
                'resolution': 'auto',
                'zoom': 3.0,
                'max_fragments': 2,
                'recording': {
                    'enabled': True,
                    'quality': 'medium',
                    'reaction_time': 0.5,
                    'max_duration': 300
                },
                'motion_detection': {
                    'sensitivity': 0.3,
                    'min_area': 500,
                    'max_zones': 5,
                    'ignore_zones': []
                },
                'display': {
                    'min_size_percent': 10,
                    'max_size_percent': 70,
                    'transition_speed': 0.8
                }
            }

    return config


def validate_configuration(config: Dict[str, Any]) -> List[str]:
    """Validate configuration and return list of errors."""
    errors = []

    # Check required sections
    required_sections = ['system', 'cameras', 'layout', 'streaming']
    for section in required_sections:
        if section not in config:
            errors.append(f"Missing required section: {section}")

    # Validate cameras
    if 'cameras' in config:
        for camera_id, camera_config in config['cameras'].items():
            if 'source' not in camera_config:
                errors.append(f"Camera {camera_id}: missing 'source'")

            zoom = camera_config.get('zoom', 3.0)
            if not 1.0 <= zoom <= 10.0:
                errors.append(f"Camera {camera_id}: zoom must be between 1.0 and 10.0")

    # Validate display resolution
    if 'system' in config and 'display' in config['system']:
        resolution = config['system']['display'].get('target_resolution', '')
        if not resolution or 'x' not in resolution:
            errors.append("Invalid display resolution format (expected: WIDTHxHEIGHT)")

    return errors


def get_nested_config_value(config: Dict, key: str) -> Any:
    """Get nested configuration value using dot notation."""
    keys = key.split('.')
    value = config

    try:
        for k in keys:
            value = value[k]
        return value
    except (KeyError, TypeError):
        return None


def set_nested_config_value(config: Dict, key: str, value: Any) -> bool:
    """Set nested configuration value using dot notation."""
    keys = key.split('.')
    current = config

    try:
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        current[keys[-1]] = value
        return True
    except (KeyError, TypeError):
        return False


def run_performance_benchmark(duration: int) -> Dict[str, Any]:
    """Run performance benchmark."""
    import time
    import numpy as np

    # Simulate camera processing
    results = {
        'duration': duration,
        'avg_cpu': 0.0,
        'avg_memory': 0.0,
        'frames_per_second': 0.0,
        'motion_detection_fps': 0.0
    }

    start_time = time.time()
    cpu_samples = []
    memory_samples = []
    frame_count = 0

    while time.time() - start_time < duration:
        # Simulate frame processing
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Simulate motion detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        frame_count += 1

        # Sample system resources
        cpu_samples.append(psutil.cpu_percent())
        memory_samples.append(psutil.virtual_memory().percent)

        time.sleep(0.033)  # ~30 FPS

    actual_duration = time.time() - start_time

    results['avg_cpu'] = np.mean(cpu_samples)
    results['avg_memory'] = np.mean(memory_samples)
    results['frames_per_second'] = frame_count / actual_duration
    results['motion_detection_fps'] = frame_count / actual_duration  # Same for this simulation

    return results


def setup_command():
    """Entry point for zoomcam-setup command."""
    cli()


def config_command():
    """Entry point for zoomcam-config command."""
    cli()


if __name__ == '__main__':
    cli()