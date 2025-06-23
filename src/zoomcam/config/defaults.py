"""
Default Configuration Generator
==============================

Creates default configuration files for ZoomCam system.
"""

import yaml
from pathlib import Path
from typing import Dict, Any


def get_default_user_config() -> Dict[str, Any]:
    """Get default user configuration."""
    return {
        "system": {
            "display": {
                "target_resolution": "1920x1080",
                "auto_detect_resolution": True,
                "refresh_rate": 30,
                "interpolation": {
                    "enabled": True,
                    "algorithm": "lanczos",
                    "quality_priority": "balanced"
                }
            },
            "performance": {
                "max_cpu_usage": 80,
                "memory_limit": 512,
                "gpu_acceleration": "auto"
            }
        },

        "cameras": {
            "camera_1": {
                "enabled": True,
                "source": "/dev/video0",
                "name": "Main Camera",
                "resolution": "auto",
                "zoom": 3.0,
                "max_fragments": 2,

                "recording": {
                    "enabled": True,
                    "quality": "medium",
                    "reaction_time": 0.5,
                    "max_duration": 300
                },

                "motion_detection": {
                    "sensitivity": 0.3,
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
        },

        "layout": {
            "algorithm": "adaptive_grid",
            "gap_size": 5,
            "border_width": 2,
            "inactive_timeout": 30,
            "css_mode": True,
            "responsive_breakpoints": [
                {
                    "resolution": "1920x1080",
                    "columns": 4,
                    "rows": 3
                },
                {
                    "resolution": "1280x720",
                    "columns": 3,
                    "rows": 2
                }
            ]
        },

        "streaming": {
            "hls_output_dir": "/tmp/zoomcam_hls",
            "segment_duration": 2,
            "segment_count": 10,
            "bitrate": "2M"
        },

        "recording": {
            "output_dir": "/var/lib/zoomcam/recordings",
            "cleanup_days": 7,
            "compression": "h264_medium"
        },

        "logging": {
            "method": "git_timeline",
            "git": {
                "enabled": True,
                "repository_path": "/var/lib/zoomcam/history",
                "auto_commit": True,
                "commit_interval": 30,
                "screenshot_interval": 5,
                "max_history_days": 30
            },
            "screenshots": {
                "enabled": True,
                "quality": 80,
                "format": "jpg"
            }
        }
    }


def get_default_auto_config() -> Dict[str, Any]:
    """Get default auto-generated configuration template."""
    return {
        "system": {
            "detected_display": {
                "actual_resolution": "1920x1080",
                "refresh_rate": 60,
                "color_depth": 24,
                "aspect_ratio": "16:9"
            },
            "performance": {
                "current_cpu_usage": 0.0,
                "memory_usage": 0,
                "gpu_acceleration": False,
                "last_optimization": None
            }
        },

        "cameras": {},

        "layout": {
            "current_grid": "1x1",
            "active_fragments": 0,
            "last_recalculation": None,
            "css_grid_state": ""
        }
    }


def get_default_css_layouts() -> Dict[str, Any]:
    """Get default CSS layout templates."""
    return {
        "templates": {
            "equal_2x2": {
                "name": "Equal 2x2 Grid",
                "css": """
                display: grid;
                grid-template-columns: 1fr 1fr;
                grid-template-rows: 1fr 1fr;
                gap: 5px;
                """
            },

            "adaptive_priority": {
                "name": "Adaptive Priority Layout",
                "css": """
                display: grid;
                grid-template-columns: 2fr 1fr;
                grid-template-rows: 2fr 1fr;
                grid-template-areas:
                  "main secondary"
                  "tertiary quaternary";
                gap: 5px;
                """
            },

            "triple_split": {
                "name": "Triple Split Layout",
                "css": """
                display: grid;
                grid-template-columns: 1.5fr 1fr 1fr;
                grid-template-rows: 1fr 1fr;
                gap: 3px;
                """
            },

            "single_focus": {
                "name": "Single Camera Focus",
                "css": """
                display: grid;
                grid-template-columns: 1fr;
                grid-template-rows: 1fr;
                gap: 0px;
                """
            }
        },

        "camera_styles": {
            "active": {
                "css": """
                border: 3px solid #00ff00;
                box-shadow: 0 0 10px rgba(0,255,0,0.5);
                z-index: 10;
                opacity: 1.0;
                transition: all 0.3s ease;
                """
            },

            "inactive": {
                "css": """
                border: 1px solid #666;
                opacity: 0.7;
                transition: all 0.3s ease;
                """
            },

            "no_signal": {
                "css": """
                background: #1a1a1a;
                border: 2px dashed #ff0000;
                display: flex;
                align-items: center;
                justify-content: center;
                color: #ff6666;
                font-size: 1.5em;
                """
            },

            "motion_detected": {
                "css": """
                border: 3px solid #ffaa00;
                box-shadow: 0 0 15px rgba(255,170,0,0.6);
                animation: motion-pulse 2s infinite;
                """
            }
        },

        "animations": {
            "motion-pulse": """
            @keyframes motion-pulse {
                0% { box-shadow: 0 0 15px rgba(255,170,0,0.6); }
                50% { box-shadow: 0 0 25px rgba(255,170,0,0.9); }
                100% { box-shadow: 0 0 15px rgba(255,170,0,0.6); }
            }
            """,

            "layout-transition": """
            .zoomcam-grid {
                transition: grid-template-columns 0.8s ease-in-out,
                           grid-template-rows 0.8s ease-in-out,
                           grid-template-areas 0.8s ease-in-out;
            }
            
            .camera-fragment {
                transition: all 0.5s ease-in-out;
            }
            """
        }
    }


def get_default_interpolation_config() -> Dict[str, Any]:
    """Get default interpolation configuration."""
    return {
        "algorithms": {
            "nearest": {
                "name": "Nearest Neighbor",
                "cv2_flag": 0,  # cv2.INTER_NEAREST
                "quality": 1,
                "performance": 10,
                "description": "Fastest, lowest quality"
            },

            "linear": {
                "name": "Bilinear",
                "cv2_flag": 1,  # cv2.INTER_LINEAR
                "quality": 4,
                "performance": 8,
                "description": "Good balance of speed and quality"
            },

            "cubic": {
                "name": "Bicubic",
                "cv2_flag": 2,  # cv2.INTER_CUBIC
                "quality": 7,
                "performance": 5,
                "description": "High quality, moderate speed"
            },

            "lanczos": {
                "name": "Lanczos",
                "cv2_flag": 4,  # cv2.INTER_LANCZOS4
                "quality": 9,
                "performance": 3,
                "description": "Highest quality, slower"
            }
        },

        "quality_profiles": {
            "performance": {
                "name": "Performance Priority",
                "upscale_algorithm": "linear",
                "downscale_algorithm": "linear",
                "sharpening": 0.0,
                "cpu_threshold": 60
            },

            "balanced": {
                "name": "Balanced Quality/Performance",
                "upscale_algorithm": "cubic",
                "downscale_algorithm": "linear",
                "sharpening": 0.1,
                "cpu_threshold": 75
            },

            "quality": {
                "name": "Quality Priority",
                "upscale_algorithm": "lanczos",
                "downscale_algorithm": "cubic",
                "sharpening": 0.2,
                "cpu_threshold": 90
            }
        },

        "adaptive_settings": {
            "enabled": True,
            "fallback_algorithm": "linear",
            "performance_monitoring": True,
            "auto_adjust_quality": True
        }
    }


def create_default_config(config_path: Path) -> None:
    """Create default configuration file."""
    config_path.parent.mkdir(parents=True, exist_ok=True)

    default_config = get_default_user_config()

    with open(config_path, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False, indent=2)


def create_all_default_configs(config_dir: Path) -> None:
    """Create all default configuration files."""
    config_dir.mkdir(parents=True, exist_ok=True)

    # User configuration
    user_config_path = config_dir / "user-config.yaml"
    if not user_config_path.exists():
        with open(user_config_path, 'w') as f:
            yaml.dump(get_default_user_config(), f, default_flow_style=False, indent=2)

    # Auto configuration template
    auto_config_path = config_dir / "auto-config.yaml"
    if not auto_config_path.exists():
        with open(auto_config_path, 'w') as f:
            yaml.dump(get_default_auto_config(), f, default_flow_style=False, indent=2)

    # CSS layouts
    css_dir = config_dir / "css"
    css_dir.mkdir(exist_ok=True)

    layouts_path = css_dir / "layouts.yaml"
    if not layouts_path.exists():
        with open(layouts_path, 'w') as f:
            yaml.dump(get_default_css_layouts(), f, default_flow_style=False, indent=2)

    # Interpolation configuration
    interpolation_dir = config_dir / "interpolation"
    interpolation_dir.mkdir(exist_ok=True)

    algorithms_path = interpolation_dir / "algorithms.yaml"
    if not algorithms_path.exists():
        with open(algorithms_path, 'w') as f:
            yaml.dump(get_default_interpolation_config(), f, default_flow_style=False, indent=2)

    # Schema validation (placeholder)
    schema_path = config_dir / "schema.yaml"
    if not schema_path.exists():
        schema_config = {
            "version": "1.0",
            "description": "ZoomCam configuration schema",
            "validation_rules": {
                "system.display.target_resolution": "string matching pattern \\d+x\\d+",
                "cameras.*.zoom": "float between 1.0 and 10.0",
                "layout.gap_size": "integer >= 0",
                "streaming.bitrate": "string ending with 'K' or 'M'"
            }
        }
        with open(schema_path, 'w') as f:
            yaml.dump(schema_config, f, default_flow_style=False, indent=2)


def get_raspberry_pi_optimized_config() -> Dict[str, Any]:
    """Get Raspberry Pi optimized configuration."""
    config = get_default_user_config()

    # Optimize for RPi performance
    config["system"]["performance"]["max_cpu_usage"] = 70
    config["system"]["performance"]["memory_limit"] = 256
    config["system"]["display"]["interpolation"]["algorithm"] = "linear"
    config["system"]["display"]["interpolation"]["quality_priority"] = "performance"

    # Reduce recording quality for better performance
    for camera_id in config["cameras"]:
        config["cameras"][camera_id]["recording"]["quality"] = "low"
        config["cameras"][camera_id]["motion_detection"]["max_zones"] = 3

    # Optimize streaming
    config["streaming"]["bitrate"] = "1M"
    config["streaming"]["segment_duration"] = 4

    # Reduce logging frequency
    config["logging"]["git"]["commit_interval"] = 60
    config["logging"]["git"]["screenshot_interval"] = 10

    return config


def create_example_configs(config_dir: Path) -> None:
    """Create example configuration files."""
    examples_dir = config_dir / "examples"
    examples_dir.mkdir(exist_ok=True)

    # Raspberry Pi example
    rpi_config_path = examples_dir / "raspberry-pi-example.yaml"
    with open(rpi_config_path, 'w') as f:
        yaml.dump(get_raspberry_pi_optimized_config(), f, default_flow_style=False, indent=2)

    # High-end system example
    high_end_config = get_default_user_config()
    high_end_config["system"]["performance"]["max_cpu_usage"] = 90
    high_end_config["system"]["performance"]["memory_limit"] = 1024
    high_end_config["system"]["display"]["interpolation"]["algorithm"] = "lanczos"
    high_end_config["system"]["display"]["interpolation"]["quality_priority"] = "quality"

    # Add more cameras for high-end system
    for i in range(2, 5):
        camera_id = f"camera_{i}"
        high_end_config["cameras"][camera_id] = {
            "enabled": True,
            "source": f"rtsp://192.168.1.{100 + i}:554/stream",
            "name": f"Camera {i}",
            "resolution": "1080p",
            "zoom": 2.0,
            "max_fragments": 3,
            "recording": {
                "enabled": True,
                "quality": "high",
                "reaction_time": 0.2,
                "max_duration": 600
            },
            "motion_detection": {
                "sensitivity": 0.4,
                "min_area": 300,
                "max_zones": 5,
                "ignore_zones": []
            },
            "display": {
                "min_size_percent": 15,
                "max_size_percent": 80,
                "transition_speed": 0.6
            }
        }

    high_end_config_path = examples_dir / "high-end-system-example.yaml"
    with open(high_end_config_path, 'w') as f:
        yaml.dump(high_end_config, f, default_flow_style=False, indent=2)

    # Security focused example
    security_config = get_default_user_config()
    security_config["recording"]["cleanup_days"] = 30  # Keep recordings longer
    security_config["logging"]["git"]["max_history_days"] = 90

    for camera_id in security_config["cameras"]:
        security_config["cameras"][camera_id]["recording"]["enabled"] = True
        security_config["cameras"][camera_id]["recording"]["quality"] = "high"
        security_config["cameras"][camera_id]["recording"]["max_duration"] = 1800  # 30 minutes
        security_config["cameras"][camera_id]["motion_detection"]["sensitivity"] = 0.2  # More sensitive

    security_config_path = examples_dir / "security-focused-example.yaml"
    with open(security_config_path, 'w') as f:
        yaml.dump(security_config, f, default_flow_style=False, indent=2)


if __name__ == "__main__":
    # Create default configs when run directly
    import sys
    from pathlib import Path

    if len(sys.argv) > 1:
        config_dir = Path(sys.argv[1])
    else:
        config_dir = Path("config")

    print(f"Creating default configurations in {config_dir}")

    create_all_default_configs(config_dir)
    create_example_configs(config_dir)

    print("Default configurations created successfully!")
    print(f"Main config: {config_dir}/user-config.yaml")
    print(f"Examples: {config_dir}/examples/")