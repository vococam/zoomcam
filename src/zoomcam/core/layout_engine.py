"""
Layout Engine - Adaptive Camera Layout Algorithm
===============================================

Intelligently calculates optimal camera layout based on activity levels,
motion detection, and screen real estate optimization.
"""

import logging
import math
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio

from zoomcam.utils.exceptions import LayoutError


@dataclass
class CameraFragment:
    """Represents a fragment of a camera view."""
    camera_id: str
    fragment_id: str
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    activity_level: float  # 0.0 to 1.0
    priority: int  # Higher number = higher priority
    last_activity: datetime


@dataclass
class LayoutCell:
    """Represents a cell in the layout grid."""
    x: int
    y: int
    width: int
    height: int
    camera_fragment: Optional[CameraFragment] = None
    css_grid_area: str = ""


@dataclass
class LayoutResult:
    """Result of layout calculation."""
    grid_columns: str
    grid_rows: str
    grid_areas: str
    cells: List[LayoutCell]
    fragments: List[CameraFragment]
    total_active_area: float
    layout_efficiency: float
    css_template: str
    timestamp: datetime


class LayoutEngine:
    """
    Adaptive layout engine that optimizes camera placement based on activity.

    Features:
    - Dynamic grid sizing based on number of active fragments
    - Activity-based size allocation
    - CSS Grid generation for web display
    - Fragment splitting for multi-zone cameras
    - Smooth transitions between layouts
    """

    def __init__(self, config: Dict[str, Any], screen_resolution: str):
        self.config = config
        self.screen_width, self.screen_height = self._parse_resolution(screen_resolution)

        # Layout parameters
        self.algorithm = config.get("algorithm", "adaptive_grid")
        self.gap_size = config.get("gap_size", 5)
        self.border_width = config.get("border_width", 2)
        self.inactive_timeout = config.get("inactive_timeout", 30)
        self.min_cell_size = config.get("min_cell_size", 100)

        # Current state
        self.current_layout: Optional[LayoutResult] = None
        self.camera_fragments: Dict[str, List[CameraFragment]] = {}
        self.activity_history: Dict[str, List[Tuple[datetime, float]]] = {}
        self.transition_speed = 0.8

        # CSS templates
        self.css_templates = {
            "equal_grid": self._generate_equal_grid_css,
            "priority_based": self._generate_priority_based_css,
            "adaptive_flow": self._generate_adaptive_flow_css
        }

        logging.info(f"Layout Engine initialized for {screen_resolution}")

    def _parse_resolution(self, resolution: str) -> Tuple[int, int]:
        """Parse resolution string to width, height tuple."""
        if "x" in resolution:
            width, height = resolution.split("x")
            return (int(width), int(height))
        elif resolution == "1080p":
            return (1920, 1080)
        elif resolution == "720p":
            return (1280, 720)
        elif resolution == "4K":
            return (3840, 2160)
        else:
            return (1920, 1080)  # Default

    async def update_camera_activity(
            self,
            camera_id: str,
            motion_zones: List[Dict[str, Any]]
    ) -> None:
        """Update camera activity and motion zones."""
        current_time = datetime.now()

        # Clear old fragments for this camera
        if camera_id not in self.camera_fragments:
            self.camera_fragments[camera_id] = []
        else:
            self.camera_fragments[camera_id].clear()

        # Create fragments based on motion zones
        if motion_zones:
            for i, zone in enumerate(motion_zones):
                fragment = CameraFragment(
                    camera_id=camera_id,
                    fragment_id=f"{camera_id}_zone_{i}",
                    bbox=zone.get("bbox", (0, 0, 100, 100)),
                    activity_level=zone.get("activity_level", 0.0),
                    priority=zone.get("priority", 1),
                    last_activity=current_time
                )
                self.camera_fragments[camera_id].append(fragment)
        else:
            # No motion - create single inactive fragment
            fragment = CameraFragment(
                camera_id=camera_id,
                fragment_id=f"{camera_id}_main",
                bbox=(0, 0, 100, 100),
                activity_level=0.0,
                priority=0,
                last_activity=current_time - timedelta(seconds=self.inactive_timeout + 1)
            )
            self.camera_fragments[camera_id].append(fragment)

        # Update activity history
        total_activity = sum(f.activity_level for f in self.camera_fragments[camera_id])
        if camera_id not in self.activity_history:
            self.activity_history[camera_id] = []

        self.activity_history[camera_id].append((current_time, total_activity))

        # Keep only recent history (last 5 minutes)
        cutoff_time = current_time - timedelta(minutes=5)
        self.activity_history[camera_id] = [
            (time, activity) for time, activity in self.activity_history[camera_id]
            if time > cutoff_time
        ]

    async def calculate_layout(self) -> LayoutResult:
        """Calculate optimal layout based on current activity."""
        try:
            # Get all active fragments
            all_fragments = []
            current_time = datetime.now()

            for camera_id, fragments in self.camera_fragments.items():
                for fragment in fragments:
                    # Check if fragment is still active
                    time_since_activity = current_time - fragment.last_activity
                    if time_since_activity.total_seconds() <= self.inactive_timeout:
                        all_fragments.append(fragment)
                    elif fragment.activity_level > 0.1:  # Keep if significant activity
                        all_fragments.append(fragment)

            # Sort fragments by priority and activity
            all_fragments.sort(
                key=lambda f: (f.priority, f.activity_level),
                reverse=True
            )

            # Determine grid size
            num_fragments = len(all_fragments)
            if num_fragments == 0:
                # No active fragments - show all cameras equally
                return await self._create_default_layout()

            # Calculate optimal grid dimensions
            grid_cols, grid_rows = self._calculate_grid_dimensions(num_fragments)

            # Allocate sizes based on activity
            if self.algorithm == "adaptive_grid":
                layout = await self._calculate_adaptive_grid(all_fragments, grid_cols, grid_rows)
            elif self.algorithm == "priority_based":
                layout = await self._calculate_priority_layout(all_fragments)
            else:
                layout = await self._calculate_equal_layout(all_fragments, grid_cols, grid_rows)

            self.current_layout = layout
            return layout

        except Exception as e:
            logging.error(f"Error calculating layout: {e}")
            raise LayoutError(f"Layout calculation failed: {e}")

    def _calculate_grid_dimensions(self, num_fragments: int) -> Tuple[int, int]:
        """Calculate optimal grid dimensions for given number of fragments."""
        if num_fragments <= 1:
            return (1, 1)
        elif num_fragments <= 2:
            return (2, 1)
        elif num_fragments <= 4:
            return (2, 2)
        elif num_fragments <= 6:
            return (3, 2)
        elif num_fragments <= 9:
            return (3, 3)
        else:
            # For more fragments, use wider grid
            cols = math.ceil(math.sqrt(num_fragments * 1.5))
            rows = math.ceil(num_fragments / cols)
            return (cols, rows)

    async def _calculate_adaptive_grid(
            self,
            fragments: List[CameraFragment],
            grid_cols: int,
            grid_rows: int
    ) -> LayoutResult:
        """Calculate adaptive grid layout with activity-based sizing."""

        # Calculate activity weights
        total_activity = sum(f.activity_level for f in fragments)
        if total_activity == 0:
            # No activity - equal distribution
            return await self._calculate_equal_layout(fragments, grid_cols, grid_rows)

        # Sort fragments by activity (highest first)
        active_fragments = sorted(fragments, key=lambda f: f.activity_level, reverse=True)

        # Create cells with proportional sizing
        cells = []
        grid_areas = []

        # Create grid template with weighted columns/rows
        col_weights = []
        row_weights = []

        # Main active fragment gets larger share
        if active_fragments:
            main_fragment = active_fragments[0]
            main_weight = max(2, int(main_fragment.activity_level * 3) + 1)
        else:
            main_weight = 1

        # Calculate column and row weights
        for col in range(grid_cols):
            if col == 0 and active_fragments:
                col_weights.append(f"{main_weight}fr")
            else:
                col_weights.append("1fr")

        for row in range(grid_rows):
            if row == 0 and active_fragments:
                row_weights.append(f"{main_weight}fr")
            else:
                row_weights.append("1fr")

        grid_columns = " ".join(col_weights)
        grid_rows = " ".join(row_weights)

        # Assign grid areas
        area_counter = 0
        for row in range(grid_rows):
            row_areas = []
            for col in range(grid_cols):
                if area_counter < len(active_fragments):
                    fragment = active_fragments[area_counter]
                    area_name = f"area_{fragment.camera_id}_{fragment.fragment_id}"

                    # Create cell
                    cell = LayoutCell(
                        x=col,
                        y=row,
                        width=1,
                        height=1,
                        camera_fragment=fragment,
                        css_grid_area=area_name
                    )
                    cells.append(cell)
                    row_areas.append(area_name)
                    area_counter += 1
                else:
                    row_areas.append(".")

            grid_areas.append(f'"{" ".join(row_areas)}"')

        grid_template_areas = "\n        ".join(grid_areas)

        # Generate CSS
        css_template = self._generate_adaptive_grid_css(
            grid_columns, grid_rows, grid_template_areas, cells
        )

        # Calculate efficiency metrics
        total_active_area = sum(f.activity_level for f in active_fragments)
        layout_efficiency = total_active_area / max(len(fragments), 1)

        return LayoutResult(
            grid_columns=grid_columns,
            grid_rows=grid_rows,
            grid_areas=grid_template_areas,
            cells=cells,
            fragments=active_fragments,
            total_active_area=total_active_area,
            layout_efficiency=layout_efficiency,
            css_template=css_template,
            timestamp=datetime.now()
        )

    async def _calculate_equal_layout(
            self,
            fragments: List[CameraFragment],
            grid_cols: int,
            grid_rows: int
    ) -> LayoutResult:
        """Calculate equal-sized grid layout."""
        cells = []
        grid_areas = []

        # Equal column and row distribution
        grid_columns = " ".join(["1fr"] * grid_cols)
        grid_rows = " ".join(["1fr"] * grid_rows)

        # Assign grid areas
        fragment_index = 0
        for row in range(grid_rows):
            row_areas = []
            for col in range(grid_cols):
                if fragment_index < len(fragments):
                    fragment = fragments[fragment_index]
                    area_name = f"area_{fragment.camera_id}_{fragment.fragment_id}"

                    cell = LayoutCell(
                        x=col,
                        y=row,
                        width=1,
                        height=1,
                        camera_fragment=fragment,
                        css_grid_area=area_name
                    )
                    cells.append(cell)
                    row_areas.append(area_name)
                    fragment_index += 1
                else:
                    row_areas.append(".")

            grid_areas.append(f'"{" ".join(row_areas)}"')

        grid_template_areas = "\n        ".join(grid_areas)

        css_template = self._generate_equal_grid_css(
            grid_columns, grid_rows, grid_template_areas, cells
        )

        return LayoutResult(
            grid_columns=grid_columns,
            grid_rows=grid_rows,
            grid_areas=grid_template_areas,
            cells=cells,
            fragments=fragments,
            total_active_area=len(fragments),
            layout_efficiency=1.0,
            css_template=css_template,
            timestamp=datetime.now()
        )

    async def _create_default_layout(self) -> LayoutResult:
        """Create default layout when no cameras are active."""
        # Show placeholder for no cameras
        placeholder_fragment = CameraFragment(
            camera_id="placeholder",
            fragment_id="no_cameras",
            bbox=(0, 0, 100, 100),
            activity_level=0.0,
            priority=0,
            last_activity=datetime.now()
        )

        cell = LayoutCell(
            x=0,
            y=0,
            width=1,
            height=1,
            camera_fragment=placeholder_fragment,
            css_grid_area="no_cameras"
        )

        css_template = '''
        .zoomcam-grid {
            display: grid;
            grid-template-columns: 1fr;
            grid-template-rows: 1fr;
            grid-template-areas: "no_cameras";
            gap: 0px;
            width: 100vw;
            height: 100vh;
        }
        .area_no_cameras {
            grid-area: no_cameras;
            background: #1a1a1a;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #666;
            font-size: 2em;
        }
        '''

        return LayoutResult(
            grid_columns="1fr",
            grid_rows="1fr",
            grid_areas='"no_cameras"',
            cells=[cell],
            fragments=[placeholder_fragment],
            total_active_area=0.0,
            layout_efficiency=0.0,
            css_template=css_template,
            timestamp=datetime.now()
        )

    def _generate_adaptive_grid_css(
            self,
            grid_columns: str,
            grid_rows: str,
            grid_areas: str,
            cells: List[LayoutCell]
    ) -> str:
        """Generate CSS for adaptive grid layout."""
        css = f'''
        .zoomcam-grid {{
            display: grid;
            grid-template-columns: {grid_columns};
            grid-template-rows: {grid_rows};
            grid-template-areas: 
                {grid_areas};
            gap: {self.gap_size}px;
            width: 100vw;
            height: 100vh;
            transition: all {self.transition_speed}s ease-in-out;
        }}
        '''

        # Add styles for each cell
        for cell in cells:
            if cell.camera_fragment:
                fragment = cell.camera_fragment
                area_class = f"area_{fragment.camera_id}_{fragment.fragment_id}"

                # Determine border color based on activity
                if fragment.activity_level > 0.5:
                    border_color = "#00ff00"  # Green for high activity
                elif fragment.activity_level > 0.1:
                    border_color = "#ffaa00"  # Orange for medium activity
                else:
                    border_color = "#666666"  # Gray for low activity

                css += f'''
        .{area_class} {{
            grid-area: {cell.css_grid_area};
            border: {self.border_width}px solid {border_color};
            border-radius: 4px;
            overflow: hidden;
            transition: all 0.3s ease;
            opacity: {max(0.3, fragment.activity_level)};
        }}

        .{area_class}:hover {{
            border-color: #ffffff;
            z-index: 10;
        }}
        '''

        return css

    def _generate_equal_grid_css(
            self,
            grid_columns: str,
            grid_rows: str,
            grid_areas: str,
            cells: List[LayoutCell]
    ) -> str:
        """Generate CSS for equal grid layout."""
        return self._generate_adaptive_grid_css(grid_columns, grid_rows, grid_areas, cells)

    def _generate_priority_based_css(
            self,
            grid_columns: str,
            grid_rows: str,
            grid_areas: str,
            cells: List[LayoutCell]
    ) -> str:
        """Generate CSS for priority-based layout."""
        return self._generate_adaptive_grid_css(grid_columns, grid_rows, grid_areas, cells)

    async def get_layout_stats(self) -> Dict[str, Any]:
        """Get current layout statistics."""
        if not self.current_layout:
            return {"status": "no_layout"}

        return {
            "timestamp": self.current_layout.timestamp.isoformat(),
            "total_fragments": len(self.current_layout.fragments),
            "active_fragments": len([f for f in self.current_layout.fragments if f.activity_level > 0.1]),
            "layout_efficiency": self.current_layout.layout_efficiency,
            "total_active_area": self.current_layout.total_active_area,
            "grid_size": f"{len(self.current_layout.grid_columns.split())}x{len(self.current_layout.grid_rows.split())}",
            "fragments": [
                {
                    "camera_id": f.camera_id,
                    "fragment_id": f.fragment_id,
                    "activity_level": f.activity_level,
                    "priority": f.priority,
                    "last_activity": f.last_activity.isoformat()
                }
                for f in self.current_layout.fragments
            ]
        }

    async def force_layout_recalculation(self) -> LayoutResult:
        """Force immediate layout recalculation."""
        logging.info("Forcing layout recalculation")
        return await self.calculate_layout()

    def get_current_layout(self) -> Optional[LayoutResult]:
        """Get current layout result."""
        return self.current_layout