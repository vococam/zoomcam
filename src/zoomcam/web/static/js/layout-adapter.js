// static/js/layout-adapter.js
class LayoutAdapter {
    constructor() {
        this.gridContainer = document.querySelector('.zoomcam-grid');
        this.cameras = new Map();
        this.currentLayout = null;
        this.transitionDuration = 800; // ms
    }

    updateLayout(layoutConfig) {
        const newCSS = this.generateCSS(layoutConfig);

        // Płynne przejście
        this.gridContainer.style.transition = `all ${this.transitionDuration}ms ease-in-out`;

        // Aplikacja nowego CSS
        this.applyDynamicCSS(newCSS);

        // Aktualizacja pozycji kamer
        this.repositionCameras(layoutConfig.cameras);

        // Zapis do historii
        this.logLayoutChange(layoutConfig);
    }

    generateCSS(config) {
        const { grid, cameras } = config;

        let css = `
            .zoomcam-grid {
                display: grid;
                grid-template-columns: ${grid.columns};
                grid-template-rows: ${grid.rows};
                grid-template-areas: ${grid.areas};
                gap: ${grid.gap}px;
                width: 100vw;
                height: 100vh;
            }
        `;

        cameras.forEach((camera, id) => {
            css += `
                .camera-${id} {
                    grid-area: ${camera.gridArea};
                    border: ${camera.active ? '3px solid #00ff00' : '1px solid #666'};
                    opacity: ${camera.active ? '1' : '0.7'};
                    transition: all 0.3s ease;
                }
            `;

            if (camera.fragments > 1) {
                camera.fragmentAreas.forEach((area, index) => {
                    css += `
                        .camera-${id}-fragment-${index} {
                            grid-area: ${area};
                            border: 2px solid #ffaa00;
                        }
                    `;
                });
            }
        });

        return css;
    }

    applyDynamicCSS(cssText) {
        let styleSheet = document.getElementById('dynamic-layout-styles');
        if (!styleSheet) {
            styleSheet = document.createElement('style');
            styleSheet.id = 'dynamic-layout-styles';
            document.head.appendChild(styleSheet);
        }
        styleSheet.textContent = cssText;
    }
}

// Real-time monitor
class RealTimeMonitor {
    constructor() {
        this.eventSource = new EventSource('/api/events');
        this.layoutAdapter = new LayoutAdapter();
        this.setupEventHandlers();
    }

    setupEventHandlers() {
        this.eventSource.addEventListener('layout_change', (event) => {
            const data = JSON.parse(event.data);
            this.layoutAdapter.updateLayout(data.layout);
            this.updateAdminPanel(data);
        });

        this.eventSource.addEventListener('motion_detected', (event) => {
            const data = JSON.parse(event.data);
            this.highlightMotionZone(data.camera, data.zone);
        });

        this.eventSource.addEventListener('config_change', (event) => {
            const data = JSON.parse(event.data);
            this.updateConfigHistory(data);
        });
    }

    updateAdminPanel(data) {
        // Aktualizacja statystyk wydajności
        document.getElementById('cpu-usage').textContent = `${data.performance.cpu}%`;
        document.getElementById('memory-usage').textContent = `${data.performance.memory}MB`;
        document.getElementById('fps-counter').textContent = data.performance.fps;

        // Aktualizacja statusu kamer
        data.cameras.forEach((camera, id) => {
            document.getElementById(`source-res-${id}`).textContent = camera.source_resolution;
            document.getElementById(`target-res-${id}`).textContent = camera.target_resolution;
            document.getElementById(`scale-ratio-${id}`).textContent = `${camera.scale_ratio}x`;
        });

        // Aktualizacja podglądu CSS
        document.getElementById('current-css').value = data.generated_css;
    }
}