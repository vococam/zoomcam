/**
 * ZoomCam Setup Wizard JavaScript
 * ===============================
 *
 * Interactive setup wizard for camera detection, RTSP testing,
 * and system configuration with real-time feedback.
 */

class SetupWizard {
    constructor() {
        this.currentStep = 0;
        this.steps = ['welcome', 'cameras', 'config', 'system', 'complete'];
        this.detectedCameras = [];
        this.configuredCameras = [];

        this.initializeEventListeners();
    }

    initializeEventListeners() {
        // RTSP checkbox toggle
        const rtspCheckbox = document.getElementById('scanRtsp');
        if (rtspCheckbox) {
            rtspCheckbox.addEventListener('change', (e) => {
                const rtspUrls = document.getElementById('rtspUrls');
                rtspUrls.style.display = e.target.checked ? 'block' : 'none';
            });
        }

        // Auto-advance progress bar
        this.updateProgressBar();
    }

    updateProgressBar() {
        const progressBar = document.getElementById('progressBar');
        const progress = ((this.currentStep + 1) / this.steps.length) * 100;
        if (progressBar) {
            progressBar.style.width = `${progress}%`;
        }
    }

    nextStep() {
        if (this.currentStep < this.steps.length - 1) {
            // Hide current step
            const currentStepEl = document.getElementById(`step-${this.steps[this.currentStep]}`);
            if (currentStepEl) {
                currentStepEl.classList.remove('active');
            }

            // Show next step
            this.currentStep++;
            const nextStepEl = document.getElementById(`step-${this.steps[this.currentStep]}`);
            if (nextStepEl) {
                nextStepEl.classList.add('active');
            }

            this.updateProgressBar();
        }
    }

    previousStep() {
        if (this.currentStep > 0) {
            // Hide current step
            const currentStepEl = document.getElementById(`step-${this.steps[this.currentStep]}`);
            if (currentStepEl) {
                currentStepEl.classList.remove('active');
            }

            // Show previous step
            this.currentStep--;
            const prevStepEl = document.getElementById(`step-${this.steps[this.currentStep]}`);
            if (prevStepEl) {
                prevStepEl.classList.add('active');
            }

            this.updateProgressBar();
        }
    }

    async detectCameras() {
        const resultsDiv = document.getElementById('cameraResults');
        resultsDiv.innerHTML = '<div class="loading">🔍 Detecting cameras...</div>';

        try {
            const scanUsb = document.getElementById('scanUsb').checked;
            const scanRtsp = document.getElementById('scanRtsp').checked;
            const rtspUrls = scanRtsp ?
                document.getElementById('rtspUrlList').value
                    .split('\n')
                    .map(url => url.trim())
                    .filter(url => url.length > 0)
                : [];

            const response = await fetch('/api/setup/detect-cameras', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    scan_usb: scanUsb,
                    scan_rtsp: scanRtsp,
                    rtsp_urls: rtspUrls,
                    timeout_seconds: 10
                })
            });

            const data = await response.json();

            if (response.ok) {
                this.detectedCameras = data.cameras;
                this.displayDetectedCameras(data.cameras);

                if (data.cameras.length > 0) {
                    setTimeout(() => {
                        this.nextStep();
                        this.setupCameraConfiguration();
                    }, 2000);
                }
            } else {
                this.showError('Camera detection failed: ' + data.detail);
            }

        } catch (error) {
            this.showError('Network error during camera detection: ' + error.message);
        }
    }

    displayDetectedCameras(cameras) {
        const resultsDiv = document.getElementById('cameraResults');

        if (cameras.length === 0) {
            resultsDiv.innerHTML = `
                <div class="alert alert-error">
                    <strong>No cameras detected</strong><br>
                    Please check your camera connections and try again.
                    <br><br>
                    <button class="btn" onclick="wizard.detectCameras()">Try Again</button>
                    <button class="btn btn-secondary" onclick="wizard.nextStep()">Skip (Manual Setup)</button>
                </div>
            `;
            return;
        }

        let html = `
            <div class="alert alert-success">
                <strong>Found ${cameras.length} camera(s)!</strong>
            </div>
        `;

        cameras.forEach((camera, index) => {
            const statusIcon = camera.status === 'detected' ? '✅' : '❌';
            html += `
                <div class="camera-item">
                    <h4>${statusIcon} ${camera.name}</h4>
                    <p><strong>Source:</strong> ${camera.source}</p>
                    <p><strong>Type:</strong> ${camera.type.toUpperCase()}</p>
                    <p><strong>Resolution:</strong> ${camera.resolution || 'Unknown'}</p>
                    ${camera.fps ? `<p><strong>FPS:</strong> ${camera.fps}</p>` : ''}
                    <button class="btn btn-secondary" onclick="wizard.testCamera('${camera.source}', ${index})">
                        Test Camera
                    </button>
                    <span id="test-result-${index}"></span>
                </div>
            `;
        });

        resultsDiv.innerHTML = html;
    }

    async testCamera(source, index) {
        const resultSpan = document.getElementById(`test-result-${index}`);
        resultSpan.innerHTML = ' <em>Testing...</em>';

        try {
            const response = await fetch('/api/setup/test-camera', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    source: source,
                    timeout_seconds: 5
                })
            });

            const data = await response.json();

            if (data.success) {
                resultSpan.innerHTML = ' ✅ <span style="color: green;">Working!</span>';
                if (data.resolution) {
                    resultSpan.innerHTML += ` (${data.resolution})`;
                }
            } else {
                resultSpan.innerHTML = ` ❌ <span style="color: red;">Failed: ${data.error || data.message}</span>`;
            }

        } catch (error) {
            resultSpan.innerHTML = ` ❌ <span style="color: red;">Test failed: ${error.message}</span>`;
        }
    }

    setupCameraConfiguration() {
        const configDiv = document.getElementById('cameraConfig');

        if (this.detectedCameras.length === 0) {
            configDiv.innerHTML = `
                <div class="alert alert-error">
                    No cameras detected. You can add cameras manually later.
                </div>
            `;
            return;
        }

        let html = '<div class="camera-list">';

        this.detectedCameras.forEach((camera, index) => {
            html += `
                <div class="camera-item">
                    <h4>Configure ${camera.name}</h4>

                    <div class="form-group">
                        <label>Camera Name:</label>
                        <input type="text" id="name-${index}" value="${camera.name}"
                               placeholder="Enter camera name">
                    </div>

                    <div class="form-group">
                        <label>Source:</label>
                        <input type="text" id="source-${index}" value="${camera.source}" readonly>
                    </div>

                    <div class="form-group">
                        <label>Resolution:</label>
                        <select id="resolution-${index}">
                            <option value="auto" selected>Auto-detect</option>
                            <option value="1920x1080">1920x1080 (Full HD)</option>
                            <option value="1280x720">1280x720 (HD)</option>
                            <option value="640x480">640x480 (VGA)</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label>Zoom Level: <span id="zoom-value-${index}">3.0x</span></label>
                        <input type="range" id="zoom-${index}" min="1" max="10" step="0.1" value="3.0"
                               oninput="wizard.updateZoomDisplay(${index}, this.value)">
                    </div>

                    <div class="form-group">
                        <label>Max Motion Zones:</label>
                        <select id="fragments-${index}">
                            <option value="1">1 zone</option>
                            <option value="2" selected>2 zones</option>
                            <option value="3">3 zones</option>
                            <option value="4">4 zones</option>
                            <option value="5">5 zones</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label>Motion Sensitivity: <span id="sensitivity-value-${index}">30%</span></label>
                        <input type="range" id="sensitivity-${index}" min="0" max="100" value="30"
                               oninput="wizard.updateSensitivityDisplay(${index}, this.value)">
                    </div>

                    <div class="form-group">
                        <label>
                            <input type="checkbox" id="recording-${index}" checked>
                            Enable recording for this camera
                        </label>
                    </div>

                    <div class="form-group">
                        <label>
                            <input type="checkbox" id="enabled-${index}" checked>
                            Enable this camera
                        </label>
                    </div>
                </div>
            `;
        });

        html += '</div>';
        configDiv.innerHTML = html;
    }

    updateZoomDisplay(index, value) {
        const display = document.getElementById(`zoom-value-${index}`);
        if (display) {
            display.textContent = `${parseFloat(value).toFixed(1)}x`;
        }
    }

    updateSensitivityDisplay(index, value) {
        const display = document.getElementById(`sensitivity-value-${index}`);
        if (display) {
            display.textContent = `${value}%`;
        }
    }

    async configureSystem() {
        // Collect camera configurations
        const cameras = [];

        this.detectedCameras.forEach((camera, index) => {
            const enabledCheckbox = document.getElementById(`enabled-${index}`);
            if (enabledCheckbox && enabledCheckbox.checked) {
                cameras.push({
                    name: document.getElementById(`name-${index}`).value,
                    source: document.getElementById(`source-${index}`).value,
                    resolution: document.getElementById(`resolution-${index}`).value,
                    zoom: parseFloat(document.getElementById(`zoom-${index}`).value),
                    max_fragments: parseInt(document.getElementById(`fragments-${index}`).value),
                    recording_enabled: document.getElementById(`recording-${index}`).checked,
                    motion_sensitivity: parseFloat(document.getElementById(`sensitivity-${index}`).value) / 100
                });
            }
        });

        if (cameras.length === 0) {
            this.showError('Please enable at least one camera.');
            return;
        }

        try {
            // Configure cameras
            const response = await fetch('/api/setup/configure-cameras', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(cameras)
            });

            const data = await response.json();

            if (response.ok) {
                this.configuredCameras = cameras;
                this.nextStep();
            } else {
                this.showError('Camera configuration failed: ' + data.detail);
            }

        } catch (error) {
            this.showError('Network error during camera configuration: ' + error.message);
        }
    }

    async completeSetup() {
        const systemConfig = {
            display_resolution: document.getElementById('displayResolution').value,
            target_fps: parseInt(document.getElementById('targetFps').value),
            quality_priority: document.getElementById('qualityPriority').value,
            recording_enabled: document.getElementById('recordingEnabled').checked,
            git_logging_enabled: document.getElementById('gitLogging').checked
        };

        try {
            // Configure system
            const systemResponse = await fetch('/api/setup/configure-system', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(systemConfig)
            });

            if (!systemResponse.ok) {
                const error = await systemResponse.json();
                throw new Error(error.detail);
            }

            // Complete setup
            const completeResponse = await fetch('/api/setup/complete', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });

            if (completeResponse.ok) {
                this.nextStep();

                // Start auto-redirect timer
                let countdown = 10;
                const updateCountdown = () => {
                    const completeDiv = document.getElementById('step-complete');
                    const button = completeDiv.querySelector('button');
                    if (button) {
                        button.textContent = `Go to ZoomCam (${countdown})`;
                    }

                    countdown--;
                    if (countdown <= 0) {
                        this.goToMainInterface();
                    } else {
                        setTimeout(updateCountdown, 1000);
                    }
                };
                updateCountdown();

            } else {
                const error = await completeResponse.json();
                throw new Error(error.detail);
            }

        } catch (error) {
            this.showError('Setup completion failed: ' + error.message);
        }
    }

    goToMainInterface() {
        // Redirect to main interface
        window.location.href = '/';
    }

    showError(message) {
        // Find current step and show error
        const currentStepEl = document.getElementById(`step-${this.steps[this.currentStep]}`);
        if (currentStepEl) {
            // Remove existing error alerts
            const existingAlerts = currentStepEl.querySelectorAll('.alert-error');
            existingAlerts.forEach(alert => alert.remove());

            // Add new error alert
            const errorDiv = document.createElement('div');
            errorDiv.className = 'alert alert-error';
            errorDiv.innerHTML = `<strong>Error:</strong> ${message}`;

            // Insert at the beginning of the step
            currentStepEl.insertBefore(errorDiv, currentStepEl.firstChild);
        }
    }

    showSuccess(message) {
        // Find current step and show success message
        const currentStepEl = document.getElementById(`step-${this.steps[this.currentStep]}`);
        if (currentStepEl) {
            // Remove existing success alerts
            const existingAlerts = currentStepEl.querySelectorAll('.alert-success');
            existingAlerts.forEach(alert => alert.remove());

            // Add new success alert
            const successDiv = document.createElement('div');
            successDiv.className = 'alert alert-success';
            successDiv.innerHTML = `<strong>Success:</strong> ${message}`;

            // Insert at the beginning of the step
            currentStepEl.insertBefore(successDiv, currentStepEl.firstChild);
        }
    }
}

// RTSPTester class for advanced RTSP testing
class RTSPTester {
    constructor() {
        this.testResults = new Map();
    }

    async testMultipleRTSP(urls, timeout = 10) {
        const results = [];

        // Test URLs in parallel with limited concurrency
        const concurrency = 3;
        for (let i = 0; i < urls.length; i += concurrency) {
            const batch = urls.slice(i, i + concurrency);
            const batchPromises = batch.map(url => this.testSingleRTSP(url, timeout));
            const batchResults = await Promise.allSettled(batchPromises);

            batchResults.forEach((result, index) => {
                const url = batch[index];
                if (result.status === 'fulfilled') {
                    results.push(result.value);
                } else {
                    results.push({
                        url: url,
                        success: false,
                        error_message: result.reason.message
                    });
                }
            });
        }

        return results;
    }

    async testSingleRTSP(url, timeout = 10) {
        try {
            const response = await fetch('/api/setup/test-camera', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    source: url,
                    timeout_seconds: timeout
                })
            });

            return await response.json();

        } catch (error) {
            return {
                url: url,
                success: false,
                error_message: error.message
            };
        }
    }

    generateCommonRTSPUrls(ipRange = '192.168.1') {
        const commonPorts = [554, 8554, 1935];
        const commonPaths = ['/stream1', '/live', '/cam/realmonitor', '/h264'];
        const urls = [];

        // Generate URLs for common IP range
        for (let i = 1; i <= 254; i++) {
            const ip = `${ipRange}.${i}`;
            for (const port of commonPorts) {
                for (const path of commonPaths) {
                    urls.push(`rtsp://${ip}:${port}${path}`);
                }
            }
        }

        return urls;
    }
}

// Camera detection utilities
class CameraDetector {
    constructor() {
        this.detectionInProgress = false;
    }

    async detectUSBCameras() {
        const cameras = [];

        // This would typically be done on the backend
        // Frontend can only provide UI for the detection process

        try {
            const response = await fetch('/api/setup/detect-cameras', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    scan_usb: true,
                    scan_rtsp: false,
                    rtsp_urls: [],
                    timeout_seconds: 5
                })
            });

            const data = await response.json();
            return data.cameras.filter(cam => cam.type === 'usb');

        } catch (error) {
            console.error('USB camera detection failed:', error);
            return [];
        }
    }

    validateRTSPUrl(url) {
        const rtspPattern = /^rtsp:\/\/[\w\.-]+(:\d+)?\/.*$/;
        return rtspPattern.test(url);
    }

    suggestRTSPUrls() {
        return [
            'rtsp://admin:password@192.168.1.100:554/stream1',
            'rtsp://192.168.1.101:554/live',
            'rtsp://192.168.1.102:8554/cam/realmonitor',
            'rtsp://admin:admin@192.168.1.103:554/h264'
        ];
    }
}

// Performance monitoring during setup
class SetupMonitor {
    constructor() {
        this.startTime = Date.now();
        this.checkpoints = [];
    }

    checkpoint(name) {
        this.checkpoints.push({
            name: name,
            timestamp: Date.now(),
            elapsed: Date.now() - this.startTime
        });

        console.log(`Setup checkpoint: ${name} (${this.getElapsedTime()})`);
    }

    getElapsedTime() {
        const elapsed = Date.now() - this.startTime;
        return `${Math.round(elapsed / 1000)}s`;
    }

    getSummary() {
        return {
            total_time: this.getElapsedTime(),
            checkpoints: this.checkpoints,
            steps_completed: this.checkpoints.length
        };
    }
}

// Global instances
const wizard = new SetupWizard();
const rtspTester = new RTSPTester();
const cameraDetector = new CameraDetector();
const setupMonitor = new SetupMonitor();

// Global functions for HTML onclick handlers
window.nextStep = () => wizard.nextStep();
window.previousStep = () => wizard.previousStep();
window.detectCameras = () => wizard.detectCameras();
window.configureSystem = () => wizard.configureSystem();
window.completeSetup = () => wizard.completeSetup();
window.goToMainInterface = () => wizard.goToMainInterface();
window.wizard = wizard;

// Auto-save functionality
class AutoSave {
    constructor() {
        this.saveInterval = 30000; // 30 seconds
        this.lastSave = Date.now();
        this.startAutoSave();
    }

    startAutoSave() {
        setInterval(() => {
            this.saveProgress();
        }, this.saveInterval);
    }

    saveProgress() {
        const progress = {
            currentStep: wizard.currentStep,
            detectedCameras: wizard.detectedCameras,
            configuredCameras: wizard.configuredCameras,
            timestamp: Date.now()
        };

        try {
            localStorage.setItem('zoomcam_setup_progress', JSON.stringify(progress));
            console.log('Setup progress auto-saved');
        } catch (error) {
            console.warn('Failed to auto-save progress:', error);
        }
    }

    loadProgress() {
        try {
            const saved = localStorage.getItem('zoomcam_setup_progress');
            if (saved) {
                const progress = JSON.parse(saved);

                // Check if save is recent (within 1 hour)
                if (Date.now() - progress.timestamp < 3600000) {
                    return progress;
                }
            }
        } catch (error) {
            console.warn('Failed to load saved progress:', error);
        }

        return null;
    }

    clearProgress() {
        try {
            localStorage.removeItem('zoomcam_setup_progress');
        } catch (error) {
            console.warn('Failed to clear progress:', error);
        }
    }
}

// Initialize auto-save
const autoSave = new AutoSave();

// Load saved progress on page load
document.addEventListener('DOMContentLoaded', () => {
    setupMonitor.checkpoint('Page loaded');

    const savedProgress = autoSave.loadProgress();
    if (savedProgress && savedProgress.currentStep > 0) {
        const userWantsRestore = confirm(
            'Found previous setup progress. Would you like to continue where you left off?'
        );

        if (userWantsRestore) {
            wizard.currentStep = savedProgress.currentStep;
            wizard.detectedCameras = savedProgress.detectedCameras || [];
            wizard.configuredCameras = savedProgress.configuredCameras || [];

            // Navigate to saved step
            const steps = document.querySelectorAll('.step');
            steps.forEach((step, index) => {
                step.classList.toggle('active', index === wizard.currentStep);
            });

            wizard.updateProgressBar();

            // Restore camera configuration if on that step
            if (wizard.currentStep === 2 && wizard.detectedCameras.length > 0) {
                wizard.setupCameraConfiguration();
            }
        } else {
            autoSave.clearProgress();
        }
    }

    setupMonitor.checkpoint('Setup wizard initialized');
});

// Clear progress on successful completion
window.addEventListener('beforeunload', () => {
    if (wizard.currentStep === wizard.steps.length - 1) {
        autoSave.clearProgress();
    }
});

// Error handling and user feedback
window.addEventListener('error', (event) => {
    console.error('Setup wizard error:', event.error);

    const errorDetails = {
        message: event.error.message,
        stack: event.error.stack,
        step: wizard.currentStep,
        timestamp: Date.now()
    };

    // Try to send error report (optional)
    try {
        fetch('/api/setup/error-report', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(errorDetails)
        }).catch(() => {
            // Ignore if error reporting fails
        });
    } catch (e) {
        // Ignore if error reporting fails
    }
});

// Network status monitoring
window.addEventListener('online', () => {
    console.log('Network connection restored');
    wizard.showSuccess('Network connection restored');
});

window.addEventListener('offline', () => {
    console.log('Network connection lost');
    wizard.showError('Network connection lost. Some features may not work.');
});