// static/js/timeline.js
class TimelineViewer {
    constructor() {
        this.autoRefresh = true;
        this.currentFilter = '';
        this.refreshInterval = 5000;
        this.startAutoRefresh();
    }

    async loadTimeline(filter = '') {
        const response = await fetch(`/api/timeline?filter=${filter}`);
        const data = await response.json();
        this.renderTimeline(data.events);
    }

    renderTimeline(events) {
        const container = document.querySelector('.timeline');
        container.innerHTML = '';

        events.forEach(event => {
            const item = this.createTimelineItem(event);
            container.appendChild(item);
        });
    }

    createTimelineItem(event) {
        const div = document.createElement('div');
        div.className = `timeline-item ${event.type}`;

        // Ikona na podstawie typu zdarzenia
        const icon = this.getEventIcon(event.type);

        div.innerHTML = `
            <div class="timeline-marker">${icon}</div>
            <div class="timeline-content">
                <div class="event-time">${this.formatTime(event.timestamp)}</div>
                <h3>${event.title}</h3>
                <div class="event-data">
                    ${this.renderEventData(event)}
                </div>
                ${event.screenshot ? `
                    <div class="screenshot-preview">
                        <img src="${event.screenshot}"
                             onclick="showFullscreen(this.src)"
                             alt="Zrzut ekranu">
                    </div>
                ` : ''}
                <div class="git-link">
                    <a href="/git/commit/${event.commit_hash}" target="_blank">
                        📋 Zobacz commit
                    </a>
                </div>
            </div>
        `;

        return div;
    }

    getEventIcon(type) {
        const icons = {
            'motion_detected': '🏃',
            'layout_change': '📐',
            'config_change': '⚙️',
            'camera_connected': '📹',
            'camera_disconnected': '📹❌',
            'performance_warning': '⚠️'
        };
        return icons[type] || '📝';
    }

    renderEventData(event) {
        let html = '<div class="changes-list">';

        if (event.changes) {
            html += '<h4>Zmiany:</h4><ul>';
            Object.entries(event.changes).forEach(([key, [from, to]]) => {
                html += `
                    <li>
                        <strong>${this.formatKey(key)}:</strong>
                        <span class="from">${from}</span> →
                        <span class="to">${to}</span>
                    </li>
                `;
            });
            html += '</ul>';
        }

        if (event.performance) {
            html += `
                <div class="performance-data">
                    <h4>Wydajność:</h4>
                    <span>CPU: ${event.performance.cpu}%</span>
                    <span>RAM: ${event.performance.memory}MB</span>
                    <span>FPS: ${event.performance.fps}</span>
                </div>
            `;
        }

        html += '</div>';
        return html;
    }
}