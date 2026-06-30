/**
 * Simple Canvas chart renderer for training metrics.
 */

export class MetricsChart {
    constructor(canvas, options = {}) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.data = [];
        this.label = options.label || '';
        this.color = options.color || '#3b82f6';
        this.fillColor = options.fillColor || 'rgba(59, 130, 246, 0.1)';
        this.maxPoints = options.maxPoints || 500;
    }

    setData(newData) {
        this.data = newData.slice(-this.maxPoints);
        this.draw();
    }

    draw() {
        const { canvas, ctx, data, color, fillColor, label } = this;
        const dpr = window.devicePixelRatio || 1;
        const rect = canvas.getBoundingClientRect();
        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;
        ctx.scale(dpr, dpr);

        const w = rect.width;
        const h = rect.height;
        const pad = { top: 24, right: 12, bottom: 20, left: 50 };
        const plotW = w - pad.left - pad.right;
        const plotH = h - pad.top - pad.bottom;

        ctx.clearRect(0, 0, w, h);

        if (data.length < 2) {
            ctx.fillStyle = '#64748b';
            ctx.font = '12px Inter, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('Waiting for data…', w / 2, h / 2);
            return;
        }

        // Compute range
        let minVal = Math.min(...data);
        let maxVal = Math.max(...data);
        if (minVal === maxVal) { minVal -= 1; maxVal += 1; }
        const range = maxVal - minVal;

        // Title
        ctx.fillStyle = '#94a3b8';
        ctx.font = '600 11px Inter, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText(label, pad.left, 14);

        // Y-axis labels
        ctx.fillStyle = '#475569';
        ctx.font = '10px JetBrains Mono, monospace';
        ctx.textAlign = 'right';
        for (let i = 0; i <= 4; i++) {
            const val = minVal + (range * i) / 4;
            const y = pad.top + plotH - (plotH * i) / 4;
            ctx.fillText(val.toFixed(val > 100 ? 0 : 1), pad.left - 6, y + 3);

            // Grid line
            ctx.strokeStyle = 'rgba(255,255,255,0.03)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(pad.left, y);
            ctx.lineTo(pad.left + plotW, y);
            ctx.stroke();
        }

        // Plot line
        ctx.beginPath();
        const stepX = plotW / (data.length - 1);
        for (let i = 0; i < data.length; i++) {
            const x = pad.left + i * stepX;
            const y = pad.top + plotH - ((data[i] - minVal) / range) * plotH;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.lineJoin = 'round';
        ctx.stroke();

        // Fill area
        ctx.lineTo(pad.left + (data.length - 1) * stepX, pad.top + plotH);
        ctx.lineTo(pad.left, pad.top + plotH);
        ctx.closePath();
        ctx.fillStyle = fillColor;
        ctx.fill();

        // Current value badge
        const lastVal = data[data.length - 1];
        ctx.fillStyle = color;
        ctx.font = '600 11px JetBrains Mono, monospace';
        ctx.textAlign = 'right';
        ctx.fillText(lastVal.toFixed(2), w - pad.right, 14);
    }
}
