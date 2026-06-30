/**
 * GameCanvas — renders the Pacman game state on an HTML5 Canvas.
 * Features: smooth movement animation, ghost glow effects, animated Pacman mouth.
 */

const TILE = 24;  // Tile size in pixels
const GHOST_COLORS = ['#FF0000', '#FFB8FF', '#00FFFF', '#FFB852'];
const GHOST_NAMES = ['Blinky', 'Pinky', 'Inky', 'Clyde'];

export class GameCanvas {
    constructor(canvasElement) {
        this.canvas = canvasElement;
        this.ctx = canvasElement.getContext('2d');
        this.state = null;
        this.prevState = null;
        this.animProgress = 1; // 0→1 interpolation between states
        this.animFrame = 0;    // global frame counter for mouth animation etc
        this.showHeatmap = false;
        this._rafId = null;
        this._lastTime = 0;
        this.animDuration = 200; // ms per state transition
    }

    /**
     * Set a new game state snapshot. Triggers smooth animation from prev → new.
     */
    setState(snapshot) {
        if (this.state) {
            this.prevState = { ...this.state };
        }
        this.state = snapshot;
        this.animProgress = 0;

        // Resize canvas to fit the grid
        if (snapshot && snapshot.width && snapshot.height) {
            const dpr = window.devicePixelRatio || 1;
            this.canvas.width = snapshot.width * TILE * dpr;
            this.canvas.height = snapshot.height * TILE * dpr;
            this.canvas.style.width = `${snapshot.width * TILE}px`;
            this.canvas.style.height = `${snapshot.height * TILE}px`;
            this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        }

        if (!this._rafId) this._startLoop();
    }

    _startLoop() {
        this._lastTime = performance.now();
        const loop = (t) => {
            const dt = t - this._lastTime;
            this._lastTime = t;

            if (this.animProgress < 1) {
                this.animProgress = Math.min(1, this.animProgress + dt / this.animDuration);
            }
            this.animFrame++;
            this._draw();
            this._rafId = requestAnimationFrame(loop);
        };
        this._rafId = requestAnimationFrame(loop);
    }

    stop() {
        if (this._rafId) cancelAnimationFrame(this._rafId);
        this._rafId = null;
    }

    /**
     * Main draw routine.
     */
    _draw() {
        const { ctx, state } = this;
        if (!state) return;
        const w = state.width * TILE;
        const h = state.height * TILE;

        ctx.clearRect(0, 0, w, h);
        ctx.fillStyle = '#000';
        ctx.fillRect(0, 0, w, h);

        this._drawWalls(state);

        // Heatmap overlay (if enabled)
        if (this.showHeatmap && state.visitCounts) {
            this._drawHeatmap(state);
        }

        this._drawFood(state);
        this._drawCapsules(state);
        this._drawGhosts(state);
        this._drawPacman(state);
    }

    /**
     * Convert grid (x, y) to canvas pixel coords.
     * Note: In Pacman, (0,0) is bottom-left; canvas (0,0) is top-left.
     */
    _toPixel(x, y, height) {
        return [x * TILE + TILE / 2, (height - 1 - y) * TILE + TILE / 2];
    }

    // ---- Wall rendering ----
    _drawWalls(state) {
        const { ctx } = this;
        const { walls, width, height } = state;

        ctx.fillStyle = '#2121DE';
        ctx.strokeStyle = '#4444FF';
        ctx.lineWidth = 1;

        for (let x = 0; x < width; x++) {
            for (let y = 0; y < height; y++) {
                if (!walls[x]?.[y]) continue;
                const px = x * TILE;
                const py = (height - 1 - y) * TILE;

                // Rounded wall tile
                const r = 3;
                ctx.beginPath();
                ctx.roundRect(px + 1, py + 1, TILE - 2, TILE - 2, r);
                ctx.fill();
                ctx.stroke();
            }
        }
    }

    // ---- Food rendering ----
    _drawFood(state) {
        const { ctx } = this;
        const { food, width, height } = state;

        for (let x = 0; x < width; x++) {
            for (let y = 0; y < height; y++) {
                if (!food[x]?.[y]) continue;
                const [px, py] = this._toPixel(x, y, height);

                ctx.fillStyle = '#FFB8FF';
                ctx.beginPath();
                ctx.arc(px, py, 2.5, 0, Math.PI * 2);
                ctx.fill();
            }
        }
    }

    // ---- Capsules ----
    _drawCapsules(state) {
        const { ctx } = this;
        const { capsules, height } = state;

        const pulse = 0.7 + 0.3 * Math.sin(this.animFrame * 0.08);
        for (const [cx, cy] of capsules) {
            const [px, py] = this._toPixel(cx, cy, height);

            // Glow
            ctx.shadowColor = '#FFFFFF';
            ctx.shadowBlur = 8 * pulse;
            ctx.fillStyle = '#FFFFFF';
            ctx.beginPath();
            ctx.arc(px, py, 5 * pulse, 0, Math.PI * 2);
            ctx.fill();
            ctx.shadowBlur = 0;
        }
    }

    // ---- Pacman rendering with smooth movement ----
    _drawPacman(state) {
        const { ctx, prevState, animProgress } = this;
        const { pacman, height } = state;

        let x = pacman.x;
        let y = pacman.y;

        // Interpolate from previous position
        if (prevState && prevState.pacman && animProgress < 1) {
            const t = this._easeInOut(animProgress);
            x = prevState.pacman.x + (pacman.x - prevState.pacman.x) * t;
            y = prevState.pacman.y + (pacman.y - prevState.pacman.y) * t;
        }

        const [px, py] = this._toPixel(x, y, height);

        // Mouth animation
        const mouthAngle = 0.25 * Math.PI * Math.abs(Math.sin(this.animFrame * 0.15));
        let startAngle = mouthAngle;
        let endAngle = 2 * Math.PI - mouthAngle;

        // Rotate based on direction
        const dirAngles = { 'East': 0, 'North': -Math.PI / 2, 'West': Math.PI, 'South': Math.PI / 2 };
        const rot = dirAngles[pacman.direction] || 0;

        // Glow effect
        ctx.shadowColor = '#FFE033';
        ctx.shadowBlur = 12;

        ctx.fillStyle = '#FFE033';
        ctx.beginPath();
        ctx.moveTo(px, py);
        ctx.arc(px, py, TILE / 2 - 2, startAngle + rot, endAngle + rot);
        ctx.closePath();
        ctx.fill();

        ctx.shadowBlur = 0;
    }

    // ---- Ghost rendering ----
    _drawGhosts(state) {
        const { ctx, prevState, animProgress } = this;
        const { ghosts, height } = state;

        ghosts.forEach((ghost, idx) => {
            let x = ghost.x;
            let y = ghost.y;

            // Interpolate
            if (prevState && prevState.ghosts && prevState.ghosts[idx] && animProgress < 1) {
                const t = this._easeInOut(animProgress);
                x = prevState.ghosts[idx].x + (ghost.x - prevState.ghosts[idx].x) * t;
                y = prevState.ghosts[idx].y + (ghost.y - prevState.ghosts[idx].y) * t;
            }

            const [px, py] = this._toPixel(x, y, height);
            const color = ghost.scared ? '#0000FF' : (GHOST_COLORS[idx] || '#FF0000');

            // Glow
            ctx.shadowColor = color;
            ctx.shadowBlur = ghost.scared ? 6 : 10;

            // Ghost body (rounded top, wavy bottom)
            const r = TILE / 2 - 2;
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.arc(px, py - 2, r, Math.PI, 0);

            // Wavy bottom
            const wave = Math.sin(this.animFrame * 0.12) * 2;
            const segments = 3;
            const segW = (r * 2) / segments;
            for (let i = 0; i < segments; i++) {
                const sx = px - r + i * segW;
                const ex = sx + segW;
                const midX = (sx + ex) / 2;
                const midY = py + r - 2 + (i % 2 === 0 ? wave : -wave);
                ctx.quadraticCurveTo(midX, midY, ex, py + r - 4);
            }
            ctx.closePath();
            ctx.fill();

            // Eyes
            ctx.shadowBlur = 0;
            const eyeOffsetX = 3.5;
            const eyeY = py - 4;

            // White sclera
            ctx.fillStyle = '#FFFFFF';
            ctx.beginPath();
            ctx.ellipse(px - eyeOffsetX, eyeY, 3, 4, 0, 0, Math.PI * 2);
            ctx.fill();
            ctx.beginPath();
            ctx.ellipse(px + eyeOffsetX, eyeY, 3, 4, 0, 0, Math.PI * 2);
            ctx.fill();

            if (!ghost.scared) {
                // Pupils — look toward Pacman
                const dx = state.pacman.x - ghost.x;
                const dy = state.pacman.y - ghost.y;
                const dist = Math.sqrt(dx * dx + dy * dy) || 1;
                const pupilOff = 1.5;
                const pdx = (dx / dist) * pupilOff;
                const pdy = -(dy / dist) * pupilOff; // flip y

                ctx.fillStyle = '#2121DE';
                ctx.beginPath();
                ctx.arc(px - eyeOffsetX + pdx, eyeY + pdy, 1.5, 0, Math.PI * 2);
                ctx.fill();
                ctx.beginPath();
                ctx.arc(px + eyeOffsetX + pdx, eyeY + pdy, 1.5, 0, Math.PI * 2);
                ctx.fill();
            }
        });

        ctx.shadowBlur = 0;
    }

    // ---- Heatmap ----
    _drawHeatmap(state) {
        const { ctx } = this;
        const { visitCounts, width, height } = state;

        let maxVisit = 0;
        for (let x = 0; x < width; x++) {
            for (let y = 0; y < height; y++) {
                if (visitCounts[x]?.[y] > maxVisit) maxVisit = visitCounts[x][y];
            }
        }
        if (maxVisit === 0) return;

        for (let x = 0; x < width; x++) {
            for (let y = 0; y < height; y++) {
                const v = visitCounts[x]?.[y] || 0;
                if (v === 0) continue;
                const intensity = v / maxVisit;
                const [px, py] = this._toPixel(x, y, height);

                ctx.fillStyle = `rgba(16, 185, 129, ${intensity * 0.4})`;
                ctx.fillRect(px - TILE / 2, py - TILE / 2, TILE, TILE);
            }
        }
    }

    // ---- Utility ----
    _easeInOut(t) {
        return t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t;
    }
}
