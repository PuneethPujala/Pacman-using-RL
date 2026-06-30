/**
 * Pacman RL Visualization Platform — Main Application
 *
 * Single-page application with hash-based routing.
 * Pages: Home, Training Dashboard, Testing Arena, Model Management.
 */

import { api } from './api.js';
import { MetricsChart } from './chart.js';
import { GameCanvas } from './gameCanvas.js';

// ===================================================================
// Router
// ===================================================================

const routes = {
    '': renderHome,
    'train': renderTraining,
    'arena': renderArena,
    'models': renderModels,
};

function navigate(page) {
    window.location.hash = page;
}

function getPage() {
    return window.location.hash.replace('#', '') || '';
}

function renderApp() {
    const app = document.getElementById('app');
    const page = getPage();
    const renderFn = routes[page] || renderHome;

    app.innerHTML = `
        ${renderNavbar(page)}
        <main class="page" id="pageContent"></main>
    `;

    renderFn(document.getElementById('pageContent'));

    // Set active nav link
    document.querySelectorAll('.nav-link').forEach(el => {
        el.classList.toggle('active', el.dataset.page === page);
    });
}

function renderNavbar(activePage) {
    return `
    <nav class="navbar">
        <a class="navbar-brand" onclick="location.hash=''">
            <span class="logo">🟡</span>
            <span>Pacman RL</span>
        </a>
        <div class="navbar-links">
            <button class="nav-link" data-page="" onclick="location.hash=''">Home</button>
            <button class="nav-link" data-page="train" onclick="location.hash='train'">Training</button>
            <button class="nav-link" data-page="arena" onclick="location.hash='arena'">Arena</button>
            <button class="nav-link" data-page="models" onclick="location.hash='models'">Models</button>
        </div>
    </nav>`;
}

// Start router
window.addEventListener('hashchange', renderApp);
window.addEventListener('DOMContentLoaded', renderApp);

// ===================================================================
// Page: Home
// ===================================================================

function renderHome(container) {
    container.innerHTML = `
    <div class="hero">
        <h1>
            <span class="gradient">Reinforcement Learning</span><br>
            Meets Pacman
        </h1>
        <p class="hero-subtitle">
            A research-grade visualization platform. Train Q-learning agents,
            watch them play in real-time with beautiful animations, and analyze
            their behavior through interactive dashboards.
        </p>
        <div class="hero-actions">
            <button class="btn btn-primary btn-lg" onclick="location.hash='train'">
                ⚡ Train an Agent
            </button>
            <button class="btn btn-secondary btn-lg" onclick="location.hash='arena'">
                🎮 Watch Agent Play
            </button>
        </div>
    </div>

    <div class="features-grid">
        <div class="card feature-card">
            <div class="feature-icon">📊</div>
            <h3>Live Training Metrics</h3>
            <p>Watch reward curves, exploration rates, and win rates update in real-time as your agent learns.</p>
        </div>
        <div class="card feature-card">
            <div class="feature-icon">🎨</div>
            <h3>Beautiful Visualization</h3>
            <p>Smooth Canvas animations, glowing ghosts, and pixel-perfect rendering bring the game to life.</p>
        </div>
        <div class="card feature-card">
            <div class="feature-icon">🔬</div>
            <h3>Research Tools</h3>
            <p>Q-value overlays, state visit heatmaps, and step-by-step execution for deep analysis.</p>
        </div>
        <div class="card feature-card">
            <div class="feature-icon">⚙️</div>
            <h3>Configurable Training</h3>
            <p>Adjust learning rate, discount factor, exploration rate, and choose from 24 different maps.</p>
        </div>
        <div class="card feature-card">
            <div class="feature-icon">💾</div>
            <h3>Model Management</h3>
            <p>Save, load, download, and compare trained models for reproducible experiments.</p>
        </div>
        <div class="card feature-card">
            <div class="feature-icon">🏎️</div>
            <h3>Speed Control</h3>
            <p>Watch at 1x for analysis or crank it up to 20x to see rapid gameplay. Step-by-step mode available.</p>
        </div>
    </div>
    `;
}


// ===================================================================
// Page: Training Dashboard
// ===================================================================

let _trainInterval = null;
let _charts = {};

function renderTraining(container) {
    container.innerHTML = `
    <div class="dashboard-header">
        <div>
            <h2>📊 Training Dashboard</h2>
            <p style="color:var(--text-secondary);font-size:0.875rem;margin-top:0.25rem">Configure and monitor your RL agent's training progress</p>
        </div>
    </div>

    <div class="card" style="margin-bottom:1.5rem">
        <div class="card-title">⚙️ Training Configuration</div>
        <div class="training-config">
            <div class="form-group">
                <label class="form-label">Episodes</label>
                <input class="input" id="cfgEpisodes" type="number" value="200" min="1" max="10000" style="width:90px">
            </div>
            <div class="form-group">
                <label class="form-label">Alpha (α)</label>
                <input class="input" id="cfgAlpha" type="number" value="0.5" step="0.05" min="0" max="1" style="width:80px">
            </div>
            <div class="form-group">
                <label class="form-label">Gamma (γ)</label>
                <input class="input" id="cfgGamma" type="number" value="0.8" step="0.05" min="0" max="1" style="width:80px">
            </div>
            <div class="form-group">
                <label class="form-label">Epsilon (ε)</label>
                <input class="input" id="cfgEpsilon" type="number" value="0.3" step="0.05" min="0" max="1" style="width:80px">
            </div>
            <div class="form-group">
                <label class="form-label">Layout</label>
                <select class="select" id="cfgLayout" style="width:160px">
                    <option value="mediumClassic">mediumClassic</option>
                    <option value="smallClassic">smallClassic</option>
                </select>
            </div>
            <div class="form-group">
                <label class="form-label">Ghosts</label>
                <select class="select" id="cfgGhostType" style="width:120px">
                    <option value="random">Random</option>
                    <option value="directional">Directional</option>
                </select>
            </div>
            <div class="form-group" style="justify-content:flex-end">
                <button class="btn btn-primary" id="btnStartTrain" onclick="window._startTraining()">
                    ⚡ Start Training
                </button>
            </div>
            <div class="form-group" style="justify-content:flex-end">
                <button class="btn btn-danger" id="btnStopTrain" onclick="window._stopTraining()" style="display:none">
                    ⏹ Stop
                </button>
            </div>
        </div>
    </div>

    <div class="metrics-row" id="metricsRow">
        <div class="card">
            <div class="stat-value blue" id="metEpisode">0</div>
            <div class="stat-label">Episode</div>
        </div>
        <div class="card">
            <div class="stat-value green" id="metWinRate">0%</div>
            <div class="stat-label">Win Rate</div>
        </div>
        <div class="card">
            <div class="stat-value purple" id="metAvgReward">0</div>
            <div class="stat-label">Avg Reward (last 10)</div>
        </div>
        <div class="card">
            <div class="stat-value amber" id="metEpsilon">0.30</div>
            <div class="stat-label">Epsilon</div>
        </div>
        <div class="card">
            <div class="stat-value" id="metStatus" style="font-size:0.9rem;padding-top:0.3rem">
                <span class="badge badge-blue">Idle</span>
            </div>
            <div class="stat-label">Status</div>
            <div class="progress-bar" id="trainProgress" style="margin-top:0.5rem">
                <div class="progress-fill" id="trainProgressFill" style="width:0%"></div>
            </div>
        </div>
    </div>

    <div class="charts-grid">
        <div class="card">
            <div class="card-title">📈 Episode Reward</div>
            <div class="chart-container"><canvas id="chartReward"></canvas></div>
        </div>
        <div class="card">
            <div class="card-title">🏆 Win Rate</div>
            <div class="chart-container"><canvas id="chartWinRate"></canvas></div>
        </div>
        <div class="card">
            <div class="card-title">🎯 Exploration Rate (ε)</div>
            <div class="chart-container"><canvas id="chartEpsilon"></canvas></div>
        </div>
        <div class="card">
            <div class="card-title">👣 Steps per Episode</div>
            <div class="chart-container"><canvas id="chartSteps"></canvas></div>
        </div>
    </div>
    `;

    // Load available layouts into dropdown
    api.getLayouts().then(data => {
        const sel = document.getElementById('cfgLayout');
        if (sel && data.layouts) {
            sel.innerHTML = data.layouts.map(l =>
                `<option value="${l}" ${l === 'smallClassic' ? 'selected' : ''}>${l}</option>`
            ).join('');
        }
    }).catch(() => { });

    // Initialize charts
    _charts.reward = new MetricsChart(document.getElementById('chartReward'),
        { label: 'Reward', color: '#3b82f6', fillColor: 'rgba(59,130,246,0.08)' });
    _charts.winRate = new MetricsChart(document.getElementById('chartWinRate'),
        { label: 'Win Rate', color: '#10b981', fillColor: 'rgba(16,185,129,0.08)' });
    _charts.epsilon = new MetricsChart(document.getElementById('chartEpsilon'),
        { label: 'Epsilon', color: '#f59e0b', fillColor: 'rgba(245,158,11,0.08)' });
    _charts.steps = new MetricsChart(document.getElementById('chartSteps'),
        { label: 'Steps', color: '#8b5cf6', fillColor: 'rgba(139,92,246,0.08)' });

    // Poll for existing training status
    _pollTrainingOnce();
}

function _pollTrainingOnce() {
    api.getTrainingStatus().then(m => {
        _updateTrainingUI(m);
        if (m.is_training) {
            _startTrainingPoll();
        }
    }).catch(() => { });
}

window._startTraining = async function () {
    const params = {
        episodes: parseInt(document.getElementById('cfgEpisodes')?.value || 200),
        alpha: parseFloat(document.getElementById('cfgAlpha')?.value || 0.5),
        gamma: parseFloat(document.getElementById('cfgGamma')?.value || 0.8),
        epsilon: parseFloat(document.getElementById('cfgEpsilon')?.value || 0.3),
        layout: document.getElementById('cfgLayout')?.value || 'smallClassic',
        ghostType: document.getElementById('cfgGhostType')?.value || 'random',
        numGhosts: 4,
    };

    try {
        await api.startTraining(params);
        _startTrainingPoll();
    } catch (e) {
        console.error('Training start error:', e);
    }
};

window._stopTraining = async function () {
    try { await api.stopTraining(); } catch (e) { }
};

function _startTrainingPoll() {
    if (_trainInterval) clearInterval(_trainInterval);
    const startBtn = document.getElementById('btnStartTrain');
    const stopBtn = document.getElementById('btnStopTrain');
    if (startBtn) startBtn.disabled = true;
    if (stopBtn) stopBtn.style.display = '';

    _trainInterval = setInterval(async () => {
        try {
            const m = await api.getTrainingStatus();
            _updateTrainingUI(m);
            if (!m.is_training) {
                clearInterval(_trainInterval);
                _trainInterval = null;
                if (startBtn) startBtn.disabled = false;
                if (stopBtn) stopBtn.style.display = 'none';
            }
        } catch {
            clearInterval(_trainInterval);
            _trainInterval = null;
        }
    }, 500);
}

function _updateTrainingUI(m) {
    const el = (id) => document.getElementById(id);

    el('metEpisode') && (el('metEpisode').textContent = `${m.current_episode}/${m.total_episodes}`);

    if (m.win_rate_history?.length) {
        const wr = m.win_rate_history[m.win_rate_history.length - 1];
        el('metWinRate') && (el('metWinRate').textContent = `${(wr * 100).toFixed(1)}%`);
    }

    if (m.episode_rewards?.length) {
        const last10 = m.episode_rewards.slice(-10);
        const avg = last10.reduce((a, b) => a + b, 0) / last10.length;
        el('metAvgReward') && (el('metAvgReward').textContent = avg.toFixed(1));
    }

    if (m.epsilon_history?.length) {
        el('metEpsilon') && (el('metEpsilon').textContent = m.epsilon_history[m.epsilon_history.length - 1].toFixed(4));
    }

    if (m.is_training) {
        el('metStatus') && (el('metStatus').innerHTML = '<span class="badge badge-green">Training</span>');
    } else if (m.training_complete) {
        el('metStatus') && (el('metStatus').innerHTML = '<span class="badge badge-blue">Complete</span>');
    }

    const pct = m.total_episodes > 0 ? (m.current_episode / m.total_episodes * 100) : 0;
    el('trainProgressFill') && (el('trainProgressFill').style.width = `${pct}%`);

    // Update charts
    if (_charts.reward && m.episode_rewards?.length) _charts.reward.setData(m.episode_rewards);
    if (_charts.winRate && m.win_rate_history?.length) _charts.winRate.setData(m.win_rate_history);
    if (_charts.epsilon && m.epsilon_history?.length) _charts.epsilon.setData(m.epsilon_history);
    if (_charts.steps && m.steps_per_episode?.length) _charts.steps.setData(m.steps_per_episode);
}


// ===================================================================
// Page: Testing Arena
// ===================================================================

let _gameCanvas = null;
let _playInterval = null;
let _playSpeed = 300; // ms between steps
let _stepMode = false;

function renderArena(container) {
    container.innerHTML = `
    <div class="dashboard-header">
        <div>
            <h2>🎮 Testing Arena</h2>
            <p style="color:var(--text-secondary);font-size:0.875rem;margin-top:0.25rem">Watch your trained agent navigate the maze</p>
        </div>
    </div>

    <div class="arena-layout">
        <div>
            <div class="game-canvas-wrapper" id="canvasWrapper">
                <canvas id="gameCanvas"></canvas>
                <div class="game-overlay" id="gameOverlay" style="display:none">
                    <div class="item"><span class="label">Score</span><span class="value" id="ovScore">0</span></div>
                    <div class="item"><span class="label">Step</span><span class="value" id="ovStep">0</span></div>
                    <div class="item"><span class="label">Food</span><span class="value" id="ovFood">0</span></div>
                </div>
            </div>
        </div>

        <div class="arena-sidebar">
            <div class="card">
                <div class="card-title">🕹️ Game Controls</div>
                <div class="form-group" style="margin-bottom:0.75rem">
                    <label class="form-label">Layout</label>
                    <select class="select" id="arenaLayout" style="width:100%">
                        <option value="smallClassic">smallClassic</option>
                    </select>
                </div>
                <div class="form-group" style="margin-bottom:0.75rem">
                    <label class="form-label">Ghost Type</label>
                    <select class="select" id="arenaGhostType" style="width:100%">
                        <option value="random">Random</option>
                        <option value="directional">Directional</option>
                    </select>
                </div>
                <div class="form-group" style="margin-bottom:0.75rem">
                    <label class="form-label">Model (Weights)</label>
                    <select class="select" id="arenaModel" style="width:100%">
                        <option value="">Current Trained Model</option>
                    </select>
                </div>
                <div class="controls-row">
                    <button class="btn btn-primary" id="btnPlay" onclick="window._arenaPlay()">▶ Play</button>
                    <button class="btn btn-secondary" id="btnPause" onclick="window._arenaPause()" style="display:none">⏸ Pause</button>
                    <button class="btn btn-secondary" id="btnStep" onclick="window._arenaStep()">⏭ Step</button>
                    <button class="btn btn-secondary" onclick="window._arenaReset()">↺ Reset</button>
                </div>
                <div class="form-group" style="margin-bottom:0.75rem">
                    <label class="form-label">Speed: <span id="speedLabel">300ms</span></label>
                    <div class="slider-container">
                        <span style="font-size:0.75rem;color:var(--text-muted)">Slow</span>
                        <input type="range" id="speedSlider" min="30" max="1000" value="300" oninput="window._setSpeed(this.value)">
                        <span style="font-size:0.75rem;color:var(--text-muted)">Fast</span>
                    </div>
                </div>
                <label style="display:flex;align-items:center;gap:0.5rem;font-size:0.85rem;color:var(--text-secondary);cursor:pointer">
                    <input type="checkbox" id="chkHeatmap" onchange="window._toggleHeatmap(this.checked)">
                    Show visit heatmap
                </label>
            </div>

            <div class="card">
                <div class="card-title">📊 Q-Values</div>
                <div class="q-values-display" id="qValuesDisplay">
                    <div class="q-val"><span class="dir">North</span><span class="val">—</span></div>
                    <div class="q-val"><span class="dir">South</span><span class="val">—</span></div>
                    <div class="q-val"><span class="dir">East</span><span class="val">—</span></div>
                    <div class="q-val"><span class="dir">West</span><span class="val">—</span></div>
                </div>
            </div>

            <div class="card">
                <div class="card-title">📈 Game Stats</div>
                <div class="grid-2">
                    <div>
                        <div class="stat-value blue" id="arenaScore" style="font-size:1.2rem">0</div>
                        <div class="stat-label">Score</div>
                    </div>
                    <div>
                        <div class="stat-value purple" id="arenaStep" style="font-size:1.2rem">0</div>
                        <div class="stat-label">Steps</div>
                    </div>
                    <div>
                        <div class="stat-value amber" id="arenaFood" style="font-size:1.2rem">0</div>
                        <div class="stat-label">Food Left</div>
                    </div>
                    <div>
                        <div class="stat-value" id="arenaStatus" style="font-size:1.2rem">
                            <span class="badge badge-blue">Ready</span>
                        </div>
                        <div class="stat-label">Status</div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    `;

    // Load layouts
    api.getLayouts().then(data => {
        const sel = document.getElementById('arenaLayout');
        if (sel && data.layouts) {
            sel.innerHTML = data.layouts.map(l =>
                `<option value="${l}" ${l === 'smallClassic' ? 'selected' : ''}>${l}</option>`
            ).join('');
        }
    }).catch(() => { });

    // Load saved models
    api.getModels().then(data => {
        const sel = document.getElementById('arenaModel');
        if (sel && data.models) {
            sel.innerHTML = '<option value="">Current Trained Model</option>' + data.models.map(m =>
                `<option value="${m.name}">${m.name}</option>`
            ).join('');
        }
    }).catch(() => { });

    // Initialize canvas renderer
    _gameCanvas = new GameCanvas(document.getElementById('gameCanvas'));
}

window._arenaPlay = async function () {
    // If no game started yet, start one
    if (!_gameCanvas?.state || _gameCanvas.state.error) {
        await _arenaStartGame();
    }

    if (_playInterval) return;

    document.getElementById('btnPlay').style.display = 'none';
    document.getElementById('btnPause').style.display = '';
    document.getElementById('arenaStatus').innerHTML = '<span class="badge badge-green">Playing</span>';
    document.getElementById('gameOverlay').style.display = '';

    _playInterval = setInterval(async () => {
        await _arenaDoStep();

        if (_gameCanvas?.state?.isWin || _gameCanvas?.state?.isLose) {
            window._arenaPause();
            _showGameOverBanner();
        }
    }, _playSpeed);
};

window._arenaPause = function () {
    if (_playInterval) {
        clearInterval(_playInterval);
        _playInterval = null;
    }
    const playBtn = document.getElementById('btnPlay');
    const pauseBtn = document.getElementById('btnPause');
    if (playBtn) playBtn.style.display = '';
    if (pauseBtn) pauseBtn.style.display = 'none';
    document.getElementById('arenaStatus').innerHTML = '<span class="badge badge-amber">Paused</span>';
};

window._arenaStep = async function () {
    if (!_gameCanvas?.state || _gameCanvas.state.error) {
        await _arenaStartGame();
    }
    window._arenaPause();
    await _arenaDoStep();
    if (_gameCanvas?.state?.isWin || _gameCanvas?.state?.isLose) {
        _showGameOverBanner();
    }
};

window._arenaReset = async function () {
    window._arenaPause();
    // Remove game over banner if present
    const banner = document.querySelector('.game-over-banner');
    if (banner) banner.remove();

    await _arenaStartGame();
};

window._setSpeed = function (val) {
    _playSpeed = 1030 - parseInt(val); // Invert so right = faster
    document.getElementById('speedLabel').textContent = `${_playSpeed}ms`;

    // If currently playing, restart interval with new speed
    if (_playInterval) {
        clearInterval(_playInterval);
        _playInterval = setInterval(async () => {
            await _arenaDoStep();
            if (_gameCanvas?.state?.isWin || _gameCanvas?.state?.isLose) {
                window._arenaPause();
                _showGameOverBanner();
            }
        }, _playSpeed);
    }

    // Update canvas animation duration
    if (_gameCanvas) _gameCanvas.animDuration = Math.min(_playSpeed * 0.8, 400);
};

window._toggleHeatmap = function (enabled) {
    if (_gameCanvas) _gameCanvas.showHeatmap = enabled;
};

async function _arenaStartGame() {
    const layout = document.getElementById('arenaLayout')?.value || 'smallClassic';
    const ghostType = document.getElementById('arenaGhostType')?.value || 'random';
    const modelName = document.getElementById('arenaModel')?.value || '';
    const modelPath = modelName ? modelName : null;

    try {
        const snapshot = await api.startGame({ layout, ghostType, numGhosts: 4, modelPath });
        _gameCanvas.setState(snapshot);
        _updateArenaStats(snapshot);
        document.getElementById('gameOverlay').style.display = '';
        document.getElementById('arenaStatus').innerHTML = '<span class="badge badge-blue">Ready</span>';
    } catch (e) {
        console.error('Failed to start game:', e);
    }
}

async function _arenaDoStep() {
    try {
        const snapshot = await api.playStep();
        _gameCanvas.setState(snapshot);
        _updateArenaStats(snapshot);
    } catch (e) {
        console.error('Step error:', e);
    }
}

function _updateArenaStats(s) {
    const el = (id) => document.getElementById(id);
    el('arenaScore') && (el('arenaScore').textContent = s.score || 0);
    el('arenaStep') && (el('arenaStep').textContent = s.step || 0);
    el('arenaFood') && (el('arenaFood').textContent = s.numFood ?? 0);
    el('ovScore') && (el('ovScore').textContent = s.score || 0);
    el('ovStep') && (el('ovStep').textContent = s.step || 0);
    el('ovFood') && (el('ovFood').textContent = s.numFood ?? 0);

    // Update Q-values
    if (s.qValues) {
        const qDiv = document.getElementById('qValuesDisplay');
        if (qDiv) {
            const dirs = ['North', 'South', 'East', 'West'];
            const vals = dirs.map(d => ({ dir: d, val: s.qValues[d] || 0 }));
            const maxVal = Math.max(...vals.map(v => v.val));

            qDiv.innerHTML = vals.map(v => {
                const isBest = v.val === maxVal && v.val !== 0;
                return `<div class="q-val ${isBest ? 'best' : ''}">
                    <span class="dir">${v.dir}</span>
                    <span class="val">${v.val.toFixed(2)}</span>
                </div>`;
            }).join('');
        }
    }
}

function _showGameOverBanner() {
    const s = _gameCanvas?.state;
    if (!s) return;

    const wrapper = document.getElementById('canvasWrapper');
    if (!wrapper) return;

    // Remove existing banner if any
    const existing = wrapper.querySelector('.game-over-banner');
    if (existing) existing.remove();

    const isWin = s.isWin;
    const banner = document.createElement('div');
    banner.className = `game-over-banner ${isWin ? 'win' : 'lose'}`;
    banner.innerHTML = `
        <h2>${isWin ? '🏆 Victory!' : '💀 Game Over'}</h2>
        <p>Final Score: ${s.score} | Steps: ${s.step}</p>
        <button class="btn ${isWin ? 'btn-success' : 'btn-primary'}" onclick="window._arenaReset()">Play Again</button>
    `;
    wrapper.appendChild(banner);

    document.getElementById('arenaStatus').innerHTML = isWin
        ? '<span class="badge badge-green">Won!</span>'
        : '<span class="badge badge-red">Lost</span>';
}


// ===================================================================
// Page: Model Management
// ===================================================================

function renderModels(container) {
    container.innerHTML = `
    <div class="dashboard-header">
        <div>
            <h2>💾 Model Management</h2>
            <p style="color:var(--text-secondary);font-size:0.875rem;margin-top:0.25rem">Save, download, and manage trained Q-table models</p>
        </div>
        <div style="display:flex;gap:0.5rem">
            <button class="btn btn-primary" onclick="window._saveModel()">💾 Save Current Model</button>
            <button class="btn btn-secondary" onclick="window._refreshModels()">↻ Refresh</button>
        </div>
    </div>

    <div class="card">
        <div class="card-title">📁 Saved Models</div>
        <div class="model-list" id="modelList">
            <p style="color:var(--text-muted);font-size:0.875rem">Loading models...</p>
        </div>
    </div>
    `;

    window._refreshModels();
}

window._saveModel = async function () {
    try {
        const result = await api.saveCurrentModel();
        window._refreshModels();
    } catch (e) {
        console.error('Save model error:', e);
    }
};

window._refreshModels = async function () {
    const list = document.getElementById('modelList');
    if (!list) return;

    try {
        const data = await api.getModels();
        if (!data.models || data.models.length === 0) {
            list.innerHTML = '<p style="color:var(--text-muted);font-size:0.875rem">No saved models yet. Train an agent and save the model.</p>';
            return;
        }

        list.innerHTML = data.models.map(m => {
            const date = new Date(m.modified * 1000).toLocaleString();
            const sizeKB = (m.size / 1024).toFixed(1);
            return `
            <div class="model-item">
                <div class="model-info">
                    <span class="model-name">📄 ${m.name}</span>
                    <span class="model-meta">${date} · ${sizeKB} KB</span>
                </div>
                <div style="display:flex;gap:0.5rem">
                    <a class="btn btn-secondary" href="${api.downloadModel(m.name)}" download style="font-size:0.8rem">
                        ⬇ Download
                    </a>
                </div>
            </div>`;
        }).join('');
    } catch (e) {
        list.innerHTML = '<p style="color:var(--text-muted);font-size:0.875rem">Could not load models. Is the backend running?</p>';
    }
};
