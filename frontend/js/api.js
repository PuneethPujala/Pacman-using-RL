/**
 * API Client — handles all communication with the FastAPI backend.
 */

const API_BASE = 'http://localhost:8000';

async function request(path, options = {}) {
    const url = `${API_BASE}${path}`;
    const res = await fetch(url, {
        headers: { 'Content-Type': 'application/json', ...options.headers },
        ...options,
    });
    if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || 'API Error');
    }
    return res.json();
}

export const api = {
    // Layouts
    getLayouts: () => request('/api/layouts'),
    getLayout: (name) => request(`/api/layouts/${name}`),

    // Training
    startTraining: (params) =>
        request('/api/train', { method: 'POST', body: JSON.stringify(params) }),
    getTrainingStatus: () => request('/api/train/status'),
    stopTraining: () => request('/api/train/stop', { method: 'POST' }),

    // Play / Test
    startGame: (params) =>
        request('/api/play/start', { method: 'POST', body: JSON.stringify(params) }),
    playStep: () => request('/api/play/step'),
    getState: () => request('/api/play/state'),
    resetGame: () => request('/api/play/reset', { method: 'POST' }),

    // Models
    getModels: () => request('/api/models'),
    saveCurrentModel: () => request('/api/model/save-current', { method: 'POST' }),
    downloadModel: (name) => `${API_BASE}/api/model/download/${name}`,
};
