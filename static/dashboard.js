import * as THREE from 'https://esm.sh/three@0.160.0';
import { GLTFLoader } from 'https://esm.sh/three@0.160.0/examples/jsm/loaders/GLTFLoader.js';

// ═══════════════════════════════════════════
// DOM
// ═══════════════════════════════════════════
const connDot = document.getElementById('conn-dot');
const connLabel = document.getElementById('conn-label');
const predLabel = document.getElementById('prediction-label');
const valPitch = document.getElementById('val-pitch');
const valRoll = document.getElementById('val-roll');
const valConf = document.getElementById('val-conf');
const valBuf = document.getElementById('val-buf');

// ═══════════════════════════════════════════
// THREE.JS — Scene setup
// ═══════════════════════════════════════════
const container = document.getElementById('scene-container');
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.2;
renderer.outputColorSpace = THREE.SRGBColorSpace;
container.appendChild(renderer.domElement);

function resizeRenderer() {
    const w = container.clientWidth;
    const h = container.clientHeight;
    renderer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
}

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(30, 1, 0.01, 100);
camera.position.set(0, 0.05, 0.45);
camera.lookAt(0, 0, 0);

// Lighting — soft studio
scene.add(new THREE.AmbientLight(0xffffff, 0.8));
const key = new THREE.DirectionalLight(0xffffff, 1.5);
key.position.set(3, 5, 4);
scene.add(key);
const fill = new THREE.DirectionalLight(0xccccdd, 0.6);
fill.position.set(-3, 2, -2);
scene.add(fill);
const rim = new THREE.PointLight(0xffffff, 0.5, 20);
rim.position.set(0, -3, 3);
scene.add(rim);

// ── Load GLB model ──
let phoneModel = null;
const loader = new GLTFLoader();
loader.load('/static/iphone_16_plus_green.glb', (gltf) => {
    phoneModel = gltf.scene;

    // Auto-scale to fit nicely in view
    const box = new THREE.Box3().setFromObject(phoneModel);
    const size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);
    const scale = 0.2 / maxDim;  // normalize to ~0.2 units
    phoneModel.scale.setScalar(scale);

    // Center the model
    const center = box.getCenter(new THREE.Vector3());
    phoneModel.position.sub(center.multiplyScalar(scale));

    scene.add(phoneModel);
    console.log('✅ iPhone model loaded', size);
}, undefined, (err) => {
    console.error('❌ Failed to load GLB:', err);
    // Fallback: create a simple box phone
    const geo = new THREE.BoxGeometry(0.075, 0.155, 0.008);
    const mat = new THREE.MeshPhysicalMaterial({
        color: 0xf5f5f0,
        metalness: 0.15,
        roughness: 0.35,
    });
    phoneModel = new THREE.Mesh(geo, mat);
    scene.add(phoneModel);
});

// ═══════════════════════════════════════════
// CHART.JS — Debug charts
// ═══════════════════════════════════════════
const CHART_LEN = 200;
const labels = Array.from({ length: CHART_LEN }, (_, i) => i);

function makeDataset(label, color) {
    return {
        label,
        data: [],
        borderColor: color,
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0.3,
        fill: false,
    };
}

const orientChart = new Chart(document.getElementById('chart-orient'), {
    type: 'line',
    data: {
        labels,
        datasets: [
            makeDataset('Pitch', '#f1c40f'),
            makeDataset('Roll', '#e67e22'),
            makeDataset('Yaw', '#9b59b6'),
        ],
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        scales: {
            x: { display: false },
            y: { grid: { color: 'rgba(0,0,0,0.06)' }, ticks: { font: { size: 10 } } },
        },
        plugins: {
            legend: { position: 'top', labels: { boxWidth: 10, font: { size: 10 } } },
        },
    },
});

const accChart = new Chart(document.getElementById('chart-acc'), {
    type: 'line',
    data: {
        labels,
        datasets: [
            makeDataset('X', '#e74c3c'),
            makeDataset('Y', '#27ae60'),
            makeDataset('Z', '#2980b9'),
        ],
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        scales: {
            x: { display: false },
            y: { grid: { color: 'rgba(0,0,0,0.06)' }, ticks: { font: { size: 10 } } },
        },
        plugins: {
            legend: { position: 'top', labels: { boxWidth: 10, font: { size: 10 } } },
        },
    },
});

const gyroChart = new Chart(document.getElementById('chart-gyro'), {
    type: 'line',
    data: {
        labels,
        datasets: [
            makeDataset('X', '#e67e22'),
            makeDataset('Y', '#8e44ad'),
            makeDataset('Z', '#16a085'),
        ],
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        scales: {
            x: { display: false },
            y: { grid: { color: 'rgba(0,0,0,0.06)' }, ticks: { font: { size: 10 } } },
        },
        plugins: {
            legend: { position: 'top', labels: { boxWidth: 10, font: { size: 10 } } },
        },
    },
});

// ═══════════════════════════════════════════
// STATE
// ═══════════════════════════════════════════
const targetQuat = new THREE.Quaternion();
const currentQuat = new THREE.Quaternion();
let useFallbackEuler = false;
let targetPitch = 0, targetRoll = 0;
let currentPitch = 0, currentRoll = 0;
const LERP = 0.15;
let lastDataTime = 0;

// ═══════════════════════════════════════════
// POLLING
// ═══════════════════════════════════════════

// Poll /state for orientation + prediction
async function fetchState() {
    try {
        const res = await fetch('/state');
        if (!res.ok) return;
        const s = await res.json();

        // Use quaternion if available from orientation sensor
        if (s.has_orientation) {
            targetQuat.set(s.qx || 0, s.qy || 0, s.qz || 0, s.qw || 1);
            useFallbackEuler = false;
        } else {
            targetPitch = (s.pitch || 0) * Math.PI / 180;
            targetRoll = (s.roll || 0) * Math.PI / 180;
            useFallbackEuler = true;
        }

        const fatigued = s.prediction === 'Fatigued';

        // Body class drives the whole color scheme
        document.body.className = fatigued ? 'fatigued' : 'optimal';
        predLabel.textContent = fatigued ? 'FATIGUED' : 'OPTIMAL';

        // Stats
        valPitch.textContent = (s.pitch || 0).toFixed(1) + '°';
        valRoll.textContent = (s.roll || 0).toFixed(1) + '°';
        valConf.textContent = Math.round((s.confidence || 0) * 100) + '%';
        valBuf.textContent = (s.buffer_fill || 0) + '/' + (s.buffer_max || 128);

        // Connection
        if ((s.buffer_fill || 0) > 0 || s.has_orientation) {
            lastDataTime = Date.now();
            connDot.classList.add('connected');
            connLabel.textContent = 'Sensor connected';
        }
    } catch (e) {
        console.error("fetchState error:", e);
    }
}
setInterval(fetchState, 60);

// Poll /data for chart updates
async function fetchData() {
    try {
        const res = await fetch('/data');
        if (!res.ok) return;
        const d = await res.json();

        accChart.data.datasets[0].data = d.acc_x;
        accChart.data.datasets[1].data = d.acc_y;
        accChart.data.datasets[2].data = d.acc_z;
        accChart.data.labels = Array.from({ length: d.length }, (_, i) => i);
        accChart.update();

        gyroChart.data.datasets[0].data = d.gyro_x;
        gyroChart.data.datasets[1].data = d.gyro_y;
        gyroChart.data.datasets[2].data = d.gyro_z;
        gyroChart.data.labels = Array.from({ length: d.length }, (_, i) => i);
        gyroChart.update();

        if (d.orient_length > 0) {
            orientChart.data.datasets[0].data = d.orient_pitch;
            orientChart.data.datasets[1].data = d.orient_roll;
            orientChart.data.datasets[2].data = d.orient_yaw;
            orientChart.data.labels = Array.from({ length: d.orient_length }, (_, i) => i);
            orientChart.update();
        }
    } catch (e) {
        console.error("fetchData error:", e);
    }
}
setInterval(fetchData, 200);

// Stale check
setInterval(() => {
    if (Date.now() - lastDataTime > 3000 && lastDataTime > 0) {
        connDot.classList.remove('connected');
        connLabel.textContent = 'Sensor stale';
    }
}, 1000);

// ═══════════════════════════════════════════
// RENDER LOOP
// ═══════════════════════════════════════════
resizeRenderer();

function animate() {
    requestAnimationFrame(animate);

    if (phoneModel) {
        if (!useFallbackEuler) {
            // Smooth quaternion interpolation (slerp)
            currentQuat.slerp(targetQuat, LERP);
            phoneModel.quaternion.copy(currentQuat);
        } else {
            // Fallback: euler from atan2 pitch/roll
            currentPitch += (targetPitch - currentPitch) * LERP;
            currentRoll += (targetRoll - currentRoll) * LERP;
            phoneModel.rotation.x = currentPitch;
            phoneModel.rotation.z = currentRoll;
        }

        // Gentle idle sway when no data
        if (lastDataTime === 0) {
            const t = Date.now() * 0.001;
            phoneModel.rotation.x = Math.sin(t * 0.5) * 0.12;
            phoneModel.rotation.z = Math.sin(t * 0.35) * 0.08;
            phoneModel.rotation.y = Math.sin(t * 0.2) * 0.05;
        }
    }

    renderer.render(scene, camera);
}
animate();

window.addEventListener('resize', resizeRenderer);
