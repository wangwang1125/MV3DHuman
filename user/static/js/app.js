/* ═══════════════════════════════════════════════════════
   MH3D 用户端 — 前端交互逻辑
   WebSocket 通信、语音引导、摄像头控制、文件上传、
   任务状态管理、3D 模型展示
   ═══════════════════════════════════════════════════════ */

const VIEW_ORDER = ["front", "right", "back", "left"];
const VIEW_LABELS = { front: "正面", right: "右侧", back: "背面", left: "左侧", unknown: "未知" };

let ws = null;
let currentMode = "import";
let currentTaskId = null;
let capturedSent = {};

// ─────── 初始化 ───────
document.addEventListener("DOMContentLoaded", () => {
    checkCloudHealth();
    setInterval(checkCloudHealth, 30000);
});

// ─────── 云端健康检查 ───────
async function checkCloudHealth() {
    const dot = document.getElementById("cloudDot");
    const text = document.getElementById("cloudText");
    try {
        const resp = await fetch("/api/health");
        const data = await resp.json();
        if (data.status === "healthy") {
            dot.className = "dot online";
            text.textContent = "云端已连接";
        } else {
            dot.className = "dot offline";
            text.textContent = "云端连接异常";
        }
    } catch {
        dot.className = "dot offline";
        text.textContent = "云端不可达";
    }
}

// ─────── Tab 切换 ───────
function switchTab(tabId, btnEl) {
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
    btnEl.classList.add("active");
    document.getElementById(tabId).classList.add("active");
    currentMode = tabId === "tab-import" ? "import" : "capture";
}

function switchPreviewTab(which, btnEl) {
    document.querySelectorAll(".preview-tab").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".preview-content").forEach(c => c.classList.remove("active"));
    btnEl.classList.add("active");
    document.getElementById("preview-" + which).classList.add("active");
}

// ═══════════════════ 导入模式 ═══════════════════

function triggerUpload(view) {
    document.getElementById("file-" + view).click();
}

async function handleFileSelect(view, input) {
    const file = input.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
        const img = document.getElementById("preview-upload-" + view);
        img.src = e.target.result;
        img.style.display = "block";
        document.getElementById("ph-" + view).style.display = "none";

        setOriginalPreview(view, e.target.result);
    };
    reader.readAsDataURL(file);

    const formData = new FormData();
    formData.append("view", view);
    formData.append("file", file);

    try {
        await fetch("/upload", { method: "POST", body: formData });
    } catch (err) {
        console.error("Upload failed:", err);
    }
}

function triggerUploadExtra(view, fileType) {
    document.getElementById("file-" + fileType + "-" + view).click();
}

async function handleFileSelectExtra(view, fileType, input) {
    const file = input.files[0];
    if (!file) return;

    const ph = document.getElementById("ph-" + fileType + "-" + view);
    if (ph) ph.textContent = file.name;

    const formData = new FormData();
    formData.append("view", view);
    formData.append("file", file);

    const url = fileType === "depth" ? "/upload/depth" : "/upload/normal";
    try {
        await fetch(url, { method: "POST", body: formData });
    } catch (err) {
        console.error("Upload " + fileType + " failed:", err);
    }
}

// ═══════════════════ 在线捕捉模式 ═══════════════════

async function startCamera() {
    try {
        await fetch("/api/camera/start", { method: "POST" });
        document.getElementById("btnStartCamera").disabled = true;
        document.getElementById("btnStopCamera").disabled = false;
        document.getElementById("btnManualCapture").disabled = false;
        document.getElementById("videoPlaceholder").style.display = "none";
        document.getElementById("videoStream").src = "/video_feed?" + Date.now();
        capturedSent = {};
        connectWebSocket();
    } catch (err) {
        console.error("Start camera failed:", err);
    }
}

async function stopCamera() {
    try {
        await fetch("/api/camera/stop", { method: "POST" });
    } catch {}
    document.getElementById("btnStartCamera").disabled = false;
    document.getElementById("btnStopCamera").disabled = true;
    document.getElementById("btnManualCapture").disabled = true;
    document.getElementById("videoStream").src = "";
    document.getElementById("videoPlaceholder").style.display = "flex";
    if (ws) { ws.close(); ws = null; }
}

async function resetCapture() {
    try {
        await fetch("/api/camera/reset", { method: "POST" });
    } catch {}
    capturedSent = {};
    VIEW_ORDER.forEach(v => {
        clearOriginalPreview(v);
        document.getElementById("prog-" + v).classList.remove("captured", "current");
    });
    updateCaptureStatus({ view: "unknown", standing_ok: false, required_view: "front", captured: {} });
}

function manualCapture() {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ action: "manual_capture" }));
    }
}

// ─────── WebSocket ───────
function connectWebSocket() {
    if (ws && ws.readyState <= WebSocket.OPEN) return;
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(proto + "//" + location.host + "/ws");

    ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        handleWsMessage(msg);
    };
    ws.onclose = () => {
        setTimeout(() => {
            if (document.getElementById("btnStopCamera").disabled === false) {
                connectWebSocket();
            }
        }, 2000);
    };
}

function handleWsMessage(msg) {
    switch (msg.type) {
        case "pose_status":
            updateCaptureStatus(msg);
            break;
        case "voice_guide":
            speak(msg.text);
            break;
        case "auto_capture":
            onAutoCapture(msg.view, msg.image_base64);
            break;
        case "task_status":
            onTaskStatus(msg);
            break;
        case "rembg_preview":
            onRembgPreview(msg.view, msg.image_base64);
            break;
    }
}

function updateCaptureStatus(data) {
    const viewEl = document.getElementById("statusView");
    viewEl.textContent = VIEW_LABELS[data.view] || "--";
    viewEl.style.color = data.view !== "unknown" ? "#16a34a" : "#6b7280";

    const standEl = document.getElementById("statusStanding");
    standEl.textContent = data.standing_ok ? "合格" : "调整中";
    standEl.style.color = data.standing_ok ? "#16a34a" : "#d97706";

    const reqView = data.required_view || null;
    document.getElementById("statusRequired").textContent = reqView ? VIEW_LABELS[reqView] : "全部完成";

    VIEW_ORDER.forEach(v => {
        const el = document.getElementById("prog-" + v);
        el.classList.remove("captured", "current");
        if (data.captured && data.captured[v]) {
            el.classList.add("captured");
        } else if (v === reqView) {
            el.classList.add("current");
        }
    });
}

function onAutoCapture(view, b64) {
    if (capturedSent[view]) return;
    capturedSent[view] = true;
    setOriginalPreview(view, "data:image/jpeg;base64," + b64);
    speak(VIEW_LABELS[view] + "已拍摄");
}

// ─────── 语音引导 ───────
let lastSpeakTime = 0;
function speak(text) {
    const enabled = document.getElementById("voiceEnabled");
    if (!enabled || !enabled.checked) return;
    const now = Date.now();
    if (now - lastSpeakTime < 2500) return;
    lastSpeakTime = now;

    if ("speechSynthesis" in window) {
        window.speechSynthesis.cancel();
        const utt = new SpeechSynthesisUtterance(text);
        utt.lang = "zh-CN";
        utt.rate = 1.1;
        window.speechSynthesis.speak(utt);
    }
}

// ═══════════════════ 生成 3D 模型 ═══════════════════

async function startGeneration() {
    const btn = document.getElementById("btnGenerate");
    btn.disabled = true;
    btn.textContent = "生成中...";

    showProgress("提交任务...");

    const params = {
        remove_background: document.getElementById("paramRembg").checked,
        seed: parseInt(document.getElementById("paramSeed").value) || 1234,
        octree_resolution: parseInt(document.getElementById("paramResolution").value) || 256,
        num_inference_steps: parseInt(document.getElementById("paramSteps").value) || 50,
        guidance_scale: parseFloat(document.getElementById("paramGuidance").value) || 5.0,
        num_chunks: parseInt(document.getElementById("paramChunks").value) || 200000,
        height_mm: parseFloat(document.getElementById("paramHeight").value) || 1750,
    };

    try {
        const resp = await fetch("/generate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ mode: currentMode === "import" ? "upload" : "capture", params }),
        });
        const data = await resp.json();

        if (data.error) {
            alert("错误: " + data.error);
            hideProgress();
            btn.disabled = false;
            btn.textContent = "生成 3D 模型";
            return;
        }

        currentTaskId = data.task_id;
        showProgress("任务已提交，等待处理...");

        if (!ws || ws.readyState !== WebSocket.OPEN) {
            connectWebSocket();
        }

        startPolling(data.task_id);
    } catch (err) {
        alert("请求失败: " + err);
        hideProgress();
        btn.disabled = false;
        btn.textContent = "生成 3D 模型";
    }
}

function startPolling(taskId) {
    const poll = async () => {
        try {
            const resp = await fetch("/task/" + taskId);
            const data = await resp.json();
            if (data.status === "completed") {
                onGenerationComplete(taskId, data);
                return;
            } else if (data.status === "error") {
                onGenerationError(data.message || "未知错误");
                return;
            }
            updateProgressText(data.status);
            setTimeout(poll, 3000);
        } catch {
            setTimeout(poll, 5000);
        }
    };
    setTimeout(poll, 3000);
}

function onTaskStatus(msg) {
    if (msg.task_id !== currentTaskId) return;
    if (msg.status === "completed") {
        onGenerationComplete(msg.task_id, msg);
    } else if (msg.status === "error") {
        onGenerationError(msg.message || "未知错误");
    } else {
        updateProgressText(msg.status);
    }
}

function onGenerationComplete(taskId, data) {
    hideProgress();
    const btn = document.getElementById("btnGenerate");
    btn.disabled = false;
    btn.textContent = "生成 3D 模型";

    showModelViewer(taskId);
    showDownload(taskId);
    loadPreviewImages(taskId);
    speak("3D 模型生成完成");
}

function onGenerationError(message) {
    hideProgress();
    const btn = document.getElementById("btnGenerate");
    btn.disabled = false;
    btn.textContent = "生成 3D 模型";
    alert("生成失败: " + message);
}

// ─────── Progress UI ───────
function showProgress(text) {
    const el = document.getElementById("generationProgress");
    el.style.display = "flex";
    document.getElementById("progressBar").className = "progress-bar indeterminate";
    document.getElementById("progressText").textContent = text;
}

function hideProgress() {
    document.getElementById("generationProgress").style.display = "none";
}

function updateProgressText(status) {
    const labels = { pending: "排队中...", processing: "GPU 推理中...", submitting: "提交中..." };
    document.getElementById("progressText").textContent = labels[status] || status;
}

// ─────── 3D Model Viewer (Three.js + OBJLoader) ───────
let threeScene = null, threeCamera = null, threeRenderer = null, threeControls = null;
let threeAnimId = null;

function initThreeViewer() {
    const canvas = document.getElementById("modelCanvas");
    const wrap = document.getElementById("modelViewerWrap");
    const w = wrap.clientWidth;
    const h = wrap.clientHeight || 500;

    threeScene = new THREE.Scene();
    threeScene.background = new THREE.Color(0xf5f5f5);

    threeCamera = new THREE.PerspectiveCamera(45, w / h, 0.1, 10000);
    threeCamera.position.set(0, 800, 2000);

    threeRenderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true });
    threeRenderer.setSize(w, h);
    threeRenderer.setPixelRatio(window.devicePixelRatio);

    threeControls = new THREE.OrbitControls(threeCamera, canvas);
    threeControls.enableDamping = true;
    threeControls.dampingFactor = 0.08;
    threeControls.target.set(0, 800, 0);

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    threeScene.add(ambientLight);
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight.position.set(500, 1500, 1000);
    threeScene.add(dirLight);
    const dirLight2 = new THREE.DirectionalLight(0xffffff, 0.3);
    dirLight2.position.set(-500, 500, -1000);
    threeScene.add(dirLight2);

    const gridHelper = new THREE.GridHelper(4000, 40, 0xcccccc, 0xe0e0e0);
    threeScene.add(gridHelper);

    function animate() {
        threeAnimId = requestAnimationFrame(animate);
        threeControls.update();
        threeRenderer.render(threeScene, threeCamera);
    }
    animate();

    window.addEventListener("resize", () => {
        const nw = wrap.clientWidth;
        const nh = wrap.clientHeight || 500;
        threeCamera.aspect = nw / nh;
        threeCamera.updateProjectionMatrix();
        threeRenderer.setSize(nw, nh);
    });
}

function showModelViewer(taskId) {
    document.getElementById("modelPlaceholder").style.display = "none";
    const canvas = document.getElementById("modelCanvas");
    canvas.style.display = "block";

    if (!threeScene) initThreeViewer();

    // Remove previous model objects
    const toRemove = [];
    threeScene.traverse(child => {
        if (child.isMesh) toRemove.push(child);
    });
    toRemove.forEach(obj => {
        obj.geometry.dispose();
        if (obj.material) {
            if (Array.isArray(obj.material)) obj.material.forEach(m => m.dispose());
            else obj.material.dispose();
        }
        threeScene.remove(obj);
    });

    const loader = new THREE.OBJLoader();
    const url = "/output/" + taskId + ".obj?" + Date.now();
    loader.load(url, (obj) => {
        const material = new THREE.MeshStandardMaterial({
            color: 0xb0b0b0,
            metalness: 0.1,
            roughness: 0.6,
            side: THREE.DoubleSide,
        });
        obj.traverse(child => {
            if (child.isMesh) {
                child.material = material;
            }
        });
        threeScene.add(obj);

        // Auto-fit camera
        const box = new THREE.Box3().setFromObject(obj);
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);
        threeCamera.position.set(center.x, center.y + maxDim * 0.3, center.z + maxDim * 1.5);
        threeControls.target.copy(center);
        threeControls.update();
    });
}

function showDownload(taskId) {
    const section = document.getElementById("downloadSection");
    section.style.display = "block";
    const btn = document.getElementById("btnDownload");
    btn.href = "/download/" + taskId;
    btn.download = taskId + ".obj";
}

// ─────── Preview images ───────
function setOriginalPreview(view, src) {
    const img = document.getElementById("view-original-" + view);
    img.src = src;
    img.classList.add("visible");
    document.getElementById("empty-" + view).style.display = "none";
}

function clearOriginalPreview(view) {
    const img = document.getElementById("view-original-" + view);
    img.src = "";
    img.classList.remove("visible");
    document.getElementById("empty-" + view).style.display = "";
}

function loadPreviewImages(taskId) {
    VIEW_ORDER.forEach(v => {
        const img = document.getElementById("view-rembg-" + v);
        img.src = "/preview/" + taskId + "/" + v + "?" + Date.now();
        img.classList.add("visible");
        document.getElementById("rembg-empty-" + v).style.display = "none";
    });
}

function onRembgPreview(view, b64) {
    const img = document.getElementById("view-rembg-" + view);
    img.src = "data:image/png;base64," + b64;
    img.classList.add("visible");
    document.getElementById("rembg-empty-" + view).style.display = "none";
}
