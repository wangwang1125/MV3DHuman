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
// 生成等待计时（参考 gradio 的 time_meta 展示已等待时间）
let generationStartTime = 0;
let progressElapsedTimer = null;
let hasSeenProcessing = false;  // 一旦见过「推理中」就不再切回「排队中」，避免闪烁
let currentProgressStatus = "submitting";

// ─────── 初始化 ───────
document.addEventListener("DOMContentLoaded", () => {
    checkCloudHealth();
    setInterval(checkCloudHealth, 30000);
    setupDragDrop();
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
        if (img.parentElement) img.parentElement.classList.add("has-preview");

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

async function clearUpload(view, fileType) {
    const formData = new FormData();
    formData.append("view", view);
    formData.append("file_type", fileType);
    try {
        await fetch("/upload/clear_one", { method: "POST", body: formData });
    } catch (err) {
        console.error("Clear upload failed:", err);
        return;
    }
    if (fileType === "rgb") {
        const img = document.getElementById("preview-upload-" + view);
        const ph = document.getElementById("ph-" + view);
        if (img) { img.src = ""; img.style.display = "none"; if (img.parentElement) img.parentElement.classList.remove("has-preview"); }
        if (ph) ph.style.display = "";
        clearOriginalPreview(view);
        const fileInput = document.getElementById("file-" + view);
        if (fileInput) fileInput.value = "";
    } else {
        const previewImg = document.getElementById("preview-" + fileType + "-" + view);
        const ph = document.getElementById("ph-" + fileType + "-" + view);
        if (previewImg) { previewImg.src = ""; previewImg.style.display = "none"; if (previewImg.parentElement) previewImg.parentElement.classList.remove("has-preview"); }
        if (ph) { ph.textContent = "可选"; ph.style.display = ""; }
        const fileInput = document.getElementById("file-" + fileType + "-" + view);
        if (fileInput) fileInput.value = "";
    }
}

async function handleFileSelectExtra(view, fileType, input) {
    const file = input.files[0];
    if (!file) return;

    const ph = document.getElementById("ph-" + fileType + "-" + view);
    const previewImg = document.getElementById("preview-" + fileType + "-" + view);

    const formData = new FormData();
    formData.append("view", view);
    formData.append("file", file);

    const url = fileType === "depth" ? "/upload/depth" : "/upload/normal";
    try {
        await fetch(url, { method: "POST", body: formData });
    } catch (err) {
        console.error("Upload " + fileType + " failed:", err);
    }

    if (previewImg) {
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImg.src = e.target.result;
            previewImg.style.display = "block";
            if (ph) ph.style.display = "none";
            if (previewImg.parentElement) previewImg.parentElement.classList.add("has-preview");
        };
        reader.onerror = () => {
            if (ph) { ph.textContent = file.name; ph.style.display = ""; }
            previewImg.style.display = "none";
        };
        if (file.type.indexOf("image") === 0) {
            reader.readAsDataURL(file);
        } else {
            if (ph) { ph.textContent = file.name; ph.style.display = ""; }
            previewImg.style.display = "none";
        }
    } else if (ph) {
        ph.textContent = file.name;
    }
}

// ─────── 拖拽上传 ───────
function setupDragDrop() {
    document.querySelectorAll(".upload-box[data-view][data-type]").forEach((box) => {
        const view = box.dataset.view;
        const type = box.dataset.type;

        box.addEventListener("dragover", (e) => {
            e.preventDefault();
            e.stopPropagation();
            if (e.dataTransfer.types.indexOf("Files") !== -1) box.classList.add("upload-box-dragover");
        });

        box.addEventListener("dragleave", (e) => {
            if (!box.contains(e.relatedTarget)) box.classList.remove("upload-box-dragover");
        });

        box.addEventListener("drop", (e) => {
            e.preventDefault();
            e.stopPropagation();
            box.classList.remove("upload-box-dragover");
            const file = e.dataTransfer.files[0];
            if (!file) return;

            if (type === "rgb") {
                if (file.type.indexOf("image") !== 0) return;
            } else if (type === "depth") {
                const n = file.name.toLowerCase();
                if (!n.endsWith(".png") && !n.endsWith(".tiff") && !n.endsWith(".tif")) return;
            } else if (type === "normal") {
                if (file.type.indexOf("image") !== 0) return;
            }

            // 直接调用上传与预览逻辑，避免依赖 input.files 赋值（部分浏览器拖拽后不触发 change）
            const fakeInput = { files: [file] };
            if (type === "rgb") {
                handleFileSelect(view, fakeInput);
            } else {
                handleFileSelectExtra(view, type, fakeInput);
            }
        });
    });
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

/** 从 data URL 截取 base64 字符串，无效则返回 null */
function dataUrlToBase64(dataUrl) {
    if (!dataUrl || typeof dataUrl !== "string" || !dataUrl.startsWith("data:")) return null;
    const i = dataUrl.indexOf(",");
    return i >= 0 ? dataUrl.slice(i + 1) : null;
}

/** 从当前页面的预览图收集 base64（含 RGB / normal / depth），重启后仍可用来提交 */
function collectPreviewBase64() {
    const images_base64 = {};
    const normals_base64 = {};
    const depths_base64 = {};
    VIEW_ORDER.forEach((view) => {
        const imgEl = document.getElementById("preview-upload-" + view);
        if (imgEl && imgEl.src) {
            const b64 = dataUrlToBase64(imgEl.src);
            if (b64) images_base64[view] = b64;
        }
        const normalEl = document.getElementById("preview-normal-" + view);
        if (normalEl && normalEl.src) {
            const b64 = dataUrlToBase64(normalEl.src);
            if (b64) normals_base64[view] = b64;
        }
        const depthEl = document.getElementById("preview-depth-" + view);
        if (depthEl && depthEl.src) {
            const b64 = dataUrlToBase64(depthEl.src);
            if (b64) depths_base64[view] = b64;
        }
    });
    return {
        images_base64: Object.keys(images_base64).length ? images_base64 : undefined,
        normals_base64: Object.keys(normals_base64).length ? normals_base64 : undefined,
        depths_base64: Object.keys(depths_base64).length ? depths_base64 : undefined,
    };
}

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
        num_chunks: parseInt(document.getElementById("paramChunks").value) || 8000,
    };

    const mode = currentMode === "import" ? "upload" : "capture";
    const payload = { mode, params };
    if (mode === "upload") {
        const preview = collectPreviewBase64();
        if (preview.images_base64) payload.images_base64 = preview.images_base64;
        if (preview.normals_base64) payload.normals_base64 = preview.normals_base64;
        if (preview.depths_base64) payload.depths_base64 = preview.depths_base64;
    }

    try {
        const resp = await fetch("/generate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
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
        generationStartTime = Date.now();
        hasSeenProcessing = false;
        currentProgressStatus = "pending";
        showProgress(getProgressLabelWithElapsed());
        if (progressElapsedTimer) clearInterval(progressElapsedTimer);
        progressElapsedTimer = setInterval(updateProgressElapsed, 1000);

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
            const data = await resp.json().catch(() => ({}));
            if (!resp.ok) {
                onGenerationError(data.error || data.message || "请求失败");
                return;
            }
            if (data.error) {
                onGenerationError(data.error);
                return;
            }
            if (data.status === "completed") {
                onGenerationComplete(taskId, data);
                return;
            }
            if (data.status === "error") {
                onGenerationError(data.message || "未知错误");
                return;
            }
            updateProgressText(data.status);
            setTimeout(poll, 2000);
        } catch {
            setTimeout(poll, 5000);
        }
    };
    setTimeout(poll, 2000);
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
    const useMatted = !!data.has_matted_views;
    loadPreviewImages(taskId, useMatted);
    speak("3D 模型生成完成");
}

function onGenerationError(message) {
    hideProgress();
    const btn = document.getElementById("btnGenerate");
    btn.disabled = false;
    btn.textContent = "生成 3D 模型";
    alert("生成失败: " + message);
}

// ─────── Progress UI（参考 gradio 的 time_meta，显示已等待时间）───────
const PROGRESS_LABELS = {
    submitting: "提交中...",
    pending: "排队中...",
    processing: "GPU 推理中...",
};
// 排队超过此秒数仍显示「推理中」，避免长时间只显示排队
const PENDING_AS_PROCESSING_AFTER_SEC = 10;
function getProgressLabel(status) {
    const sec = getElapsedSeconds();
    if (status === "pending" && sec >= PENDING_AS_PROCESSING_AFTER_SEC)
        return PROGRESS_LABELS.processing;
    return PROGRESS_LABELS[status] || status;
}
function getElapsedSeconds() {
    return generationStartTime ? Math.floor((Date.now() - generationStartTime) / 1000) : 0;
}
function getProgressLabelWithElapsed() {
    const label = getProgressLabel(currentProgressStatus);
    const sec = getElapsedSeconds();
    return sec > 0 ? label + " 已等待 " + sec + " 秒" : label;
}
function updateProgressElapsed() {
    const el = document.getElementById("generationProgress");
    if (!el || el.style.display !== "flex") return;
    document.getElementById("progressText").textContent = getProgressLabelWithElapsed();
}
function showProgress(text) {
    const el = document.getElementById("generationProgress");
    el.style.display = "flex";
    document.getElementById("progressBar").className = "progress-bar indeterminate";
    document.getElementById("progressText").textContent = text;
}

function hideProgress() {
    if (progressElapsedTimer) {
        clearInterval(progressElapsedTimer);
        progressElapsedTimer = null;
    }
    document.getElementById("generationProgress").style.display = "none";
}

function updateProgressText(status) {
    if (status === "processing") hasSeenProcessing = true;
    // 一旦见过「推理中」就不再显示「排队中」，避免排队/推理来回闪
    const displayStatus = status === "pending" && hasSeenProcessing ? "processing" : status;
    currentProgressStatus = displayStatus;
    document.getElementById("progressText").textContent = getProgressLabelWithElapsed();
}

// ─────── 3D Model Viewer (Google model-viewer) ───────

function initModelViewerEvents() {
    const viewer = document.getElementById("modelViewer");
    if (!viewer) return;

    viewer.addEventListener("load", () => {
        if (!viewer.model || !viewer.model.materials) return;
        viewer.model.materials.forEach((mat) => {
            const pbr = mat.pbrMetallicRoughness;
            pbr.setBaseColorFactor([0.18, 0.18, 0.19, 1.0]);
            pbr.setMetallicFactor(0.3);
            pbr.setRoughnessFactor(0.35);
        });
    });
}

document.addEventListener("DOMContentLoaded", () => {
    initModelViewerEvents();
});

function showModelViewer(taskId) {
    document.getElementById("modelPlaceholder").style.display = "none";
    const viewer = document.getElementById("modelViewer");

    if (!customElements.get("model-viewer")) {
        document.getElementById("modelPlaceholder").style.display = "flex";
        document.getElementById("modelPlaceholder").innerHTML =
            "<p>3D 预览组件加载失败（网络超时），请刷新页面后重试。</p><p class='small'>您仍可下载下方模型文件。</p>";
        return;
    }

    const glbUrl = "/output/" + taskId + ".glb?" + Date.now();
    viewer.src = glbUrl;
    viewer.style.display = "block";
}

let currentDownloadTaskId = null;

function updateDownloadLink() {
    if (!currentDownloadTaskId) return;
    const format = document.getElementById("downloadFormat").value || "glb";
    const btn = document.getElementById("btnDownload");
    btn.href = "/download/" + currentDownloadTaskId + "?format=" + format;
    btn.download = currentDownloadTaskId + "." + format;
}

function showDownload(taskId) {
    currentDownloadTaskId = taskId;
    const section = document.getElementById("downloadSection");
    section.style.display = "block";
    updateDownloadLink();
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

function loadPreviewImages(taskId, useMatted = false) {
    const suffix = useMatted ? "_matted" : "";
    VIEW_ORDER.forEach(v => {
        const img = document.getElementById("view-rembg-" + v);
        img.src = "/preview/" + taskId + "/" + v + suffix + "?" + Date.now();
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
