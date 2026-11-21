const els = {
  fileInput: document.querySelector("#fileInput"),
  browseButton: document.querySelector("#browseButton"),
  dropzone: document.querySelector("#dropzone"),
  loadSample: document.querySelector("#loadSample"),
  progressBar: document.querySelector("#progressBar"),
  pipelineState: document.querySelector("#pipelineState"),
  runtimeStatus: document.querySelector("#runtimeStatus"),
  results: document.querySelector("#results"),
  metricsGrid: document.querySelector("#metricsGrid"),
  chart: document.querySelector("#chart"),
  ragSummary: document.querySelector("#ragSummary"),
  evidenceList: document.querySelector("#evidenceList"),
  evidenceCount: document.querySelector("#evidenceCount"),
  confusionMatrix: document.querySelector("#confusionMatrix"),
  windowCards: document.querySelector("#windowCards"),
  windowCount: document.querySelector("#windowCount"),
  downloadCsv: document.querySelector("#downloadCsv"),
  steps: [...document.querySelectorAll(".pipeline-step")],
  windowSize: document.querySelector("#windowSize"),
  baselinePoints: document.querySelector("#baselinePoints"),
  topK: document.querySelector("#topK"),
  sceneSelect: document.querySelector("#sceneSelect"),
  docInput: document.querySelector("#docInput"),
  kbCsvInput: document.querySelector("#kbCsvInput"),
  sceneNameInput: document.querySelector("#sceneNameInput"),
  buildKnowledge: document.querySelector("#buildKnowledge"),
  knowledgeStatus: document.querySelector("#knowledgeStatus"),
  knowledgePreview: document.querySelector("#knowledgePreview"),
  sensorPreview: document.querySelector("#sensorPreview"),
  patternPreview: document.querySelector("#patternPreview")
};

let latestRows = [];
let latestCsv = "";
const API_BASE = window.location.protocol === "file:" ? "http://127.0.0.1:8765" : "";

const knowledgeBase = [
  {
    title: "阀门部分关闭",
    tags: ["Volume Flow RateRMS_low", "Temperature_high"],
    text: "Volume Flow RateRMS 显著下降且 Temperature 上升时，常见解释是阀门开度减小或管路阻力增加，流速降低导致热量积累。"
  },
  {
    title: "轴承或旋转部件振动增强",
    tags: ["Accelerometer1RMS_high", "Accelerometer2RMS_high"],
    text: "Accelerometer1RMS 或 Accelerometer2RMS 的 z-score 持续高于正常范围时，可能对应轴承磨损、不平衡或机械松动。"
  },
  {
    title: "多变量联合偏移",
    tags: ["multi_feature"],
    text: "多个传感器同时出现 |z| > 1.5 的偏移时，单点噪声的可能性降低，应结合温度、流量、振动方向共同判断。"
  },
  {
    title: "温度异常",
    tags: ["Temperature_high", "Temperature_low"],
    text: "Temperature 极端偏高或偏低代表热平衡异常，可能由冷却不足、负载变化或传感器漂移造成。"
  },
  {
    title: "正常运行基线",
    tags: ["normal"],
    text: "当主要变量的 |z| 均小于 1.5，且振动比例处于相对平衡区间时，窗口通常可视为稳定运行状态。"
  },
  {
    title: "流量异常",
    tags: ["Volume Flow RateRMS_high", "Volume Flow RateRMS_low"],
    text: "流量变量偏移需要结合温度和振动同步判断；单独短时偏移可能是工况波动，持续偏移更可能对应异常。"
  },
  {
    title: "气穴或复合故障",
    tags: ["Accelerometer1RMS_high", "Temperature_high", "Volume Flow RateRMS_low"],
    text: "振动增强、温度升高和流量下降同时出现时，可能说明泵或管路状态进入复合异常，需要优先复核。"
  }
];

const featureAliases = {
  Accelerometer1RMS: ["Accelerometer1RMS", "Accelerometer1", "acc1", "Accel1"],
  Accelerometer2RMS: ["Accelerometer2RMS", "Accelerometer2", "acc2", "Accel2"],
  Temperature: ["Temperature", "temp"],
  "Volume Flow RateRMS": ["Volume Flow RateRMS", "VolumeFlowRateRMS", "Rate", "rate", "Flow", "Volume Flow Rate"]
};

document.querySelectorAll("[data-step]").forEach((button) => {
  button.addEventListener("click", () => {
    const input = document.querySelector(`#${button.dataset.step}`);
    const delta = Number(button.dataset.delta);
    const next = Math.min(Number(input.max), Math.max(Number(input.min), Number(input.value) + delta));
    input.value = next;
  });
});

els.browseButton.addEventListener("click", () => els.fileInput.click());
els.fileInput.addEventListener("change", () => {
  const files = [...(els.fileInput.files || [])];
  if (files.length) handleFiles(files);
});

["dragenter", "dragover"].forEach((eventName) => {
  els.dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    els.dropzone.classList.add("dragover");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  els.dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    els.dropzone.classList.remove("dragover");
  });
});

els.dropzone.addEventListener("drop", (event) => {
  const files = [...(event.dataTransfer.files || [])];
  if (files.length) handleFiles(files);
});

els.loadSample.addEventListener("click", () => {
  alert("真实运行模式需要上传 CSV 或 ZIP。请点击 Browse files 选择待检测数据。");
});

els.downloadCsv.addEventListener("click", () => {
  const blob = new Blob([latestCsv], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "llm_rag_predictions.csv";
  link.click();
  URL.revokeObjectURL(url);
});

els.buildKnowledge?.addEventListener("click", buildKnowledgeBase);
els.sceneSelect?.addEventListener("change", applySceneDefaults);
loadScenes();

async function loadScenes(selectedId = null) {
  if (!els.sceneSelect) return;
  try {
    const response = await fetch(`${API_BASE}/api/knowledge/scenes`);
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "知识库列表加载失败");
    renderSceneOptions(payload.scenes || [], selectedId);
    applySceneDefaults();
  } catch (error) {
    els.knowledgeStatus.textContent = `知识库列表加载失败：${error.message}`;
  }
}

function renderSceneOptions(scenes, selectedId = null) {
  const current = selectedId || els.sceneSelect.value || "default";
  els.sceneSelect.innerHTML = scenes.map((scene) => {
    const label = scene.builtin
      ? scene.name
      : `${scene.name}（${scene.matched_sensor_count || 0}/${scene.sensor_count || 0}字段）`;
    return `<option value="${escapeHtml(scene.id)}">${escapeHtml(label)}</option>`;
  }).join("");
  els.sceneSelect.value = scenes.some((scene) => scene.id === current) ? current : "default";
  applySceneDefaults();
}

function applySceneDefaults() {
  if (!els.sceneSelect) return;
  if (els.sceneSelect.value === "hai") {
    els.windowSize.value = 30;
    els.baselinePoints.value = 180;
    els.topK.value = 6;
    els.runtimeStatus.textContent = "HAI 模式：上传字段兼容的本地 CSV";
    els.knowledgeStatus.textContent = "当前使用 HAI 迁移知识库；公开仓库不附带数据文件。";
  } else if (els.sceneSelect.value === "default") {
    els.windowSize.value = 30;
    els.baselinePoints.value = 400;
    els.topK.value = 6;
    els.runtimeStatus.textContent = "SKAB 模式：等待上传 CSV/ZIP";
    els.knowledgeStatus.textContent = "默认使用 SKAB 知识库；上传工程资料后可生成新场景知识库。";
  }
}

async function buildKnowledgeBase() {
  const documentFile = els.docInput?.files?.[0];
  const csvFile = els.kbCsvInput?.files?.[0];
  if (!documentFile || !csvFile) {
    alert("请同时选择工程资料文档和字段匹配 CSV。");
    return;
  }

  const form = new FormData();
  form.append("document", documentFile);
  form.append("csv", csvFile);
  form.append("scene_name", els.sceneNameInput?.value || "");

  els.buildKnowledge.disabled = true;
  els.knowledgeStatus.textContent = "正在抽取工程资料并构建 FAISS 知识库...";
  els.knowledgePreview.classList.add("hidden");

  try {
    const response = await fetch(`${API_BASE}/api/knowledge/build`, {
      method: "POST",
      body: form
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "知识库构建失败");

    renderSceneOptions(payload.scenes || [], payload.scene.id);
    els.sceneSelect.value = payload.scene.id;
    els.sensorPreview.textContent = payload.scene.sensor_preview || "";
    els.patternPreview.textContent = payload.scene.pattern_preview || "";
    els.knowledgePreview.classList.remove("hidden");

    const mode = payload.scene.extraction_mode === "fallback" ? "规则兜底" : "LLM抽取";
    els.knowledgeStatus.textContent = `已生成 ${payload.scene.name}，${mode}，匹配 ${payload.scene.matched_sensor_count}/${payload.scene.sensor_count} 个传感器字段。`;
  } catch (error) {
    els.knowledgeStatus.textContent = `知识库构建失败：${error.message}`;
  } finally {
    els.buildKnowledge.disabled = false;
  }
}

async function handleFiles(files) {
  runDetection(files);
}

async function runDetection(files) {
  els.results.classList.add("hidden");
  const fileLabel = files.length === 1 ? files[0].name : `${files.length} 个文件`;
  els.runtimeStatus.textContent = `正在上传 ${fileLabel}`;
  latestRows = [];
  latestCsv = "";

  try {
    await advance(0, "上传 CSV 文件", 12);
    const form = new FormData();
    files.forEach((file) => form.append("files", file));
    form.append("window_size", els.windowSize.value);
    form.append("baseline_points", els.baselinePoints.value);
    form.append("top_k", els.topK.value);
    form.append("scene_id", els.sceneSelect?.value || "default");

    await advance(1, "后端识别字段", 24);
    await advance(2, "后端构造窗口", 36);
    await advance(3, "后端执行 RAG 检索", 48);
    await advance(4, "后端调用 LLM 判定", 64);

    const response = await fetch(`${API_BASE}/api/detect`, {
      method: "POST",
      body: form
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "后端检测失败");
    }

    await advance(5, "后端计算指标", 88);
    const normalized = normalizeBackendResult(payload);
    await advance(6, "渲染真实检测结果", 100);
    latestRows = normalized.windows;
    latestCsv = normalized.predictionsCsv;
    renderResults(normalized.windows, normalized.metrics, normalized.fileName, normalized.evidenceSummary);
    setState("检测完成", 6);
    els.runtimeStatus.textContent = `完成：${normalized.windows.length} 个窗口`;
  } catch (error) {
    setState("检测失败", 0);
    els.runtimeStatus.textContent = "后端未完成检测";
    alert(`${error.message}\n\n请确认已运行：python frontend/server.py，并且 Ollama 服务已启动。`);
  }
}

function parseCsv(text) {
  const normalized = text.replace(/\r\n/g, "\n").replace(/\r/g, "\n").trim();
  if (!normalized) return { headers: [], records: [], delimiter: "," };
  const lines = normalized.split("\n").filter(Boolean);
  const delimiter = guessDelimiter(lines[0]);
  const headers = splitCsvLine(lines[0], delimiter).map((h) => h.trim());
  const records = lines.slice(1).map((line) => {
    const values = splitCsvLine(line, delimiter);
    return Object.fromEntries(headers.map((header, index) => [header, values[index]?.trim() ?? ""]));
  });
  return { headers, records, delimiter };
}

function guessDelimiter(headerLine) {
  const candidates = [",", ";", "\t"];
  return candidates.map((d) => [d, headerLine.split(d).length]).sort((a, b) => b[1] - a[1])[0][0];
}

function splitCsvLine(line, delimiter) {
  const out = [];
  let current = "";
  let quoted = false;
  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];
    if (char === '"') {
      if (quoted && line[i + 1] === '"') {
        current += '"';
        i += 1;
      } else {
        quoted = !quoted;
      }
    } else if (char === delimiter && !quoted) {
      out.push(current);
      current = "";
    } else {
      current += char;
    }
  }
  out.push(current);
  return out;
}

function resolveSchema(headers) {
  const features = {};
  const normalizedHeaders = headers.map((header) => ({ raw: header, key: normalize(header) }));
  for (const [feature, aliases] of Object.entries(featureAliases)) {
    const found = normalizedHeaders.find((header) => aliases.some((alias) => header.key === normalize(alias)));
    if (found) features[feature] = found.raw;
  }
  return {
    features,
    missing: Object.keys(featureAliases).filter((feature) => !features[feature])
  };
}

function normalize(value) {
  return String(value).replace(/[\s_\-()]/g, "").toLowerCase();
}

function toNumber(value) {
  const number = Number(String(value).replace(",", "."));
  return Number.isFinite(number) ? number : NaN;
}

function buildWindows(rows) {
  const windowSize = Number(els.windowSize.value);
  const baselinePoints = Math.min(Number(els.baselinePoints.value), rows.length);
  const baselineRows = rows.slice(0, baselinePoints);
  const baseline = {};

  Object.keys(featureAliases).forEach((feature) => {
    const values = baselineRows.map((row) => row.values[feature]);
    const mean = avg(values);
    const std = Math.max(stdev(values, mean), 0.000001);
    baseline[feature] = { mean, std };
  });

  const windows = [];
  for (let start = 0; start + windowSize <= rows.length; start += windowSize) {
    const slice = rows.slice(start, start + windowSize);
    const stats = {};
    Object.keys(featureAliases).forEach((feature) => {
      const values = slice.map((row) => row.values[feature]);
      const mean = avg(values);
      const std = stdev(values, mean);
      const max = Math.max(...values);
      const z = (mean - baseline[feature].mean) / baseline[feature].std;
      stats[feature] = { mean, std, max, z };
    });
    const accelRatio = stats.Accelerometer2RMS.mean
      ? stats.Accelerometer1RMS.mean / stats.Accelerometer2RMS.mean
      : 0;
    windows.push({
      id: windows.length,
      start,
      end: start + windowSize - 1,
      label: slice.some((row) => row.anomaly === 1) ? 1 : 0,
      stats,
      accelRatio,
      maxAbsZ: Math.max(...Object.values(stats).map((item) => Math.abs(item.z)))
    });
  }
  return windows;
}

function retrieveEvidence(window) {
  const abnormalTags = Object.entries(window.stats)
    .filter(([, stats]) => Math.abs(stats.z) > 1.5)
    .map(([feature, stats]) => `${feature}_${stats.z > 0 ? "high" : "low"}`);

  if (abnormalTags.length >= 2) abnormalTags.push("multi_feature");
  if (!abnormalTags.length) abnormalTags.push("normal");

  const ranked = knowledgeBase.map((item) => {
    const score = item.tags.reduce((sum, tag) => sum + (abnormalTags.includes(tag) ? 2 : 0), 0)
      + (item.tags.includes("multi_feature") && abnormalTags.includes("multi_feature") ? 1 : 0);
    return { ...item, score };
  }).sort((a, b) => b.score - a.score);

  return ranked.slice(0, Number(els.topK.value)).filter((item, index) => item.score > 0 || index < 2);
}

function simulateLlmDecision(window) {
  const zs = window.stats;
  const largeCount = Object.values(zs).filter((item) => Math.abs(item.z) > 1.5).length;
  const highVibration = zs.Accelerometer1RMS.z > 2 || zs.Accelerometer2RMS.z > 2;
  const valvePattern = zs["Volume Flow RateRMS"].z < -2 && zs.Temperature.z > 1;
  const extremeTemp = Math.abs(zs.Temperature.z) > 3.5;
  const ratioIssue = window.accelRatio > 3 || window.accelRatio < 0.3;
  const normal = largeCount === 0 && window.accelRatio >= 0.5 && window.accelRatio <= 2;

  const prediction = normal ? 0 : Number(largeCount >= 2 || highVibration || valvePattern || extremeTemp || ratioIssue);
  let reason = "All key z-scores remain within the normal band, and vibration balance is stable.";
  if (prediction) {
    const strongest = Object.entries(zs).sort((a, b) => Math.abs(b[1].z) - Math.abs(a[1].z))[0];
    const evidence = window.evidence[0]?.title || "domain knowledge";
    reason = `${strongest[0]} shows the strongest deviation (z=${strongest[1].z.toFixed(2)}); ${evidence} supports an anomaly decision for this window.`;
  }
  return { prediction, reason };
}

function calculateMetrics(windows) {
  const tp = windows.filter((w) => w.label === 1 && w.prediction === 1).length;
  const tn = windows.filter((w) => w.label === 0 && w.prediction === 0).length;
  const fp = windows.filter((w) => w.label === 0 && w.prediction === 1).length;
  const fn = windows.filter((w) => w.label === 1 && w.prediction === 0).length;
  const precision = divide(tp, tp + fp);
  const recall = divide(tp, tp + fn);
  return {
    total: windows.length,
    tp,
    tn,
    fp,
    fn,
    f1: divide(2 * precision * recall, precision + recall),
    far: divide(fp, fp + tn),
    mar: divide(fn, fn + tp),
    precision,
    recall
  };
}

function divide(a, b) {
  return b ? a / b : 0;
}

function avg(values) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function stdev(values, mean = avg(values)) {
  return Math.sqrt(values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / Math.max(values.length - 1, 1));
}

function normalizeBackendResult(payload) {
  const metrics = {
    total: payload.metrics.total,
    tp: payload.metrics.TP,
    tn: payload.metrics.TN,
    fp: payload.metrics.FP,
    fn: payload.metrics.FN,
    f1: payload.metrics.F1,
    far: payload.metrics.FAR,
    mar: payload.metrics.MAR,
    precision: payload.metrics.Precision,
    recall: payload.metrics.Recall,
    accuracy: payload.metrics.Accuracy
  };
  const windows = payload.windows.map((window, index) => ({
    id: window.id ?? index,
    start: window.start ?? index * Number(els.windowSize.value),
    end: window.end ?? (index + 1) * Number(els.windowSize.value) - 1,
    label: Number(window.label ?? 0),
    prediction: Number(window.prediction ?? 0),
    maxAbsZ: Number(window.maxAbsZ ?? 0),
    reason: window.reason || "LLM 未返回解释。",
    stats: window.stats || {},
    evidence: window.evidence || []
  }));
  return {
    fileName: payload.file_name || "uploaded.csv",
    metrics,
    windows,
    evidenceSummary: payload.evidence_summary || [],
    predictionsCsv: payload.predictions_csv || buildPredictionCsv(windows),
  };
}

function renderResults(windows, metrics, fileName, evidenceSummary = null) {
  els.results.classList.remove("hidden");
  renderMetrics(metrics, fileName);
  renderConfusionMatrix(metrics);
  drawChart(windows);
  renderRagSummary(windows, metrics, evidenceSummary);
  renderEvidence(windows, evidenceSummary);
  renderWindowCards(windows);
}

function renderMetrics(metrics, fileName) {
  const cards = [
    ["文件", fileName],
    ["窗口数", metrics.total],
    ["F1", metrics.f1.toFixed(4)],
    ["FAR", `${(metrics.far * 100).toFixed(2)}%`],
    ["MAR", `${(metrics.mar * 100).toFixed(2)}%`],
    ["TP / TN / FP / FN", `${metrics.tp}/${metrics.tn}/${metrics.fp}/${metrics.fn}`]
  ];
  els.metricsGrid.innerHTML = cards.map(([label, value], index) => `
    <article class="metric-card ${index === 2 ? "primary" : ""}">
      <span>${escapeHtml(label)}</span>
      <strong>${escapeHtml(value)}</strong>
    </article>
  `).join("");
}

function renderConfusionMatrix(metrics) {
  els.confusionMatrix.innerHTML = `
    <div class="matrix-cell axis"></div>
    <div class="matrix-cell axis">预测正常</div>
    <div class="matrix-cell axis">预测异常</div>
    <div class="matrix-cell axis">真实正常</div>
    <div class="matrix-cell good"><span>TN</span><strong>${metrics.tn}</strong></div>
    <div class="matrix-cell bad"><span>FP</span><strong>${metrics.fp}</strong></div>
    <div class="matrix-cell axis">真实异常</div>
    <div class="matrix-cell bad"><span>FN</span><strong>${metrics.fn}</strong></div>
    <div class="matrix-cell good"><span>TP</span><strong>${metrics.tp}</strong></div>
  `;
}

function drawChart(windows) {
  const canvas = els.chart;
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  const pad = { left: 54, top: 24, right: 28, bottom: 42 };
  const innerW = width - pad.left - pad.right;
  const innerH = height - pad.top - pad.bottom;
  const primaryFeature = choosePrimaryFeature(windows);
  const values = windows.map((w) => Number(w.stats[primaryFeature]?.z ?? 0));
  const maxAbs = Math.max(4, ...windows.map((w) => Math.abs(w.maxAbsZ)));

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);

  ctx.strokeStyle = "rgba(35, 104, 209, 0.12)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 8; i += 1) {
    const y = pad.top + (innerH / 8) * i;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(width - pad.right, y);
    ctx.stroke();
  }

  windows.forEach((w, i) => {
    const x = pad.left + (i / Math.max(windows.length - 1, 1)) * innerW;
    const bandW = innerW / Math.max(windows.length, 1);
    if (w.label === 1) {
      ctx.fillStyle = "rgba(226, 85, 99, 0.18)";
      ctx.fillRect(x - bandW / 2, pad.top, Math.max(2, bandW), innerH);
    }
    if (w.prediction === 1) {
      ctx.strokeStyle = "rgba(31, 157, 99, 0.72)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x, pad.top);
      ctx.lineTo(x, pad.top + innerH);
      ctx.stroke();
    }
  });

  [-2, 0, 2].forEach((level) => {
    const y = pad.top + innerH / 2 - (level / maxAbs) * (innerH / 2);
    ctx.strokeStyle = level === 0 ? "rgba(18,32,51,0.20)" : "rgba(96,112,137,0.38)";
    ctx.setLineDash(level === 0 ? [] : [8, 8]);
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(width - pad.right, y);
    ctx.stroke();
    ctx.fillStyle = "#607089";
    ctx.fillText(String(level), 18, y + 4);
  });
  ctx.setLineDash([]);

  ctx.strokeStyle = "#2368d1";
  ctx.lineWidth = 3;
  ctx.beginPath();
  values.forEach((value, i) => {
    const x = pad.left + (i / Math.max(values.length - 1, 1)) * innerW;
    const y = pad.top + innerH / 2 - (value / maxAbs) * (innerH / 2);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  ctx.fillStyle = "#122033";
  ctx.font = "18px Microsoft YaHei UI";
  ctx.fillText(`${primaryFeature} z-score / anomaly window`, pad.left, 24);
  ctx.fillStyle = "#607089";
  ctx.font = "14px Microsoft YaHei UI";
  ctx.fillText("Window Index", width / 2 - 46, height - 12);
}

function choosePrimaryFeature(windows) {
  const preferred = "Volume Flow RateRMS";
  if (windows.some((w) => w.stats?.[preferred])) return preferred;
  const scores = new Map();
  windows.forEach((window) => {
    Object.entries(window.stats || {}).forEach(([feature, stats]) => {
      scores.set(feature, Math.max(scores.get(feature) || 0, Math.abs(Number(stats.z ?? stats.z_score ?? 0))));
    });
  });
  return [...scores.entries()].sort((a, b) => b[1] - a[1])[0]?.[0] || preferred;
}

function renderRagSummary(windows, metrics, evidenceSummary = null) {
  const anomalyWindows = windows.filter((w) => w.prediction === 1);
  const strongest = [...windows].sort((a, b) => b.maxAbsZ - a.maxAbsZ)[0];
  const topFeature = strongest
    ? Object.entries(strongest.stats).sort((a, b) => Math.abs(b[1].z) - Math.abs(a[1].z))[0]
    : null;
  const evidenceCounts = evidenceSummary?.length ? evidenceSummary : summarizeEvidence(windows);
  const topEvidence = evidenceCounts.slice(0, 3).map((item) => item.title).join("、") || "正常运行基线";
  const strongestEvidence = strongest?.evidence?.[0]?.title || "检索证据";
  const rows = [
    ["证据归纳", `根据窗口偏移方向，RAG 检索频次最高的证据为：${topEvidence}。这些证据被写入 LLM 上下文，用于解释异常判定。`],
    ["窗口解释依据", topFeature ? `最大偏移窗口为 #${strongest.id}，主导特征是 ${topFeature[0]}（z=${topFeature[1].z.toFixed(2)}），对应证据片段为“${strongestEvidence}”。` : "未发现明显偏移。"],
    ["判定结果", `系统共处理 ${windows.length} 个窗口，其中 ${anomalyWindows.length} 个窗口被判定为异常；该数量来自逐窗口 LLM Decision 汇总。`],
    ["评价指标", `CSV 含 anomaly 标签，因此按窗口标签计算 F1=${metrics.f1.toFixed(4)}、FAR=${(metrics.far * 100).toFixed(2)}%、MAR=${(metrics.mar * 100).toFixed(2)}。`]
  ];
  els.ragSummary.innerHTML = rows.map(([title, text]) => `
    <div class="summary-line">
      <strong>${escapeHtml(title)}</strong>
      <p>${escapeHtml(text)}</p>
    </div>
  `).join("");
}

function renderEvidence(windows, evidenceSummary = null) {
  const items = evidenceSummary?.length ? evidenceSummary : summarizeEvidence(windows);
  els.evidenceCount.textContent = `${items.length} 条`;
  els.evidenceList.innerHTML = items.map((item) => `
    <article class="evidence-card">
      <strong>${escapeHtml(item.title)} · 命中 ${item.count} 次</strong>
      <p>${escapeHtml(item.text)}</p>
    </article>
  `).join("");
}

function summarizeEvidence(windows) {
  const map = new Map();
  windows.forEach((window) => {
    window.evidence.forEach((item) => {
      if (!map.has(item.title)) map.set(item.title, { ...item, count: 0 });
      map.get(item.title).count += 1;
    });
  });
  return [...map.values()].sort((a, b) => b.count - a.count);
}

function renderWindowCards(windows) {
  els.windowCount.textContent = `${windows.length} windows`;
  const orderedWindows = [...windows].sort((a, b) => Number(a.id) - Number(b.id));
  els.windowCards.innerHTML = orderedWindows.map((w) => {
    const evidence = w.evidence || [];
    const statusClass = w.label === w.prediction ? "is-hit" : "is-miss";
    const evidenceTags = evidence.length
      ? evidence.map((item) => `<span>${escapeHtml(item.title)}</span>`).join("")
      : "<span>未检索到证据</span>";
    const evidenceItems = evidence.length
      ? evidence.map((item, index) => `
        <section class="evidence-item">
          <b>${index + 1}. ${escapeHtml(item.title)}</b>
          <p>${escapeHtml(item.text || item.title)}</p>
        </section>
      `).join("")
      : `<section class="evidence-item"><p>该窗口没有返回可展示的 RAG 证据。</p></section>`;
    return `
    <article class="window-card ${statusClass}">
      <div class="window-card-head">
        <div>
          <strong>#${w.id}</strong>
          <small>${w.start}-${w.end}</small>
        </div>
        <div>
          ${pill(w.label)}
          ${pill(w.prediction)}
        </div>
      </div>
      <div class="zscore">最大 |z| = ${w.maxAbsZ.toFixed(2)}</div>
      <div class="window-card-body">
        <div class="reason-block">
          <strong>LLM Reason</strong>
          <p>${escapeHtml(w.reason)}</p>
        </div>
        <div class="evidence-block">
          <strong>RAG 证据</strong>
          <div class="evidence-tags">${evidenceTags}</div>
          <div class="evidence-items">${evidenceItems}</div>
        </div>
      </div>
    </article>
  `;
  }).join("");
}

function pill(value) {
  return value
    ? '<span class="pill anomaly">Anomaly</span>'
    : '<span class="pill normal">Normal</span>';
}

function buildPredictionCsv(windows) {
  const header = ["window_idx", "start", "end", "label", "prediction", "max_abs_z", "llm_reason", "rag_evidence"];
  const lines = windows.map((w) => [
    w.id,
    w.start,
    w.end,
    w.label,
    w.prediction,
    w.maxAbsZ.toFixed(6),
    w.reason,
    w.evidence.map((item) => item.title).join("; ")
  ].map(csvCell).join(","));
  return `${header.join(",")}\n${lines.join("\n")}`;
}

function csvCell(value) {
  const text = String(value ?? "");
  return /[",\n;]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

function setState(label, stepIndex) {
  els.pipelineState.textContent = label;
  els.steps.forEach((step, index) => {
    step.classList.toggle("done", index < stepIndex);
    step.classList.toggle("active", index === stepIndex);
  });
}

async function advance(stepIndex, label, progress) {
  setState(label, stepIndex);
  els.progressBar.style.width = `${progress}%`;
  await wait(240);
}

function wait(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function createSampleCsv() {
  const rows = ["datetime;Accelerometer1RMS;Accelerometer2RMS;Temperature;Volume Flow RateRMS;anomaly"];
  const start = new Date("2026-03-31T09:00:00");
  for (let i = 0; i < 720; i += 1) {
    const anomaly = (i >= 420 && i < 510) || (i >= 600 && i < 660) ? 1 : 0;
    const phase = i / 13;
    const acc1 = 0.034 + Math.sin(phase) * 0.002 + (anomaly && i < 180 ? 0.024 : 0);
    const acc2 = 0.032 + Math.cos(phase / 1.4) * 0.002 + (anomaly && i >= 260 ? 0.02 : 0);
    const temp = 82 + Math.sin(phase / 2) * 0.8 + (anomaly ? 6.5 : 0);
    const flow = 78 + Math.cos(phase / 2.5) * 2.2 - (anomaly ? 24 : 0);
    const time = new Date(start.getTime() + i * 1000).toISOString();
    rows.push([time, acc1.toFixed(5), acc2.toFixed(5), temp.toFixed(3), flow.toFixed(3), anomaly].join(";"));
  }
  return rows.join("\n");
}
