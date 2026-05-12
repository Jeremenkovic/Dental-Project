import { useRef, useState, useEffect, useCallback } from "react";
import { selectKeyframes } from "../lib/keyframeSelector";
import { uploadFrames, pollScan, startDemoScan } from "../lib/uploader";
import QualityBar from "./QualityBar";
import CoverageHeatmap from "./CoverageHeatmap";

const CAPTURE_FPS = 15;
const FRAME_W = 1280;
const FRAME_H = 720;

export default function CaptureView({ onScanReady }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const intervalRef = useRef(null);
  const framesRef = useRef([]);
  const prevDataRef = useRef(null);
  const imuRef = useRef([]);

  const [state, setState] = useState("idle");
  const [sharpness, setSharpness] = useState(0);
  const [motion, setMotion] = useState(0);
  const [frameCount, setFrameCount] = useState(0);
  const [uploadPct, setUploadPct] = useState(0);
  const [scanStatus, setScanStatus] = useState(null);
  const [covered, setCovered] = useState(new Set());
  const [error, setError] = useState(null);
  const [imuAvail, setImuAvail] = useState(false);

  // ── IMU via DeviceOrientationEvent (available on mobile / Electron with motion sensor)
  useEffect(() => {
    const handler = (e) => {
      imuRef.current.push({
        alpha: e.alpha ?? 0,
        beta: e.beta ?? 0,
        gamma: e.gamma ?? 0,
        t: Date.now(),
      });
      setImuAvail(true);
    };
    window.addEventListener("deviceorientation", handler);
    return () => window.removeEventListener("deviceorientation", handler);
  }, []);

  const startCamera = useCallback(async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: FRAME_W, height: FRAME_H },
        audio: false,
      });
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      setState("streaming");
    } catch (e) {
      setError(`Camera access denied: ${e.message}`);
    }
  }, []);

  const stopCamera = useCallback(() => {
    videoRef.current?.srcObject?.getTracks().forEach((t) => t.stop());
    clearInterval(intervalRef.current);
  }, []);

  const captureFrame = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    canvas.width = FRAME_W;
    canvas.height = FRAME_H;
    ctx.drawImage(video, 0, 0, FRAME_W, FRAME_H);

    // Downsampled ImageData for quality metrics (fast)
    const small = ctx.getImageData(0, 0, 320, 180);
    const lap = lapVar(small);
    const mot = motionScore(prevDataRef.current, small);
    prevDataRef.current = small;
    setSharpness(lap);
    setMotion(mot);

    // Update coverage heatmap (in production: derived from IMU/VO pose)
    const lastImu = imuRef.current[imuRef.current.length - 1];
    if (lastImu) {
      const seg = Math.floor(((lastImu.alpha ?? 0) / 360) * 12) % 12;
      setCovered((prev) => new Set([...prev, seg]));
    } else {
      setCovered((prev) => {
        const next = new Set(prev);
        if (next.size < 12) next.add(Math.floor(Math.random() * 12));
        return next;
      });
    }

    const fullData = ctx.getImageData(0, 0, FRAME_W, FRAME_H);
    canvas.toBlob(
      (blob) => {
        framesRef.current.push({
          blob,
          imageData: fullData,
          timestamp: Date.now(),
          imu: lastImu ?? null,
        });
        setFrameCount(framesRef.current.length);
      },
      "image/jpeg",
      0.85,
    );
  }, []);

  const startRecording = useCallback(() => {
    framesRef.current = [];
    imuRef.current = [];
    prevDataRef.current = null;
    setCovered(new Set());
    setFrameCount(0);
    setState("recording");
    intervalRef.current = setInterval(captureFrame, 1000 / CAPTURE_FPS);
  }, [captureFrame]);

  const stopAndUpload = useCallback(async () => {
    clearInterval(intervalRef.current);
    setState("uploading");
    setError(null);
    try {
      const keyframes = selectKeyframes(framesRef.current, { targetCount: 120 });
      const { scan_id } = await uploadFrames(keyframes, setUploadPct);
      setState("processing");
      await pollScan(scan_id, setScanStatus);
      onScanReady(scan_id);
    } catch (e) {
      setError(e.message);
      setState("streaming");
    }
  }, [onScanReady]);

  const runDemo = useCallback(async () => {
    setState("processing");
    setError(null);
    try {
      const { scan_id } = await startDemoScan();
      setScanStatus({ stage: "generating demo arch…", progress: 0.1 });
      await pollScan(scan_id, setScanStatus);
      onScanReady(scan_id);
    } catch (e) {
      setError(e.message);
      setState("idle");
    }
  }, [onScanReady]);

  useEffect(() => () => stopCamera(), [stopCamera]);

  return (
    <div className="flex flex-col gap-4">
      {/* Video preview */}
      <div className="relative rounded-xl overflow-hidden bg-black aspect-video">
        <video ref={videoRef} className="w-full h-full object-cover" playsInline muted />
        {state === "recording" && (
          <div className="absolute top-3 left-3 flex items-center gap-2">
            <span className="w-3 h-3 rounded-full bg-red-500 animate-pulse" />
            <span className="text-sm font-medium text-white shadow">
              {frameCount} frames {imuAvail && "· IMU ✓"}
            </span>
          </div>
        )}
        {state === "idle" && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3">
            <p className="text-gray-400 text-sm">Camera not started</p>
          </div>
        )}
        {/* Sweep guide overlay during recording */}
        {state === "recording" && <SweepGuide />}
      </div>

      <canvas ref={canvasRef} className="hidden" />

      {/* Live quality indicators */}
      {(state === "streaming" || state === "recording") && (
        <div className="grid grid-cols-2 gap-3">
          <QualityBar sharpness={sharpness} motion={motion} />
          <CoverageHeatmap coveredSegments={covered} />
        </div>
      )}

      {/* Progress bars */}
      {state === "uploading" && (
        <ProgressBar
          label={`Uploading keyframes… ${Math.round(uploadPct * 100)}%`}
          pct={uploadPct}
        />
      )}
      {state === "processing" && scanStatus && (
        <ProgressBar
          label={`${scanStatus.stage || "Processing"}… ${Math.round((scanStatus.progress || 0) * 100)}%`}
          pct={scanStatus.progress || 0}
        />
      )}

      {error && <p className="text-red-400 text-sm bg-red-950 px-3 py-2 rounded-lg">{error}</p>}

      {/* Action buttons */}
      <div className="flex gap-3">
        {state === "idle" && (
          <>
            <button onClick={startCamera} className="flex-1 px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 text-white font-medium transition-colors">
              Start Camera
            </button>
            <button onClick={runDemo} className="px-4 py-2 rounded-lg bg-gray-700 hover:bg-gray-600 text-white text-sm font-medium transition-colors">
              Demo Mode
            </button>
          </>
        )}
        {state === "streaming" && (
          <button onClick={startRecording} className="flex-1 px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 text-white font-medium transition-colors">
            Record Scan
          </button>
        )}
        {state === "recording" && (
          <button
            onClick={stopAndUpload}
            disabled={frameCount < 10}
            className="flex-1 px-4 py-2 rounded-lg bg-red-600 hover:bg-red-500 disabled:opacity-40 text-white font-medium transition-colors"
          >
            Stop &amp; Reconstruct ({frameCount} frames)
          </button>
        )}
      </div>

      {state === "recording" && frameCount < 10 && (
        <p className="text-xs text-gray-500 text-center">
          Keep scanning — 10 frames minimum (recommend 60–90 s for full arch)
        </p>
      )}
    </div>
  );
}

function SweepGuide() {
  return (
    <div className="absolute bottom-3 left-0 right-0 flex justify-center pointer-events-none">
      <div className="px-3 py-1.5 rounded-full bg-black/60 text-xs text-gray-200">
        Sweep buccal → occlusal → lingual · Keep 5–30 mm from teeth
      </div>
    </div>
  );
}

function ProgressBar({ label, pct }) {
  return (
    <div className="flex flex-col gap-1.5">
      <span className="text-sm text-gray-300">{label}</span>
      <div className="h-2 rounded bg-gray-800 overflow-hidden">
        <div
          className="h-full bg-blue-500 transition-all duration-300"
          style={{ width: `${Math.round(pct * 100)}%` }}
        />
      </div>
    </div>
  );
}

// ── Quality metric helpers ───────────────────────────────────────────────────

function lapVar(imageData) {
  const { data, width, height } = imageData;
  let sum = 0, count = 0;
  for (let y = 1; y < height - 1; y += 2) {
    for (let x = 1; x < width - 1; x += 2) {
      const i = (y * width + x) * 4;
      const g = (data[i] + data[i + 1] + data[i + 2]) / 3;
      const t = ((y - 1) * width + x) * 4;
      const b = ((y + 1) * width + x) * 4;
      const r = (y * width + x + 1) * 4;
      const l = (y * width + x - 1) * 4;
      const lap =
        4 * g -
        (data[t] + data[b] + data[r] + data[l]) / 3;
      sum += lap * lap;
      count++;
    }
  }
  return count ? sum / count : 0;
}

function motionScore(prev, curr) {
  if (!prev) return 0;
  let diff = 0, n = 0;
  for (let i = 0; i < curr.data.length; i += 16) {
    diff += Math.abs(curr.data[i] - prev.data[i]);
    n++;
  }
  return diff / (n * 255);
}
