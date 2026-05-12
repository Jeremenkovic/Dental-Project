const API = import.meta.env.VITE_API_URL
  ? import.meta.env.VITE_API_URL
  : "/api";

export async function uploadFrames(keyframes, onProgress) {
  const form = new FormData();
  keyframes.forEach((frame, i) => {
    form.append("frames", frame.blob, `frame_${String(i).padStart(4, "0")}.jpg`);
  });

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && onProgress) onProgress(e.loaded / e.total);
    };
    xhr.onload = () => {
      if (xhr.status === 201) resolve(JSON.parse(xhr.responseText));
      else reject(new Error(`Upload failed: ${xhr.status} — ${xhr.responseText}`));
    };
    xhr.onerror = () => reject(new Error("Network error during upload"));
    xhr.open("POST", `${API}/scans`);
    xhr.send(form);
  });
}

export async function startDemoScan() {
  const res = await fetch(`${API}/scans/demo`, { method: "POST" });
  if (!res.ok) throw new Error(`Demo scan failed: ${res.status}`);
  return res.json();
}

export async function pollScan(scanId, onStatus, intervalMs = 1500) {
  return new Promise((resolve, reject) => {
    const tick = async () => {
      try {
        const res = await fetch(`${API}/scans/${scanId}`);
        if (!res.ok) throw new Error(`Poll failed: ${res.status}`);
        const data = await res.json();
        if (onStatus) onStatus(data);
        if (data.status === "done") return resolve(data);
        if (data.status === "error") return reject(new Error(data.error ?? "Reconstruction failed"));
        setTimeout(tick, intervalMs);
      } catch (e) {
        reject(e);
      }
    };
    tick();
  });
}

export function resultUrl(scanId) {
  return `${API}/scans/${scanId}/result`;
}
