import { useState } from "react";
import CaptureView from "./components/CaptureView";
import ModelViewer from "./components/ModelViewer";

export default function App() {
  const [scanId, setScanId] = useState(null);
  const [view, setView] = useState("capture"); // capture | result

  function handleScanReady(id) {
    setScanId(id);
    setView("result");
  }

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      <header className="flex items-center justify-between px-6 py-4 border-b border-gray-800">
        <div>
          <h1 className="text-lg font-semibold tracking-tight">Dental 3D</h1>
          <p className="text-xs text-gray-500">POC · Capture → Reconstruct → View</p>
        </div>
        <nav className="flex gap-2">
          <TabBtn active={view === "capture"} onClick={() => setView("capture")}>Capture</TabBtn>
          {scanId && (
            <TabBtn active={view === "result"} onClick={() => setView("result")}>3D Viewer</TabBtn>
          )}
        </nav>
      </header>

      <main className="max-w-2xl mx-auto px-6 py-8">
        {view === "capture" && <CaptureView onScanReady={handleScanReady} />}
        {view === "result" && scanId && (
          <div className="flex flex-col gap-4">
            <ModelViewer scanId={scanId} />
            <div className="flex gap-3">
              <a
                href={`/api/scans/${scanId}/result`}
                download={`scan_${scanId}.obj`}
                className="px-4 py-2 rounded-lg bg-gray-700 hover:bg-gray-600 text-sm font-medium transition-colors"
              >
                Download OBJ
              </a>
              <button
                onClick={() => { setScanId(null); setView("capture"); }}
                className="px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 text-sm font-medium transition-colors"
              >
                New Scan
              </button>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

function TabBtn({ active, onClick, children }) {
  return (
    <button
      onClick={onClick}
      className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
        active ? "bg-blue-600 text-white" : "text-gray-400 hover:text-white"
      }`}
    >
      {children}
    </button>
  );
}
