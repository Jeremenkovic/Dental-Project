export default function QualityBar({ sharpness = 0, motion = 0 }) {
  // sharpness: Laplacian variance (higher = sharper), motion: 0–1 (lower = steadier)
  const focusPct = Math.min(100, (sharpness / 300) * 100);
  const motionPct = Math.min(100, motion * 100);

  const focusColor = focusPct > 60 ? "bg-green-500" : focusPct > 30 ? "bg-yellow-400" : "bg-red-500";
  const motionColor = motionPct < 20 ? "bg-green-500" : motionPct < 50 ? "bg-yellow-400" : "bg-red-500";

  return (
    <div className="flex gap-4 px-4 py-2 bg-gray-900 rounded-lg text-sm">
      <Meter label="Focus" pct={focusPct} colorClass={focusColor} />
      <Meter label="Steady" pct={100 - motionPct} colorClass={motionColor} />
    </div>
  );
}

function Meter({ label, pct, colorClass }) {
  return (
    <div className="flex flex-col gap-1 flex-1">
      <span className="text-gray-400 text-xs">{label}</span>
      <div className="h-2 rounded bg-gray-700 overflow-hidden">
        <div className={`h-full rounded transition-all duration-150 ${colorClass}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}
