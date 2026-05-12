/**
 * Simplified arch coverage heatmap.
 * In production: maintain a 3D voxel grid updated from camera pose estimates
 * (from IMU or lightweight visual odometry) and project to this SVG arc.
 */
const SEGMENTS = 12; // tooth-level granularity

export default function CoverageHeatmap({ coveredSegments = new Set() }) {
  const arcPath = buildArcPath(SEGMENTS);

  return (
    <div className="p-3 bg-gray-900 rounded-lg">
      <p className="text-xs text-gray-400 mb-2 text-center">Arch coverage</p>
      <svg viewBox="-60 -10 120 80" className="w-full max-w-xs mx-auto">
        {arcPath.map(({ x, y, segIdx }) => {
          const covered = coveredSegments.has(segIdx);
          return (
            <circle
              key={segIdx}
              cx={x}
              cy={y}
              r={6}
              className={`transition-colors duration-300 ${covered ? "fill-green-400" : "fill-gray-600"}`}
              stroke={covered ? "#22c55e" : "#374151"}
              strokeWidth={1}
            />
          );
        })}
      </svg>
      <p className="text-xs text-center text-gray-400 mt-1">
        {coveredSegments.size} / {SEGMENTS} regions
      </p>
    </div>
  );
}

function buildArcPath(n) {
  return Array.from({ length: n }, (_, i) => {
    const theta = (Math.PI * i) / (n - 1); // 0 → π
    const r = 45;
    return {
      segIdx: i,
      x: r * Math.cos(Math.PI - theta),
      y: r * Math.sin(Math.PI - theta) - r + 10,
    };
  });
}
