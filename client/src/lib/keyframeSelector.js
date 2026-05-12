/**
 * On-device keyframe selection (Section 4.5 of spec).
 *
 * From a raw array of captured frames (ImageBitmap or HTMLVideoElement snapshots),
 * selects 80-150 keyframes using:
 *   1. Laplacian variance (focus quality)
 *   2. Motion magnitude threshold (rejects blur)
 *   3. Greedy farthest-point sampling over time to ensure viewpoint diversity
 */

const LAPLACIAN_KERNEL = [
  [0, 1, 0],
  [1, -4, 1],
  [0, 1, 0],
];

function laplacianVariance(imageData) {
  const { data, width, height } = imageData;
  let sum = 0, sumSq = 0, n = 0;

  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      let lap = 0;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const idx = ((y + dy) * width + (x + dx)) * 4;
          const gray = 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];
          lap += LAPLACIAN_KERNEL[dy + 1][dx + 1] * gray;
        }
      }
      sum += lap;
      sumSq += lap * lap;
      n++;
    }
  }
  const mean = sum / n;
  return sumSq / n - mean * mean;
}

function motionScore(prev, curr) {
  if (!prev) return 0;
  const { data: d1, width, height } = prev;
  const { data: d2 } = curr;
  let diff = 0;
  const step = 4 * 4; // sample every 4th pixel for speed
  let count = 0;
  for (let i = 0; i < d1.length; i += step) {
    diff += Math.abs(d1[i] - d2[i]) + Math.abs(d1[i + 1] - d2[i + 1]) + Math.abs(d1[i + 2] - d2[i + 2]);
    count++;
  }
  return diff / (count * 3 * 255);
}

/**
 * @param {Array<{blob: Blob, imageData: ImageData, timestamp: number}>} allFrames
 * @param {object} opts
 * @returns {Array<{blob: Blob, timestamp: number, score: number}>}
 */
export function selectKeyframes(allFrames, opts = {}) {
  const {
    minLaplacian = 80,
    maxMotion = 0.15,
    targetCount = 120,
  } = opts;

  // 1. Focus + motion filter
  const candidates = [];
  for (let i = 0; i < allFrames.length; i++) {
    const frame = allFrames[i];
    const sharpness = laplacianVariance(frame.imageData);
    const motion = motionScore(allFrames[i - 1]?.imageData, frame.imageData);

    if (sharpness >= minLaplacian && motion <= maxMotion) {
      candidates.push({ ...frame, sharpness, motion });
    }
  }

  if (candidates.length <= targetCount) return candidates;

  // 2. Greedy farthest-point sampling over time index for diversity
  const selected = new Set();
  selected.add(0);
  selected.add(candidates.length - 1);

  while (selected.size < targetCount) {
    let bestIdx = -1, bestDist = -1;
    for (let i = 0; i < candidates.length; i++) {
      if (selected.has(i)) continue;
      let minDist = Infinity;
      for (const s of selected) {
        minDist = Math.min(minDist, Math.abs(i - s));
      }
      if (minDist > bestDist) {
        bestDist = minDist;
        bestIdx = i;
      }
    }
    if (bestIdx === -1) break;
    selected.add(bestIdx);
  }

  return [...selected].sort((a, b) => a - b).map((i) => candidates[i]);
}
