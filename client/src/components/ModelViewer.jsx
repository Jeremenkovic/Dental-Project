import { Suspense, useEffect, useRef, useState } from "react";
import { Canvas, useThree } from "@react-three/fiber";
import { OrbitControls, Environment, Grid } from "@react-three/drei";
import * as THREE from "three";
import { OBJLoader } from "three/examples/jsm/loaders/OBJLoader";
import { resultUrl } from "../lib/uploader";

export default function ModelViewer({ scanId }) {
  const [obj, setObj] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!scanId) return;
    const loader = new OBJLoader();
    loader.load(
      resultUrl(scanId),
      (object) => {
        // Ensure vertex colors are used if present
        object.traverse((child) => {
          if (child.isMesh) {
            if (!child.material) {
              child.material = new THREE.MeshStandardMaterial({ vertexColors: true });
            } else {
              child.material.vertexColors = true;
              child.material.side = THREE.DoubleSide;
            }
          }
        });

        // Center the model
        const box = new THREE.Box3().setFromObject(object);
        const center = box.getCenter(new THREE.Vector3());
        object.position.sub(center);

        setObj(object);
        setLoading(false);
      },
      undefined,
      (err) => {
        setError("Failed to load model");
        setLoading(false);
        console.error(err);
      },
    );
  }, [scanId]);

  return (
    <div className="relative w-full aspect-square rounded-xl overflow-hidden bg-gray-950">
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center text-gray-400">
          Loading 3D model…
        </div>
      )}
      {error && (
        <div className="absolute inset-0 flex items-center justify-center text-red-400">{error}</div>
      )}
      <Canvas camera={{ position: [0, 20, 60], fov: 45 }} shadows>
        <ambientLight intensity={0.6} />
        <directionalLight position={[30, 40, 20]} intensity={1.2} castShadow />
        <directionalLight position={[-20, 10, -20]} intensity={0.4} />
        <Suspense fallback={null}>
          {obj && <primitive object={obj} />}
          <Environment preset="studio" />
        </Suspense>
        <Grid args={[200, 200]} cellSize={5} cellThickness={0.5} cellColor="#374151" sectionColor="#4b5563" position={[0, -20, 0]} />
        <OrbitControls makeDefault enableDamping dampingFactor={0.05} />
      </Canvas>

      <div className="absolute bottom-3 left-3 text-xs text-gray-500 bg-black/50 px-2 py-1 rounded">
        Drag to rotate · Scroll to zoom · Right-click to pan
      </div>
    </div>
  );
}
