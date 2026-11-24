import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

// Interactive viewer for a result mesh (.glb). Display only: never used to compute anything.
// The demo mesh is Z-up (height-normalised); three.js is Y-up, so the model is rotated -90° about X.
export default function MeshViewer({ url, height = 460 }) {
  const mount = useRef(null);
  const [state, setState] = useState("loading");

  useEffect(() => {
    const el = mount.current;
    if (!el) return undefined;
    const width = el.clientWidth || 600;
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(width, height);
    renderer.setClearColor(0xf4f5f7);
    el.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.add(new THREE.HemisphereLight(0xffffff, 0x667788, 1.4));
    const sun = new THREE.DirectionalLight(0xffffff, 1.6);
    sun.position.set(2, 3, 4);
    scene.add(sun);
    const camera = new THREE.PerspectiveCamera(30, width / height, 0.01, 100);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    let frame = 0;
    let disposed = false;
    new GLTFLoader().load(
      url,
      (gltf) => {
        if (disposed) return;
        const model = gltf.scene;
        model.rotation.x = -Math.PI / 2;
        model.traverse((o) => {
          if (o.isMesh) {
            o.material = new THREE.MeshStandardMaterial({ color: 0xc9ccd3, roughness: 0.75, metalness: 0.0,
              side: THREE.DoubleSide });
          }
        });
        scene.add(model);
        const box = new THREE.Box3().setFromObject(model);
        const size = box.getSize(new THREE.Vector3());
        const centre = box.getCenter(new THREE.Vector3());
        controls.target.copy(centre);
        camera.position.set(centre.x, centre.y + size.y * 0.1, centre.z + size.y * 2.4);
        controls.update();
        const grid = new THREE.GridHelper(size.y * 1.2, 12, 0xbfc4cc, 0xdde0e5);
        grid.position.y = box.min.y;
        scene.add(grid);
        setState("ready");
      },
      undefined,
      () => !disposed && setState("error"),
    );

    const tick = () => {
      frame = requestAnimationFrame(tick);
      controls.update();
      renderer.render(scene, camera);
    };
    tick();
    const onResize = () => {
      const w = el.clientWidth || width;
      renderer.setSize(w, height);
      camera.aspect = w / height;
      camera.updateProjectionMatrix();
    };
    window.addEventListener("resize", onResize);
    return () => {
      disposed = true;
      cancelAnimationFrame(frame);
      window.removeEventListener("resize", onResize);
      controls.dispose();
      renderer.dispose();
      el.removeChild(renderer.domElement);
    };
  }, [url, height]);

  return (
    <div className="viewer">
      <div ref={mount} className="viewer-canvas" style={{ height }} />
      {state === "loading" && <div className="viewer-msg">Loading mesh…</div>}
      {state === "error" && <div className="viewer-msg bad">Could not load the mesh.</div>}
      {state === "ready" && <div className="viewer-hint">drag to rotate · scroll to zoom · right-drag to pan</div>}
    </div>
  );
}
