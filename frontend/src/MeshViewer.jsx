import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

// Interactive viewer for a result mesh (.glb). Display only: never used to compute anything.
// The demo mesh is Z-up (height-normalised); three.js is Y-up, so the model is rotated -90° about X.
// `polished`: refined mesh look (satin body, soft shadow, platform); vertices flagged estimated in the GLB's green
// vertex channel are shown pale. The look never changes the geometry.
const BODY = new THREE.Color("#a8dcc8");
const PALE = new THREE.Color("#eef3f1");

export default function MeshViewer({ url, height = 460, polished = false, onError }) {
  const mount = useRef(null);
  const [state, setState] = useState("loading");

  useEffect(() => {
    const el = mount.current;
    if (!el) return undefined;
    const width = el.clientWidth || 600;
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true }); // background comes from CSS (.viewer)
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(width, height);
    renderer.setClearColor(0x000000, 0);
    if (polished) {
      renderer.shadowMap.enabled = true;
      renderer.shadowMap.type = THREE.PCFSoftShadowMap;
      renderer.toneMapping = THREE.ACESFilmicToneMapping;
    }
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
          if (!o.isMesh) return;
          if (polished) {
            const g = o.geometry;
            const src = g.getAttribute("color");
            const n = g.getAttribute("position").count;
            const cols = new Float32Array(n * 3);
            for (let i = 0; i < n; i++) {
              const c = src && src.getY(i) > 0.5 ? PALE : BODY;
              cols[3 * i] = c.r; cols[3 * i + 1] = c.g; cols[3 * i + 2] = c.b;
            }
            g.setAttribute("color", new THREE.BufferAttribute(cols, 3));
            if (!g.getAttribute("normal")) g.computeVertexNormals();
            o.material = new THREE.MeshPhysicalMaterial({ vertexColors: true, roughness: 0.6, metalness: 0.0,
              sheen: 0.5, sheenRoughness: 0.6, clearcoat: 0.08, clearcoatRoughness: 0.6, side: THREE.DoubleSide });
            o.castShadow = true;
          } else {
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
        if (polished) {
          sun.castShadow = true;
          sun.shadow.mapSize.set(2048, 2048);
          const sc = sun.shadow.camera;
          sc.left = sc.bottom = -size.y; sc.right = sc.top = size.y; sc.near = 0.1; sc.far = 20;
          sun.position.set(centre.x + size.y, centre.y + size.y * 1.5, centre.z + size.y * 1.2);
          sun.target.position.copy(centre);
          scene.add(sun.target);
          const plat = new THREE.Mesh(new THREE.CylinderGeometry(size.y * 0.2, size.y * 0.21, size.y * 0.03, 64),
            new THREE.MeshPhysicalMaterial({ color: 0xf9fafb, roughness: 0.35, clearcoat: 0.3 }));
          plat.position.set(centre.x, box.min.y - size.y * 0.015, centre.z);
          plat.receiveShadow = true;
          scene.add(plat);
          const ground = new THREE.Mesh(new THREE.PlaneGeometry(8, 8), new THREE.ShadowMaterial({ opacity: 0.15 }));
          ground.rotation.x = -Math.PI / 2;
          ground.position.y = box.min.y - size.y * 0.03;
          ground.receiveShadow = true;
          scene.add(ground);
        } else {
          const grid = new THREE.GridHelper(size.y * 1.2, 12, 0x8a929c, 0x8a929c);
          grid.material.transparent = true;
          grid.material.opacity = 0.35;
          grid.position.y = box.min.y;
          scene.add(grid);
        }
        setState("ready");
      },
      undefined,
      () => { if (!disposed) { setState("error"); onError?.(); } },
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
  }, [url, height, polished]);

  return (
    <div className="viewer">
      <div ref={mount} className="viewer-canvas" style={{ height }} />
      {state === "loading" && <div className="viewer-msg">Loading mesh…</div>}
      {state === "error" && <div className="viewer-msg bad">Could not load the mesh.</div>}
      {state === "ready" && <div className="viewer-hint">Drag to rotate · scroll to zoom · right-drag to pan</div>}
    </div>
  );
}
