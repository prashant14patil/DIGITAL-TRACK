import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

// Body avatar viewer (display only). The mesh comes from /files/mesh_regions.glb: region id is encoded in the red
// vertex channel as id * 40 / 255 (backend experimental/body_regions.py). Mesh is Z-up and height-normalised.
const REGION_NAMES = ["head", "trunk", "left_arm", "right_arm", "left_leg", "right_leg"];
const BASE = new THREE.Color("#dfe6ea");
const AZIMUTH = { front: 0, side: Math.PI / 2, back: Math.PI };
const SLOT_TOP = [22, 38, 60]; // % of stage height for callout slots 0..2

export default function AvatarViewer({ meshUrl, regions, regionColors, view, callouts, height = 560 }) {
  const stage = useRef(null);
  const mount = useRef(null);
  const sceneRef = useRef(null); // { camera, controls, geometry, ids, anchors, renderer }
  const cardRefs = useRef({});
  const lineRefs = useRef({});
  const dotRefs = useRef({});
  const targetAz = useRef(0);
  const [state, setState] = useState("loading");

  // ---- scene (once per mesh) ----------------------------------------------------------------------
  useEffect(() => {
    const el = mount.current;
    const width = el.clientWidth || 800;
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(width, height);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    el.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.add(new THREE.HemisphereLight(0xffffff, 0xb9c4cc, 1.5));
    const sun = new THREE.DirectionalLight(0xffffff, 1.7);
    sun.position.set(-1.2, 3, 2);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    Object.assign(sun.shadow.camera, { left: -1.5, right: 1.5, top: 1.5, bottom: -1.5, near: 0.1, far: 8 });
    sun.shadow.radius = 6;
    scene.add(sun);

    const camera = new THREE.PerspectiveCamera(28, width / height, 0.01, 50);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.enablePan = false;
    controls.minDistance = 1.6;
    controls.maxDistance = 5;
    controls.minPolarAngle = controls.maxPolarAngle = Math.PI / 2 - 0.12; // drag to turn (horizontal only)

    const ctx = { camera, controls, renderer, geometry: null, ids: null, anchors: {} };
    sceneRef.current = ctx;
    let frame = 0;
    let disposed = false;

    new GLTFLoader().load(meshUrl, (gltf) => {
      if (disposed) return;
      let mesh = null;
      gltf.scene.traverse((o) => { if (o.isMesh && !mesh) mesh = o; });
      if (!mesh) { setState("error"); return; }
      const g = mesh.geometry;
      const src = g.getAttribute("color");
      const n = g.getAttribute("position").count;
      const ids = new Uint8Array(n);
      for (let i = 0; i < n; i++) ids[i] = src ? Math.round((src.getX(i) * 255) / 40) : 1;
      g.setAttribute("color", new THREE.BufferAttribute(new Float32Array(n * 3), 3));
      if (!g.getAttribute("normal")) g.computeVertexNormals();
      mesh.material = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.55, metalness: 0.0,
        side: THREE.DoubleSide });
      mesh.castShadow = true;
      ctx.geometry = g;
      ctx.ids = ids;

      const body = new THREE.Group(); // Z-up -> Y-up, mirror view (subject's left on screen left)
      gltf.scene.rotation.x = -Math.PI / 2;
      body.add(gltf.scene);
      body.scale.x = -1;
      scene.add(body);
      body.updateMatrixWorld(true);
      const box = new THREE.Box3().setFromObject(body);
      const size = box.getSize(new THREE.Vector3());
      const centre = box.getCenter(new THREE.Vector3());

      const platform = new THREE.Mesh(new THREE.BoxGeometry(size.y * 0.32, size.y * 0.03, size.y * 0.24),
        new THREE.MeshStandardMaterial({ color: 0xf7f8fa, roughness: 0.4 }));
      platform.position.set(centre.x, box.min.y - size.y * 0.015, centre.z);
      platform.castShadow = platform.receiveShadow = true;
      scene.add(platform);
      const ground = new THREE.Mesh(new THREE.PlaneGeometry(6, 6), new THREE.ShadowMaterial({ opacity: 0.16 }));
      ground.rotation.x = -Math.PI / 2;
      ground.position.y = box.min.y - size.y * 0.03;
      ground.receiveShadow = true;
      scene.add(ground);

      for (const r of regions?.regions || []) {
        if (!r.anchor) continue;
        ctx.anchors[r.name] = gltf.scene.localToWorld(new THREE.Vector3(...r.anchor));
      }
      controls.target.set(centre.x, centre.y - size.y * 0.02, centre.z);
      camera.position.set(centre.x, centre.y, centre.z + size.y * 2.9);
      controls.update();
      ctx.paint?.();
      setState("ready");
    }, undefined, () => !disposed && setState("error"));

    const v = new THREE.Vector3();
    const tick = () => {
      frame = requestAnimationFrame(tick);
      // ease towards the requested Front/Side/Back azimuth (user dragging overrides until the next button press)
      if (targetAz.current !== null) {
        const cur = controls.getAzimuthalAngle();
        let d = targetAz.current - cur;
        d = Math.atan2(Math.sin(d), Math.cos(d));
        if (Math.abs(d) < 0.002) targetAz.current = null;
        else {
          const off = camera.position.clone().sub(controls.target);
          off.applyAxisAngle(new THREE.Vector3(0, 1, 0), d * 0.15);
          camera.position.copy(controls.target).add(off);
        }
      }
      controls.update();
      renderer.render(scene, camera);
      // callout leader lines
      const rect = renderer.domElement.getBoundingClientRect();
      const srect = stage.current?.getBoundingClientRect();
      if (!srect) return;
      for (const [key, a] of Object.entries(ctx.anchors)) {
        const card = cardRefs.current[key], line = lineRefs.current[key], dot = dotRefs.current[key];
        if (!card || !line || !dot) continue;
        v.copy(a).project(camera);
        const x = (v.x * 0.5 + 0.5) * rect.width + (rect.left - srect.left);
        const y = (-v.y * 0.5 + 0.5) * rect.height + (rect.top - srect.top);
        const left = card.dataset.side === "left";
        const x1 = left ? card.offsetLeft + card.offsetWidth : card.offsetLeft;
        const y1 = card.offsetTop + card.offsetHeight / 2;
        line.setAttribute("x1", x1); line.setAttribute("y1", y1);
        line.setAttribute("x2", x); line.setAttribute("y2", y);
        dot.setAttribute("cx", x); dot.setAttribute("cy", y);
      }
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
      sceneRef.current = null;
    };
  }, [meshUrl, regions, height]);

  // ---- region colours (no reload) ------------------------------------------------------------------
  useEffect(() => {
    const ctx = sceneRef.current;
    if (!ctx) return;
    ctx.paint = () => {
      if (!ctx.geometry) return;
      const cols = REGION_NAMES.map((name) => {
        const c = BASE.clone();
        const hex = regionColors?.[name];
        return hex ? c.lerp(new THREE.Color(hex), 0.8) : c;
      });
      const attr = ctx.geometry.getAttribute("color");
      for (let i = 0; i < ctx.ids.length; i++) {
        const c = cols[ctx.ids[i]] || BASE;
        attr.setXYZ(i, c.r, c.g, c.b);
      }
      attr.needsUpdate = true;
    };
    ctx.paint();
  }, [regionColors, state]);

  useEffect(() => { targetAz.current = AZIMUTH[view] ?? 0; }, [view, state]);

  return (
    <div className="avatar-stage" ref={stage} style={{ height }}>
      <div ref={mount} className="avatar-canvas" />
      {state === "ready" && (
        <>
          <svg className="avatar-lines">
            {callouts.map((c) => (
              <g key={c.key}>
                <line ref={(e) => (lineRefs.current[c.key] = e)} stroke="#9aa6b2" strokeWidth="1.2" />
                <circle ref={(e) => (dotRefs.current[c.key] = e)} r="6" fill="#fff" stroke="#2457c5" strokeWidth="2.5" />
              </g>
            ))}
          </svg>
          {callouts.map((c) => (
            <div key={c.key} ref={(e) => (cardRefs.current[c.key] = e)} data-side={c.side}
              className={`callout ${c.side}`} style={{ top: `${SLOT_TOP[c.slot]}%` }}>
              <small>{c.label}</small>
              <div><strong>{c.value}</strong> <span>{c.unit}</span></div>
              {c.demo && <em>DEMO</em>}
            </div>
          ))}
        </>
      )}
      {state === "loading" && <div className="viewer-msg">Loading avatar…</div>}
      {state === "error" && <div className="viewer-msg bad">Could not load the avatar mesh.</div>}
      <div className="avatar-foot left">⟨|⟩ Drag to turn · mirror view</div>
    </div>
  );
}
