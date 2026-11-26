import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { RoundedBoxGeometry } from "three/examples/jsm/geometries/RoundedBoxGeometry.js";

// Body avatar viewer (display only). The mesh comes from /files/mesh_regions.glb: region id is encoded in the red
// vertex channel as id * 40 / 255 (backend experimental/body_regions.py). Mesh is Z-up and height-normalised.
const REGION_NAMES = ["head", "trunk", "left_arm", "right_arm", "left_leg", "right_leg"];
const BASE = new THREE.Color("#e4eaee"); // regions without a reading (and the head) stay neutral
const TINT = 0.7; // how strongly a region takes its category colour
const ESTIMATED = new THREE.Color("#ffffff"); // gap-filled display areas are blended towards this
const AZIMUTH = { front: 0, side: Math.PI / 2, back: Math.PI };
const CARD_GAP = 12; // px between stacked callout cards

export default function AvatarViewer({ meshUrl, regions, regionColors, view, callouts, height = 560 }) {
  const stage = useRef(null);
  const mount = useRef(null);
  const sceneRef = useRef(null); // { camera, controls, geometry, ids, anchors, renderer }
  const cardRefs = useRef({});
  const lineRefs = useRef({});
  const dotRefs = useRef({});
  const targetAz = useRef(0);
  const [state, setState] = useState("loading");
  const [hasEstimated, setHasEstimated] = useState(false);

  // ---- scene (once per mesh) ----------------------------------------------------------------------
  useEffect(() => {
    const el = mount.current;
    const width = el.clientWidth || 800;
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(width, height);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.05;
    el.appendChild(renderer.domElement);

    // soft studio lighting: sky/ground fill, key light with a soft shadow, side fill, cool rim from behind
    const scene = new THREE.Scene();
    scene.add(new THREE.HemisphereLight(0xffffff, 0xc9d3da, 1.6));
    const fill = new THREE.DirectionalLight(0xffffff, 0.6);
    fill.position.set(2, 1.2, 1.5);
    scene.add(fill);
    const rim = new THREE.DirectionalLight(0xdfeaff, 1.1);
    rim.position.set(0.5, 2, -3);
    scene.add(rim);
    const sun = new THREE.DirectionalLight(0xffffff, 1.6);
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
    controls.minDistance = 1.2;
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
      const est = new Uint8Array(n); // green channel: 1 = estimated (gap bridged for display, not measured)
      for (let i = 0; i < n; i++) {
        ids[i] = src ? Math.round((src.getX(i) * 255) / 40) : 1;
        est[i] = src && src.getY(i) > 0.5 ? 1 : 0;
      }
      g.setAttribute("color", new THREE.BufferAttribute(new Float32Array(n * 3), 3));
      if (!g.getAttribute("normal")) g.computeVertexNormals();
      // satin skin-like finish; opaque on purpose: a see-through scan would show its inner (doubled) layers
      mesh.material = new THREE.MeshPhysicalMaterial({ vertexColors: true, roughness: 0.62, metalness: 0.0,
        sheen: 0.5, sheenRoughness: 0.6, sheenColor: new THREE.Color(0xffffff), clearcoat: 0.08,
        clearcoatRoughness: 0.6, side: THREE.DoubleSide });
      mesh.castShadow = true;
      ctx.geometry = g;
      ctx.ids = ids;
      ctx.est = est;
      setHasEstimated(est.some((x) => x));

      const body = new THREE.Group(); // Z-up -> Y-up, mirror view (subject's left on screen left)
      gltf.scene.rotation.x = -Math.PI / 2;
      body.add(gltf.scene);
      body.scale.x = -1;
      scene.add(body);
      body.updateMatrixWorld(true);
      const box = new THREE.Box3().setFromObject(body);
      const size = box.getSize(new THREE.Vector3());
      const centre = box.getCenter(new THREE.Vector3());

      const platform = new THREE.Mesh(new RoundedBoxGeometry(size.y * 0.34, size.y * 0.035, size.y * 0.26, 4, size.y * 0.012),
        new THREE.MeshPhysicalMaterial({ color: 0xf9fafb, roughness: 0.35, clearcoat: 0.3 }));
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
      camera.position.set(centre.x, centre.y, centre.z + size.y * 2.3);
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
      // Each card sits at the height of its body point (cards on one side are pushed apart so they never overlap);
      // the card is centred on `top` (CSS translateY(-50%)), so the line starts at its vertical middle.
      const rect = renderer.domElement.getBoundingClientRect();
      const srect = stage.current?.getBoundingClientRect();
      if (!srect) return;
      const pts = [];
      for (const [key, a] of Object.entries(ctx.anchors)) {
        const card = cardRefs.current[key], line = lineRefs.current[key], dot = dotRefs.current[key];
        if (!card || !line || !dot) continue;
        v.copy(a).project(camera);
        pts.push({ card, line, dot, x: (v.x * 0.5 + 0.5) * rect.width + (rect.left - srect.left),
          y: (-v.y * 0.5 + 0.5) * rect.height + (rect.top - srect.top) });
      }
      for (const side of ["left", "right"]) {
        const items = pts.filter((p) => p.card.dataset.side === side).sort((p, q) => p.y - q.y);
        const half = (p) => p.card.offsetHeight / 2;
        let prev = -Infinity;
        for (const p of items) { // downward pass: keep order, enforce spacing
          p.cy = Math.max(p.y, half(p) + CARD_GAP, prev + (prev === -Infinity ? 0 : half(p) + CARD_GAP));
          prev = p.cy + half(p);
        }
        let next = srect.height - 44; // upward pass: stay above the footer text
        for (const p of items.reverse()) {
          p.cy = Math.min(p.cy, next - half(p));
          next = p.cy - half(p) - CARD_GAP;
        }
      }
      for (const p of pts) {
        p.card.style.top = `${p.cy}px`;
        const left = p.card.dataset.side === "left";
        const x1 = left ? p.card.offsetLeft + p.card.offsetWidth : p.card.offsetLeft;
        p.line.setAttribute("x1", x1); p.line.setAttribute("y1", p.cy);
        p.line.setAttribute("x2", p.x); p.line.setAttribute("y2", p.y);
        p.dot.setAttribute("cx", p.x); p.dot.setAttribute("cy", p.y);
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
        return hex ? c.lerp(new THREE.Color(hex), TINT) : c;
      });
      const attr = ctx.geometry.getAttribute("color");
      const tmp = new THREE.Color();
      for (let i = 0; i < ctx.ids.length; i++) {
        const c = cols[ctx.ids[i]] || BASE;
        if (ctx.est?.[i]) { // estimated areas: pale, so they never read as measured surface
          tmp.copy(c).lerp(ESTIMATED, 0.65);
          attr.setXYZ(i, tmp.r, tmp.g, tmp.b);
        } else attr.setXYZ(i, c.r, c.g, c.b);
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
              className={`callout ${c.side}`} style={{ top: `${22 + 19 * c.slot}%` }}>
              <small>{c.label}</small>
              <div><strong>{c.value}</strong> <span>{c.unit}</span></div>
              {c.demo && <em>DEMO</em>}
            </div>
          ))}
        </>
      )}
      {state === "ready" && hasEstimated && (
        <div className="avatar-est-note">Pale areas: gaps bridged for display (estimated, not measured)</div>
      )}
      {state === "loading" && <div className="viewer-msg">Loading avatar…</div>}
      {state === "error" && <div className="viewer-msg bad">Could not load the avatar mesh.</div>}
      <div className="avatar-foot left">Drag to turn · mirror view</div>
    </div>
  );
}
