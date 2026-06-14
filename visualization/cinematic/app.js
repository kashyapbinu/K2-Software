/**
 * K2 Aerospace — Cinematic Flight View
 * =====================================
 * three.js renderer fed live telemetry from the K2 simulation engine via
 * QWebChannel. Visual-only: all physics happens Python-side; this file just
 * draws the latest state sample as prettily as the GPU allows.
 *
 * Frames: K2 sim is Z-up (z = altitude); three.js is Y-up.
 *         three.(x, y, z) = k2.(x, altitude, y)
 */
import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

// ── Renderer / composer ──────────────────────────────────────────────────────
const container = document.getElementById('scene');
const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
container.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.1, 200000);
camera.position.set(14, 4, 14);

const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
const bloom = new UnrealBloomPass(
  new THREE.Vector2(window.innerWidth, window.innerHeight), 0.55, 0.5, 0.78);
composer.addPass(bloom);
composer.addPass(new OutputPass());

// PBR environment (neutral studio IBL so metallic surfaces have reflections)
const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  composer.setSize(window.innerWidth, window.innerHeight);
});

// ── Sky dome (altitude-aware gradient) + stars + sun ─────────────────────────
const skyUniforms = {
  topColor:     { value: new THREE.Color(0.16, 0.34, 0.72) },
  bottomColor:  { value: new THREE.Color(0.66, 0.78, 0.95) },
  topSpace:     { value: new THREE.Color(0.001, 0.002, 0.01) },
  bottomSpace:  { value: new THREE.Color(0.01, 0.02, 0.06) },
  altFrac:      { value: 0.0 },          // 0 ground → 1 edge of space
};
const skyMat = new THREE.ShaderMaterial({
  uniforms: skyUniforms,
  side: THREE.BackSide, depthWrite: false, fog: false,
  vertexShader: `
    varying vec3 vDir;
    void main() {
      vDir = normalize(position);
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }`,
  fragmentShader: `
    uniform vec3 topColor, bottomColor, topSpace, bottomSpace;
    uniform float altFrac;
    varying vec3 vDir;
    void main() {
      float h = pow(max(vDir.y, 0.0), 0.55);
      vec3 top = mix(topColor, topSpace, altFrac);
      vec3 bot = mix(bottomColor, bottomSpace, altFrac);
      gl_FragColor = vec4(mix(bot, top, h), 1.0);
    }`,
});
const sky = new THREE.Mesh(new THREE.SphereGeometry(90000, 32, 16), skyMat);
scene.add(sky);

// Stars — fade in with altitude
const starGeo = new THREE.BufferGeometry();
{
  const n = 2500, pos = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const u = Math.random() * 2 - 1, t = Math.random() * Math.PI * 2;
    const r = 80000, s = Math.sqrt(1 - u * u);
    pos[i*3] = r*s*Math.cos(t); pos[i*3+1] = Math.abs(r*u) + 2000; pos[i*3+2] = r*s*Math.sin(t);
  }
  starGeo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
}
const starMat = new THREE.PointsMaterial({ color: 0xffffff, size: 90,
  sizeAttenuation: true, transparent: true, opacity: 0, depthWrite: false });
const stars = new THREE.Points(starGeo, starMat);
scene.add(stars);

// Sun: directional light + glow sprite (bloom does the rest)
const sun = new THREE.DirectionalLight(0xfff3e0, 2.6);
sun.position.set(9000, 14000, 6000);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.camera.near = 1; sun.shadow.camera.far = 40000;
const shadowExtent = 60;
sun.shadow.camera.left = -shadowExtent; sun.shadow.camera.right = shadowExtent;
sun.shadow.camera.top = shadowExtent;   sun.shadow.camera.bottom = -shadowExtent;
scene.add(sun);
scene.add(new THREE.AmbientLight(0x8899bb, 0.45));
const sunSprite = new THREE.Sprite(new THREE.SpriteMaterial({
  map: radialTexture('#fffdf2', 0.5), color: 0xfff6d8,
  transparent: true, opacity: 0.95, depthWrite: false }));
sunSprite.scale.setScalar(9000);
sunSprite.position.copy(sun.position).multiplyScalar(4);
scene.add(sunSprite);

// Atmospheric fog (ground haze; sky dome opted out)
scene.fog = new THREE.FogExp2(new THREE.Color(0.62, 0.72, 0.86), 1.6e-5);

// ── Ground + launch pad ──────────────────────────────────────────────────────
function groundTexture() {
  const c = document.createElement('canvas'); c.width = c.height = 1024;
  const g = c.getContext('2d');
  const grad = g.createRadialGradient(512, 512, 40, 512, 512, 512);
  grad.addColorStop(0, '#3a4438'); grad.addColorStop(0.5, '#2e372f');
  grad.addColorStop(1, '#222a26');
  g.fillStyle = grad; g.fillRect(0, 0, 1024, 1024);
  g.strokeStyle = 'rgba(150,170,160,0.10)'; g.lineWidth = 1;
  for (let i = 0; i <= 32; i++) {
    const p = i * 32;
    g.beginPath(); g.moveTo(p, 0); g.lineTo(p, 1024); g.stroke();
    g.beginPath(); g.moveTo(0, p); g.lineTo(1024, p); g.stroke();
  }
  const tex = new THREE.CanvasTexture(c);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping; tex.repeat.set(60, 60);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}
const ground = new THREE.Mesh(
  new THREE.CircleGeometry(60000, 64),
  new THREE.MeshStandardMaterial({ map: groundTexture(), roughness: 1.0, metalness: 0.0 }));
ground.rotation.x = -Math.PI / 2;
ground.receiveShadow = true;
scene.add(ground);

// Launch pad — rebuilt to scale with the rocket so the rail doesn't dwarf it
const pad = new THREE.Group();
scene.add(pad);
function buildPad(L, bodyR) {
  pad.clear();
  const concrete = new THREE.MeshStandardMaterial({ color: 0x9aa0a6, roughness: 0.9 });
  const steel = new THREE.MeshStandardMaterial({ color: 0x6f7a85, roughness: 0.4, metalness: 0.9 });
  const slab = new THREE.Mesh(new THREE.CylinderGeometry(L * 1.4, L * 1.55, L * 0.08, 32), concrete);
  slab.position.y = L * 0.04; slab.receiveShadow = true; pad.add(slab);
  const railH = L * 1.35;                               // rail ≈ 1.3 rocket lengths
  const railW = Math.max(bodyR * 0.5, 0.012);
  const rail = new THREE.Mesh(new THREE.BoxGeometry(railW, railH, railW), steel);
  rail.position.set(bodyR * 1.6, railH / 2 + L * 0.06, 0);
  rail.castShadow = true; pad.add(rail);
  for (const a of [0, Math.PI/2, Math.PI, 3*Math.PI/2]) {
    const towH = L * 1.6, towD = L * 3.0;
    const tower = new THREE.Mesh(new THREE.BoxGeometry(L * 0.06, towH, L * 0.06), steel);
    tower.position.set(Math.cos(a) * towD, towH / 2, Math.sin(a) * towD);
    tower.castShadow = true; pad.add(tower);
    const lamp = new THREE.Mesh(new THREE.SphereGeometry(L * 0.05, 8, 8),
      new THREE.MeshBasicMaterial({ color: 0xffeebb }));
    lamp.position.set(Math.cos(a) * towD, towH + L * 0.04, Math.sin(a) * towD);
    pad.add(lamp);                                  // emissive → bloom glow
  }
}
buildPad(1.2, 0.033);

// ── Rocket built from K2 geometry dict ───────────────────────────────────────
const rocket = new THREE.Group();
scene.add(rocket);
let rocketLen = 1.2, maxThrust = 200, recovery = { drogue_cd_area: 0.5, main_cd_area: 3.0 };

function buildRocket(geo) {
  rocket.clear();
  const L = Math.max(geo.length || 1.2, 0.3);
  const R = Math.max(geo.body_radius || 0.033, 0.01);
  const NL = Math.min(geo.nose_length || L * 0.25, L * 0.8);
  rocketLen = L;

  // Body+nose as a lathe: base y=0 → nose tip y=L (tangent-ogive profile)
  const prof = [];
  prof.push(new THREE.Vector2(R * 0.82, 0));               // nozzle lip
  prof.push(new THREE.Vector2(R, 0.04 * L));
  prof.push(new THREE.Vector2(R, L - NL));
  for (let i = 1; i <= 12; i++) {                          // ogive
    const f = i / 12, y = L - NL + f * NL;
    const rho = (R * R + NL * NL) / (2 * R);
    const r = Math.sqrt(Math.max(rho * rho - Math.pow(f * NL, 2), 0)) - (rho - R);
    prof.push(new THREE.Vector2(Math.max(r * (1 - f * 0.02), 0.001), y));
  }
  prof.push(new THREE.Vector2(0.001, L));
  const bodyMat = new THREE.MeshStandardMaterial({
    color: 0xf4f6f8, metalness: 0.65, roughness: 0.28 });
  const body = new THREE.Mesh(new THREE.LatheGeometry(prof, 48), bodyMat);
  body.castShadow = true;
  rocket.add(body);

  // Nose accent band
  const band = new THREE.Mesh(
    new THREE.CylinderGeometry(R * 1.002, R * 1.002, L * 0.05, 48),
    new THREE.MeshStandardMaterial({ color: 0xd64533, metalness: 0.4, roughness: 0.35 }));
  band.position.y = L - NL - L * 0.03;
  rocket.add(band);

  // Fins (trapezoid extrusions)
  const n = geo.fin_count || 4;
  const cr = geo.fin_root || L * 0.12, ct = geo.fin_tip || cr * 0.5;
  const h = geo.fin_height || R * 1.6;
  const sweep = Math.tan((geo.fin_sweep_deg || 30) * Math.PI / 180) * h;
  const finShape = new THREE.Shape();
  finShape.moveTo(0, 0); finShape.lineTo(0, cr);
  finShape.lineTo(h, cr - sweep > 0 ? cr - sweep : cr * 0.2);
  finShape.lineTo(h, Math.max(cr - sweep - ct, 0)); finShape.lineTo(0, 0);
  const finGeo = new THREE.ExtrudeGeometry(finShape,
    { depth: geo.fin_thick || 0.003, bevelEnabled: false });
  finGeo.translate(R * 0.98, 0.05 * L, -(geo.fin_thick || 0.003) / 2);
  const finMat = new THREE.MeshStandardMaterial({
    color: 0x20262e, metalness: 0.55, roughness: 0.35 });
  for (let i = 0; i < n; i++) {
    const fin = new THREE.Mesh(finGeo, finMat);
    fin.castShadow = true;
    const holder = new THREE.Group();
    holder.add(fin); holder.rotation.y = (i / n) * Math.PI * 2;
    rocket.add(holder);
  }

  // Nozzle (emissive inner)
  const noz = new THREE.Mesh(new THREE.CylinderGeometry(R * 0.5, R * 0.78, L * 0.06, 24),
    new THREE.MeshStandardMaterial({ color: 0x2a2f36, metalness: 0.9, roughness: 0.4 }));
  noz.position.y = -L * 0.02;
  rocket.add(noz);

  buildPad(L, R);
  rebuildChute();
  rocket.position.set(0, 0.3, 0);
}

// ── Exhaust: flame sprites + dynamic light + smoke pool ──────────────────────
function radialTexture(color, mid = 0.25) {
  const c = document.createElement('canvas'); c.width = c.height = 128;
  const g = c.getContext('2d');
  const grad = g.createRadialGradient(64, 64, 2, 64, 64, 64);
  grad.addColorStop(0, color);
  grad.addColorStop(mid, hexA(color, 0.55));
  grad.addColorStop(1, hexA(color, 0));
  g.fillStyle = grad; g.fillRect(0, 0, 128, 128);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace;
  return t;
}
function hexA(hex, a) {
  const c = new THREE.Color(hex);
  return `rgba(${(c.r*255)|0},${(c.g*255)|0},${(c.b*255)|0},${a})`;
}

// Plume: vertically-stretched additive sprites anchored at the nozzle
// (sprite.center y=1 → scaling extends downward in group-local space),
// plus a chain of mach diamonds inside the core. The whole group is
// rotated with the rocket so "down" is always along the body axis.
function plumeSprite(hex, w, h) {
  const s = new THREE.Sprite(new THREE.SpriteMaterial({
    map: radialTexture(hex, 0.3), color: hex, blending: THREE.AdditiveBlending,
    transparent: true, depthWrite: false }));
  s.center.set(0.5, 1.0);          // anchor at top edge — grows downward
  s.userData = { w, h };
  return s;
}
const flameGroup = new THREE.Group();
const flameOuter = plumeSprite('#ff4d12', 1.5, 2.6);
const flameMid   = plumeSprite('#ff9a2e', 1.0, 3.4);
const flameCore  = plumeSprite('#fff3c4', 0.55, 4.0);
flameGroup.add(flameOuter, flameMid, flameCore);

const DIAMONDS = 5;
const diamonds = [];
for (let i = 0; i < DIAMONDS; i++) {
  const d = new THREE.Sprite(new THREE.SpriteMaterial({
    map: radialTexture('#ffffff', 0.2), color: 0xfff8e0,
    blending: THREE.AdditiveBlending, transparent: true, depthWrite: false }));
  d.visible = false;
  flameGroup.add(d);
  diamonds.push(d);
}
const flameLight = new THREE.PointLight(0xffa040, 0, 120, 2);
flameGroup.add(flameLight);
const groundGlow = new THREE.PointLight(0xff8830, 0, 60, 2);  // pad wash at liftoff
scene.add(flameGroup, groundGlow);
let flick = 1.0;                                   // low-pass flame flutter

// Smoke: pooled sprites with per-particle rotation, clumpy multi-blob
// texture, flame-lit birth colour, in/out opacity ramp, and a ground-hugging
// pad cloud at liftoff.
function smokeTexture(seed) {
  const c = document.createElement('canvas'); c.width = c.height = 256;
  const g = c.getContext('2d');
  let s = seed;
  const rnd = () => (s = (s * 16807) % 2147483647) / 2147483647;
  for (let i = 0; i < 14; i++) {
    const a = rnd() * Math.PI * 2, r = rnd() * 62;
    const x = 128 + Math.cos(a) * r, y = 128 + Math.sin(a) * r;
    const rad = 36 + rnd() * 46;
    const grad = g.createRadialGradient(x, y, 2, x, y, rad);
    const al = 0.10 + rnd() * 0.13;
    grad.addColorStop(0, `rgba(235,238,242,${al})`);
    grad.addColorStop(0.6, `rgba(225,228,233,${al * 0.55})`);
    grad.addColorStop(1, 'rgba(220,224,230,0)');
    g.fillStyle = grad;
    g.beginPath(); g.arc(x, y, rad, 0, Math.PI * 2); g.fill();
  }
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace;
  return t;
}
const smokeTextures = [smokeTexture(1234567), smokeTexture(7654321)];
const SMOKE_N = 420;
const smoke = [];
const _birthTint = new THREE.Color(1.0, 0.82, 0.62);  // flame-lit
const _hotGrey  = new THREE.Color(0.93, 0.93, 0.95);  // fresh steam-white
const _coldGrey = new THREE.Color(0.62, 0.64, 0.68);  // dispersed
for (let i = 0; i < SMOKE_N; i++) {
  const s = new THREE.Sprite(new THREE.SpriteMaterial({
    map: smokeTextures[i % 2], transparent: true, opacity: 0, depthWrite: false,
    rotation: Math.random() * Math.PI * 2 }));
  s.visible = false; scene.add(s);
  smoke.push({ spr: s, age: 0, life: 1, grow: 1, rot: 0, vel: new THREE.Vector3() });
}
let smokeCursor = 0;
function spawnSmoke(pos, bodyDir, exhaustSpeed, opts = {}) {
  const p = smoke[smokeCursor]; smokeCursor = (smokeCursor + 1) % SMOKE_N;
  p.spr.visible = true; p.age = 0;
  p.life = opts.life ?? (5 + Math.random() * 5);
  p.grow = opts.grow ?? (0.9 + Math.random() * 0.5);
  p.rot = (Math.random() - 0.5) * 1.4;
  p.spr.position.copy(pos);
  p.spr.position.x += (Math.random() - 0.5) * rocketLen * 0.25;
  p.spr.position.z += (Math.random() - 0.5) * rocketLen * 0.25;
  if (opts.vel) p.vel.copy(opts.vel);
  else {
    p.vel.copy(bodyDir).multiplyScalar(-exhaustSpeed * (0.12 + Math.random() * 0.1));
    p.vel.x += (Math.random() - 0.5) * 3.5;
    p.vel.z += (Math.random() - 0.5) * 3.5;
  }
  p.spr.scale.setScalar(opts.size ?? rocketLen * (0.3 + Math.random() * 0.25));
  p.spr.material.opacity = 0;
}

// ── Trail ribbon (velocity-coloured, additive → bloom glow) ──────────────────
const TRAIL_MAX = 4000;
const trailPos = new Float32Array(TRAIL_MAX * 3);
const trailCol = new Float32Array(TRAIL_MAX * 3);
let trailCount = 0;
const trailGeo = new THREE.BufferGeometry();
trailGeo.setAttribute('position', new THREE.BufferAttribute(trailPos, 3));
trailGeo.setAttribute('color', new THREE.BufferAttribute(trailCol, 3));
const trail = new THREE.Line(trailGeo, new THREE.LineBasicMaterial({
  vertexColors: true, transparent: true, opacity: 0.85,
  blending: THREE.AdditiveBlending, depthWrite: false }));
trail.frustumCulled = false;
scene.add(trail);
const lastTrailPt = new THREE.Vector3(1e9, 1e9, 1e9);
function pushTrail(p, mach) {
  if (trailCount >= TRAIL_MAX) return;
  if (p.distanceTo(lastTrailPt) < rocketLen * 0.6) return;
  lastTrailPt.copy(p);
  const i = trailCount * 3;
  trailPos[i] = p.x; trailPos[i+1] = p.y; trailPos[i+2] = p.z;
  const c = new THREE.Color();
  if (mach < 0.6) c.setRGB(0.35, 0.62, 1.0);
  else if (mach < 1.0) c.lerpColors(new THREE.Color(0.35,0.62,1.0), new THREE.Color(1,1,1), (mach-0.6)/0.4);
  else c.lerpColors(new THREE.Color(1,1,1), new THREE.Color(1,0.45,0.15), Math.min((mach-1)/1.5, 1));
  trailCol[i] = c.r; trailCol[i+1] = c.g; trailCol[i+2] = c.b;
  trailCount++;
  trailGeo.setDrawRange(0, trailCount);
  trailGeo.attributes.position.needsUpdate = true;
  trailGeo.attributes.color.needsUpdate = true;
}

// ── Transonic vapour cone ────────────────────────────────────────────────────
const vapor = new THREE.Mesh(
  new THREE.ConeGeometry(1, 1.6, 32, 1, true),
  new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0,
    side: THREE.DoubleSide, depthWrite: false }));
vapor.visible = false;
scene.add(vapor);

// ── Parachute ────────────────────────────────────────────────────────────────
let chute = null;
function rebuildChute() {
  if (chute) { scene.remove(chute); chute = null; }
  chute = new THREE.Group();
  const canopy = new THREE.Mesh(
    new THREE.SphereGeometry(1, 24, 12, 0, Math.PI * 2, 0, Math.PI / 2),
    new THREE.MeshStandardMaterial({ color: 0xff7733, roughness: 0.85,
      side: THREE.DoubleSide, transparent: true, opacity: 0.92 }));
  chute.add(canopy);
  const lineMat = new THREE.LineBasicMaterial({ color: 0xccd2da, transparent: true, opacity: 0.7 });
  const linePts = [];
  for (let i = 0; i < 8; i++) {
    const a = (i / 8) * Math.PI * 2;
    linePts.push(new THREE.Vector3(Math.cos(a) * 0.95, -0.05, Math.sin(a) * 0.95));
    linePts.push(new THREE.Vector3(0, -2.2, 0));
  }
  chute.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(linePts), lineMat));
  chute.visible = false;
  scene.add(chute);
}

// ── Telemetry state (latest sample; render side smooths) ─────────────────────
const T = {
  t: 0, x: 0, y: 0, alt: 0, vx: 0, vy: 0, vz: 0,
  pitch: Math.PI / 2, yaw: 0, roll: 0,
  thrust: 0, mach: 0, acc: 0, q: 0, phase: 'Pre-Launch', running: false,
};
const rocketPos = new THREE.Vector3(0, 0.3, 0);
const rocketQuat = new THREE.Quaternion();
const tmpV = new THREE.Vector3(), tmpV2 = new THREE.Vector3();
const UP = new THREE.Vector3(0, 1, 0);

function applyTelemetry(d) {
  // Record while flying, plus the first post-flight sample (the landed state)
  // so a replay reaches touchdown and post-flight overlays rebuild.
  if (!replay.active && (d.running ||
      (rec.samples.length && rec.samples[rec.samples.length - 1].running)))
    rec.push(d);
  Object.assign(T, d);
  hud.update();
  graphs.push();
  graphs.draw();
  events.check(d.phase, d);
}

// ── HUD ──────────────────────────────────────────────────────────────────────
const hud = {
  el: {
    alt: document.getElementById('v-alt'),   bAlt: document.getElementById('b-alt'),
    vel: document.getElementById('v-vel'),   bVel: document.getElementById('b-vel'),
    mach: document.getElementById('v-mach'), bMach: document.getElementById('b-mach'),
    thr: document.getElementById('v-thrust'),bThr: document.getElementById('b-thrust'),
    met: document.getElementById('met'),     phase: document.getElementById('phase'),
  },
  maxAlt: 1, maxVel: 1,
  update() {
    this.maxAlt = Math.max(this.maxAlt, T.alt);
    const speed = Math.hypot(T.vx, T.vy, T.vz);
    this.maxVel = Math.max(this.maxVel, speed);
    this.el.alt.textContent = T.alt >= 10000 ? (T.alt/1000).toFixed(2) + 'k' : T.alt.toFixed(0);
    this.el.vel.textContent = speed.toFixed(0);
    this.el.mach.textContent = T.mach.toFixed(2);
    this.el.thr.textContent = T.thrust.toFixed(0);
    this.el.bAlt.style.width  = Math.min(T.alt / this.maxAlt * 100, 100) + '%';
    this.el.bVel.style.width  = Math.min(speed / this.maxVel * 100, 100) + '%';
    this.el.bMach.style.width = Math.min(T.mach / 2 * 100, 100) + '%';
    this.el.bThr.style.width  = Math.min(T.thrust / Math.max(maxThrust, 1) * 100, 100) + '%';
    const h = Math.floor(T.t / 3600), m = Math.floor((T.t % 3600) / 60), s = Math.floor(T.t % 60);
    this.el.met.textContent =
      `T+ ${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}`;
    this.el.phase.textContent = T.phase;
    this.el.phase.className = '';
    if (/boost|ignition/i.test(T.phase)) this.el.phase.classList.add('boost');
    else if (/descent/i.test(T.phase)) this.el.phase.classList.add('chute');
    else if (/landed/i.test(T.phase)) this.el.phase.classList.add('landed');
  },
};

// ── Mini telemetry graphs (sparkline strip, mirrors Mission Visualizer) ─────
const graphs = {
  specs: [
    { id: 'g-alt',  label: 'ALT m',     get: () => T.alt },
    { id: 'g-vel',  label: 'VEL m/s',   get: () => Math.hypot(T.vx, T.vy, T.vz) },
    { id: 'g-mach', label: 'MACH',      get: () => T.mach },
    { id: 'g-acc',  label: 'ACC m/s²',  get: () => T.acc },
    { id: 'g-q',    label: 'DYN-Q Pa',  get: () => T.q },
  ],
  MAX: 2400, DPR: Math.min(window.devicePixelRatio || 1, 2),
  t: [], series: [], lastT: -1, lastDraw: 0,
  init() {
    this.series = this.specs.map(() => []);
    this.cv = this.specs.map(s => document.getElementById(s.id));
    this.cx = this.cv.map(c => {
      c.width = 108 * this.DPR; c.height = 54 * this.DPR;
      return c.getContext('2d');
    });
    this.draw(true);
  },
  push() {
    if (!T.running || T.t <= this.lastT) return;
    this.lastT = T.t;
    this.t.push(T.t);
    this.specs.forEach((s, i) => this.series[i].push(s.get()));
    if (this.t.length > this.MAX) {            // halve: drop every 2nd sample
      const keep = (_, i) => i % 2 === 0;
      this.t = this.t.filter(keep);
      this.series = this.series.map(a => a.filter(keep));
    }
  },
  clear() {
    this.t = []; this.series = this.specs.map(() => []); this.lastT = -1;
    this.draw(true);
  },
  draw(force = false) {
    const now = performance.now();
    if (!force && now - this.lastDraw < 300) return;
    this.lastDraw = now;
    const D = this.DPR;
    for (let k = 0; k < this.specs.length; k++) {
      const g = this.cx[k], W = this.cv[k].width, H = this.cv[k].height;
      g.clearRect(0, 0, W, H);
      g.fillStyle = 'rgba(255,255,255,0.40)';
      g.font = `600 ${8.5 * D}px Bahnschrift, "Segoe UI", sans-serif`;
      g.textAlign = 'left'; g.textBaseline = 'top';
      g.fillText(this.specs[k].label, 2 * D, 1 * D);
      const data = this.series[k], n = data.length;
      if (n < 2) continue;
      let lo = Infinity, hi = -Infinity;
      for (const v of data) { if (v < lo) lo = v; if (v > hi) hi = v; }
      if (hi - lo < 1e-9) { lo -= 0.5; hi += 0.5; }
      const t0 = this.t[0], t1 = this.t[n - 1], dt = Math.max(t1 - t0, 1e-9);
      const yPad = 13 * D, plotH = H - yPad - 2 * D;
      g.strokeStyle = 'rgba(255,255,255,0.85)'; g.lineWidth = 1.1 * D;
      g.beginPath();
      for (let i = 0; i < n; i++) {
        const x = (this.t[i] - t0) / dt * (W - 2 * D) + D;
        const y = yPad + (1 - (data[i] - lo) / (hi - lo)) * plotH;
        i === 0 ? g.moveTo(x, y) : g.lineTo(x, y);
      }
      g.stroke();
      const last = data[n - 1];
      g.fillStyle = '#ffffff';
      g.font = `300 ${10 * D}px Bahnschrift, "Segoe UI", sans-serif`;
      g.textAlign = 'right';
      g.fillText(last >= 10000 ? (last / 1000).toFixed(1) + 'k'
                 : last >= 100 ? last.toFixed(0) : last.toFixed(2),
                 W - 2 * D, 1 * D);
    }
  },
};
graphs.init();
window.__k2graphs = graphs;          // debug/test handle

// ── 3D event markers (glowing pin + label at the trajectory point) ──────────
const markers = [];
const MARKER_COLORS = {
  'LIFTOFF': '#f0883e', 'MAX-Q': '#f85149', 'BURNOUT': '#d29922',
  'APOGEE': '#58a6ff', 'DROGUE': '#7ee787', 'MAIN': '#7ee787',
  'TOUCHDOWN': '#3fb950',
};
function labelTexture(title, sub, colorHex) {
  const c = document.createElement('canvas'); c.width = 512; c.height = 144;
  const g = c.getContext('2d');
  g.fillStyle = 'rgba(13,17,23,0.72)';
  g.beginPath(); g.roundRect(8, 8, 496, 128, 22); g.fill();
  g.strokeStyle = colorHex; g.lineWidth = 4; g.stroke();
  g.fillStyle = colorHex;
  g.font = '700 52px "Segoe UI", sans-serif';
  g.textAlign = 'center';
  g.fillText(title, 256, 66);
  g.fillStyle = '#c9d1d9';
  g.font = '400 36px "Cascadia Code", monospace';
  g.fillText(sub, 256, 114);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace;
  return t;
}
const markerLog = [];               // names in firing order (debug/test)
function addMarker(name, pos, alt, t) {
  markerLog.push(name);
  const node = document.getElementById('tl-' + name);
  if (node) {
    node.classList.add('done');
    // Show the altitude this stage was reached at, under its label.
    const altEl = node.querySelector('b');
    if (altEl) altEl.textContent = alt >= 1000
      ? (alt / 1000).toFixed(2) + ' km' : Math.round(alt) + ' m';
  }
  const color = new THREE.Color(MARKER_COLORS[name] || '#58a6ff');
  const grp = new THREE.Group();
  // glowing tip
  const tip = new THREE.Mesh(new THREE.SphereGeometry(rocketLen * 0.18, 12, 12),
    new THREE.MeshBasicMaterial({ color }));
  grp.add(tip);
  // vertical beam (additive — bloom picks it up)
  const beamH = rocketLen * 5;
  const beam = new THREE.Mesh(new THREE.CylinderGeometry(rocketLen * 0.03, rocketLen * 0.03, beamH, 8),
    new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.4,
      blending: THREE.AdditiveBlending, depthWrite: false }));
  beam.position.y = -beamH / 2;
  grp.add(beam);
  // floating label, distance-scaled in the loop
  const sub = `${alt >= 1000 ? (alt/1000).toFixed(2) + ' km' : alt.toFixed(0) + ' m'}  ·  T+${t.toFixed(1)}s`;
  const label = new THREE.Sprite(new THREE.SpriteMaterial({
    map: labelTexture(name, sub, MARKER_COLORS[name] || '#58a6ff'),
    transparent: true, depthWrite: false }));
  label.center.set(0.5, 0);
  label.position.y = rocketLen * 0.6;
  grp.add(label);
  grp.userData.label = label;
  grp.position.copy(pos);
  scene.add(grp);
  markers.push(grp);
}
function clearMarkers() {
  for (const m of markers) scene.remove(m);
  markers.length = 0;
  markerLog.length = 0;
  document.querySelectorAll('.tl-node').forEach(n => {
    n.classList.remove('done');
    const b = n.querySelector('b');
    if (b) b.textContent = '';
  });
}

// ── Event banners + marker placement ─────────────────────────────────────────
const events = {
  prev: 'Pre-Launch', maxQfired: false, peakQ: 0,
  peakQpos: new THREE.Vector3(), peakQalt: 0, peakQt: 0,
  banner(text) {
    const b = document.getElementById('banner');
    b.textContent = text; b.style.opacity = 1;
    clearTimeout(this._to);
    this._to = setTimeout(() => { b.style.opacity = 0; }, 1800);
  },
  fire(name, d) {
    this.banner(name === 'DROGUE' ? 'DROGUE DEPLOY' : name === 'MAIN' ? 'MAIN DEPLOY' : name);
    addMarker(name, rocketPos.clone(), d.alt ?? T.alt, d.t ?? T.t);
    if (name === 'TOUCHDOWN') envelope.build();
  },
  check(phase, d) {
    // Max-Q: track peak dynamic pressure; fire once it clearly falls off
    const q = d.q || 0;
    if (q > this.peakQ) {
      this.peakQ = q;
      this.peakQpos.copy(rocketPos); this.peakQalt = d.alt; this.peakQt = d.t;
    } else if (!this.maxQfired && this.peakQ > 500 && q < this.peakQ * 0.95) {
      this.maxQfired = true;
      this.banner('MAX-Q');
      addMarker('MAX-Q', this.peakQpos.clone(), this.peakQalt, this.peakQt);
    }
    if (phase !== this.prev) {
      if (/boost/i.test(phase) && /ignition|pre/i.test(this.prev)) this.fire('LIFTOFF', d);
      else if (/coast/i.test(phase) && /boost/i.test(this.prev)) this.fire('BURNOUT', d);
      else if (/apogee/i.test(phase)) this.fire('APOGEE', d);
      else if (/drogue/i.test(phase)) this.fire('DROGUE', d);
      else if (/main/i.test(phase) && !/pre/i.test(phase)) this.fire('MAIN', d);
      else if (/landed/i.test(phase)) this.fire('TOUCHDOWN', d);
      this.prev = phase;
    }
  },
};

// ── Flight recorder + replay ─────────────────────────────────────────────────
const rec = {
  samples: [], MAX: 36000,                 // ≈20 min at 30 Hz before halving
  push(d) {
    this.samples.push(d);
    if (this.samples.length > this.MAX)
      this.samples = this.samples.filter((_, i) => i % 2 === 0);
  },
  clear() { this.samples = []; },
  t0() { return this.samples.length ? this.samples[0].t : 0; },
  duration() {
    const s = this.samples;
    return s.length ? s[s.length - 1].t - s[0].t : 0;
  },
};

function fmtClock(t) {
  const m = Math.floor(t / 60), s = Math.floor(t % 60);
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

const replay = {
  active: false, playing: false, speed: 1, time: 0, idx: 0,
  el: {
    bar:  document.getElementById('replay'),
    play: document.getElementById('rp-play'),
    scrub: document.getElementById('rp-scrub'),
    fill: document.getElementById('rp-fill'),
    dot:  document.getElementById('rp-dot'),
    time: document.getElementById('rp-time'),
    live: document.getElementById('rp-live'),
  },
  enter() {
    if (rec.samples.length < 2) return;
    this.active = true; this.playing = true;
    this.idx = 0; this.time = rec.t0();
    resetVisuals();
  },
  exit() {
    if (!this.active) return;
    this.active = false; this.playing = false;
    if (rec.samples.length) this.seekFeed(rec.samples.length - 1);  // back to latest
    T.running = false;              // flight is over; keep the replay bar offered
  },
  // Rebuild flight state up to samples[endIdx] after a visual reset.
  // Fast path: trail/markers/graphs per sample, HUD/DOM only once at the end.
  seekFeed(endIdx) {
    resetVisuals();
    const s = rec.samples;
    for (let i = 0; i <= endIdx; i++) {
      const d = s[i];
      Object.assign(T, d);
      rocketPos.set(d.x, Math.max(d.alt, 0) + 0.3, d.y);
      if (d.running && d.alt > 1) pushTrail(rocketPos, d.mach);
      graphs.push();
      events.check(d.phase, d);
    }
    this.idx = endIdx;
    this.time = s[endIdx].t;
    hud.update();
    graphs.draw(true);
  },
  seekFrac(f) {
    if (rec.samples.length < 2) return;
    const tt = rec.t0() + rec.duration() * THREE.MathUtils.clamp(f, 0, 1);
    const s = rec.samples;
    let lo = 0;
    while (lo + 1 < s.length && s[lo + 1].t <= tt) lo++;
    this.seekFeed(lo);
  },
  step(dt) {
    const show = this.active || (rec.samples.length > 10 && !T.running);
    this.el.bar.classList.toggle('visible', show);
    if (!show) return;
    this.el.live.style.display = this.active ? '' : 'none';
    if (this.active && this.playing) {
      this.time += dt * this.speed;
      const s = rec.samples;
      while (this.idx + 1 < s.length && s[this.idx + 1].t <= this.time) {
        this.idx++;
        applyTelemetry(s[this.idx]);
      }
      if (this.idx >= s.length - 1) this.playing = false;   // hold on last frame
    }
    const f = rec.duration() > 0
      ? THREE.MathUtils.clamp((this.time - rec.t0()) / rec.duration(), 0, 1) : 0;
    const pct = (this.active ? f : 1) * 100;
    this.el.fill.style.width = pct + '%';
    this.el.dot.style.left = pct + '%';
    this.el.time.textContent =
      `${fmtClock(this.active ? this.time - rec.t0() : rec.duration())} / ${fmtClock(rec.duration())}`;
    this.el.play.textContent = !this.active ? '▶ REPLAY' : (this.playing ? '❚❚' : '▶');
  },
};
window.__k2replay = { rec, replay };       // debug/test handle
window.__k2feed = applyTelemetry;          // debug/test handle

replay.el.play.addEventListener('click', () => {
  if (!replay.active || replay.idx >= rec.samples.length - 1) replay.enter();
  else replay.playing = !replay.playing;
});
replay.el.live.addEventListener('click', () => replay.exit());
document.querySelectorAll('#rp-speeds button').forEach(b => {
  b.addEventListener('click', () => {
    replay.speed = parseFloat(b.dataset.spd);
    document.querySelectorAll('#rp-speeds button')
      .forEach(x => x.classList.toggle('active', x === b));
  });
});
{ // scrub: click or drag anywhere on the track
  let scrubbing = false, wasPlaying = false;
  const toFrac = (e) => {
    const r = replay.el.scrub.getBoundingClientRect();
    return (e.clientX - r.left) / Math.max(r.width, 1);
  };
  replay.el.scrub.addEventListener('pointerdown', (e) => {
    if (rec.samples.length < 2) return;
    if (!replay.active) { replay.enter(); replay.playing = false; }
    scrubbing = true; wasPlaying = replay.playing; replay.playing = false;
    replay.el.scrub.setPointerCapture(e.pointerId);
    replay.seekFrac(toFrac(e));
  });
  replay.el.scrub.addEventListener('pointermove', (e) => {
    if (scrubbing) replay.seekFrac(toFrac(e));
  });
  replay.el.scrub.addEventListener('pointerup', () => {
    if (scrubbing) { scrubbing = false; replay.playing = wasPlaying; }
  });
}

// ── Flight envelope (post-flight mission overlay, mirrors Mission Visualizer) ─
let envParams = { target_apogee: 0, recovery_radius: 1000,
                  wind_speed: 0, wind_dir_deg: 0, descent_rate: 5 };
const ENV = { ok: 0x3fb950, warn: 0xd29922, bad: 0xf85149,
              predicted: 0x56d364, drift: 0xd29922, target: 0x3fb950 };

function groundEllipse(cx, cz, a, b, ang, n = 64, y = 0.4) {
  const ca = Math.cos(ang), sa = Math.sin(ang), pts = [];
  for (let i = 0; i < n; i++) {
    const t = 2 * Math.PI * i / n, ex = a * Math.cos(t), ez = b * Math.sin(t);
    pts.push(new THREE.Vector3(cx + ex * ca - ez * sa, y, cz + ex * sa + ez * ca));
  }
  return pts;
}
function fanGeometry(apex, rim) {
  const pts = [apex, ...rim];
  const pos = new Float32Array(pts.length * 3);
  pts.forEach((p, i) => { pos[i * 3] = p.x; pos[i * 3 + 1] = p.y; pos[i * 3 + 2] = p.z; });
  const idx = [];
  for (let i = 0; i < rim.length; i++) idx.push(0, 1 + i, 1 + (i + 1) % rim.length);
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setIndex(idx);
  geo.computeVertexNormals();
  return geo;
}

const envelope = {
  group: null, sprites: [], visible: true,
  clear() {
    if (this.group) { scene.remove(this.group); this.group = null; }
    this.sprites.length = 0;
  },
  label(text, sub, pos, colorHex) {
    const spr = new THREE.Sprite(new THREE.SpriteMaterial({
      map: labelTexture(text, sub, colorHex), transparent: true, depthWrite: false }));
    spr.center.set(0.5, 0);
    spr.position.copy(pos);
    this.group.add(spr);
    this.sprites.push(spr);
  },
  flat(mat) { return new THREE.MeshBasicMaterial({
    transparent: true, side: THREE.DoubleSide, depthWrite: false, ...mat }); },
  build() {
    this.clear();
    const s = rec.samples;
    if (s.length < 2) return;
    this.group = new THREE.Group();
    this.group.visible = this.visible;

    const launch = s[0], land = s[s.length - 1];
    let apo = s[0];
    for (const d of s) if (d.alt > apo.alt) apo = d;

    const R = Math.max(envParams.recovery_radius, 1);
    const dist = Math.hypot(land.x - launch.x, land.y - launch.y);
    const outCol = dist <= R * 0.8 ? ENV.ok : dist <= R ? ENV.warn : ENV.bad;
    const outName = dist <= R * 0.8 ? 'SAFE' : dist <= R ? 'MARGINAL' : 'OUTSIDE';
    const hex = '#' + outCol.toString(16).padStart(6, '0');

    // Recovery zone: outcome-coloured ring + faint fill (k2 x,y → three x,z)
    const ring = new THREE.Mesh(new THREE.RingGeometry(R * 0.985, R, 96),
      this.flat({ color: outCol, opacity: 0.85 }));
    ring.rotation.x = -Math.PI / 2;
    ring.position.set(launch.x, 0.35, launch.y);
    this.group.add(ring);
    const fill = new THREE.Mesh(new THREE.CircleGeometry(R, 96),
      this.flat({ color: outCol, opacity: 0.05 }));
    fill.rotation.x = -Math.PI / 2;
    fill.position.set(launch.x, 0.25, launch.y);
    this.group.add(fill);
    this.label('RECOVERY ZONE', `R ${R.toFixed(0)} m`,
      new THREE.Vector3(launch.x, 2, launch.y + R), hex);

    // Launch pad marker
    const pad = new THREE.Mesh(
      new THREE.RingGeometry(R * 0.006, R * 0.012, 32),
      this.flat({ color: 0xf0f6fc, opacity: 0.9 }));
    pad.rotation.x = -Math.PI / 2;
    pad.position.set(launch.x, 0.45, launch.y);
    this.group.add(pad);

    // Target apogee plane
    if (envParams.target_apogee > 0) {
      const size = Math.max(apo.alt, R * 2.5) * 0.8;
      const plane = new THREE.Mesh(new THREE.PlaneGeometry(size, size),
        this.flat({ color: ENV.target, opacity: 0.07 }));
      plane.rotation.x = -Math.PI / 2;
      plane.position.set(launch.x, envParams.target_apogee, launch.y);
      this.group.add(plane);
      const delta = apo.alt - envParams.target_apogee;
      this.label('TARGET APOGEE', `${envParams.target_apogee.toFixed(0)} m  ·  ${delta >= 0 ? '+' : ''}${delta.toFixed(0)} m actual`,
        new THREE.Vector3(launch.x - size / 2, envParams.target_apogee, launch.y), '#3fb950');
    }

    // Predicted landing (wind drift from apogee) + drift cone
    const dr = Math.max(envParams.descent_rate, 0.01);
    if (apo.alt > 1) {
      const drift = envParams.wind_speed * (apo.alt / dr);
      const blow = (envParams.wind_dir_deg + 180) * Math.PI / 180;
      const px = apo.x + drift * Math.cos(blow);
      const pz = apo.y + drift * Math.sin(blow);
      const a = Math.max(R * 0.2, drift * 0.25, 50), b = a * 0.6;
      const rim = groundEllipse(px, pz, a, b, blow);
      const zone = new THREE.Mesh(
        fanGeometry(new THREE.Vector3(px, 0.45, pz), rim),
        this.flat({ color: ENV.predicted, opacity: 0.12 }));
      this.group.add(zone);
      const outline = new THREE.LineLoop(
        new THREE.BufferGeometry().setFromPoints(rim),
        new THREE.LineBasicMaterial({ color: ENV.predicted, transparent: true, opacity: 0.8 }));
      this.group.add(outline);
      this.label('EXPECTED LANDING',
        `${Math.hypot(px - launch.x, pz - launch.y).toFixed(0)} m downrange`,
        new THREE.Vector3(px, 2, pz + b), '#56d364');

      // Drift cone: apogee → predicted landing ellipse
      const cone = new THREE.Mesh(
        fanGeometry(new THREE.Vector3(apo.x, apo.alt, apo.y), rim),
        this.flat({ color: ENV.drift, opacity: 0.08 }));
      this.group.add(cone);
    }

    // Actual landing
    this.label(`LANDING · ${outName}`, `${dist.toFixed(0)} m from pad`,
      new THREE.Vector3(land.x, 2, land.y), hex);

    scene.add(this.group);
  },
};
window.__k2envelope = {                      // debug/test handle
  get group() { return envelope.group; },
  setParams(p) { envParams = { ...envParams, ...p }; },
  params() { return envParams; },
};
document.getElementById('env-btn').addEventListener('click', (e) => {
  envelope.visible = !envelope.visible;
  if (envelope.group) envelope.group.visible = envelope.visible;
  e.target.classList.toggle('active', envelope.visible);
});

// ── Camera rig ───────────────────────────────────────────────────────────────
let camMode = 'chase';
let orbitAz = 0;
let zoom = 1.0;                     // user wheel zoom: <1 closer, >1 farther
const camPos = new THREE.Vector3(14, 4, 14);
const camTarget = new THREE.Vector3(0, 2, 0);
document.querySelectorAll('#cams button[data-cam]').forEach(btn => {
  btn.addEventListener('click', () => {
    camMode = btn.dataset.cam;
    dragAz = 0; dragEl = 0;        // fresh framing per mode
    document.querySelectorAll('#cams button[data-cam]')
      .forEach(b => b.classList.toggle('active', b === btn));
  });
});
window.addEventListener('wheel', (e) => {
  // Ctrl+wheel (and trackpad pinch, which Chromium delivers as ctrl+wheel)
  // would zoom the whole page — claim it for camera zoom instead.
  if (e.ctrlKey) e.preventDefault();
  // Post-flight, allow zooming far out so the mission envelope is frameable
  const zMax = T.running ? 6.0 : 60.0;
  zoom = THREE.MathUtils.clamp(zoom * Math.exp(e.deltaY * 0.0011), 0.25, zMax);
}, { passive: false });
window.addEventListener('keydown', (e) => {
  // Block browser page-zoom shortcuts (Ctrl +/-/0)
  if (e.ctrlKey && (e.key === '+' || e.key === '-' || e.key === '=' || e.key === '0'))
    e.preventDefault();
});

// Click-drag to rotate the POV around the rocket (chase/side/orbit modes).
// Azimuth/elevation offsets ride on top of each mode's base placement.
let dragAz = 0, dragEl = 0;
let dragging = false, lastPX = 0, lastPY = 0;
renderer.domElement.addEventListener('pointerdown', (e) => {
  dragging = true; lastPX = e.clientX; lastPY = e.clientY;
  renderer.domElement.setPointerCapture(e.pointerId);
});
renderer.domElement.addEventListener('pointermove', (e) => {
  if (!dragging) return;
  dragAz -= (e.clientX - lastPX) * 0.006;
  dragEl = THREE.MathUtils.clamp(
    dragEl + (e.clientY - lastPY) * 0.005, -0.9, 1.25);
  lastPX = e.clientX; lastPY = e.clientY;
});
window.addEventListener('pointerup', () => { dragging = false; });

// Rotate an offset (relative to a centre point) by the user's drag angles
const _sph = new THREE.Spherical();
function applyDrag(desired, centre) {
  if (dragAz === 0 && dragEl === 0) return;
  const off = desired.clone().sub(centre);
  _sph.setFromVector3(off);
  _sph.theta += dragAz;
  _sph.phi = THREE.MathUtils.clamp(_sph.phi + dragEl, 0.12, Math.PI - 0.12);
  desired.copy(centre).add(off.setFromSpherical(_sph));
}

// Smoothed follow point: the ONLY smoothed quantity. Every camera offset is
// rigid relative to it, so the rocket can never leave the frame — smoothing
// a camera position toward a stale target is what caused liftoff to outrun
// the old chase cam.
const followPos = new THREE.Vector3(0, 0.3, 0);

function updateCamera(dt) {
  const L = rocketLen;
  const vel = new THREE.Vector3(T.vx, T.vz, T.vy);   // k2 → three frame
  const speed = vel.length();

  // Follow point: speed-adaptive smoothing + hard lag cap of one body length.
  const kf = 8 + speed * 0.2;
  followPos.lerp(rocketPos, 1 - Math.exp(-kf * dt));
  if (followPos.distanceTo(rocketPos) > L)
    followPos.sub(rocketPos).setLength(L).add(rocketPos);

  const desired = tmpV, lookAt = tmpV2;
  lookAt.copy(followPos).addScaledVector(UP, L * 0.5);
  let targetFov = 55 + Math.min(T.thrust / Math.max(maxThrust, 1), 1) * 6;

  switch (camMode) {
    case 'chase': {
      // Horizontal standoff that drifts behind the horizontal velocity —
      // never directly under the rocket (vertical flight = degenerate view).
      const hv = new THREE.Vector3(vel.x, 0, vel.z);
      if (hv.lengthSq() > 4) {
        const behind = hv.normalize().multiplyScalar(-1);
        chaseDir.lerp(behind, 1 - Math.exp(-0.8 * dt)).normalize();
      }
      desired.copy(followPos)
        .addScaledVector(chaseDir, L * 8 * zoom)
        .addScaledVector(UP, L * 2.5 * Math.sqrt(zoom));
      break;
    }
    case 'pad': {
      // Fixed tracking camera: stays put, zooms like a real launch lens.
      desired.set(16, 2.5, 16);
      const dist = Math.max(desired.distanceTo(followPos), 1);
      const halfAngle = Math.atan((L * 5 * zoom) / dist);   // wheel = lens zoom
      targetFov = THREE.MathUtils.clamp(THREE.MathUtils.radToDeg(halfAngle) * 2, 2, 70);
      break;
    }
    case 'side':
      desired.copy(followPos).add(new THREE.Vector3(L * 9 * zoom, L * 0.8 * zoom, 0));
      break;
    case 'orbit':
      orbitAz += dt * 0.25;
      desired.copy(followPos).add(new THREE.Vector3(
        Math.cos(orbitAz) * L * 11 * zoom, L * 3.5 * zoom, Math.sin(orbitAz) * L * 11 * zoom));
      break;
    case 'onboard':
      desired.copy(rocketPos)
        .addScaledVector(UP.clone().applyQuaternion(rocketQuat), L * 1.08)
        .add(new THREE.Vector3(L * 0.4, 0, L * 0.4));
      lookAt.copy(rocketPos).add(new THREE.Vector3(0, -L * 3, 0));
      break;
  }

  // User drag rotates the viewpoint around the rocket (not pad/onboard —
  // pad is a fixed ground camera, onboard is bolted to the airframe)
  if (camMode === 'chase' || camMode === 'side' || camMode === 'orbit')
    applyDrag(desired, followPos);

  // Rigid placement from the follow point; light extra easing only on mode
  // switches (converges fast, bounded by the followPos cap afterwards).
  camPos.lerp(desired, 1 - Math.exp(-10 * dt));
  if (camPos.distanceTo(desired) > L * 0.8)
    camPos.sub(desired).setLength(L * 0.8).add(desired);
  camTarget.lerp(lookAt, 1 - Math.exp(-10 * dt));
  if (camTarget.distanceTo(lookAt) > L * 0.5)
    camTarget.sub(lookAt).setLength(L * 0.5).add(lookAt);

  camera.position.copy(camPos);
  camera.lookAt(camTarget);
  camera.fov += (targetFov - camera.fov) * (1 - Math.exp(-6 * dt));
  camera.updateProjectionMatrix();
}
const chaseDir = new THREE.Vector3(0.94, 0, 0.34);   // initial standoff azimuth

// ── Main loop ────────────────────────────────────────────────────────────────
const clock = new THREE.Clock();
let smokeAcc = 0, padAcc = 0;       // fractional particle-emission accumulators
function animate() {
  requestAnimationFrame(animate);
  const dt = Math.min(clock.getDelta(), 0.05);

  replay.step(dt);                  // advance playback + replay bar UI

  // Pose from latest telemetry (smoothed)
  const targetPos = new THREE.Vector3(T.x, Math.max(T.alt, 0) + 0.3, T.y);
  rocketPos.lerp(targetPos, 1 - Math.exp(-12 * dt));

  const dir = new THREE.Vector3(
    Math.cos(T.pitch) * Math.cos(T.yaw),
    Math.sin(T.pitch),
    Math.cos(T.pitch) * Math.sin(T.yaw));
  const qAlign = new THREE.Quaternion().setFromUnitVectors(UP, dir.normalize());
  const qRoll = new THREE.Quaternion().setFromAxisAngle(UP, T.roll);
  rocketQuat.slerp(qAlign.multiply(qRoll), 1 - Math.exp(-10 * dt));
  rocket.position.copy(rocketPos);
  rocket.quaternion.copy(rocketQuat);

  // Sky/fog altitude response
  const altFrac = THREE.MathUtils.clamp(T.alt / 38000, 0, 1);
  skyUniforms.altFrac.value = altFrac;
  starMat.opacity = THREE.MathUtils.smoothstep(altFrac, 0.35, 0.9);
  scene.fog.density = 1.6e-5 * (1 - altFrac * 0.9);
  sky.position.copy(rocketPos);          // dome follows so it never clips
  stars.position.x = rocketPos.x; stars.position.z = rocketPos.z;

  // Exhaust plume
  const thr = Math.max(T.thrust, 0) / Math.max(maxThrust, 1);
  if (thr > 0.01) {
    const bodyUp = UP.clone().applyQuaternion(rocketQuat);
    const bodyDown = bodyUp.clone().multiplyScalar(-1);
    const nozzle = rocketPos.clone().addScaledVector(bodyDown, rocketLen * 0.02);
    flameGroup.position.copy(nozzle);
    flameGroup.quaternion.copy(rocketQuat);       // plume-local -Y = aft

    // Low-pass flutter (white noise resonates; filtered noise looks like fire)
    flick += ((0.9 + Math.random() * 0.35) - flick) * Math.min(dt * 16, 1);

    // Plume widens with altitude (under-expansion) and lengthens with thrust
    const expand = 1 + THREE.MathUtils.clamp(T.alt / 25000, 0, 1) * 1.6;
    const L = rocketLen;
    for (const s of [flameOuter, flameMid, flameCore]) {
      const u = s.userData;
      s.scale.set(u.w * L * 0.6 * (0.7 + 0.45 * thr) * expand * flick,
                  u.h * L * 0.55 * (0.45 + 0.75 * thr) * flick, 1);
      s.position.set(0, -L * 0.01, 0);
    }
    // Mach diamonds: bright cells along the core, spacing ∝ thrust,
    // shrinking + dimming downstream; jitter keeps them alive
    const spacing = L * 0.32 * (0.5 + 0.7 * thr) * expand;
    for (let i = 0; i < DIAMONDS; i++) {
      const d = diamonds[i];
      const fade = 1 - i / DIAMONDS;
      d.visible = thr > 0.15;
      d.position.set(0, -spacing * (i + 0.9), 0);
      d.scale.setScalar(L * 0.16 * fade * (0.7 + 0.5 * thr) * flick);
      d.material.opacity = 0.85 * fade * Math.min(thr * 1.6, 1) * (0.8 + Math.random() * 0.2);
    }
    flameLight.intensity = 320 * thr * flick;
    flameLight.position.set(0, -L * 0.45, 0);
    flameGroup.visible = true;

    // Ground wash light + glow while close to the pad
    if (T.alt < 30) {
      groundGlow.position.set(rocketPos.x, 1.2, rocketPos.z);
      groundGlow.intensity = 500 * thr * (1 - T.alt / 30) * flick;
    } else groundGlow.intensity = 0;

    // Smoke emission — continuous rate (per second, not per frame)
    if (T.alt < 18000) {
      const speed = Math.hypot(T.vx, T.vy, T.vz);
      const exhaustV = Math.max(speed * 0.2, 25);
      const tail = nozzle.clone().addScaledVector(bodyDown, L * (0.8 + thr * 0.8));
      smokeAcc += dt * (50 + 60 * thr);
      while (smokeAcc >= 1) {
        smokeAcc -= 1;
        spawnSmoke(tail, bodyUp, exhaustV);
      }
      // Pad cloud: exhaust deflects off the ground and rolls outward
      if (T.alt < 8) {
        padAcc += dt * 90 * thr;
        while (padAcc >= 1) {
          padAcc -= 1;
          const a = Math.random() * Math.PI * 2;
          const v = 6 + Math.random() * 14;
          spawnSmoke(
            new THREE.Vector3(rocketPos.x, 0.4 + Math.random() * 0.8, rocketPos.z),
            UP, 0,
            { vel: new THREE.Vector3(Math.cos(a) * v, 1.5 + Math.random() * 2, Math.sin(a) * v),
              size: L * (0.5 + Math.random() * 0.5),
              life: 7 + Math.random() * 6, grow: 1.6 });
        }
      }
    }
  } else {
    flameGroup.visible = false;
    flameLight.intensity = 0;
    groundGlow.intensity = 0;
  }

  // Smoke aging: fade in fast, drift, expand, tumble, fade out slow
  for (const p of smoke) {
    if (!p.spr.visible) continue;
    p.age += dt;
    if (p.age >= p.life) { p.spr.visible = false; continue; }
    const f = p.age / p.life;
    p.vel.y += 0.7 * dt;                                  // buoyancy
    p.spr.position.addScaledVector(p.vel, dt);
    p.vel.multiplyScalar(1 - 1.1 * dt);                   // drag
    p.spr.scale.setScalar(p.spr.scale.x + dt * rocketLen * p.grow);
    p.spr.material.rotation += p.rot * dt;
    // opacity envelope: ramp in over 8% of life, ease out after 35%
    const ramp = Math.min(p.age / (p.life * 0.08), 1);
    p.spr.material.opacity = ramp * (f < 0.35 ? 0.4 : 0.4 * (1 - (f - 0.35) / 0.65));
    // colour: flame-lit at birth → white steam → dispersed grey
    const c = p.spr.material.color;
    if (p.age < 0.35) c.lerpColors(_birthTint, _hotGrey, p.age / 0.35);
    else c.lerpColors(_hotGrey, _coldGrey, Math.min((f - 0.1) / 0.7, 1));
  }

  // Trail
  if (T.running && T.alt > 1) pushTrail(rocketPos, T.mach);

  // Vapour cone near Mach 1
  if (T.mach > 0.92 && T.mach < 1.08) {
    vapor.visible = true;
    const noseW = rocketPos.clone().addScaledVector(UP.clone().applyQuaternion(rocketQuat), rocketLen * 0.8);
    vapor.position.copy(noseW);
    vapor.quaternion.copy(rocketQuat);
    const s = rocketLen * 0.9;
    vapor.scale.set(s, s, s);
    vapor.material.opacity = 0.28 * (1 - Math.abs(T.mach - 1) / 0.08) * (0.8 + Math.random() * 0.2);
  } else vapor.visible = false;

  // Parachute
  const chuteOpen = /descent/i.test(T.phase);
  if (chute) {
    chute.visible = chuteOpen;
    if (chuteOpen) {
      const isMain = /main/i.test(T.phase);
      const cda = isMain ? recovery.main_cd_area : recovery.drogue_cd_area;
      const rC = Math.max(Math.sqrt(cda / Math.PI) * 1.25, rocketLen * 0.25);
      chute.position.copy(rocketPos).addScaledVector(UP, rocketLen + rC * 2.0);
      const target = new THREE.Vector3(rC, rC, rC);
      chute.scale.lerp(target, 1 - Math.exp(-3 * dt));     // inflation
    } else chute.scale.setScalar(0.01);
  }

  // Marker labels: keep readable size regardless of camera distance
  for (const m of markers) {
    const lbl = m.userData.label;
    const d = camera.position.distanceTo(m.position);
    const s = Math.max(d * 0.045, rocketLen * 0.8);
    lbl.scale.set(s * 3.2, s * 0.9, 1);
  }
  for (const spr of envelope.sprites) {
    const d = camera.position.distanceTo(spr.position);
    const s = Math.max(d * 0.045, rocketLen * 0.8);
    spr.scale.set(s * 3.2, s * 0.9, 1);
  }

  // Sun shadow camera follows the action at low altitude
  sun.target.position.copy(rocketPos); sun.target.updateMatrixWorld();

  updateCamera(dt);
  composer.render();
}

// ── Reset ────────────────────────────────────────────────────────────────────
// Visual state only — used by both a real flight reset and replay rewinds.
function resetVisuals() {
  trailCount = 0; trailGeo.setDrawRange(0, 0); lastTrailPt.set(1e9, 1e9, 1e9);
  for (const p of smoke) p.spr.visible = false;
  smokeAcc = 0; padAcc = 0;
  clearMarkers();
  envelope.clear();
  events.maxQfired = false; events.peakQ = 0;
  hud.maxAlt = 1; hud.maxVel = 1;
  graphs.clear();
  events.prev = 'Pre-Launch';
  Object.assign(T, { t: 0, x: 0, y: 0, alt: 0, vx: 0, vy: 0, vz: 0,
                     pitch: Math.PI/2, yaw: 0, roll: 0, thrust: 0, mach: 0,
                     acc: 0, q: 0, phase: 'Pre-Launch', running: false });
  rocketPos.set(0, 0.3, 0);
}

// Full reset for a new flight: drop the recording and leave replay mode.
function resetFlight() {
  replay.active = false; replay.playing = false;
  rec.clear();
  zoom = 1.0; dragAz = 0; dragEl = 0;
  resetVisuals();
}

// ── Qt bridge ────────────────────────────────────────────────────────────────
buildRocket({});                                   // placeholder until init arrives
if (typeof qt !== 'undefined' && qt.webChannelTransport) {
  new QWebChannel(qt.webChannelTransport, (channel) => {
    const k2 = channel.objects.k2;
    // initSig only (re)configures the scene — it also fires on tab switches
    // (showEvent), so it must NOT reset flight state or the recording would
    // be wiped mid-flight. resetSig (sim start) owns the reset.
    k2.initSig.connect((json) => {
      const d = JSON.parse(json);
      if (d.geometry) buildRocket(d.geometry);
      if (d.max_thrust > 0) maxThrust = d.max_thrust;
      if (d.recovery) recovery = d.recovery;
      if (d.envelope) envParams = { ...envParams, ...d.envelope };
    });
    k2.tickSig.connect((json) => {
      const d = JSON.parse(json);
      if (replay.active) {
        if (!d.running) return;     // idle live ticks don't disturb a replay
        resetFlight();              // new sim launched mid-replay → go live
      }
      applyTelemetry(d);
    });
    k2.resetSig.connect(resetFlight);
    if (k2.ready) k2.ready();
  });
} else {
  // Standalone browser demo: canned flight so the scene can be previewed
  let t = 0;
  setInterval(() => {
    if (replay.active) return;
    t += 0.05;
    const burn = 2.5, alt = t < burn ? 60 * t * t : Math.max(375 + 150 * (t - burn) - 4.9 * (t - burn) ** 2, 0);
    const vz = t < burn ? 120 * t : 150 - 9.8 * (t - burn);
    const landed = t > burn && alt <= 0;
    applyTelemetry({ t, x: t * 2, y: 0, alt, vx: landed ? 0 : 2, vy: 0,
      vz: landed ? 0 : vz,
      pitch: Math.PI / 2 - t * 0.01, yaw: 0, roll: landed ? 0 : t * 0.4,
      thrust: t < burn ? 180 : 0, mach: landed ? 0 : Math.abs(vz) / 340,
      acc: landed ? 0 : (t < burn ? 110 : 9.8),
      q: landed ? 0 : 0.5 * 1.225 * Math.exp(-alt / 8500) * vz * vz,
      phase: landed ? 'Landed'
        : t < burn ? 'Boost'
        : (vz > 4 ? 'Coast'
        : (vz > -4 ? 'Apogee'
        : (alt > 150 ? 'Drogue Descent' : 'Main Descent'))),
      running: !landed });
  }, 33);
}

// Debug/test handle: where is the rocket on screen? (NDC; |x|,|y|<1 = in frame)
window.__k2dbg = {
  screenPos() {
    const p = rocketPos.clone().project(camera);
    return { x: p.x, y: p.y, z: p.z, alt: T.alt, mode: camMode };
  },
  markers() { return markerLog.slice(); },
  zoom() { return zoom; },
  project(x, y, z) {                 // world point → NDC (test hook)
    const p = new THREE.Vector3(x, y, z).project(camera);
    return { x: p.x, y: p.y };
  },
  drag() { return { az: dragAz, el: dragEl }; },
};

animate();
