/* K2 AeroSim — launch intro.
 *
 * A pinned three.js scene scrubbed by scroll: an establishing shot of a dusk
 * midnight mountain range, one orbit in to the rocket, then four phases
 * (design → propulsion → analysis → liftoff) before it fades into the
 * normal hero. Every visual is a pure function of scroll progress, so
 * scrolling back rewinds it; only flame flicker, sparks and streamline flow
 * use wall-clock time.
 *
 * Classic script on purpose: a module script would not load from file:// and
 * the page would stay locked behind the loader. three.js is pulled in with a
 * dynamic import() only when the intro actually plays.
 *
 * The <head> snippet in index.html decides whether to play (every visit unless
 * reduced motion; ?intro forces it, ?nointro skips) by adding .intro-on. The
 * scene starts as soon as it has loaded; Skip / Esc drops it for this visit.
 */
(function () {
  "use strict";
  window.__k2introBooted = true;

  const THREE_URL = "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.min.js";

  const root = document.documentElement;
  const intro = document.getElementById("intro");
  if (!intro || !root.classList.contains("intro-on")) return;

  // ── Timeline ────────────────────────────────────────────────────────────────
  // Scroll progress `s` (0–1 over the intro) drives the landscape orbit and the
  // final fade; the rocket story runs on its own clock `q`, which starts once
  // the orbit has landed on the rocket. All story constants below are in q.
  const S_ORBIT = 0.11;                    // establishing shot + one orbit
  const S_LIFTOFF = 0.62;                  // scroll point where the story reaches liftoff
  const S_FADE = [0.93, 0.99];             // fade into the site, just after "Fly it."
  function story(s) {
    if (s <= S_ORBIT) return 0;
    if (s <= S_LIFTOFF) return 0.58 * (s - S_ORBIT) / (S_LIFTOFF - S_ORBIT);
    return 0.58 + (s - S_LIFTOFF) * 0.83;  // q ≈ 0.88 at the end: mid-climb, in the clouds
  }
  const PHASE_START = [0, 0.22, 0.40, 0.58];                 // rail highlight
  const TEXT_WINDOWS = [[0.05, 0.20], [0.24, 0.38], [0.42, 0.56], [0.70, 0.81]];
  const COUNT = [[0.60, "3"], [0.62, "2"], [0.64, "1"], [0.66, "Liftoff"], [0.69, null]];
  const P_IGNITE = 0.655, P_LIFT = 0.68, P_BURNOUT = 0.86, P_APOGEE = 0.95;

  const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
  const lerp = (a, b, t) => a + (b - a) * t;
  const smooth = (t) => t * t * (3 - 2 * t);
  const range = (p, a, b) => clamp((p - a) / (b - a));
  // 0 outside [a, b], 1 inside, with soft edges of width f
  const band = (p, a, b, f = 0.02) => smooth(clamp((p - a) / f)) * smooth(clamp((b - p) / f));

  // Rocket height in scene units: accelerating boost, then a decelerating
  // coast. Velocity is continuous at burnout.
  function altitude(p) {
    if (p <= P_LIFT) return 0;
    if (p <= P_BURNOUT) return 180 * Math.pow(range(p, P_LIFT, P_BURNOUT), 2);
    const s = range(p, P_BURNOUT, P_APOGEE);
    return 180 + 90 * (1 - (1 - s) * (1 - s));
  }

  // HUD numbers follow the flight in the hero terminal: burnout T+1.34 s at
  // 168.4 m/s, apogee 1,191 m at T+15.6 s.
  function telemetry(p) {
    let t, alt, vel, stage;
    if (p < P_LIFT) { t = 0; alt = 0; vel = 0; stage = "IGNITION"; }
    else if (p < P_BURNOUT) {
      t = 1.34 * range(p, P_LIFT, P_BURNOUT);
      vel = 168.4 * (t / 1.34); alt = 0.5 * 168.4 * 1.34 * Math.pow(t / 1.34, 2); stage = "BOOST";
    } else {
      const s = range(p, P_BURNOUT, P_APOGEE), k = 1.227;
      t = 1.34 + 14.26 * s;
      vel = 168.4 * Math.pow(1 - s, k); alt = 112.8 + 1078.2 * (1 - Math.pow(1 - s, k + 1)); stage = "COAST";
    }
    return { t, alt, vel, stage };
  }

  // ── DOM ─────────────────────────────────────────────────────────────────────
  const $ = (s) => intro.querySelector(s);
  const stage = $(".intro-stage");
  const canvas = $(".intro-canvas");
  const gate = $(".intro-gate");
  const loadPct = $(".intro-load-pct");
  const loadBar = $(".intro-load-bar span");
  const phases = Array.from(intro.querySelectorAll(".intro-phase"));
  const rail = $(".intro-rail");
  const railItems = Array.from(intro.querySelectorAll(".intro-rail li"));
  const countEl = $(".intro-count");
  const hud = $(".intro-hud");
  const hudT = $(".intro-hud [data-k='t']");
  const hudStage = $(".intro-hud [data-k='stage']");
  const hudAlt = $(".intro-hud [data-k='alt']");
  const hudVel = $(".intro-hud [data-k='vel']");
  const hint = $(".intro-hint");
  const fade = $(".intro-fade");
  const soundBtn = $(".intro-sound");
  const thrustPath = $(".intro-thrust path");

  const debug = /[?&]introdebug\b/.test(location.search);
  const lowPower = matchMedia("(max-width: 760px)").matches ||
    (navigator.hardwareConcurrency || 8) <= 4;

  let scene = null, audio = null, started = false;
  let forcedP = null, shownP = 0, lastCount = -1, rafId = 0;

  function unlock() { root.classList.remove("intro-locked"); }

  // Bail out quietly (no WebGL, CDN blocked…) and show the normal site
  function abort(err) {
    if (debug && err) console.warn("[intro] aborted:", err);
    cancelAnimationFrame(rafId);
    if (scene) scene.dispose();
    root.classList.remove("intro-on", "intro-locked", "intro-active");
    intro.remove();
  }

  // Skip / Esc: drop the intro for this visit and land on the hero
  function skip() {
    if (audio) audio.stop();
    cancelAnimationFrame(rafId);
    if (scene) scene.dispose();
    root.classList.remove("intro-on", "intro-locked", "intro-active");
    intro.remove();
    window.scrollTo(0, 0);
  }

  // Scene loaded → fade the loader out and hand scrolling back
  function start() {
    if (started) return;
    started = true;
    gate.classList.add("is-gone");
    setTimeout(() => (gate.hidden = true), 900);
    unlock();
  }

  // Browsers only allow audio after a click/tap/key (scrolling doesn't count),
  // so sound starts on the first such gesture — or from the speaker button.
  function setSoundUI(on) {
    soundBtn.classList.toggle("is-muted", !on);
    soundBtn.setAttribute("aria-label", on ? "Mute" : "Turn sound on");
  }
  function enableSound() {
    if (audio || !document.body.contains(intro)) return;
    audio = makeAudio();
    if (audio) setSoundUI(true);
  }
  function onGesture(e) {
    if (e.key === "Escape") return;
    if (e.target && e.target.closest && e.target.closest(".intro-skip, .intro-sound")) return;
    enableSound();
    if (audio) ["pointerdown", "keydown"].forEach((t) => window.removeEventListener(t, onGesture));
  }
  ["pointerdown", "keydown"].forEach((t) => window.addEventListener(t, onGesture));

  intro.querySelectorAll(".intro-skip").forEach((b) => b.addEventListener("click", skip));
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && document.body.contains(intro)) skip();
  });
  soundBtn.addEventListener("click", () => {
    if (!audio) return enableSound();
    const muted = !soundBtn.classList.contains("is-muted");
    setSoundUI(!muted);
    audio.mute(muted);
  });
  document.addEventListener("visibilitychange", () => {
    if (audio) document.hidden ? audio.suspend() : audio.resume();
  });

  // ── Boot: load three.js behind the loader ───────────────────────────────────
  boot();

  async function boot() {
    window.scrollTo(0, 0);
    const probe = document.createElement("canvas");
    if (!(probe.getContext("webgl2") || probe.getContext("webgl"))) return abort("no WebGL");

    // Download progress is not observable for import(); ease toward 80 %
    // while it is in flight, then finish on scene build.
    let pct = 0;
    const tick = setInterval(() => setLoad((pct += (80 - pct) * 0.08)), 60);
    let THREE;
    try {
      THREE = await import(THREE_URL);
    } catch (e) {
      clearInterval(tick);
      return abort(e);
    }
    clearInterval(tick);
    setLoad(88);
    try {
      scene = buildScene(THREE);
    } catch (e) {
      return abort(e);
    }
    setLoad(100);
    if (debug) window.__k2intro = { setP: (p) => (forcedP = p), start, skip, scene };
    rafId = requestAnimationFrame(loop);
    setTimeout(start, 350);                         // let 100 % register, then go
  }

  function setLoad(v) {
    loadPct.textContent = Math.round(v) + "%";
    loadBar.style.transform = `scaleX(${v / 100})`;
  }

  // ── Frame loop ──────────────────────────────────────────────────────────────
  let lastT = performance.now();
  function loop(now) {
    rafId = requestAnimationFrame(loop);
    const dt = Math.min((now - lastT) / 1000, 0.1);
    lastT = now;

    const rect = intro.getBoundingClientRect();
    const total = intro.offsetHeight - innerHeight;
    const target = forcedP ?? clamp(-rect.top / total);
    // Inertia on the scene only — native scrolling is left alone
    shownP = forcedP != null ? forcedP : shownP + (target - shownP) * (1 - Math.exp(-dt * 7));
    if (Math.abs(target - shownP) < 1e-4) shownP = target;

    const inView = rect.bottom > innerHeight * 0.35;
    root.classList.toggle("intro-active", inView);
    if (audio) inView ? audio.resume() : audio.suspend();
    if (rect.bottom <= 0) return;                     // scrolled past: idle

    const s = shownP, q = story(s);
    scene.render(q, now / 1000, s);
    updateOverlay(q, s);
    if (audio) audio.update(q);
  }

  function updateOverlay(p, s) {        // p: story clock, s: scroll progress
    phases.forEach((el, i) => {
      const [a, b] = TEXT_WINDOWS[i];
      const v = band(p, a, b, 0.025);
      el.style.opacity = v.toFixed(3);
      el.style.transform = `translateY(${((1 - v) * 24).toFixed(1)}px)`;
      el.style.visibility = v > 0.001 ? "visible" : "hidden";
    });
    if (thrustPath) {
      const v = range(p, 0.25, 0.34);
      thrustPath.style.strokeDashoffset = (1 - v).toFixed(3);
    }

    let idx = 0;
    PHASE_START.forEach((a, i) => { if (p >= a) idx = i; });
    railItems.forEach((li, i) => li.classList.toggle("is-on", i === idx));
    rail.style.opacity = range(s, S_ORBIT - 0.04, S_ORBIT).toFixed(3);   // appears as the orbit lands
    stage.classList.toggle("has-copy", s > S_ORBIT - 0.02);           // phone scrim only once copy shows

    // Countdown — beep only when scrolling forward into a new number
    let ci = -1;
    COUNT.forEach(([s], i) => { if (p >= s) ci = i; });
    if (ci !== lastCount) {
      const label = ci >= 0 ? COUNT[ci][1] : null;
      if (label && audio && ci > lastCount) audio.beep(label === "Liftoff" ? 1320 : 880);
      countEl.textContent = label || "";
      countEl.classList.toggle("is-word", label === "Liftoff");
      countEl.classList.remove("pop");
      void countEl.offsetWidth;                      // restart the pop animation
      if (label) countEl.classList.add("pop");
      lastCount = ci;
    }

    const hv = band(p, 0.655, 0.97, 0.01);
    hud.style.opacity = hv.toFixed(3);
    if (hv > 0) {
      const tm = telemetry(p);
      hudT.textContent = "T+ " + tm.t.toFixed(2).padStart(5, "0") + " s";
      hudStage.textContent = tm.stage;
      hudAlt.textContent = Math.round(tm.alt).toLocaleString() + " m";
      hudVel.textContent = tm.vel.toFixed(1) + " m/s";
    }

    hint.style.opacity = started ? (1 - range(s, 0.005, 0.03)).toFixed(3) : "0";
    fade.style.opacity = range(s, S_FADE[0], S_FADE[1]).toFixed(3);
  }

  // ── Sound: generated live with Web Audio, no files ──────────────────────────
  function makeAudio() {
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) return null;
    const ctx = new AC();
    const master = ctx.createGain();
    master.gain.value = 0;
    master.connect(ctx.destination);
    master.gain.setTargetAtTime(0.7, ctx.currentTime, 0.8);

    const buf = ctx.createBuffer(1, ctx.sampleRate * 2, ctx.sampleRate);
    const ch = buf.getChannelData(0);
    for (let i = 0; i < ch.length; i++) ch[i] = Math.random() * 2 - 1;
    const noise = () => { const s = ctx.createBufferSource(); s.buffer = buf; s.loop = true; s.start(); return s; };
    const chain = (...nodes) => { nodes.reduce((a, b) => (a.connect(b), b)); return nodes[nodes.length - 1]; };
    const gain = (v = 0) => { const g = ctx.createGain(); g.gain.value = v; return g; };
    const filter = (type, f, q = 0.7) => { const b = ctx.createBiquadFilter(); b.type = type; b.frequency.value = f; b.Q.value = q; return b; };

    // Low ambient drone
    const drone = gain(0);
    drone.connect(master);
    [[55, "sine", 0.5], [82.41, "sine", 0.28], [110, "triangle", 0.05]].forEach(([f, type, v]) => {
      const o = ctx.createOscillator(); o.type = type; o.frequency.value = f;
      chain(o, gain(v), drone); o.start();
    });
    // Engine: sub rumble + mid roar; wind for the climb
    const rumble = gain(0); chain(noise(), filter("lowpass", 150), rumble, master);
    const roar = gain(0); chain(noise(), filter("bandpass", 650, 0.6), roar, master);
    const windF = filter("bandpass", 420, 1.1);
    const wind = gain(0); chain(noise(), windF, wind, master);

    const set = (param, v) => param.setTargetAtTime(v, ctx.currentTime, 0.12);
    let muted = false;
    return {
      update(p) {
        const h = altitude(p);
        const thrust = range(p, P_IGNITE, P_IGNITE + 0.02) * (1 - range(p, P_BURNOUT + 0.015, P_BURNOUT + 0.03));
        const near = 1 - range(h, 0, 140);
        set(drone.gain, 0.11 * (1 - range(p, 0.95, 1)));
        set(rumble.gain, thrust * (0.55 + 0.45 * near));
        set(roar.gain, thrust * (0.08 + 0.3 * near));
        set(wind.gain, band(p, 0.80, 0.985, 0.03) * 0.3);
        set(windF.frequency, 300 + 500 * range(p, 0.80, 0.95));
      },
      beep(f) {
        if (muted) return;
        const o = ctx.createOscillator(); o.type = "sine"; o.frequency.value = f;
        const g = gain(0); chain(o, g, master);
        g.gain.setValueAtTime(0, ctx.currentTime);
        g.gain.linearRampToValueAtTime(0.18, ctx.currentTime + 0.01);
        g.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 0.35);
        o.start(); o.stop(ctx.currentTime + 0.4);
      },
      mute(m) { muted = m; master.gain.setTargetAtTime(m ? 0 : 0.7, ctx.currentTime, 0.15); },
      suspend() { if (ctx.state === "running") ctx.suspend(); },
      resume() { if (ctx.state === "suspended" && !document.hidden) ctx.resume(); },
      stop() { master.gain.setTargetAtTime(0, ctx.currentTime, 0.2); setTimeout(() => ctx.close(), 800); },
    };
  }

  // ── Scene ───────────────────────────────────────────────────────────────────
  function buildScene(THREE) {
    const V3 = THREE.Vector3;
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: !lowPower, powerPreference: "high-performance" });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, lowPower ? 1.25 : 1.75));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.05;

    const scene = new THREE.Scene();
    // Midnight palette (deep-blue horizon → near-black zenith), to space on the climb
    const HORIZON = new THREE.Color(0x0c1433), GLOW = new THREE.Color(0x1d2c66);
    const ZENITH = new THREE.Color(0x010209), SPACE = new THREE.Color(0x010106);
    const FOG = new THREE.Color(0x0a1029);
    // No scene.background: a Color background force-clears on every render(),
    // which would wipe the reflection pass. The sky dome covers the frame.
    renderer.setClearColor(ZENITH, 1);
    scene.fog = new THREE.Fog(FOG.clone(), 40, 1300);
    const camera = new THREE.PerspectiveCamera(38, 1, 0.05, 4000);

    scene.add(new THREE.HemisphereLight(0x5a6cb0, 0x020208, 0.7));
    const key = new THREE.DirectionalLight(0xc4d2ff, 2.3); key.position.set(4, 6, 5); scene.add(key);   // moonlight
    const rimWarm = new THREE.DirectionalLight(0xff7a3d, 0.9); rimWarm.position.set(-5, 2.5, -4); scene.add(rimWarm);
    const rimCool = new THREE.DirectionalLight(0x4cc9f0, 1.1); rimCool.position.set(5, 1, -5); scene.add(rimCool);

    // Canvas textures
    function radialTex(inner, outer = "rgba(255,255,255,0)", mid = 0.3) {
      const c = document.createElement("canvas"); c.width = c.height = 128;
      const g = c.getContext("2d");
      const gr = g.createRadialGradient(64, 64, 1, 64, 64, 64);
      gr.addColorStop(0, inner); gr.addColorStop(mid, inner.replace(/[\d.]+\)$/, "0.45)")); gr.addColorStop(1, outer);
      g.fillStyle = gr; g.fillRect(0, 0, 128, 128);
      const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; return t;
    }
    function puffTex(seed, blobs = 14) {
      const c = document.createElement("canvas"); c.width = c.height = 256;
      const g = c.getContext("2d");
      let s = seed; const rnd = () => (s = (s * 16807) % 2147483647) / 2147483647;
      for (let i = 0; i < blobs; i++) {
        const a = rnd() * Math.PI * 2, r = rnd() * 60;
        const x = 128 + Math.cos(a) * r, y = 128 + Math.sin(a) * r, rad = 34 + rnd() * 48;
        const al = 0.12 + rnd() * 0.14;
        const gr = g.createRadialGradient(x, y, 2, x, y, rad);
        gr.addColorStop(0, `rgba(240,242,246,${al})`);
        gr.addColorStop(0.6, `rgba(228,231,236,${al * 0.5})`);
        gr.addColorStop(1, "rgba(220,224,230,0)");
        g.fillStyle = gr; g.beginPath(); g.arc(x, y, rad, 0, Math.PI * 2); g.fill();
      }
      const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; return t;
    }
    function bodyTex() {
      const c = document.createElement("canvas"); c.width = 512; c.height = 1024;
      const g = c.getContext("2d");
      g.fillStyle = "#f1f3f6"; g.fillRect(0, 0, 512, 1024);
      g.fillStyle = "#ff6b35"; g.fillRect(0, 70, 512, 18);
      g.fillStyle = "#4cc9f0"; g.fillRect(0, 96, 512, 6);
      g.save(); g.translate(150, 600); g.rotate(-Math.PI / 2);
      g.fillStyle = "#151a24"; g.font = "700 120px Arial, sans-serif"; g.fillText("K2", 0, 0);
      g.font = "500 40px Arial, sans-serif"; g.fillStyle = "#5b6577"; g.fillText("AEROSIM", 190, -4);
      g.restore();
      const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 4; return t;
    }

    // ── Rocket (same proportions/colours as the app's Advanced Visualizer) ──
    const L = 2.4, R = 0.085, NL = 0.62;
    const rocket = new THREE.Group();
    scene.add(rocket);
    const edgeMat = new THREE.LineBasicMaterial({ color: 0x4cc9f0, transparent: true, opacity: 1, depthWrite: false });
    const parts = [];               // { obj, from: {pos, rotZ}, mats, edges, delay }

    // wireGeo: optional low-poly twin so smooth lathes/tubes show their
    // profile lines in the blueprint (plain edges would only draw the rims)
    function addPart(mesh, from, delay, wireGeo) {
      const holder = new THREE.Group();
      holder.add(mesh);
      const edges = new THREE.LineSegments(new THREE.EdgesGeometry(wireGeo || mesh.geometry, wireGeo ? 1 : 28), edgeMat);
      mesh.add(edges);
      const mats = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
      mats.forEach((m) => { m.transparent = true; m.opacity = 0; });
      rocket.add(holder);
      parts.push({ holder, mesh, from, delay, mats });
      return holder;
    }

    // Nose: tangent ogive
    const noseProf = [];
    const rho = (R * R + NL * NL) / (2 * R);
    for (let i = 0; i <= 16; i++) {
      const f = i / 16, y = f * NL;
      const r = Math.sqrt(Math.max(rho * rho - y * y, 0)) - (rho - R);
      noseProf.push(new THREE.Vector2(Math.max(r, 0.0008), y));
    }
    const noseMat = new THREE.MeshStandardMaterial({ color: 0xff6b35, metalness: 0.25, roughness: 0.38 });
    const nose = new THREE.Mesh(new THREE.LatheGeometry(noseProf, 48), noseMat);
    nose.position.y = L - NL;
    addPart(nose, { pos: new V3(0, 0.55, 0), rotZ: 0.3 }, 0, new THREE.LatheGeometry(noseProf.filter((_, i) => i % 4 === 0 || i === 16), 10));

    const bandMat = new THREE.MeshStandardMaterial({ color: 0xd64533, metalness: 0.4, roughness: 0.35 });
    const bandMesh = new THREE.Mesh(new THREE.CylinderGeometry(R * 1.01, R * 1.01, 0.09, 48), bandMat);
    bandMesh.position.y = L - NL - 0.05;
    addPart(bandMesh, { pos: new V3(0, 0.3, 0), rotZ: -0.2 }, 0.012);

    const tubeLen = L - NL - 0.1 - 0.06;
    const bodyMat = new THREE.MeshStandardMaterial({ map: bodyTex(), metalness: 0.3, roughness: 0.35 });
    const tube = new THREE.Mesh(new THREE.CylinderGeometry(R, R, tubeLen, 48, 1, true), bodyMat);
    tube.position.y = 0.06 + tubeLen / 2;
    addPart(tube, { pos: new V3(0, 0, 0), rotZ: 0 }, 0.02, new THREE.CylinderGeometry(R, R, tubeLen, 10, 4, true));

    const finMat = new THREE.MeshStandardMaterial({ color: 0x1d232c, metalness: 0.5, roughness: 0.35 });
    const cr = 0.38, ct = 0.15, fh = 0.19, sweep = 0.14, th = 0.012;
    const finShape = new THREE.Shape();
    finShape.moveTo(0, 0); finShape.lineTo(0, cr); finShape.lineTo(fh, cr - sweep); finShape.lineTo(fh, cr - sweep - ct); finShape.lineTo(0, 0);
    const finGeo = new THREE.ExtrudeGeometry(finShape, { depth: th, bevelEnabled: false });
    finGeo.translate(R * 0.98, 0.07, -th / 2);
    for (let i = 0; i < 4; i++) {
      const fin = new THREE.Mesh(finGeo, finMat.clone());
      const h = addPart(fin, { pos: new V3(0.5, -0.2, 0), rotZ: 0, finSpin: 0.8 }, 0.03 + i * 0.008);
      h.rotation.y = (i / 4) * Math.PI * 2;
    }

    const nozMat = new THREE.MeshStandardMaterial({ color: 0x2a2f36, metalness: 0.85, roughness: 0.4, emissive: 0xff5a1f, emissiveIntensity: 0 });
    const noz = new THREE.Mesh(new THREE.CylinderGeometry(R * 0.82, R * 0.6, 0.1, 32), nozMat);
    noz.position.y = 0.01;
    addPart(noz, { pos: new V3(0, -0.5, 0), rotZ: 0.25 }, 0.05, new THREE.CylinderGeometry(R * 0.82, R * 0.6, 0.1, 10, 1, true));

    // Motor (phase 2) — slides up into the aft end while the airframe goes x-ray
    const motorMat = new THREE.MeshStandardMaterial({ color: 0xff8a3d, emissive: 0xff5a1f, emissiveIntensity: 0.6, metalness: 0.3, roughness: 0.4, transparent: true });
    const motor = new THREE.Mesh(new THREE.CylinderGeometry(R * 0.66, R * 0.66, 0.85, 32), motorMat);
    const motorEdges = new THREE.LineSegments(new THREE.EdgesGeometry(motor.geometry, 28),
      new THREE.LineBasicMaterial({ color: 0xffb347, transparent: true, depthWrite: false }));
    motor.add(motorEdges);
    rocket.add(motor);

    // Sparks at the nozzle as the motor arms
    const SPARKS = lowPower ? 40 : 90;
    const sparkGeo = new THREE.BufferGeometry();
    const sparkPos = new Float32Array(SPARKS * 3);
    sparkGeo.setAttribute("position", new THREE.BufferAttribute(sparkPos, 3));
    const sparkSeed = Array.from({ length: SPARKS }, () => [Math.random(), Math.random() * Math.PI * 2, 0.4 + Math.random()]);
    const sparks = new THREE.Points(sparkGeo, new THREE.PointsMaterial({
      color: 0xffc070, size: 0.035, map: radialTex("rgba(255,255,255,1)"), transparent: true,
      blending: THREE.AdditiveBlending, depthWrite: false }));
    rocket.add(sparks);

    // ── Phase 3: CFD look — streamlines, pressure wash, shock cone ──
    const bodyRadiusAt = (y) => {
      if (y < 0 || y > L) return 0;
      if (y < L - NL) return R;
      const yy = y - (L - NL);
      return Math.max(Math.sqrt(Math.max(rho * rho - yy * yy, 0)) - (rho - R), 0);
    };
    const flowGroup = new THREE.Group();
    rocket.add(flowGroup);
    const flowMats = [];
    const LINES = lowPower ? 20 : 36;
    for (let i = 0; i < LINES; i++) {
      const theta = (i / LINES) * Math.PI * 2 + (i % 3) * 0.37;
      const r0 = R * (0.55 + (i % 4) * 0.55);
      const pts = [];
      for (let k = 0; k <= 80; k++) {
        const y = L + 1.0 - (k / 80) * (L + 2.2);
        const rb = bodyRadiusAt(y) + (y > -0.1 && y < 0.5 ? 0.02 : 0);
        const rr = Math.sqrt(r0 * r0 + (rb * 1.6) * (rb * 1.6)) + rb * 0.4;
        pts.push(new V3(Math.cos(theta) * rr, y, Math.sin(theta) * rr));
      }
      const geo = new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), 90, lowPower ? 0.006 : 0.0045, 4, false);
      const mat = new THREE.ShaderMaterial({
        uniforms: { uTime: { value: 0 }, uOpacity: { value: 0 }, uSeed: { value: Math.random() } },
        vertexShader: "varying float vU; void main(){ vU = uv.x; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }",
        fragmentShader: `uniform float uTime, uOpacity, uSeed; varying float vU;
          void main(){
            float f = fract(vU*5.0 - uTime*0.55 + uSeed);
            float streak = smoothstep(0.0,0.22,f) * smoothstep(1.0,0.55,f);
            vec3 hot = vec3(1.0,0.42,0.18), cool = vec3(0.30,0.80,0.95);
            vec3 col = mix(hot, cool, smoothstep(0.30,0.48,vU));
            float edge = smoothstep(0.0,0.06,vU) * smoothstep(1.0,0.9,vU);
            gl_FragColor = vec4(col, uOpacity * edge * (0.18 + 0.82*streak));
          }`,
        transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
      });
      flowMats.push(mat);
      flowGroup.add(new THREE.Mesh(geo, mat));
    }

    // Pressure wash: full outer mould line coloured by a Cp-like scalar
    const fullProf = [new THREE.Vector2(R * 0.82, 0), new THREE.Vector2(R, 0.06), new THREE.Vector2(R, L - NL)];
    noseProf.slice(1).forEach((v) => fullProf.push(new THREE.Vector2(v.x, v.y + L - NL)));
    const washMat = new THREE.ShaderMaterial({
      uniforms: { uOpacity: { value: 0 } },
      vertexShader: `varying float vY; varying vec3 vN; varying vec3 vV;
        void main(){ vY = position.y; vec4 mv = modelViewMatrix*vec4(position,1.0);
          vN = normalize(normalMatrix*normal); vV = normalize(-mv.xyz); gl_Position = projectionMatrix*mv; }`,
      fragmentShader: `uniform float uOpacity; varying float vY; varying vec3 vN; varying vec3 vV;
        vec3 ramp(float s){
          vec3 c0=vec3(0.10,0.25,0.85), c1=vec3(0.15,0.80,0.95), c2=vec3(0.30,0.90,0.45), c3=vec3(1.0,0.85,0.25), c4=vec3(1.0,0.30,0.15);
          if(s<0.25) return mix(c0,c1,s/0.25); if(s<0.5) return mix(c1,c2,(s-0.25)/0.25);
          if(s<0.75) return mix(c2,c3,(s-0.5)/0.25); return mix(c3,c4,(s-0.75)/0.25);
        }
        void main(){
          float y = vY / ${L.toFixed(3)};
          float s = 0.32 + 0.68*smoothstep(0.86,1.0,y)        // stagnation at the tip
                  - 0.30*smoothstep(0.70,0.76,y)*(1.0-smoothstep(0.76,0.84,y)) // suction past the shoulder
                  + 0.12*(1.0-smoothstep(0.0,0.2,y));          // fin-root recompression
          float fres = pow(1.0 - abs(dot(vN, vV)), 2.0);
          gl_FragColor = vec4(ramp(clamp(s,0.0,1.0)) * (0.85 + 0.6*fres), uOpacity*0.92);
        }`,
      transparent: true, depthWrite: false,
    });
    const wash = new THREE.Mesh(new THREE.LatheGeometry(fullProf, 64), washMat);
    wash.scale.set(1.015, 1.002, 1.015);
    rocket.add(wash);

    const shockMat = new THREE.ShaderMaterial({
      uniforms: { uOpacity: { value: 0 } },
      vertexShader: `varying vec3 vN; varying vec3 vV; varying float vY;
        void main(){ vY = uv.y; vec4 mv = modelViewMatrix*vec4(position,1.0);
          vN = normalize(normalMatrix*normal); vV = normalize(-mv.xyz); gl_Position = projectionMatrix*mv; }`,
      fragmentShader: `uniform float uOpacity; varying vec3 vN; varying vec3 vV; varying float vY;
        void main(){ float f = pow(1.0 - abs(dot(vN, vV)), 3.0);
          gl_FragColor = vec4(vec3(0.55,0.88,1.0), uOpacity * f * smoothstep(0.0,0.5,vY)); }`,
      transparent: true, depthWrite: false, side: THREE.DoubleSide, blending: THREE.AdditiveBlending,
    });
    const shock = new THREE.Mesh(new THREE.ConeGeometry(0.75, 1.7, 64, 1, true), shockMat);
    shock.position.y = L + 0.03 - 0.85;
    rocket.add(shock);

    // ── Landscape: sky dome, mountain ridges, moonlit lake ──
    const PEAK_DIR = Math.atan2(4.6, 3.4) + Math.PI + 0.12;   // behind the rocket in the opening shot
    const MOON_DIR = PEAK_DIR - 0.42, MOON_EL = 0.3;
    const moonVec = new THREE.Vector3(Math.cos(MOON_DIR) * Math.cos(MOON_EL), Math.sin(MOON_EL), Math.sin(MOON_DIR) * Math.cos(MOON_EL));
    const skyMat = new THREE.ShaderMaterial({
      uniforms: {
        uHorizon: { value: HORIZON.clone() }, uGlow: { value: GLOW.clone() }, uZenith: { value: ZENITH.clone() },
        uGlowAmt: { value: 1 }, uGlowDir: { value: new THREE.Vector2(Math.cos(PEAK_DIR), Math.sin(PEAK_DIR)) },
      },
      vertexShader: "varying vec3 vP; void main(){ vP = normalize(position); gl_Position = projectionMatrix*modelViewMatrix*vec4(position,1.0); }",
      fragmentShader: `uniform vec3 uHorizon, uGlow, uZenith; uniform float uGlowAmt; uniform vec2 uGlowDir; varying vec3 vP;
        void main(){
          float e = max(vP.y, 0.0);
          vec3 c = mix(uHorizon, uZenith, pow(smoothstep(0.0, 0.6, e), 0.7));
          float az = max(dot(normalize(vP.xz + 1e-5), uGlowDir), 0.0);
          c += uGlow * uGlowAmt * (0.55 * exp(-e * 10.0) + 0.6 * pow(az, 5.0) * exp(-e * 4.0));
          gl_FragColor = vec4(c, 1.0);
          #include <colorspace_fragment>
        }`,
      side: THREE.BackSide, depthWrite: false, fog: false,
    });
    const sky = new THREE.Mesh(new THREE.SphereGeometry(1500, 48, 24), skyMat);
    sky.renderOrder = -1;
    scene.add(sky);

    // Mountain ridges: rings of silhouettes whose heights are sums of ridged
    // sines (integer frequencies → seamless all the way round). Fog hazes the
    // far rings toward the horizon colour for depth. Each ring also gets an
    // upside-down twin under the waterline, seen through the lake as its
    // reflection.
    function ridge(radius, base, amps, seed, peak) {
      const N = 512, pos = new Float32Array((N + 1) * 6), col = new Float32Array((N + 1) * 6);
      let sd = seed; const rnd = () => (sd = (sd * 16807) % 2147483647) / 2147483647;
      const waves = amps.map((a, i) => ({ a, f: [4, 9, 17, 31, 57][i], ph: rnd() * 6.28 }));
      const lo = new THREE.Color(0x010103), hi = new THREE.Color(0x0a0e22);
      for (let i = 0; i <= N; i++) {
        const th = (i / N) * Math.PI * 2;
        let hgt = base;
        waves.forEach((w) => { hgt += w.a * Math.pow(1 - Math.abs(Math.sin(w.f * th * 0.5 + w.ph)), 2); });
        if (peak) {
          const dth = Math.atan2(Math.sin(th - peak.dir), Math.cos(th - peak.dir));
          hgt += peak.h * Math.exp(-Math.pow(dth / peak.w, 2)) * (0.85 + 0.15 * Math.abs(Math.sin(th * 40)));
        }
        const x = Math.cos(th) * radius, z = Math.sin(th) * radius;
        pos.set([x, 0, z, x, hgt, z], i * 6);
        col.set([lo.r, lo.g, lo.b, hi.r, hi.g, hi.b], i * 6);
      }
      const idx = [];
      for (let i = 0; i < N; i++) { const a = i * 2; idx.push(a, a + 1, a + 2, a + 1, a + 3, a + 2); }
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
      g.setAttribute("color", new THREE.BufferAttribute(col, 3));
      g.setIndex(idx);
      const mat = new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide });
      const m = new THREE.Mesh(g, mat);
      const mirror = new THREE.Mesh(g, mat);
      mirror.scale.y = -1;
      scene.add(m, mirror);
      return m;
    }
    ridge(250, 4, [14, 9, 5, 2.5, 1.2], 1111);
    ridge(430, 10, [34, 20, 11, 5, 2.5], 2222, { dir: PEAK_DIR - 0.5, h: 55, w: 0.12 });
    ridge(760, 24, [80, 46, 22, 11, 5], 3333, { dir: PEAK_DIR, h: 230, w: 0.16 });

    // Still lake: reflects the sky gradient, lets the mirrored ridges show
    // through, and carries a rippling moon path toward the camera.
    const waterMat = new THREE.ShaderMaterial({
      uniforms: {
        uHorizon: { value: HORIZON.clone() }, uZenith: { value: ZENITH.clone() }, uGlow: { value: GLOW.clone() },
        uFog: { value: FOG.clone() }, uMoon: { value: moonVec.clone() }, uMoonCol: { value: new THREE.Color(0xdfe6ff) },
        uTime: { value: 0 }, uNear: { value: 120 }, uFar: { value: 1500 },
      },
      vertexShader: "varying vec3 vW; void main(){ vW = (modelMatrix*vec4(position,1.0)).xyz; gl_Position = projectionMatrix*viewMatrix*vec4(vW,1.0); }",
      fragmentShader: `uniform vec3 uHorizon, uZenith, uGlow, uFog, uMoon, uMoonCol; uniform float uTime, uNear, uFar; varying vec3 vW;
        float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
        float noise(vec2 p){ vec2 i = floor(p), f = fract(p); f = f*f*(3.0-2.0*f);
          return mix(mix(hash(i), hash(i+vec2(1,0)), f.x), mix(hash(i+vec2(0,1)), hash(i+vec2(1,1)), f.x), f.y); }
        void main(){
          vec3 d = normalize(vW - cameraPosition);
          float dist = length(vW - cameraPosition);
          // gentle ripples, smoothed out with distance so the far lake stays glassy
          vec2 q = vW.xz;
          float n1 = noise(q * 0.22 + vec2(uTime * 0.06, 0.0)) - 0.5;
          float n2 = noise(q * 0.9 - vec2(0.0, uTime * 0.11)) - 0.5;
          float k = 1.0 / (1.0 + dist * 0.02);
          vec3 nrm = normalize(vec3(n1 * 0.10 * k + n2 * 0.05 * k, 1.0, n2 * 0.10 * k - n1 * 0.04 * k));
          vec3 r = reflect(d, nrm);
          float e = max(r.y, 0.0);
          vec3 sky = mix(uHorizon, uZenith, pow(smoothstep(0.0, 0.6, e), 0.7)) + uGlow * 0.5 * exp(-e * 10.0);
          // moon path: broken into glitter by fine ripples, faded out near the camera
          float m = max(dot(r, uMoon), 0.0);
          float glitter = smoothstep(0.55, 0.95, noise(q * 2.6 + vec2(uTime * 0.4, -uTime * 0.25)));
          float path = (pow(m, 2500.0) * 1.6 * glitter + pow(m, 160.0) * 0.08) * smoothstep(8.0, 60.0, dist);
          vec3 c = sky * 0.55 + uMoonCol * path;
          c = mix(c, uFog, smoothstep(uNear, uFar, dist));
          // mostly see-through at grazing angles so the mirrored ridges read as reflections
          float graze = pow(1.0 - clamp(-d.y, 0.0, 1.0), 3.0);
          gl_FragColor = vec4(c, mix(0.74, 0.55, graze));
          #include <colorspace_fragment>
        }`,
      transparent: true, depthWrite: false,
    });
    const water = new THREE.Mesh(new THREE.CircleGeometry(1400, 96), waterMat);
    water.rotation.x = -Math.PI / 2;
    water.renderOrder = -0.2;            // under the rocket's transparent parts
    scene.add(water);

    // ── Holographic disc under the blueprint ──
    const holo = new THREE.Mesh(new THREE.CircleGeometry(1.6, 48),
      new THREE.MeshBasicMaterial({ map: radialTex("rgba(76,201,240,0.55)"), transparent: true, depthWrite: false }));
    holo.rotation.x = -Math.PI / 2; holo.position.y = 0.02;
    scene.add(holo);

    const starGeo = new THREE.BufferGeometry();
    const starN = lowPower ? 900 : 2200, starArr = new Float32Array(starN * 3);
    for (let i = 0; i < starN; i++) {
      const u = Math.random() * 2 - 1, a = Math.random() * Math.PI * 2, s = Math.sqrt(1 - u * u);
      starArr.set([Math.cos(a) * s * 1300, Math.abs(u) * 1300, Math.sin(a) * s * 1300], i * 3);
    }
    starGeo.setAttribute("position", new THREE.BufferAttribute(starArr, 3));
    const starMat = new THREE.PointsMaterial({ color: 0xffffff, size: 2.2, sizeAttenuation: false, transparent: true, opacity: 0.85, fog: false, depthWrite: false });
    const stars = new THREE.Points(starGeo, starMat);
    scene.add(stars);

    // Moon: disc + halo on the sky, off to the side of the main peak
    const moon = new THREE.Group();
    moon.position.copy(moonVec).multiplyScalar(1250);
    const moonSprite = (tex, size, opacity) => {
      const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false,
        fog: false, blending: THREE.AdditiveBlending, opacity }));
      s.scale.setScalar(size); moon.add(s); return s;
    };
    moonSprite(radialTex("rgba(120,150,255,0.35)"), 380, 0.45);      // halo
    moonSprite(radialTex("rgba(235,240,255,1)", "rgba(235,240,255,0)", 0.42), 70, 1);  // disc
    moon.renderOrder = -0.5;
    scene.add(moon);

    // Clouds: a band the climb punches through + a deck seen from above
    const cloudTex = [puffTex(1234567, 18), puffTex(7654321, 18), puffTex(2468013, 18)];
    const clouds = [];
    const CLOUDS = lowPower ? 40 : 80;
    let cs = 99991; const crnd = () => (cs = (cs * 16807) % 2147483647) / 2147483647;
    for (let i = 0; i < CLOUDS; i++) {
      const deck = i % 2 === 0;
      const a = crnd() * Math.PI * 2;
      const r = deck ? 25 + crnd() * 200 : 7 + crnd() * 45;
      const y = deck ? 96 + crnd() * 14 : 125 + crnd() * 70;
      const m = new THREE.SpriteMaterial({ map: cloudTex[i % 3], color: 0xc8d4e8, transparent: true, depthWrite: false, opacity: 0, rotation: crnd() * 6.28 });
      const s = new THREE.Sprite(m);
      s.position.set(Math.cos(a) * r, y, Math.sin(a) * r);
      s.scale.setScalar(deck ? 40 + crnd() * 50 : 16 + crnd() * 26);
      scene.add(s); clouds.push({ s, deck, base: deck ? 0.85 : 0.42 });
    }

    // ── Exhaust: plume sprites + lights ──
    const plume = new THREE.Group();
    plume.position.y = -0.04;
    rocket.add(plume);
    const plumeSprite = (col, w, h) => {
      const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: radialTex(col), transparent: true,
        blending: THREE.AdditiveBlending, depthWrite: false }));
      s.center.set(0.5, 1.0); s.userData = { w, h }; plume.add(s); return s;
    };
    const flames = [plumeSprite("rgba(255,77,18,1)", 0.55, 1.5), plumeSprite("rgba(255,154,46,1)", 0.34, 1.9), plumeSprite("rgba(255,243,196,1)", 0.16, 2.2)];
    const flameLight = new THREE.PointLight(0xffa040, 0, 14, 2);
    plume.add(flameLight);
    const burnDecal = new THREE.Mesh(new THREE.CircleGeometry(3.2, 48),
      new THREE.MeshBasicMaterial({ map: radialTex("rgba(255,140,60,0.9)"), transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, opacity: 0 }));
    burnDecal.rotation.x = -Math.PI / 2; burnDecal.position.y = 0.025;
    scene.add(burnDecal);
    const padGlow = new THREE.PointLight(0xff8830, 0, 16, 2);
    padGlow.position.set(0, 0.4, 0);
    scene.add(padGlow);

    // Smoke: one Points draw call; each particle's state is a closed-form
    // function of scroll progress, so it rewinds exactly.
    const SMOKE = lowPower ? 260 : 620;
    const smokeGeo = new THREE.BufferGeometry();
    const sPos = new Float32Array(SMOKE * 3), sSize = new Float32Array(SMOKE), sAlpha = new Float32Array(SMOKE),
      sRot = new Float32Array(SMOKE), sTint = new Float32Array(SMOKE);
    smokeGeo.setAttribute("position", new THREE.BufferAttribute(sPos, 3));
    smokeGeo.setAttribute("aSize", new THREE.BufferAttribute(sSize, 1));
    smokeGeo.setAttribute("aAlpha", new THREE.BufferAttribute(sAlpha, 1));
    smokeGeo.setAttribute("aRot", new THREE.BufferAttribute(sRot, 1));
    smokeGeo.setAttribute("aTint", new THREE.BufferAttribute(sTint, 1));
    const smokeSeed = [];
    let ss = 13579; const srnd = () => (ss = (ss * 16807) % 2147483647) / 2147483647;
    for (let i = 0; i < SMOKE; i++) {
      const padCloud = i % 10 < 3;
      smokeSeed.push({
        padCloud,
        tb: padCloud ? lerp(P_IGNITE + 0.004, 0.725, srnd()) : lerp(P_LIFT, P_BURNOUT + 0.012, Math.pow(srnd(), 0.8)),
        dir: srnd() * Math.PI * 2, speed: 0.6 + srnd() * 1.2, life: 3 + srnd() * 3.5,
        grow: 0.8 + srnd() * 0.7, jx: srnd() - 0.5, jz: srnd() - 0.5, rot: srnd() * 6.28, spin: (srnd() - 0.5) * 0.8,
      });
    }
    const smokeMat = new THREE.ShaderMaterial({
      uniforms: { uTex: { value: puffTex(424242, 14) }, uScale: { value: 600 } },
      vertexShader: `uniform float uScale; attribute float aSize, aAlpha, aRot, aTint; varying float vA, vR, vT;
        void main(){ vA = aAlpha; vR = aRot; vT = aTint; vec4 mv = modelViewMatrix*vec4(position,1.0);
          gl_PointSize = aSize * uScale / max(-mv.z, 0.1); gl_Position = projectionMatrix*mv; }`,
      fragmentShader: `uniform sampler2D uTex; varying float vA, vR, vT;
        void main(){ vec2 c = gl_PointCoord - 0.5; float s = sin(vR), co = cos(vR);
          vec2 uv = vec2(co*c.x - s*c.y, s*c.x + co*c.y) + 0.5;
          vec4 t = texture2D(uTex, uv);
          vec3 warm = vec3(1.0,0.78,0.55), grey = vec3(0.78,0.80,0.85);
          gl_FragColor = vec4(mix(warm, grey, vT) * t.rgb, t.a * vA * 2.2); }`,
      transparent: true, depthWrite: false,
    });
    const smoke = new THREE.Points(smokeGeo, smokeMat);
    smoke.frustumCulled = false;
    scene.add(smoke);

    // ── Camera path: [q, position, target, followPos, followTarget] ──
    // positions/targets are offsets; follow factors add the rocket's height.
    // Before these keys run, the opening orbit (see orbitAt) brings the camera
    // from the wide landscape shot to KEYS[0].
    const KEYS = [
      [0.00, [3.4, 1.9, 4.6], [0, 1.4, 0], 0, 0],
      [0.10, [3.1, 1.8, 4.2], [0, 1.35, 0], 0, 0],
      [0.21, [-2.7, 2.2, 3.7], [0, 1.25, 0], 0, 0],
      [0.30, [-2.1, 0.6, 3.1], [0, 0.2, 0], 0, 0],
      [0.39, [1.8, 0.45, 2.6], [0, 0.5, 0], 0, 0],
      [0.48, [7.4, 1.6, 2.7], [0, 1.25, 0], 0, 0],
      [0.57, [4.2, 3.4, 4.0], [0, 1.5, 0], 0, 0],
      [0.63, [6.8, 0.9, 8.4], [0, 1.4, 0], 0, 0],
      [0.675, [5.2, 0.45, 6.4], [0, 1.6, 0], 0, 0],
      [0.76, [5.6, 0.6, 7.2], [0, 1.4, 0], 0, 1],
      [0.83, [3.0, -3.4, 4.0], [0, 1.4, 0], 1, 1],
      [0.90, [3.6, 4.6, 4.6], [0, 0.8, 0], 1, 1],
    ];
    const camPos = new V3(), camTgt = new V3(), tmpA = new V3(), tmpB = new V3();
    function cameraAt(p, h) {
      let k = 0;
      while (k < KEYS.length - 2 && p > KEYS[k + 1][0]) k++;
      const a = KEYS[k], b = KEYS[k + 1];
      const t = smooth(range(p, a[0], b[0]));
      const fp = lerp(a[3], b[3], t), ft = lerp(a[4], b[4], t);
      camPos.copy(tmpA.fromArray(a[1])).lerp(tmpB.fromArray(b[1]), t); camPos.y += h * fp;
      camTgt.copy(tmpA.fromArray(a[2])).lerp(tmpB.fromArray(b[2]), t); camTgt.y += h * ft;
    }
    // One full turn around the pad, spiralling in from the landscape shot and
    // ending exactly on KEYS[0] so the hand-over is seamless.
    const ORBIT_END = Math.atan2(KEYS[0][1][2], KEYS[0][1][0]);
    const ORBIT_R = Math.hypot(KEYS[0][1][0], KEYS[0][1][2]);
    function orbitAt(s) {
      const t = smooth(range(s, 0.02, S_ORBIT));
      const ang = ORBIT_END - Math.PI * 2 * (1 - t);
      const r = lerp(60, ORBIT_R, t);
      camPos.set(Math.cos(ang) * r, lerp(4.5, KEYS[0][1][1], t), Math.sin(ang) * r);
      camTgt.set(0, lerp(17, KEYS[0][2][1], smooth(t)), 0);
    }

    let W = 0, H = 0;
    const moonAnchor = new V3();            // moon rides with the camera like the sky
    function resize() {
      W = stage.clientWidth; H = stage.clientHeight;
      renderer.setSize(W, H, false);
      camera.aspect = W / H;
      camera.fov = camera.aspect < 0.8 ? 58 : camera.aspect < 1.2 ? 46 : 38;
      camera.updateProjectionMatrix();
      smokeMat.uniforms.uScale.value = H * renderer.getPixelRatio() * 0.6;
    }
    resize();
    window.addEventListener("resize", resize);

    function render(p, time, s) {          // p: story clock, s: scroll progress
      const h = altitude(p);
      if (s < S_ORBIT) orbitAt(s); else cameraAt(p, h);

      // Phase 1 — assembly from an exploded blueprint
      const solid = smooth(range(p, 0.10, 0.21));
      const xray = band(p, 0.25, 0.385, 0.03);
      parts.forEach((pt) => {
        const a = smooth(range(p, 0.065 + pt.delay, 0.165 + pt.delay));
        const k = 1 - a;
        pt.mesh.position.x = (pt.from.finSpin ? pt.from.pos.x : 0) * k;
        pt.holder.position.set(pt.from.finSpin ? 0 : pt.from.pos.x * k, pt.from.pos.y * k, pt.from.pos.z * k);
        pt.holder.rotation.z = pt.from.rotZ * k;
        if (pt.from.finSpin) pt.mesh.rotation.x = pt.from.finSpin * k;
        pt.mats.forEach((m) => {
          const o = solid * (pt.mesh === tube || pt.from.finSpin ? 1 - 0.75 * xray : 1);
          m.opacity = o; m.transparent = o < 0.999; m.depthWrite = o > 0.5;
        });
      });
      edgeMat.opacity = Math.max(0.9 * (1 - solid), 0.55 * xray, 0.0);
      // keep the faint blueprint lines just long enough to read as "assembled"
      edgeMat.visible = edgeMat.opacity > 0.01;

      // Showcase spin while in the studio, then hold heading for the flight
      rocket.rotation.y = 0.6 + 4.2 * smooth(range(p, 0, 0.58));
      rocket.rotation.z = 0.06 * band(p, 0.42, 0.57, 0.04) + 0.12 * range(p, P_BURNOUT, P_APOGEE);
      rocket.position.y = h;

      // Phase 2 — motor in, nozzle arms
      const mIn = smooth(range(p, 0.25, 0.33));
      motor.visible = p > 0.235 && p < 0.405;
      motor.position.y = lerp(-1.3, 0.5, mIn);
      motorMat.opacity = band(p, 0.235, 0.40, 0.02);
      motorMat.emissiveIntensity = 0.4 + 0.9 * range(p, 0.30, 0.36);
      motorEdges.material.opacity = motorMat.opacity;
      const arm = band(p, 0.33, 0.40, 0.015);
      nozMat.emissiveIntensity = 2.5 * arm + 3.0 * range(p, P_IGNITE, P_IGNITE + 0.01) * (1 - range(p, P_BURNOUT, P_BURNOUT + 0.02));
      sparks.visible = arm > 0.01;
      if (sparks.visible) {
        for (let i = 0; i < SPARKS; i++) {
          const [o, a, sp] = sparkSeed[i];
          const tt = (time * sp + o) % 1;
          sparkPos[i * 3] = Math.cos(a) * tt * 0.35 * sp;
          sparkPos[i * 3 + 1] = -0.05 - tt * 0.6 - tt * tt * 0.4;
          sparkPos[i * 3 + 2] = Math.sin(a) * tt * 0.35 * sp;
        }
        sparkGeo.attributes.position.needsUpdate = true;
        sparks.material.opacity = arm;
      }

      // Phase 3 — CFD overlay
      const cfd = band(p, 0.41, 0.575, 0.03);
      flowGroup.visible = cfd > 0.005;
      flowMats.forEach((m) => { m.uniforms.uOpacity.value = cfd; m.uniforms.uTime.value = time; });
      wash.visible = cfd > 0.005; washMat.uniforms.uOpacity.value = cfd * 0.9;
      shock.visible = cfd > 0.005; shockMat.uniforms.uOpacity.value = cfd * 0.55;

      // Holographic disc while the blueprint is on it; sky deepens on the climb
      holo.material.opacity = 0.8 * (1 - range(p, 0.45, 0.6));
      holo.visible = holo.material.opacity > 0.005;
      const climb = range(h, 10, 220);
      skyMat.uniforms.uZenith.value.copy(ZENITH).lerp(SPACE, climb);
      skyMat.uniforms.uHorizon.value.copy(HORIZON).lerp(ZENITH, climb * 0.6);
      skyMat.uniforms.uGlowAmt.value = 1 - 0.6 * climb;
      waterMat.uniforms.uTime.value = time;
      starMat.opacity = 0.85 + 0.15 * climb;

      // the deck only reads as a cloud sea from above — hide it while the camera is under it
      const aboveDeck = range(camPos.y, 105, 135);
      clouds.forEach((c) => (c.s.material.opacity = c.base * range(p, 0.70, 0.76) * (c.deck ? aboveDeck : 1)));

      // Phase 4 — ignition, flame, smoke
      const ign = range(p, P_IGNITE, P_IGNITE + 0.015);
      const burn = 1 - range(p, P_BURNOUT, P_BURNOUT + 0.012);
      const fl = ign * burn;
      plume.visible = fl > 0.005;
      if (plume.visible) {
        const flick = 0.88 + 0.08 * Math.sin(time * 41) + 0.05 * Math.sin(time * 23.7);
        const boostLen = 1 + 0.8 * range(p, P_LIFT, P_LIFT + 0.05);
        flames.forEach((s, i) => {
          s.scale.set(s.userData.w * fl * (0.95 + 0.05 * flick), s.userData.h * fl * boostLen * flick * (1 - i * 0.05), 1);
          s.material.opacity = fl;
        });
      }
      flameLight.intensity = 9 * fl;
      padGlow.intensity = 14 * fl * (1 - range(h, 0, 12));
      burnDecal.material.opacity = fl * (1 - range(h, 0, 25));

      const age0 = 22;                                    // progress → seconds-ish
      for (let i = 0; i < SMOKE; i++) {
        const s = smokeSeed[i];
        const age = (p - s.tb) * age0;
        if (age <= 0 || age > s.life) { sSize[i] = 0; sAlpha[i] = 0; continue; }
        const life = age / s.life;
        const fadeIn = clamp(age / 0.08);
        if (s.padCloud) {
          const out = (1 - Math.exp(-age * 1.6)) * s.speed * 3.2;
          sPos[i * 3] = Math.cos(s.dir) * out;
          sPos[i * 3 + 1] = 0.15 + age * 0.35 + out * 0.08;
          sPos[i * 3 + 2] = Math.sin(s.dir) * out;
          sSize[i] = (0.9 + age * 1.9) * s.grow;
        } else {
          const hb = altitude(s.tb);
          const spread = 0.12 + age * 0.35 * (1 + hb / 60);
          sPos[i * 3] = s.jx * spread;
          sPos[i * 3 + 1] = hb - 0.15 - age * 0.25 - s.jz * 0.3 * (1 + hb / 40);
          sPos[i * 3 + 2] = s.jz * spread;
          sSize[i] = (0.9 + age * 1.7) * s.grow * (1 + hb / 45);
        }
        sAlpha[i] = fadeIn * (1 - life) * (1 - life) * (s.padCloud ? 0.75 : 0.6);
        sRot[i] = s.rot + s.spin * age;
        sTint[i] = clamp(age / 0.5);
      }
      smokeGeo.attributes.position.needsUpdate = true;
      smokeGeo.attributes.aSize.needsUpdate = true;
      smokeGeo.attributes.aAlpha.needsUpdate = true;
      smokeGeo.attributes.aRot.needsUpdate = true;
      smokeGeo.attributes.aTint.needsUpdate = true;

      // Camera, with the frame nudged away from whichever side the copy is on
      camera.position.copy(camPos);
      camera.lookAt(camTgt);
      sky.position.copy(camPos);
      stars.position.copy(camPos);
      moon.position.sub(moonAnchor).add(camPos); moonAnchor.copy(camPos);
      let side = 0;
      TEXT_WINDOWS.forEach(([a, b], i) => {
        const el = phases[i];
        side += band(p, a - 0.02, b + 0.02, 0.04) * (el.dataset.side === "left" ? 1 : -1);
      });
      if (camera.aspect > 1.15) camera.setViewOffset(W, H, -side * W * 0.13, 0, W, H);
      else if (camera.aspect < 0.8) camera.setViewOffset(W, H, 0, H * 0.14, W, H);  // portrait: lift the scene above the copy
      else camera.clearViewOffset();

      // Reflection pass: the rocket mirrored under the waterline (three flips
      // the face winding for the negative scale), drawn first so the lake's
      // translucent surface tints it like the mirrored ridges
      renderer.clear();
      const vis = mirrorHide.map((o) => o.visible);
      mirrorHide.forEach((o) => (o.visible = false));
      rocket.scale.y = -1; rocket.position.y = -h;
      renderer.render(scene, camera);
      rocket.scale.y = 1; rocket.position.y = h;
      mirrorHide.forEach((o, i) => (o.visible = vis[i]));
      renderer.render(scene, camera);
    }

    // Everything except the rocket and the lights sits out the reflection pass
    const mirrorHide = scene.children.filter((o) => o !== rocket && !o.isLight);
    renderer.autoClear = false;

    // Warm up shaders behind the loader so the first scroll doesn't hitch
    render(0, 0, 0);
    renderer.compile(scene, camera);

    return {
      render,
      dispose() {
        window.removeEventListener("resize", resize);
        renderer.dispose();
        scene.traverse((o) => {
          if (o.geometry) o.geometry.dispose();
          const ms = o.material ? (Array.isArray(o.material) ? o.material : [o.material]) : [];
          ms.forEach((m) => { if (m.map) m.map.dispose(); m.dispose(); });
        });
      },
    };
  }
})();
