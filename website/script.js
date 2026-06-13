/* K2 AeroSim site — starfield, scroll reveal, counters, hero terminal */
(function () {
  "use strict";

  /* ── Starfield canvas ── */
  const canvas = document.getElementById("starfield");
  const ctx = canvas.getContext("2d");
  let stars = [];
  let w, h;
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function resize() {
    w = canvas.width = window.innerWidth;
    h = canvas.height = window.innerHeight;
    const n = Math.min(220, Math.floor((w * h) / 9000));
    stars = Array.from({ length: n }, () => ({
      x: Math.random() * w,
      y: Math.random() * h,
      r: Math.random() * 1.3 + 0.2,
      v: Math.random() * 0.25 + 0.05,
      tw: Math.random() * Math.PI * 2,
    }));
  }

  function tick(t) {
    ctx.clearRect(0, 0, w, h);
    for (const s of stars) {
      const a = 0.35 + 0.45 * Math.sin(t / 900 + s.tw);
      ctx.globalAlpha = a;
      ctx.fillStyle = s.r > 1 ? "#7fd4ff" : "#cfd8ea";
      ctx.beginPath();
      ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
      ctx.fill();
      s.y -= s.v;
      if (s.y < -2) { s.y = h + 2; s.x = Math.random() * w; }
    }
    ctx.globalAlpha = 1;
    requestAnimationFrame(tick);
  }

  resize();
  window.addEventListener("resize", resize);
  if (!reduced) requestAnimationFrame(tick);
  else {
    // static field
    for (const s of stars) {
      ctx.globalAlpha = 0.5;
      ctx.fillStyle = "#cfd8ea";
      ctx.beginPath();
      ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  /* ── Scroll reveal ── */
  const io = new IntersectionObserver(
    (entries) => {
      for (const e of entries) {
        if (e.isIntersecting) {
          e.target.classList.add("in");
          io.unobserve(e.target);
        }
      }
    },
    { threshold: 0.12 }
  );
  document.querySelectorAll(".reveal").forEach((el) => io.observe(el));

  /* ── Animated counters ── */
  function animateCount(el) {
    const target = parseInt(el.dataset.count, 10);
    const suffix = el.dataset.suffix || "";
    const dur = 1600;
    const start = performance.now();
    function step(now) {
      const p = Math.min((now - start) / dur, 1);
      const eased = 1 - Math.pow(1 - p, 3);
      el.textContent = Math.round(target * eased).toLocaleString() + (p === 1 ? suffix : "");
      if (p < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }
  const cio = new IntersectionObserver(
    (entries) => {
      for (const e of entries) {
        if (e.isIntersecting) {
          animateCount(e.target);
          cio.unobserve(e.target);
        }
      }
    },
    { threshold: 0.6 }
  );
  document.querySelectorAll(".metric-num").forEach((el) => {
    if (el.dataset.count === undefined) return;
    if (reduced) {
      el.textContent =
        parseInt(el.dataset.count, 10).toLocaleString() + (el.dataset.suffix || "");
    } else {
      cio.observe(el);
    }
  });

  /* ── Screenshot lightbox ── */
  const lb = document.getElementById("lightbox");
  if (lb) {
    const lbImg = document.getElementById("lbImg");
    const lbCap = document.getElementById("lbCap");
    const lbClose = document.getElementById("lbClose");
    document.querySelectorAll(".shot img, .wsd-shot img").forEach((img) => {
      img.addEventListener("click", () => {
        lbImg.src = img.src;
        lbImg.alt = img.alt;
        const fig = img.closest(".shot, .wsd-shot");
        const cap = fig ? fig.querySelector("figcaption") : null;
        lbCap.innerHTML = cap ? cap.innerHTML : img.alt;
        lb.hidden = false;
        document.body.style.overflow = "hidden";
      });
    });
    function closeLb() {
      lb.hidden = true;
      lbImg.src = "";
      document.body.style.overflow = "";
    }
    lbClose.addEventListener("click", closeLb);
    lb.addEventListener("click", (e) => { if (e.target === lb) closeLb(); });
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && !lb.hidden) closeLb();
    });
  }

  /* ── Hero terminal typewriter ── */
  const lines = [
    "$ python main.py",
    "[K2] AeroSim initialized — 12 workspaces ready",
    "[SIM] 6DOF · RK45 Dormand-Prince · dt adaptive",
    "[SIM] T+0.00s   liftoff        thrust 1977 N",
    "[SIM] T+1.34s   burnout        v = 168.8 m/s   M 0.50",
    "[SIM] T+15.7s   APOGEE         1,223 m AGL",
    "[SIM] T+16.1s   drogue deploy  descent 21.2 m/s",
    "[SIM] T+60.4s   main deploy    descent 5.1 m/s",
    "[SIM] T+95.8s   touchdown nominal — recovery DEPLOYED ✓",
  ];
  const term = document.getElementById("termBody");
  if (term) {
    if (reduced) {
      term.textContent = lines.join("\n");
    } else {
      let li = 0, ci = 0, out = "";
      function type() {
        if (li >= lines.length) return;
        const line = lines[li];
        if (ci < line.length) {
          out += line[ci++];
          term.textContent = out + "▌";
          setTimeout(type, li === 0 ? 38 : 9);
        } else {
          out += "\n";
          term.textContent = out + "▌";
          li++; ci = 0;
          setTimeout(type, 260);
        }
      }
      setTimeout(type, 600);
    }
  }
})();

/* Download buttons link directly to the GitHub Releases asset
   (releases/latest/download/K2-Setup.exe) — a stable URL that always resolves
   to the newest release, so no client-side tag lookup is needed. */
