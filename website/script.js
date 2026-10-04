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
      // The first frame's timestamp can predate `start`; clamp so it never counts below zero
      const p = Math.min(Math.max((now - start) / dur, 0), 1);
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

  /* ── Promo video: autoplay (muted) while in view, overlay controls ── */
  const vp = document.querySelector(".vp");
  if (vp) {
    const video = vp.querySelector("video");
    const bigBtn = vp.querySelector(".vp-big");
    const muteBtn = vp.querySelector(".vp-mute");
    const fsBtn = vp.querySelector(".vp-fs");
    const bar = vp.querySelector(".vp-progress span");
    let userPaused = false; // a pause the visitor chose — scrolling back must not override it

    video.removeAttribute("controls");
    vp.querySelectorAll("[hidden]").forEach((el) => (el.hidden = false));
    vp.classList.add("is-ready", "is-muted");

    const sync = () => {
      vp.classList.toggle("is-paused", video.paused);
      bigBtn.setAttribute("aria-label", video.paused ? "Play video" : "Pause video");
    };
    const toggle = () => {
      if (video.paused) {
        userPaused = false;
        video.play().catch(() => {});
      } else {
        userPaused = true;
        video.pause();
      }
    };
    video.addEventListener("play", sync);
    video.addEventListener("pause", sync);
    video.addEventListener("click", toggle);
    bigBtn.addEventListener("click", toggle);
    video.addEventListener("timeupdate", () => {
      if (video.duration) bar.style.transform = `scaleX(${video.currentTime / video.duration})`;
    });

    muteBtn.addEventListener("click", () => {
      video.muted = !video.muted;
      vp.classList.toggle("is-muted", video.muted);
      muteBtn.setAttribute("aria-label", video.muted ? "Unmute" : "Mute");
    });
    fsBtn.addEventListener("click", () => {
      if (document.fullscreenElement) document.exitFullscreen();
      else if (vp.requestFullscreen) vp.requestFullscreen();
      else if (video.webkitEnterFullscreen) video.webkitEnterFullscreen(); // iOS Safari
    });

    if (!reduced) {
      new IntersectionObserver(
        ([e]) => {
          if (e.isIntersecting && !userPaused) video.play().catch(() => {});
          else if (!e.isIntersecting && !video.paused) video.pause();
        },
        { threshold: 0.5 }
      ).observe(vp);
    }
  }

  /* ── Workspace marquee ── */
  // Duplicate each track so translateX(-50%) loops seamlessly. The copies are
  // decorative: hidden from screen readers and skipped by keyboard focus.
  if (!reduced) {
    document.querySelectorAll(".ws-marquee").forEach((mq) => {
      mq.querySelectorAll(".ws-track").forEach((track) => {
        Array.from(track.children).forEach((tile) => {
          const copy = tile.cloneNode(true);
          copy.setAttribute("aria-hidden", "true");
          copy.tabIndex = -1;
          track.appendChild(copy);
        });
      });
      mq.classList.add("is-running");
    });
  }

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
        // Carry the card's accent colour into the viewer (empty for cards without one)
        lb.style.setProperty("--c", fig ? fig.style.getPropertyValue("--c") : "");
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
    "[SIM] 6DOF · RK4 · dt adaptive",
    "[SIM] T+0.00s   liftoff        thrust 2106 N",
    "[SIM] T+1.34s   burnout        v = 168.4 m/s   M 0.50",
    "[SIM] T+15.6s   APOGEE         1,191 m AGL",
    "[SIM] T+16.6s   drogue deploy  descent 20.8 m/s",
    "[SIM] T+59.1s   main deploy    descent 7.7 m/s",
    "[SIM] T+96.2s   touchdown nominal — recovery DEPLOYED ✓",
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
