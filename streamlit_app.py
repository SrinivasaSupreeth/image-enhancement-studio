"""Image Enhancement Studio - Streamlit web app.

Run:   streamlit run streamlit_app.py
"""
import base64
import glob
import hashlib
import io
import os
import time

import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Image Enhancement Studio", page_icon="🎞️", layout="wide")

import enhance as E  # noqa: E402  (imports TensorFlow, after page config)

import streamlit.components.v1 as components

BASE = os.path.dirname(os.path.abspath(__file__))

# Each effect has its own ink colour, used everywhere that effect appears.
EFFECTS = {
    "superres": {
        "name": "Super-resolution", "color": "#0E8A96", "file": "1_super_resolution.png",
        "short": "Turn a small, blurry photo into a sharp one, 4× bigger.",
        "model": "ESRGAN, a pretrained CNN",
        "steps": ["Your photo is reduced to a low-resolution version (160 px).",
                  "ESRGAN, a deep CNN trained with a GAN, predicts the missing detail.",
                  "The result is 4× wider and 4× taller with sharp edges and texture."],
        "run": "Upscale photo",
    },
    "denoise": {
        "name": "Remove noise and blur", "color": "#B8326A", "file": "2_denoise_deblur.png",
        "short": "Clean up grain and the streaks left by shaky hands.",
        "model": "DnCNN-style CNN, trained by us",
        "steps": ["We damaged thousands of clean photos with camera-shake blur and noise.",
                  "A 7-layer CNN with dilated convolutions learned to predict that damage.",
                  "The predicted damage is subtracted from your photo (residual learning)."],
        "run": "Clean photo",
    },
    "style": {
        "name": "Style transfer", "color": "#B7860B", "file": "3_style_transfer.png",
        "short": "Repaint your photo with the textures and colours of a painting.",
        "model": "Magenta arbitrary style transfer, pretrained",
        "steps": ["One CNN reads the style picture and summarises its textures and colours.",
                  "A second CNN keeps the shapes and objects of your photo.",
                  "Both are combined into a new image in a single pass."],
        "run": "Apply style",
    },
    "colorize": {
        "name": "Colourise black and white", "color": "#3552C4", "file": "4_colorization.png",
        "short": "Give an old black-and-white photo realistic colour.",
        "model": "U-Net CNN, trained by us",
        "steps": ["The photo is split into lightness (L) and colour (a, b) in Lab colour space.",
                  "A U-Net trained on flower photos predicts a and b from L alone.",
                  "The predicted colour is joined with the original lightness."],
        "run": "Colourise photo",
    },
}

# ------------------------------------------------------------------ styles
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
:root { --ink:#16202E; --graphite:#5B6675; --table:#EEF1F4; --paper:#FFFFFF; --rule:#D5DCE3; --film:#1B2230; }
html, body, [class*="css"], .stMarkdown, .stText, p, li, label, .stRadio, .stButton button, .stDownloadButton button {
  font-family: 'IBM Plex Sans', system-ui, sans-serif; }
h1, h2, h3, .display { font-family: 'Bricolage Grotesque', 'IBM Plex Sans', sans-serif !important; letter-spacing: -0.02em; color: var(--ink); }
.block-container { padding-top: 2.2rem; max-width: 1200px; }

/* hero */
.hero h1 { font-size: clamp(2.4rem, 5vw, 4.2rem); line-height: 0.98; font-weight: 800; margin: 0 0 0.6rem; }
.hero p.lede { font-size: 1.12rem; color: var(--graphite); max-width: 60ch; margin: 0 0 1.6rem; }
.filmstrip { background: var(--film); border-radius: 6px; padding: 18px 14px; display: grid;
  grid-template-columns: repeat(4, 1fr); gap: 12px; position: relative; }
.filmstrip::before, .filmstrip::after { content: ""; position: absolute; left: 10px; right: 10px; height: 7px;
  background-image: radial-gradient(circle, #EEF1F4 0 2.5px, transparent 3px); background-size: 18px 7px; opacity: .55; }
.filmstrip::before { top: 5px; } .filmstrip::after { bottom: 5px; }
.frame { margin: 0; }
.frame img { width: 100%; aspect-ratio: 4 / 3; object-fit: cover; display: block; border-radius: 2px; }
.frame figcaption { color: #E7EBF0; font-size: .82rem; margin-top: 8px; display: flex; gap: 8px; align-items: center; }
.swatch { width: 10px; height: 10px; border-radius: 50%; flex: none; display: inline-block; }
@media (max-width: 760px) { .filmstrip { grid-template-columns: repeat(2, 1fr); } }

/* effect card in the sidebar + explanation cards */
.effect-card { background: var(--paper); border: 1px solid var(--rule); border-left: 5px solid var(--c); border-radius: 8px; padding: 12px 14px; margin: 4px 0 14px; }
.effect-card b { display: block; font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.05rem; }
.effect-card span { color: var(--graphite); font-size: .9rem; }
.how { background: var(--paper); border: 1px solid var(--rule); border-top: 5px solid var(--c); border-radius: 8px; padding: 18px 20px; height: 100%; }
.how h3 { margin: 0 0 2px; font-size: 1.25rem; }
.how .model { color: var(--graphite); font-size: .9rem; margin-bottom: 10px; }
.how ol { margin: 0; padding-left: 1.2rem; } .how li { margin-bottom: 6px; font-size: .95rem; }

/* result area */
.empty { border: 2px dashed var(--rule); border-radius: 10px; padding: 48px 24px; text-align: center; color: var(--graphite); background: #F6F8FA; }
.empty b { color: var(--ink); font-size: 1.05rem; display: block; margin-bottom: 4px; }
.result-head { display: flex; align-items: baseline; gap: 10px; margin-bottom: 6px; }
.result-head h3 { margin: 0; font-size: 1.4rem; }
.note { color: var(--graphite); font-size: .95rem; margin: 4px 0 12px; }
[data-testid="stMetricValue"] { font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.35rem; }
.stButton button[kind="primary"] { padding: .7rem 1rem; font-weight: 600; border-radius: 8px; }
.footer { color: var(--graphite); font-size: .85rem; margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--rule); }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ helpers
def to_pil(arr):
    return Image.fromarray((np.clip(arr, 0, 1) * 255).round().astype(np.uint8))


def png_bytes(arr):
    buf = io.BytesIO()
    to_pil(arr).save(buf, "PNG")
    return buf.getvalue()


@st.cache_data(show_spinner=False)
def hero_thumb(path):
    """Cut the 'after' half out of a notebook result figure and make a small JPEG."""
    img = np.asarray(Image.open(path).convert("L"))
    rgb = Image.open(path).convert("RGB")
    h, w = img.shape
    right = img[:, w // 2:]
    rows = np.where((right < 235).mean(axis=1) > 0.3)[0]        # rows that are mostly picture, not title text
    cols = np.where((right < 235).mean(axis=0) > 0.3)[0]
    box = (w // 2 + cols.min(), rows.min(), w // 2 + cols.max(), rows.max()) if len(rows) and len(cols) else (w // 2, 0, w, h)
    thumb = rgb.crop(box)
    thumb.thumbnail((420, 420))
    buf = io.BytesIO()
    thumb.save(buf, "JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode()


@st.cache_data(show_spinner=False)
def samples():
    return {k: (v / 255.0).astype(np.float32) for k, v in E.sample_images().items()}


def _jpeg_b64(im):
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def comparison(before, after, height=430):
    """Before/after slider: drag (or use arrow keys) to reveal the original over the result."""
    a = to_pil(after)
    b = to_pil(before).resize(a.size, Image.BICUBIC)
    components.html(f"""
    <style>
      body {{ margin: 0; font-family: 'IBM Plex Sans', system-ui, sans-serif; }}
      .cmp {{ position: relative; height: {height}px; background: #1B2230; border-radius: 8px; overflow: hidden; }}
      .cmp img {{ position: absolute; inset: 0; width: 100%; height: 100%; object-fit: contain; user-select: none; }}
      .line {{ position: absolute; top: 0; bottom: 0; width: 2px; background: #fff; transform: translateX(-1px); pointer-events: none; }}
      .knob {{ position: absolute; top: 50%; left: 50%; width: 34px; height: 34px; border-radius: 50%; background: #fff;
               transform: translate(-50%, -50%); box-shadow: 0 1px 6px rgba(0,0,0,.35); display: grid; place-items: center;
               font-size: 14px; color: #16202E; }}
      .lbl {{ position: absolute; top: 10px; padding: 3px 9px; border-radius: 4px; background: rgba(22,32,46,.78);
              color: #fff; font-size: 13px; pointer-events: none; }}
      .lbl.l {{ left: 10px; }} .lbl.r {{ right: 10px; }}
      input {{ position: absolute; inset: 0; width: 100%; height: 100%; margin: 0; opacity: 0; cursor: ew-resize; }}
      input:focus-visible + .ring {{ outline: 3px solid #0E8A96; }}
      .ring {{ position: absolute; inset: 0; border-radius: 8px; pointer-events: none; }}
    </style>
    <div class="cmp">
      <img src="data:image/jpeg;base64,{_jpeg_b64(a)}" alt="Result">
      <img id="top" src="data:image/jpeg;base64,{_jpeg_b64(b)}" alt="Original">
      <div class="line" id="line"><span class="knob">⇆</span></div>
      <span class="lbl l">Original</span><span class="lbl r">After</span>
      <input type="range" min="0" max="100" value="50" step="0.5" id="r" aria-label="Compare original and result">
      <div class="ring"></div>
    </div>
    <script>
      const r = document.getElementById('r'), topImg = document.getElementById('top'), line = document.getElementById('line');
      function update() {{ topImg.style.clipPath = `inset(0 ${{100 - r.value}}% 0 0)`; line.style.left = r.value + '%'; }}
      r.addEventListener('input', update); update();
    </script>""", height=height + 4)


# ------------------------------------------------------------------ sidebar: controls
with st.sidebar:
    st.markdown("<h2 style='margin-top:0'>Studio controls</h2>", unsafe_allow_html=True)
    effect = st.radio("Effect", list(EFFECTS), format_func=lambda k: EFFECTS[k]["name"], key="effect")
    fx = EFFECTS[effect]
    st.markdown(f"<div class='effect-card' style='--c:{fx['color']}'><b>{fx['name']}</b><span>{fx['short']}</span></div>",
                unsafe_allow_html=True)

    style_file, simulate = None, True
    if effect == "style":
        style_file = st.file_uploader("Style picture (optional)", type=["jpg", "jpeg", "png"],
                                      help="A painting or texture. Leave empty to use a Kandinsky painting.")
        if not style_file:
            st.caption("No style picture chosen: a Kandinsky painting will be used.")
    if effect == "denoise":
        simulate = st.toggle("Add camera shake and noise first", value=True,
                             help="Keep on for a clear demo. Turn off if your photo is already noisy or blurry.")

    st.divider()
    ready = E.status()
    st.markdown("**Models**")
    for k, label in [("superres", "ESRGAN"), ("denoise", "Denoiser"), ("style", "Style transfer"), ("colorize", "Colouriser")]:
        ok = ready.get(k, True)
        st.markdown(f"<span class='swatch' style='background:{EFFECTS[k]['color']}'></span>&nbsp; {label}: "
                    f"{'ready' if ok else '<b>not trained</b>'}", unsafe_allow_html=True)
    if not all(ready.values()):
        st.caption("Run `python train_models.py` once to train the missing models.")


# ------------------------------------------------------------------ hero
frames = "".join(
    f"<figure class='frame'><img src='data:image/jpeg;base64,{hero_thumb(os.path.join(BASE, 'static', 'results', v['file']))}' alt='{v['name']} result'>"
    f"<figcaption><span class='swatch' style='background:{v['color']}'></span>{v['name']}</figcaption></figure>"
    for v in EFFECTS.values() if os.path.exists(os.path.join(BASE, "static", "results", v["file"])))
st.markdown(f"""
<div class="hero">
  <h1>Image Enhancement Studio</h1>
  <p class="lede">Sharpen, clean, restyle and colourise photos with convolutional neural networks,
  built in Python and TensorFlow. Upload a picture and compare the original with the result.</p>
  {f'<div class="filmstrip">{frames}</div>' if frames else ''}
</div>
""", unsafe_allow_html=True)
st.write("")

tab_studio, tab_results, tab_how = st.tabs(["Studio", "Notebook results", "How it works"])

# ------------------------------------------------------------------ studio
with tab_studio:
    left, right = st.columns([5, 7], gap="large")

    with left:
        st.markdown("### Your photo")
        source = st.radio("Source", ["Upload a photo", "Use a sample"], horizontal=True, label_visibility="collapsed")
        img, img_id = None, None
        if source == "Upload a photo":
            up = st.file_uploader("Photo", type=["jpg", "jpeg", "png", "bmp", "webp"], label_visibility="collapsed")
            if up:
                data = up.getvalue()
                try:
                    img = E.read_image(data)
                    img_id = hashlib.md5(data).hexdigest()
                except Exception:
                    st.error("This file could not be opened as an image. Use a JPG or PNG.")
        else:
            name = st.selectbox("Sample", list(samples()), label_visibility="collapsed")
            img, img_id = samples()[name], f"sample-{name}"

        if img is not None:
            st.image(to_pil(img), use_container_width=True)
        run = st.button(fx["run"], type="primary", use_container_width=True, disabled=img is None)

    with right:
        if run and img is not None:
            style = E.read_image(style_file.getvalue()) if style_file else None
            start = time.time()
            with st.spinner(f"Running {fx['model']}… the first run loads the model and can take up to a minute."):
                try:
                    result = E.run(effect, img, style=style, simulate=simulate)
                    result["seconds"] = time.time() - start
                    result["effect"] = effect
                    st.session_state["result"] = result
                    st.session_state.pop("error", None)
                except E.ModelNotTrained as e:
                    st.session_state["error"] = str(e)
                    st.session_state.pop("result", None)
                except Exception as e:
                    hint = " The first use downloads the model, so check the internet connection." if effect in ("superres", "style") else ""
                    st.session_state["error"] = f"The effect could not run: {e}.{hint}"
                    st.session_state.pop("result", None)

        if "error" in st.session_state:
            st.error(st.session_state["error"])
        elif "result" in st.session_state:
            r = st.session_state["result"]
            rfx = EFFECTS[r["effect"]]
            st.markdown(f"<div class='result-head'><span class='swatch' style='background:{rfx['color']};width:14px;height:14px'></span>"
                        f"<h3>{rfx['name']}</h3></div><p class='note'>{r['info']}</p>", unsafe_allow_html=True)

            metrics = dict(r.get("metrics", {}))
            metrics["Time"] = f"{r['seconds']:.1f} s"
            cols = st.columns(len(metrics))
            for c, (k, v) in zip(cols, metrics.items()):
                c.metric(k, v)

            st.caption("Drag across the picture to compare the original with the result.")
            comparison(r["before"], r["after"])

            st.markdown("##### Side by side")
            c1, c2 = st.columns(2)
            c1.image(to_pil(r["before"]), caption="Original", use_container_width=True)
            c2.image(to_pil(r["after"]), caption="After", use_container_width=True)
            if r.get("style") is not None:
                st.image(to_pil(r["style"]), caption="Style picture used", width=140)
            st.download_button("Download result (PNG)", png_bytes(r["after"]), file_name=f"{r['effect']}_result.png",
                               mime="image/png", use_container_width=True)
        else:
            st.markdown(f"""<div class='empty'><b>Your result will appear here</b>
            Pick an effect on the left, choose a photo, then press “{fx['run']}”.</div>""", unsafe_allow_html=True)

# ------------------------------------------------------------------ notebook results
with tab_results:
    st.markdown("### Results from our Colab notebook")
    st.markdown("<p class='note'>Each picture shows the original on the left and the result on the right, "
                "produced by <code>Image_Quality_CNN.ipynb</code>.</p>", unsafe_allow_html=True)
    items = [v for v in EFFECTS.values() if os.path.exists(os.path.join(BASE, "static", "results", v["file"]))]
    for i in range(0, len(items), 2):
        cols = st.columns(2, gap="large")
        for c, v in zip(cols, items[i:i + 2]):
            with c:
                st.markdown(f"<div class='result-head'><span class='swatch' style='background:{v['color']}'></span>"
                            f"<b>{v['name']}</b></div>", unsafe_allow_html=True)
                st.image(os.path.join(BASE, "static", "results", v["file"]), use_container_width=True)

# ------------------------------------------------------------------ how it works
with tab_how:
    st.markdown("### How each effect works")
    st.markdown("<p class='note'>All four effects use convolutional neural networks (CNNs): networks that learn "
                "patterns such as edges, textures and shapes from many example images.</p>", unsafe_allow_html=True)
    keys = list(EFFECTS)
    for i in range(0, 4, 2):
        cols = st.columns(2, gap="large")
        for c, k in zip(cols, keys[i:i + 2]):
            v = EFFECTS[k]
            steps = "".join(f"<li>{s}</li>" for s in v["steps"])
            c.markdown(f"<div class='how' style='--c:{v['color']}'><h3>{v['name']}</h3>"
                       f"<div class='model'>{v['model']}</div><ol>{steps}</ol></div>", unsafe_allow_html=True)
        st.write("")

st.markdown("<div class='footer'>Improving Image Quality: picture transformation and enhancement using CNNs. "
            "Built with Python, TensorFlow and Streamlit.</div>", unsafe_allow_html=True)
