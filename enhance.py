"""The CNN image-processing functions used by the web page.

1. super_resolve  - low-res / blurry image -> sharp 4x image      (pretrained ESRGAN)
2. denoise        - remove noise + motion blur (shaky hands)       (our DnCNN-style CNN)
3. stylize        - neural style transfer                          (pretrained Magenta CNN)
4. colorize       - black & white -> colour                        (our U-Net CNN)
"""
import io
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")                                   # hide TensorFlow log spam
_hub_dir = os.path.join(MODEL_DIR, "hub_cache")
if not os.access(MODEL_DIR, os.W_OK):                                              # read-only host: use temp folder
    import tempfile
    _hub_dir = os.path.join(tempfile.gettempdir(), "hub_cache")
os.environ.setdefault("TFHUB_CACHE_DIR", _hub_dir)                                 # keep downloads in one place

# On Streamlit Community Cloud (code lives in /mount/src) memory is limited,
# so only one model is kept in memory at a time.
LOW_MEMORY = os.environ.get("LOW_MEMORY") == "1" or os.path.exists("/mount/src")

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from scipy.ndimage import rotate
from skimage import color
from tensorflow.keras import layers

ESRGAN_URL = "https://tfhub.dev/captain-pool/esrgan-tf2/1"
STYLE_MODEL_URL = "https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2"
DEFAULT_STYLE_URL = ("https://storage.googleapis.com/download.tensorflow.org/example_images/"
                     "Vassily_Kandinsky%2C_1913_-_Composition_7.jpg")
DENOISER_PATH = os.path.join(MODEL_DIR, "denoiser.keras")
COLORIZER_PATH = os.path.join(MODEL_DIR, "colorizer.keras")

SR_INPUT_SIDE = 160     # super-resolution works on a small image (output is 4x = 640 px)
MAX_SIDE = 512          # other tasks: longest side of the processed image (keeps it fast on a laptop)


class ModelNotTrained(Exception):
    pass


# ----------------------------------------------------------------- model cache
_models = {}


def _model(name, loader):
    """Load each model only once, then keep it in memory."""
    if name not in _models:
        if LOW_MEMORY and _models:
            import gc
            _models.clear()
            gc.collect()
        print(f"Loading {name} model ...", flush=True)
        _models[name] = loader()
    return _models[name]


def status():
    return {"denoise": os.path.exists(DENOISER_PATH), "colorize": os.path.exists(COLORIZER_PATH)}


# ----------------------------------------------------------------- image helpers
def read_image(data):
    """Uploaded file bytes -> float32 RGB array with values 0..1."""
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(data)))      # respect phone-photo rotation
    return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0


def to_png_bytes(img):
    arr = (np.clip(img, 0, 1) * 255).round().astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    return buf.getvalue()


def resize_max(img, max_side):
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s == 1.0:
        return img
    return tf.image.resize(img, (max(1, round(h * s)), max(1, round(w * s))), antialias=True).numpy()


def psnr(a, b):
    return float(tf.image.psnr(np.clip(a, 0, 1), np.clip(b, 0, 1), max_val=1.0))


# ----------------------------------------------------------------- simulated damage (task 2)
def motion_kernel(length, angle):
    """A straight line kernel = the streak a shaking camera leaves."""
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1.0
    k = rotate(k, angle, reshape=False, order=1)
    return k / k.sum()


def shake_and_noise(img, rng, length=None, sigma=None):
    """Add motion blur in a random direction + grainy noise to one image."""
    if length is None:
        length = int(rng.choice([1, 3, 5, 7]))          # 1 = no blur
    out = img
    if length > 1:
        k = np.repeat(motion_kernel(length, rng.uniform(0, 180))[:, :, None, None], 3, axis=2)
        out = tf.nn.depthwise_conv2d(img[None], k, [1, 1, 1, 1], "SAME")[0].numpy()
    if sigma is None:
        sigma = rng.uniform(0, 0.08)
    out = out + rng.normal(0, sigma, out.shape).astype(np.float32)
    return np.clip(out, 0, 1).astype(np.float32)


# ----------------------------------------------------------------- the two CNNs we train ourselves
def build_denoiser(filters=48):
    """DnCNN-style network: predicts the damage, then subtracts it from the input."""
    inp = layers.Input((None, None, 3))
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(inp)
    for d in (1, 2, 4, 2, 1):                           # dilated convs see a wider area -> undo blur streaks
        x = layers.Conv2D(filters, 3, padding="same", dilation_rate=d, activation="relu")(x)
    damage = layers.Conv2D(3, 3, padding="same")(x)
    return tf.keras.Model(inp, layers.Subtract()([inp, damage]), name="denoiser")


def build_colorizer():
    """U-Net: from lightness L (64x64) predict the colour channels a,b."""
    def block(x, f):
        x = layers.Conv2D(f, 3, padding="same", activation="relu")(x)
        return layers.Conv2D(f, 3, padding="same", activation="relu")(x)

    i = layers.Input((64, 64, 1))
    e1 = block(i, 32)
    e2 = block(layers.MaxPooling2D()(e1), 64)
    e3 = block(layers.MaxPooling2D()(e2), 128)
    b = block(layers.MaxPooling2D()(e3), 256)
    d3 = block(layers.Concatenate()([layers.UpSampling2D()(b), e3]), 128)
    d2 = block(layers.Concatenate()([layers.UpSampling2D()(d3), e2]), 64)
    d1 = block(layers.Concatenate()([layers.UpSampling2D()(d2), e1]), 32)
    return tf.keras.Model(i, layers.Conv2D(2, 1, activation="tanh")(d1), name="colorizer")


def _load_trained(path, name):
    if not os.path.exists(path):
        raise ModelNotTrained(f"The {name} model is not trained yet. Stop the app and run: python train_models.py")
    return _model(name, lambda: tf.keras.models.load_model(path, compile=False))


# ----------------------------------------------------------------- task 1: super-resolution
def super_resolve(img):
    import tensorflow_hub as hub
    model = _model("super-resolution (ESRGAN)", lambda: hub.load(ESRGAN_URL))
    low = resize_max(img, SR_INPUT_SIDE)
    h, w = low.shape[:2]
    low = low[: max(4, h - h % 4), : max(4, w - w % 4)]
    out = model(tf.constant(low[None] * 255.0))[0].numpy() / 255.0
    out = np.clip(out, 0, 1)
    return {"before": low, "after": out,
            "metrics": {"Input (px)": f"{low.shape[1]}×{low.shape[0]}", "Output (px)": f"{out.shape[1]}×{out.shape[0]}", "Upscale": "4×"},
            "info": f"Low-resolution input {low.shape[1]}×{low.shape[0]} px → "
                    f"sharp output {out.shape[1]}×{out.shape[0]} px (4× bigger)"}


# ----------------------------------------------------------------- task 2: noise + motion blur removal
def denoise(img, simulate=True):
    model = _load_trained(DENOISER_PATH, "noise & blur removal")
    clean = resize_max(img, MAX_SIDE)
    before = shake_and_noise(clean, np.random.default_rng(7), length=5, sigma=0.05) if simulate else clean
    after = np.clip(model(before[None], training=False)[0].numpy(), 0, 1)
    metrics = {"Size (px)": f"{clean.shape[1]}×{clean.shape[0]}"}
    if simulate:
        metrics["PSNR before"] = f"{psnr(before, clean):.1f} dB"
        metrics["PSNR after"] = f"{psnr(after, clean):.1f} dB"
        info = (f"Shaky-hands blur + noise were added to your photo, then removed by the CNN. "
                f"Quality (PSNR, higher = better): {psnr(before, clean):.1f} dB → {psnr(after, clean):.1f} dB")
    else:
        info = "Your photo was cleaned by the CNN."
    return {"before": before, "after": after, "info": info, "metrics": metrics}


# ----------------------------------------------------------------- task 3: neural style transfer
def stylize(content, style=None):
    import tensorflow_hub as hub
    model = _model("style transfer", lambda: hub.load(STYLE_MODEL_URL))
    used_default = style is None
    if used_default:
        path = tf.keras.utils.get_file("kandinsky.jpg", DEFAULT_STYLE_URL, cache_dir=MODEL_DIR, cache_subdir="styles")
        with open(path, "rb") as f:
            style = read_image(f.read())
    content = resize_max(content, MAX_SIDE)
    style_in = tf.image.resize(style, (256, 256), antialias=True)     # the model was trained on 256 px styles
    out = np.clip(model(tf.constant(content[None]), tf.constant(style_in[None]))[0][0].numpy(), 0, 1)
    info = ("Textures and colours of the style picture were painted onto your photo."
            + (" (No style picture chosen, so a Kandinsky painting was used.)" if used_default else ""))
    return {"before": content, "after": out, "info": info, "style": resize_max(style, 256),
            "metrics": {"Size (px)": f"{out.shape[1]}×{out.shape[0]}", "Style": "Kandinsky" if used_default else "Yours"}}


# ----------------------------------------------------------------- task 4: colourisation
def colorize(img):
    model = _load_trained(COLORIZER_PATH, "colourisation")
    img = resize_max(img, MAX_SIDE)
    gray = color.rgb2gray(img)                                               # the black & white photo
    L_full = color.rgb2lab(np.stack([gray] * 3, -1))[..., :1] / 100.0        # lightness channel
    L_small = tf.image.resize(L_full[None], (64, 64), antialias=True)
    ab_small = model(L_small, training=False).numpy()                        # predicted colour
    ab_full = tf.image.resize(ab_small, gray.shape, method="bicubic")[0].numpy()
    ab_full = np.clip(ab_full * 1.3, -1, 1)                                  # slightly stronger colours
    out = np.clip(color.lab2rgb(np.concatenate([L_full * 100, ab_full * 128], -1)), 0, 1)
    return {"before": np.stack([gray] * 3, -1).astype(np.float32), "after": out,
            "metrics": {"Size (px)": f"{out.shape[1]}×{out.shape[0]}", "Colour space": "Lab"},
            "info": "Colours were predicted by the CNN from the black & white image. "
                    "It was trained on flower photos, so nature pictures work best."}


def run(task, img, style=None, simulate=True):
    if task == "superres":
        return super_resolve(img)
    if task == "denoise":
        return denoise(img, simulate)
    if task == "style":
        return stylize(img, style)
    if task == "colorize":
        return colorize(img)
    raise ValueError(f"Unknown task: {task}")


def sample_images():
    """Built-in sample photos (no internet needed)."""
    from skimage import data
    return {"Cat": data.chelsea(), "Coffee cup": data.coffee(), "Astronaut": data.astronaut(), "Rocket launch": data.rocket()}
