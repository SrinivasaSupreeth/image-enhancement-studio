"""One-time setup: train the two small CNNs (tasks 2 and 4) and download the
two pretrained models (tasks 1 and 3). Takes about 5-10 minutes on a laptop.

    python train_models.py            # normal
    python train_models.py --quick    # ~1 minute, lower quality (for a quick test)
"""
import argparse
import glob
import os
import time

import numpy as np
import tensorflow as tf

import enhance as E

FLOWERS_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"


def load_photos(folder, count, seed=0):
    if folder:
        files = [f for ext in ("jpg", "jpeg", "png") for f in glob.glob(os.path.join(folder, "**", f"*.{ext}"), recursive=True)]
    else:
        print("Downloading training photos (flowers, ~220 MB, one time only) ...")
        tf.keras.utils.get_file("flower_photos.tgz", FLOWERS_URL, extract=True, cache_dir=E.MODEL_DIR, cache_subdir="data")
        files = glob.glob(os.path.join(E.MODEL_DIR, "data", "**", "*.jpg"), recursive=True)
    if not files:
        raise SystemExit("No training photos found.")
    np.random.default_rng(seed).shuffle(files)
    files = files[:count]

    def load(path):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        return tf.image.resize(tf.image.convert_image_dtype(img, tf.float32), (64, 64), antialias=True)

    ds = tf.data.Dataset.from_tensor_slices(files).map(load, num_parallel_calls=tf.data.AUTOTUNE)
    photos = np.stack(list(ds.as_numpy_iterator())).astype("float32")
    print(f"Loaded {len(photos)} training photos (64x64)")
    return photos


def train_denoiser(photos, epochs):
    print(f"\n[1/2] Training noise & blur removal CNN ({epochs} epochs) ...")
    rng = np.random.default_rng(0)
    damaged = np.stack([E.shake_and_noise(p, rng) for p in photos])      # make (damaged, clean) pairs
    model = E.build_denoiser()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mae")
    model.fit(damaged, photos, batch_size=32, epochs=epochs, verbose=2)
    model.save(E.DENOISER_PATH)
    print(f"Saved {E.DENOISER_PATH}")


def train_colorizer(photos, epochs):
    print(f"\n[2/2] Training colourisation U-Net ({epochs} epochs) ...")
    from skimage import color
    lab = color.rgb2lab(photos).astype("float32")
    L, ab = lab[..., :1] / 100.0, lab[..., 1:] / 128.0                    # input: lightness, target: colour
    model = E.build_colorizer()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="huber")
    model.fit(L, ab, batch_size=32, epochs=epochs, verbose=2)
    model.save(E.COLORIZER_PATH)
    print(f"Saved {E.COLORIZER_PATH}")


def download_pretrained():
    print("\nDownloading the pretrained models (super-resolution + style transfer) ...")
    try:
        import tensorflow_hub as hub
        hub.load(E.ESRGAN_URL)
        hub.load(E.STYLE_MODEL_URL)
        tf.keras.utils.get_file("kandinsky.jpg", E.DEFAULT_STYLE_URL, cache_dir=E.MODEL_DIR, cache_subdir="styles")
        print("Done.")
    except Exception as e:
        print(f"Could not download them now ({e}). They will be downloaded the first time you use those tasks.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fast test run (lower quality)")
    ap.add_argument("--data", help="optional folder with your own colour photos for training")
    ap.add_argument("--if-missing", action="store_true", help="skip training if the model files already exist")
    args = ap.parse_args()

    if args.if_missing and all(E.status().values()):
        print("Trained models found in models/ - skipping training.")
        download_pretrained()
        return

    t = time.time()
    count, ep_denoise, ep_color = (400, 2, 3) if args.quick else (2500, 6, 12)
    photos = load_photos(args.data, count)
    train_denoiser(photos[:2000], ep_denoise)
    train_colorizer(photos, ep_color)
    download_pretrained()
    print(f"\nAll set in {(time.time() - t) / 60:.1f} minutes. Now start the web page with:  streamlit run streamlit_app.py")


if __name__ == "__main__":
    main()
