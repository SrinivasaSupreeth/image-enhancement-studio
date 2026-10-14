IMAGE ENHANCEMENT STUDIO
Task 2: Web page for "Improving Image Quality - Picture Transformation & Enhancement using CNN"

Built with: Python, TensorFlow, Streamlit, Docker


WHAT IT IS

A web page where you choose one of 4 effects, upload a photo (or use a
sample photo), and compare the Original with the After image:

  1. Super-resolution            small, blurry photo -> sharp, 4x bigger photo
  2. Remove noise and blur        grainy, shaky-hands photo -> clear photo
  3. Style transfer               photo repainted in the style of a painting
  4. Colourise black and white    old B&W photo -> colour photo

The page has 3 tabs:
  Studio            upload a photo, apply an effect, drag the slider to compare,
                    see the two images side by side, download the result
  Notebook results  the 4 results from our Google Colab notebook
  How it works      a simple explanation of the CNN behind each effect


FILES

streamlit_app.py       the web page (Streamlit)
enhance.py             the 4 CNN image functions (same logic as our notebook)
train_models.py        trains the CNNs for effects 2 and 4, downloads the
                       pretrained CNNs for effects 1 and 3
requirements.txt       Python libraries needed
Dockerfile             recipe to build the app as a Docker image
docker-compose.yml     starts the app with one command
.streamlit/config.toml colours and settings of the page
static/results/        the 4 result images from our Colab notebook


==========================================================================
OPTION A: RUN WITH DOCKER  (recommended, works on any computer with Docker)
==========================================================================

You only need Docker Desktop installed and running (whale icon in the menu bar).
No Python setup is needed.

1. Unzip the project and open Terminal in that folder:
       cd ~/Downloads/Image_Enhancement_Studio

2. Build and start the app (first time takes about 15-25 minutes, because it
   installs TensorFlow and trains the models; it needs internet):
       docker compose up --build

3. When you see "You can now view your Streamlit app", open in your browser:
       http://localhost:8501

4. To stop it: press Ctrl+C in Terminal (or run: docker compose down).

Next time, start it in a few seconds with:
       docker compose up

Run it in the background (Terminal can be closed):
       docker compose up -d
Stop it:
       docker compose down

On ANOTHER computer: copy this folder (or the zip) to that computer, install
Docker Desktop there, and run the same command: docker compose up --build

Other people on the SAME Wi-Fi can open it while it runs on your computer:
find your computer's IP address (Mac: System Settings > Wi-Fi > Details),
then they open  http://YOUR-IP-ADDRESS:8501


==========================================================================
OPTION B: RUN WITHOUT DOCKER (with Python)
==========================================================================

Needs Python 3.10, 3.11 or 3.12.

   python3 -m venv venv
   source venv/bin/activate              (Windows: venv\Scripts\activate)
   pip install -r requirements.txt
   python train_models.py                (one time, about 5-10 minutes)
   streamlit run streamlit_app.py

The page opens at http://localhost:8501


HOW TO USE THE PAGE

1. In the left panel, choose an effect.
2. In the Studio tab, upload a photo or choose "Use a sample".
3. Style transfer only: optionally upload a painting in the left panel
   (otherwise a Kandinsky painting is used).
   Noise removal only: keep "Add camera shake and noise first" switched on
   for a clear demo, or switch it off if your photo is already noisy.
4. Press the button (Upscale photo / Clean photo / Apply style / Colourise photo).
5. Drag across the picture to compare Original and After, look at the
   side-by-side view, and press "Download result (PNG)" to save it.

The first time each effect runs it takes longer (model loading). After that
it is quick.


HOW EACH EFFECT WORKS (simple version)

1. Super-resolution: the photo is made small (low resolution). ESRGAN, a
   pretrained CNN, makes it 4 times bigger and fills in the missing detail.

2. Noise and blur removal: we damaged clean photos on purpose (camera-shake
   blur and noise) and trained a CNN to predict the damage. The predicted
   damage is subtracted from the photo. PSNR shows the quality (higher = better).

3. Style transfer: a pretrained CNN combines the content (shapes, objects) of
   your photo with the style (textures, colours) of the painting.

4. Colourisation: the photo is split into lightness (black and white part) and
   colour. A U-Net CNN trained on flower photos predicts the colour from the
   lightness. Nature photos give the best colours.


COMMON PROBLEMS

"Cannot connect to the Docker daemon"   open Docker Desktop first and wait
                                        until it says it is running.
"port is already allocated"             something else uses port 8501; stop it,
                                        or change "8501:8501" to "8502:8501" in
                                        docker-compose.yml and open localhost:8502
Effect 1 or 3 shows an error           check the internet connection; these
                                        download their model the first time.
Build stops while training             in Docker Desktop > Settings > Resources,
                                        give Docker at least 4 GB of memory.


==========================================================================
OPTION C: PUT IT ONLINE (a link anyone can open, your laptop can be off)
==========================================================================

We use Streamlit Community Cloud (free). It runs streamlit_app.py on
Streamlit's servers and gives a link like https://your-app.streamlit.app

Step 1. Copy the trained models out of Docker (one time)
   The two CNNs we trained live inside the Docker image. Copy them into a
   "models" folder in this project folder:
       cd ~/Downloads/Image_Enhancement_Studio_Cloud
       mkdir -p models
       docker cp image-enhancement-studio:/app/models/denoiser.keras models/
       docker cp image-enhancement-studio:/app/models/colorizer.keras models/
       ls models
   You should see: colorizer.keras  denoiser.keras

Step 2. Put the code on GitHub
   1. Create a free account at github.com
   2. Click "+" (top right) > New repository
      Name: image-enhancement-studio, Public, click "Create repository"
   3. Click "uploading an existing file"
   4. In Finder press Cmd + Shift + . (dot) to show hidden files, then drag
      EVERYTHING from this folder into the browser, including the folders
      .streamlit, models and static.
   5. Click "Commit changes"

Step 3. Deploy on Streamlit
   1. Go to share.streamlit.io and "Continue with GitHub"
   2. Click "Create app" > "Deploy a public app from GitHub"
   3. Repository: your-username/image-enhancement-studio
      Branch: main
      Main file path: streamlit_app.py
      App URL: choose a name, e.g. image-enhancement-studio
   4. Advanced settings > Python version: 3.11 > Save
   5. Click "Deploy". The first start takes about 5-10 minutes.

Step 4. Share the link, e.g. https://image-enhancement-studio.streamlit.app
   It works on any laptop or phone. Your laptop and Docker can be off.

Notes
- If nobody opens the app for 12 hours it goes to sleep. The next visitor
  sees "Yes, get this app back up!" - one click and it wakes in about a minute.
- On the cloud only one model stays in memory at a time (the free server has
  2.7 GB of memory), so switching effects takes a few extra seconds.
