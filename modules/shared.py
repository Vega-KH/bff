import os

class State:
    interrupted = False

class Options:
    def __init__(self):
        self.ESRGAN_tile = 192
        self.ESRGAN_tile_overlap = 8
        self.enable_upscale_progressbar = True

opts = Options()
state = State()
face_restorers = []
sd_upscalers = []
models_path = os.path.abspath('models')
