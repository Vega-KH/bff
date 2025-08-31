import os
from abc import abstractmethod

from PIL import Image

import modules.shared
from modules import modelloader, shared

LANCZOS = (Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
NEAREST = (Image.Resampling.NEAREST if hasattr(Image, 'Resampling') else Image.NEAREST)


class Upscaler:
    name: str | None = None
    model_path: str | None = None
    model_name: str | None = None
    model_url: str | None = None
    enable = True
    filter = None
    model = None
    user_path: str | None = None
    scalers: list
    tile = True

    def __init__(self, create_dirs=False):
        self.model_download_path: str | None = None

        if self.model_path is None and self.name:
            self.model_path = os.path.join(shared.models_path, self.name)
        if self.model_path and create_dirs:
            os.makedirs(self.model_path, exist_ok=True)

    @abstractmethod
    def do_upscale(self, img: Image.Image, selected_model: str | None):
        return img

    def upscale(self, img: Image.Image, scale, selected_model: str | None = None):
        self.scale = scale
        dest_w = int((img.width * scale) // 8 * 8)
        dest_h = int((img.height * scale) // 8 * 8)

        for i in range(3):
            if img.width >= dest_w and img.height >= dest_h and (i > 0 or scale != 1):
                break

            if shared.state.interrupted:
                break

            shape = (img.width, img.height)

            img = self.do_upscale(img, selected_model)

            if shape == (img.width, img.height):
                break

        if img.width != dest_w or img.height != dest_h:
            img = img.resize((int(dest_w), int(dest_h)), resample=LANCZOS)

        return img

    @abstractmethod
    def load_model(self, path: str):
        pass

    def find_models(self, ext_filter=None) -> list:
        return modelloader.load_models(model_path=self.model_path, model_url=self.model_url, ext_filter=ext_filter)


class UpscalerData:
    name: str | None = None
    data_path: str | None = None
    scale: int = 4
    scaler: Upscaler | None = None
    model = None

    def __init__(self, name: str, path: str | None, upscaler: Upscaler | None = None, scale: int = 4, model=None):
        self.name = name
        self.data_path = path
        self.local_data_path = path
        self.scaler = upscaler
        self.scale = scale
        self.model = model

    def __repr__(self):
        return f"<UpscalerData name={self.name} path={self.data_path} scale={self.scale}>"


class UpscalerNone(Upscaler):
    name = "None"
    scalers = []

    def load_model(self, path):
        pass

    def do_upscale(self, img, selected_model=None):
        return img

    def __init__(self, dirname=None):
        super().__init__(False)
        self.scalers = [UpscalerData("None", None, self)]