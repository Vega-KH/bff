from modules import modelloader, devices, errors
from modules.shared import opts, sd_upscalers
from modules.upscaler import Upscaler, UpscalerData
from modules.upscaler_utils import upscale_with_model

esrgan_upscaler: Upscaler | None = None

class UpscalerESRGAN(Upscaler):
    def __init__(self, model_path="models/ESRGAN"):
        self.name = "ESRGAN"
        self.user_path = model_path
        self.user_path = model_path
        self.scalers = []
        super().__init__()

        default_models = [
            {
                "url": "https://huggingface.co/VegaKH/Photo_Bigify_ESRGAN/resolve/main/4x_Vega_Photo_Bigify.pth?download=true",
                "name": "4x_Vega_Photo_Bigify"
            },
            {
                "url": "https://huggingface.co/VegaKH/Photo_Bigify_ESRGAN/resolve/main/2X_Vega_Photo_Bigify.pth?download=true",
                "name": "2X_Vega_Photo_Bigify"
            }
        ]

        model_paths = self.find_models(ext_filter=[".pt", ".pth"])
        local_model_names = [modelloader.friendly_name(p) for p in model_paths]

        # Add default models if they are not found locally
        for model in default_models:
            if model["name"] not in local_model_names:
                model_paths.append(model["url"])

        for file in model_paths:
            if file.startswith("http"):
                # Find the name from the default models list
                matching_model = next((m for m in default_models if m["url"] == file), None)
                if matching_model:
                    name = matching_model["name"]
                else:
                    # Fallback for other URLs
                    name = modelloader.friendly_name(file)
            else:
                # For local files
                name = modelloader.friendly_name(file)
            
            scale = 4
            if "2x" in name.lower():
                scale = 2

            scaler_data = UpscalerData(name=name, path=file, upscaler=self, scale=scale)
            self.scalers.append(scaler_data)
    def do_upscale(self, img, selected_model: str | None):
        try:
            model = self.load_model(selected_model)
            if model is None:
                return img
        except Exception:
            errors.report(f"Unable to load ESRGAN model {selected_model}", exc_info=True)
            return img
        model.to(devices.device_esrgan)
        return esrgan_upscale(model, img)

    def load_model(self, path: str | None):
        if path is None:
            return None
        if path.startswith("http"):
            filename = modelloader.load_file_from_url(
                url=path,
                model_dir=self.user_path or "models/ESRGAN",
                file_name=f"{modelloader.friendly_name(path)}.pth",
            )
        else:
            filename = path

        model_descriptor = modelloader.load_spandrel_model(
            filename,
            device=('cpu' if devices.device_esrgan.type == 'mps' else None),
            expected_architecture='ESRGAN',
        )
        return model_descriptor.model


def esrgan_upscale(model, img):
    return upscale_with_model(
        model,
        img,
        tile_size=opts.ESRGAN_tile,
        tile_overlap=opts.ESRGAN_tile_overlap,
    )

def setup_model(model_path: str):
    global esrgan_upscaler
    try:
        esrgan_upscaler = UpscalerESRGAN(model_path=model_path)
        for scaler in esrgan_upscaler.scalers:
            sd_upscalers.append(scaler)
    except Exception:
        errors.report("Error setting up ESRGAN", exc_info=True)