"""Custom handler for huggingface/diffusers models."""
import base64
import io
import logging
import os
from typing import Any, List, Sequence, Tuple

from google.cloud import storage

from diffusers import AutoencoderKL
from diffusers import DiffusionPipeline
from diffusers import StableDiffusionXLPipeline
from diffusers import StableDiffusionPipeline
import numpy as np
from PIL import Image
import torch
from ts.torch_handler.base_handler import BaseHandler

STABLE_DIFFUSION_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"

# GCS prefixes
GCS_URI_PREFIX = 'gs://'
LOCAL_MODEL_DIR = '/tmp/model_dir'


# Tasks
TEXT_TO_IMAGE = "text-to-image"
IMAGE_TO_IMAGE = "image-to-image"
IMAGE_INPAINTING = "image-inpainting"
CONTROLNET = "controlnet"
CONDITIONED_SUPER_RES = "conditioned-super-res"
TEXT_TO_IMAGE_SDXL = "text-to-image-sdxl"
TEXT_TO_IMAGE_REFINER = "text-to-image-refiner"


def image_to_base64(image: Image.Image) -> str:
    """Convert a PIL image to a base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    image_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return image_str


def base64_to_image(image_str: str) -> Image.Image:
    """Convert a base64 string to a PIL image."""
    image = Image.open(io.BytesIO(base64.b64decode(image_str)))
    return image


def is_gcs_path(input_path: str) -> bool:
    """Checks if the input path is a Google Cloud Storage (GCS) path.
    Args:
        input_path: The input path to be checked.
    Returns:
        True if the input path is a GCS path, False otherwise.
    """
    return input_path.startswith(GCS_URI_PREFIX)


def download_gcs_dir_to_local(
    gcs_dir: str, local_dir: str, skip_hf_model_bin: bool = False
):
    """Downloads files in a GCS directory to a local directory.
    For example:
        download_gcs_dir_to_local(gs://bucket/foo, /tmp/bar)
        gs://bucket/foo/a -> /tmp/bar/a
        gs://bucket/foo/b/c -> /tmp/bar/b/c
    Arguments:
        gcs_dir: A string of directory path on GCS.
        local_dir: A string of local directory path.
        skip_hf_model_bin: True to skip downloading HF model bin files.
    """
    if not is_gcs_path(gcs_dir):
        raise ValueError(f'{gcs_dir} is not a GCS path starting with gs://.')
    bucket_name = gcs_dir.split('/')[2]
    prefix = gcs_dir[len(constants.GCS_URI_PREFIX + bucket_name) :].strip('/')
    client = storage.Client()
    blobs = client.list_blobs(bucket_name, prefix=prefix)
    for blob in blobs:
        if blob.name[-1] == '/':
            continue
        file_path = blob.name[len(prefix) :].strip('/')
        local_file_path = os.path.join(local_dir, file_path)
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        if (
            file_path.endswith(constants.HF_MODEL_WEIGHTS_SUFFIX)
            and skip_hf_model_bin
        ):
            logging.info('Skip downloading model bin %s', file_path)
            with open(local_file_path, 'w') as f:
                f.write(f'{constants.GCS_URI_PREFIX}{bucket_name}/{prefix}/{file_path}')
        else:
            logging.info('Downloading %s to %s', file_path, local_file_path)
            blob.download_to_filename(local_file_path)


class DiffusersHandler(BaseHandler):
    """Custom handler for Stable Diffusion XL models."""
    
    def initialize(self, context: Any):
        """Custom initialize."""
        
        properties = context.system_properties
        self.map_location = (
            "cuda"
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )
        self.device = torch.device(
            self.map_location + ":" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else self.map_location
        )
        self.manifest = context.manifest
    
        self.model_id = os.environ["MODEL_ID"]
        if self.model_id.startswith(GCS_URI_PREFIX):
            gcs_path = self.model_id[len(GCS_URI_PREFIX) :]
            local_model_dir = os.path.join(LOCAL_MODEL_DIR, gcs_path)
            logging.info(f"Download {self.model_id} to {local_model_dir}")
            download_gcs_dir_to_local(self.model_id, local_model_dir)
            self.model_id = local_model_dir

        self.refiner_model_id = os.environ.get("REFINER_MODEL_ID", "")

        self.task = os.environ.get("TASK", TEXT_TO_IMAGE)
        logging.info(f"Using task:{self.task}, model:{self.model_id}")

        if self.task == TEXT_TO_IMAGE:
            vae = AutoencoderKL.from_pretrained(
                "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
            )
            pipeline = DiffusionPipeline.from_pretrained(
                self.model_id,
                vae=vae, 
                torch_dtype=torch.float16, 
                variant="fp16",
                use_safetensors=True
            )
            pipeline = pipeline.to(self.map_location)
            
            logging.info(f"Using refiner:{self.refiner_model_id}")
            refiner = DiffusionPipeline.from_pretrained(
                self.refiner_model_id,
                text_encoder_2=pipeline.text_encoder_2,
                vae=pipeline.vae,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
            )
            refiner = refiner.to(self.map_location)

            # Reduce memory footprint.
            pipeline.enable_xformers_memory_efficient_attention()
            pipeline.enable_attention_slicing()
            refiner.enable_xformers_memory_efficient_attention()
            refiner.enable_attention_slicing()
            # Optimization for pytorch > 2.0
            # pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead", fullgraph=True)
            # refiner.unet = torch.compile(refiner.unet, mode="reduce-overhead", fullgraph=True)
        else:
            raise ValueError(f"Invalid TASK: {self.task}")
        self.pipeline = pipeline
        self.refiner = refiner
        self.initialized = True
        logging.info("Handler initialization done.")
    
    def preprocess(self, data: Any) -> Tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any]:
        """Preprocess input data."""
        prompts = [item["prompt"] for item in data]
        
        negative_prompts = None
        images = None
        mask_images = None
        denoising_end = None
        num_inference_steps=20
        generator = None
        height = None
        width = None
        
        if "negative_prompt" in data[0]:
            negative_prompts = [item["negative_prompt"] for item in data]
        if "height" in data[0]:
            height = data[0]["height"]
        if "width" in data[0]:
            width = data[0]["width"]
        if "denoising_end" in data[0]:
            denoising_end = data[0]["denoising_end"]
        if "num_inference_steps" in data[0]:
            num_inference_steps = data[0]["num_inference_steps"]
        if "seed" in data[0]:
            generator = torch.manual_seed(data[0]["seed"])
        if "image" in data[0]:
            images = [
              base64_to_image(item["image"]) for item in data
            ]
        if "mask_image" in data[0]:
            mask_images = [
              base64_to_image(item["mask_image"])
              for item in data
            ]
        return prompts, negative_prompts, height, width, denoising_end, num_inference_steps, generator, images, mask_images
    
    def inference(self, data: Any, *args, **kwargs) -> List[Image.Image]:
        """Run the inference."""
        prompts, negative_prompts, height, width, denoising_end, num_inference_steps, generator, images, mask_images = data
        if self.task == TEXT_TO_IMAGE:
            predicted_images = self.pipeline(
                prompt=prompts,
                negative_prompt=negative_prompts,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                denoising_end=denoising_end,
                output_type="latent",
                generator=generator
            ).images
            refined_images = self.refiner(
                prompt=prompts,
                negative_prompt=negative_prompts,
                num_inference_steps=num_inference_steps,
                denoising_start=denoising_end,
                image=predicted_images,
                generator=generator
            ).images
            
        else:
            raise ValueError(f"Invalid TASK: {self.task}")
        return refined_images
    
    def postprocess(self, data: Any) -> List[str]:
        """Convert the images to base64 string."""
        outputs = []
        for prediction in data:
            if isinstance(prediction, bytes):
                # This is the video bytes.
                outputs.append(base64.b64encode(prediction).decode("utf-8"))
            else:
                outputs.append(image_to_base64(prediction))
        return outputs