from datasets import load_dataset
from dataclasses import dataclass
from torchvision import transforms
import matplotlib.pyplot as plt
import os

@dataclass
class TrainingConfig:
    image_size = 96  # the generated image resolution
    train_batch_size = 8
    eval_batch_size = 64  # how many images to sample during evaluation
    num_epochs = 200
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 20
    save_model_epochs = 30
    mixed_precision = 'fp16'  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = 'ddpm-butterflies-128'  # the model namy locally and on the HF Hub

    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_private_repo = False
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0

def transform(examples):
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"images": images}

import torch


from diffusers import UNet2DModel


from diffusers import DDPMScheduler


import torch
from PIL import Image

#noise = torch.randn(sample_image.shape)
#timesteps = torch.LongTensor([50])
#noisy_image = noise_scheduler.add_noise(sample_image, noise, timesteps)

#Image.fromarray(((noisy_image.permute(0, 2, 3, 1) + 1.0) * 127.5).type(torch.uint8).numpy()[0])

import torch.nn.functional as F

#noise_pred = model(noisy_image, timesteps).sample
#loss = F.mse_loss(noise_pred, noise)


from diffusers import DDPMPipeline

import math

def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid

def evaluate(config, epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        batch_size = config.eval_batch_size,
        generator=torch.manual_seed(config.seed),
    ).images

    # Make a grid out of the images
    image_grid = make_grid(images, rows=4, cols=4)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")


from accelerate import Accelerator
from huggingface_hub import HfFolder, Repository, whoami

from tqdm.auto import tqdm
from pathlib import Path
import os


def get_full_repo_name(model_id: str, organization: str = None, token: str = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir="."
    )
    #logging_dir = os.path.join(config.output_dir, "logs")
    if accelerator.is_main_process:
        if config.push_to_hub:
            repo_name = get_full_repo_name(Path(config.output_dir).name)
            repo = Repository(config.output_dir, clone_from=repo_name)
        elif config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("train_example")

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch['images']
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bs,), device=clean_images.device).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, pipeline)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                if config.push_to_hub:
                    repo.push_to_hub(commit_message=f"Epoch {epoch}", blocking=True)
                else:
                    pipeline.save_pretrained(config.output_dir)


def train():
    config = TrainingConfig()
    # Or just load images from a local folder!
    config.dataset_name = "faces"
    dataset = load_dataset(config.dataset_name, data_dir="./", cache_dir="./caches", trust_remote_code=True)
    print(f"dataset:{dataset}")

    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    for i, image in enumerate(dataset['train'][:4]["image"]):
        axs[i].imshow(image)
        axs[i].set_axis_off()
    fig.show()

    preprocess = transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    dataset.set_transform(transform)

    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    for i, image in enumerate(dataset['train'][:4]["images"]):
        axs[i].imshow(image.permute(1, 2, 0).numpy() / 2 + 0.5)
        axs[i].set_axis_off()
    fig.show()
    train_dataloader = torch.utils.data.DataLoader(dataset['train'], batch_size=config.train_batch_size, shuffle=True)
    model = UNet2DModel.from_pretrained("/workspace/pixel/ddpm-butterflies-128/unet/", use_safetensors=True)
    #model = UNet2DModel(
    #    sample_size=config.image_size,  # the target image resolution
    #    in_channels=3,  # the number of input channels, 3 for RGB images
    #    out_channels=3,  # the number of output channels
    #    layers_per_block=2,  # how many ResNet layers to use per UNet block
    #    block_out_channels=(96, 96, 192, 192, 384, 384),  # the number of output channes for each UNet block
    #    down_block_types=(
    #        "DownBlock2D",  # a regular ResNet downsampling block
    #        "DownBlock2D",
    #        "DownBlock2D",
    #        "DownBlock2D",
    #        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
    #        "DownBlock2D",
    #    ),
    #    up_block_types=(
    #        "UpBlock2D",  # a regular ResNet upsampling block
    #        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
    #        "UpBlock2D",
    #        "UpBlock2D",
    #        "UpBlock2D",
    #        "UpBlock2D"
    #    ),
    #)

    sample_image = dataset['train'][0]['images'].unsqueeze(0)
    print('Input shape:', sample_image.shape)

    print('Output shape:', model(sample_image, timestep=0).sample.shape)
    noise_scheduler = DDPMScheduler.from_pretrained("/workspace/pixel/ddpm-butterflies-128", subfolder="scheduler")
    #noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    from diffusers.optimization import get_cosine_schedule_with_warmup

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )
    #from accelerate import notebook_launcher
    #args = (config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)
    train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)

    #notebook_launcher(train_loop, args, num_processes=8)

def eval():
    config = TrainingConfig()
    pipeline = DDPMPipeline.from_pretrained("./ddpm-butterflies-128/", use_safetensors=True)

    images = pipeline(
        batch_size = 4,
        generator=torch.manual_seed(19900710),
    ).images

    # Make a grid out of the images
    image_grid = make_grid(images, rows=4, cols=4)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/eval.png")

if __name__=="__main__":
    #train()
    eval()