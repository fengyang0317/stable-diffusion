import argparse
import functools
import logging
import multiprocessing
import os
import pathlib

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import webdataset as wds
from PIL import Image
from omegaconf import OmegaConf
from torchvision.transforms import transforms

from ldm.util import instantiate_from_config

_SPLITS = {
    'train': 'imagenet1k-train-{0000..1023}.tar',
    'val': 'imagenet1k-validation-{00..63}.tar',
}


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def write_fn(args, queue):
    total = 0
    os.makedirs(f'{args.output_dir}/val', exist_ok=True)
    os.makedirs(f'{args.output_dir}/recon', exist_ok=True)
    while True:
        task = queue.get()
        if task is None:
            break

        for image, recon, name in zip(*task):
            path = pathlib.Path(name)

            image = np.clip((image + 1) / 2, 0, 1)
            image = Image.fromarray((np.transpose(image, [1, 2, 0]) * 255).astype(np.uint8))
            image.save(f"{args.output_dir}/val/{path.stem}.png")

            recon = np.clip((recon + 1) / 2, 0, 1)
            recon = Image.fromarray((np.transpose(recon, [1, 2, 0]) * 255).astype(np.uint8))
            recon.save(f"{args.output_dir}/recon/{path.stem}.png")

            total += 1

        logging.info('Write %d samples', total)


def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    print("missing keys:")
    print(m)
    print("unexpected keys:")
    print(u)

    model.cuda()
    model.eval()
    return model


def cv2_gaussian_blur(pil_image, blur_sigma):
    if not blur_sigma:
        return pil_image
    img_np = np.array(pil_image)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    blurred_img = cv2.GaussianBlur(img_np, (0, 0), blur_sigma)
    blurred_img = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2RGB)
    blurred_pil = Image.fromarray(blurred_img)
    return blurred_pil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--im_size", default=256, type=int, help="Image size.")
    parser.add_argument("--eval_size", default=None, type=int, help="Image eval size.")
    parser.add_argument("--num_workers", default=8, type=int, help="Number of workers.")
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
    parser.add_argument("--output_dir", default="/tmp/imagenet_1k", type=str, help="Output directory.")
    parser.add_argument('--vae_path', default="models/first_stage_models/kl-f16/model.ckpt", type=str,
                        help='images input size')
    parser.add_argument('--hf_token', default=None, type=str, help='Hugging Face token')
    parser.add_argument('--config', default='models/first_stage_models/kl-f16/config.yaml', type=str,
                        help='Config file')
    parser.add_argument('--split', default='val', type=str, help='Split to use')
    parser.add_argument('--blur_sigma', default=None, type=float, help='Blur sigma')
    args, unknown = parser.parse_known_args()
    if args.eval_size is None:
        args.eval_size = args.im_size

    # augmentation following DiT and ADM
    transform_train = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.im_size)),
        transforms.Lambda(functools.partial(cv2_gaussian_blur, blur_sigma=args.blur_sigma)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    split = _SPLITS[args.split]
    url = f'https://huggingface.co/datasets/timm/imagenet-1k-wds/resolve/main/{split}'
    url = f"pipe:curl -s -L {url} -H 'Authorization:Bearer {args.hf_token}'"
    dataset_train = wds.WebDataset(url)
    dataset_train = dataset_train.decode('pil').to_tuple('jpg', 'json')
    dataset_train = dataset_train.map_tuple(transform_train, lambda x: x)

    config = OmegaConf.load(args.config)
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(config, cli)
    vae = load_model_from_config(config, args.vae_path)
    if 'use_ema' in config['model']['params'] and config['model']['params']['use_ema']:
        vae.model_ema.copy_to(vae)
    state_dict = vae.state_dict()
    keys = list(state_dict.keys())
    for key in keys:
        if key.startswith("loss") or key.startswith("model_ema"):
            del state_dict[key]
    torch.save({"model": state_dict}, "converted.ckpt")

    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    queue = multiprocessing.Queue()
    processes = []
    for _ in range(args.num_workers):
        p = multiprocessing.Process(target=write_fn, args=(args, queue))
        p.start()
        processes.append(p)

    for data in tqdm.tqdm(train_loader, total=50000 // args.batch_size):
        with torch.no_grad():
            image = data[0].cuda()
            posterior = vae.encode(image)
            recon = vae.decode(posterior.sample())
            if args.eval_size != recon.shape[-1]:
                recon = F.interpolate(recon, size=(args.eval_size, args.eval_size), mode='area')
                image = F.interpolate(image, size=(args.eval_size, args.eval_size), mode='area')
        queue.put((image.cpu().numpy(), recon.cpu().numpy(), data[1]['filename']))

    for _ in range(args.num_workers):
        queue.put(None)
    for p in processes:
        p.join()
    logging.info('Done')


if __name__ == '__main__':
    main()
