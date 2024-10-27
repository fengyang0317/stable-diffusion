import argparse
import logging
import multiprocessing
import os
import pathlib

import numpy as np
import torch
import tqdm
import webdataset as wds
from PIL import Image
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from torchvision.transforms import transforms


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


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--im_size", default=256, type=int, help="Image size.")
    parser.add_argument("--num_workers", default=8, type=int, help="Number of workers.")
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
    parser.add_argument("--output_dir", default="/tmp/imagenet_1k", type=str, help="Output directory.")
    parser.add_argument('--vae_path', default="models/first_stage_models/kl-f16/model.ckpt", type=str,
                        help='images input size')
    parser.add_argument("--use_ema", default=False, action="store_true", help="Use ema.")
    parser.add_argument('--hf_token', default=None, type=str, help='Hugging Face token')
    args = parser.parse_args()

    # augmentation following DiT and ADM
    transform_train = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.im_size)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    url = 'https://huggingface.co/datasets/timm/imagenet-1k-wds/resolve/main/imagenet1k-validation-{00..63}.tar'
    url = f"pipe:curl -s -L {url} -H 'Authorization:Bearer {args.hf_token}'"
    dataset_train = wds.WebDataset(url)
    dataset_train = dataset_train.decode('pil').to_tuple('jpg', 'json')
    dataset_train = dataset_train.map_tuple(transform_train, lambda x: x)

    config = OmegaConf.load('models/first_stage_models/kl-f16/config.yaml')
    if args.use_ema:
        config['model']['params']['use_ema'] = True
    vae = load_model_from_config(config, args.vae_path)
    if args.use_ema:
        vae.model_ema.copy_to(vae)

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
        queue.put((data[0].numpy(), recon.cpu().numpy(), data[1]['filename']))

    for _ in range(args.num_workers):
        queue.put(None)
    for p in processes:
        p.join()
    logging.info('Done')


if __name__ == '__main__':
    main()
