import os
import time
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import clip

from src.preprocess import CLIPPreprocessor

def load_clip_model(model_name, device):
    """
    Loads CLIP model + tokenizer
    """
    print(f"Loading CLIP model: {model_name}")
    model, preprocess = clip.load(model_name, device=device)
    return model,preprocess

def compute_image_embeddings(model,df,preprocessor,device,batch_size):
    """
    Computes CLIP embeddings for a list of image paths
    """
    all_embeddings = []

    image_paths = df['file_path'].tolist()

    for i in tqdm(range(0,len(image_paths),batch_size),desc="Image Embeddings"):
        batch_paths = image_paths[i:i+batch_size]

        images = torch.stack([
            preprocessor.preprocess_image(p) for p in batch_paths if os.path.exists(p)
        ]).to(device)

        with torch.no_grad():
            embeddings = model.encode_image(images)

        all_embeddings.append(embeddings.cpu().numpy())
    all_embeddings = np.concatenate(all_embeddings,axis=0)
    return all_embeddings

def compute_caption_embeddings(model,df,preprocessor,device,batch_size):
    """
    Computes CLIP embeddings for captions
    If multiple captions exist per image -> averages them.
    """
    all_embeddings = []

    captions = df['caption'].tolist()

    for i in tqdm(range(0,len(captions),batch_size),desc="Caption Embeddings"):
        batch_caps = captions[i:i+batch_size]
        normalized_caps = [preprocessor.preprocess_text(c) for c in batch_caps]

        tokens = clip.tokenize(normalized_caps).to(device)

        with torch.no_grad():
            embeddings = model.encode_text(tokens)

        all_embeddings.append(embeddings.cpu().numpy())
    all_embeddings = np.concatenate(all_embeddings,axis=0)
    return all_embeddings

def l2_normalize(x):
    return x / np.linalg.norm(x,axis=1,keepdims=True)

def main(args):
    start_time = time.time()

    # Load metadata CSV : file_path, caption
    df = pd.read_csv(args.metadata)
    print(f"Loaded {len(df)} records from metadata.")

    # Load config
    import yaml
    config = yaml.safe_load(open(args.config))

    device = config['device']
    batch_size = config['batch_size']

    # Preprocessor
    preproc = CLIPPreprocessor(image_size=config['image_size'])

    # Load CLIP model
    model, _ = clip.load(config['model_name'], device=device)

    # Compute embeddings
    print("\n Computing image embeddings...")
    image_embeds = compute_image_embeddings(model,df,preproc,device,batch_size)

    print("\n Computing caption embeddings...")
    caption_embeds = compute_caption_embeddings(model,df,preproc,device,batch_size)

    # Optional L2 normalization
    if config['normalize_embeddings']:
        print("Normalizing embeddings...")
        image_embeds = l2_normalize(image_embeds)
        caption_embeds = l2_normalize(caption_embeds)

    # Save embeddings
    os.makedirs(args.output_dir, exist_ok=True)

    img_path  = os.path.join(args.output_dir,'image_embeds.npy')
    cap_path = os.path.join(args.output_dir,'caption_embeds.npy')

    np.save(img_path,image_embeds.astype(config['dtype']))
    np.save(cap_path,caption_embeds.astype(config['dtype']))

    # Write statistics
    end_time = time.time()
    runtime = end_time - start_time

    stats_path = "reports/embeddings_stats.md"
    os.makedirs("reports",exist_ok=True)

    with open(stats_path,'w') as f:
        f.write("Embedding generation stats \n")
        f.write(f"Model: {config['model_name']} \n")
        f.write(f"Device : {device} \n")
        f.write(f"Batch size : {batch_size} \n")
        f.write(f"Run time: {runtime} \n")
        f.write(f"Total images : {len(df)}\n")
    print(f"\n Embeddings saved")
    print(f"Images -> {img_path}")
    print(f"Captions -> {cap_path}")
    print(f"Stats written to reports/embeddings_stats.md")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--metadata', type=str, required=True, help='CSV with columns: file_path, caption')
    parser.add_argument('--config', type=str, default='configs/clip.yaml')
    parser.add_argument('--output_dir', type=str, default='data/embeddings')

    args = parser.parse_args()
    main(args)