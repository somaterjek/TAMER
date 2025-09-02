import pickle
import os
import io
from PIL import Image
import numpy as np
from datasets import load_dataset


def tamer_dataset_creator(dataset_dict, out_dir):
    """
    Convert a local DatasetDict-like object to hme100k format for TAMER.
    Args:
        dataset_dict: dict-like object with split names as keys and datasets as values. Each dataset must have 'image' and 'text' fields.
        out_dir: Output directory for the formatted dataset.
    """
    all_captions = []
    for split_name, split_data in dataset_dict.items():
        split_dir = os.path.join(out_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        captions_lines = []
        images = {}
        for idx, item in enumerate(split_data):
            image = item['image']
            if isinstance(image, bytes):
                img = Image.open(io.BytesIO(image))
            else:
                img = image
            if img.mode != 'L':
                img = img.convert('L')
            img_np = np.array(img)
            img_filename = f'{split_name}_{idx}.jpg'
            caption = item['text'] if 'text' in item else item.get('caption', '')
            captions_lines.append(f'{img_filename}\t{caption}')
            images[img_filename] = img_np
            all_captions.append(caption)
        with open(os.path.join(split_dir, 'caption.txt'), 'w', encoding='utf-8') as f:
            for line in captions_lines:
                f.write(line + '\n')
        with open(os.path.join(split_dir, 'images.pkl'), 'wb') as f:
            pickle.dump(images, f)
        print(f"Saved {split_name} set: {len(split_data)} images and captions in hme100k format (numpy arrays).")

    # Tokenize captions into symbols (split by space)
    symbols = set()
    for caption in all_captions:
        for token in caption.split():
            symbols.add(token)

    # Save dictionary.txt
    dict_path = os.path.join(out_dir, 'dictionary.txt')
    with open(dict_path, 'w', encoding='utf-8') as f:
        for symbol in sorted(symbols):
            f.write(symbol + '\n')
    print(f"Saved dictionary.txt with {len(symbols)} unique symbols.")
    print(f'Saved dictionary.txt with {len(symbols)} unique symbols.')



if __name__ == "__main__":
    # Example usage for local dataset_dict
    # from datasets import load_from_disk
    # dataset_dict = load_from_disk('path_to_local_dataset')
    # out_dir = 'output_hme100k_folder'
    # tamer_dataset_creator(dataset_dict, out_dir)
    print("This script is now intended for local DatasetDict-like objects only.\n"
          "Please import and call tamer_dataset_creator(dataset_dict, out_dir) from your own script or notebook.")
