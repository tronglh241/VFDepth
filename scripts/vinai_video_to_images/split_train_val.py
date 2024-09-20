import argparse
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('list_file')
    parser.add_argument('--out-dir', default='output')
    parser.add_argument('--train-ratio', type=float, default=0.8)
    args = parser.parse_args()

    list_file = Path(args.list_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = [file.strip() for file in list_file.open()]
    split_point = round(args.train_ratio * len(files))
    train_files = files[:split_point]
    val_files = files[split_point:]

    with out_dir.joinpath('train.txt').open(mode='w') as f:
        for file in train_files:
            f.write(f'{file}\n')

    with out_dir.joinpath('val.txt').open(mode='w') as f:
        for file in val_files:
            f.write(f'{file}\n')
