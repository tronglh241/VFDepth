import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid

from datasets.fisheyevinai_dataset import FisheyeVinAIDataset
from models.camera import Fisheye

if __name__ == '__main__':
    width = 640
    height = 400
    dataset = FisheyeVinAIDataset(
        dataroot=(
            'data/PHOTOMETRIC_ALIGNMENT_FISHEYE_VIDEO_calib17Feb2023/'
            '20_PA- 3D view at 4 overlap zones When passing throught shade of building or tree/'
            'PA- 3D view at 4 overlap zones When passing throught shade of building or tree_Sunny/'
        ),
        mask_dir='datasets/masks/vinaicar/',
        calib_file=(
            'data/PHOTOMETRIC_ALIGNMENT_FISHEYE_VIDEO_calib17Feb2023/'
            'VFe34_Calib_17Feb2023/output/cameraData.json'
        ),
        token_list_file=(
            'data/PHOTOMETRIC_ALIGNMENT_FISHEYE_VIDEO_calib17Feb2023/'
            '20_PA- 3D view at 4 overlap zones When passing throught shade of building or tree/'
            'PA- 3D view at 4 overlap zones When passing throught shade of building or tree_Sunny/'
            'file_list.txt'
        ),
        cam_names=['front', 'left', 'rear', 'right'],
        image_shape=(width, height),
        jittering=(0.2, 0.2, 0.2, 0.05),
        crop_train_borders=(),
        crop_eval_borders=(),
        ref_extrinsic_idx=0,
    )
    loader = DataLoader(dataset)
    sample = next(iter(loader))
    images = torch.cat(sample[:3], dim=0)
    image = make_grid(images.reshape(12, 3, height, width), nrow=4)
    image = transforms.ToPILImage()(image)
    image.show()
    points_3d = torch.Tensor([
        [0, 10, 10],
        [10, 0, 10],
        [10, 10, 0],
    ])

    camera = Fisheye(
        width=width,
        height=height,
        extrinsic=sample[5],
        intrinsic=sample[4],
        distortion=sample[6],
        eps=1e-8,
        max_count=10,
    )
    points_2d, valid, depth = camera.world_to_im(points_3d, normalize=True)
    print(points_2d[:, 0])
    print(valid[:, 0])
    print(depth[:, 0])
