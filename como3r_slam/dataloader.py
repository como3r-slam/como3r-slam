import pathlib
import cv2
from natsort import natsorted
import numpy as np
import torch

from como3r_slam.mast3r_utils import resize_img
from como3r_slam.config import config

HAS_TORCHCODEC = True
try:
    from torchcodec.decoders import VideoDecoder
except Exception as e:
    HAS_TORCHCODEC = False


class MonocularDataset(torch.utils.data.Dataset):
    def __init__(self, dtype=np.float32):
        self.dtype = dtype
        self.rgb_files = []
        self.timestamps = []
        self.img_size = 512
        self.camera_intrinsics = None
        self.use_calibration = config["use_calib"]

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        img = self.get_image(idx)
        timestamp = self.get_timestamp(idx)
        return timestamp, img

    def get_timestamp(self, idx):
        return self.timestamps[idx]

    def read_img(self, idx):
        img = cv2.imread(self.rgb_files[idx])
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def get_image(self, idx):
        img = self.read_img(idx)
        if self.use_calibration:
            img = self.camera_intrinsics.remap(img)
        return img.astype(self.dtype) / 255.0

    def get_img_shape(self):
        # First entry is the post-resize (and post-remap, in calib mode) shape
        # actually used by every downstream pipeline -- SharedKeyframes buffer
        # allocation, frame creation, and CUDA kernel sizing all assume this.
        # Second entry is the raw on-disk shape, used by Intrinsics.from_calib
        # to compute the K_orig -> K_resized scale factors.
        img = self.read_img(0)
        raw_img_shape = img.shape
        if self.use_calibration and self.camera_intrinsics is not None:
            img = self.camera_intrinsics.remap(img)
        img = resize_img(img, self.img_size)
        # 3XHxW, HxWx3 -> HxW, HxW
        return img["img"][0].shape[1:], raw_img_shape[:2]

    def subsample(self, subsample):
        self.rgb_files = self.rgb_files[::subsample]
        self.timestamps = self.timestamps[::subsample]


class MP4Dataset(MonocularDataset):
    def __init__(self, dataset_path):
        super().__init__()
        self.use_calibration = False
        self.dataset_path = pathlib.Path(dataset_path)
        if HAS_TORCHCODEC:
            self.decoder = VideoDecoder(str(self.dataset_path))
            self.fps = self.decoder.metadata.average_fps
            self.total_frames = self.decoder.metadata.num_frames
        else:
            print("torchcodec is not installed. This may slow down the dataloader")
            self.cap = cv2.VideoCapture(str(self.dataset_path))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.stride = config["dataset"]["subsample"]

    def __len__(self):
        return self.total_frames // self.stride

    def read_img(self, idx):
        if HAS_TORCHCODEC:
            img = self.decoder[idx * self.stride]  # c,h,w
            img = img.permute(1, 2, 0)
            img = img.numpy()
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx * self.stride)
            ret, img = self.cap.read()
            if not ret:
                raise ValueError("Failed to read image")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(self.dtype)
        timestamp = idx / self.fps
        self.timestamps.append(timestamp)
        return img


class RGBFiles(MonocularDataset):
    def __init__(self, dataset_path):
        super().__init__()
        self.use_calibration = False
        self.dataset_path = pathlib.Path(dataset_path)
        candidate_dir = self.dataset_path
        if (self.dataset_path / "color").is_dir() and not any(
            self.dataset_path.glob("*.png")
        ) and not any(self.dataset_path.glob("*.jpg")):
            candidate_dir = self.dataset_path / "color"
        files = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG"):
            files.extend(candidate_dir.glob(ext))
        self.rgb_files = natsorted(files)
        if not self.rgb_files:
            raise FileNotFoundError(
                f"no images found in {candidate_dir}"
                + ("" if candidate_dir.is_dir() else " (directory does not exist)")
            )
        self.timestamps = np.arange(0, len(self.rgb_files)).astype(self.dtype) / 30.0


class Intrinsics:
    def __init__(self, img_size, W, H, K_orig, K, distortion, mapx, mapy):
        self.img_size = img_size
        self.W, self.H = W, H
        self.K_orig = K_orig
        self.K = K
        self.distortion = distortion
        self.mapx = mapx
        self.mapy = mapy
        _, (scale_w, scale_h, half_crop_w, half_crop_h) = resize_img(
            np.zeros((H, W, 3)), self.img_size, return_transformation=True
        )
        self.K_frame = self.K.copy()
        self.K_frame[0, 0] = self.K[0, 0] / scale_w
        self.K_frame[1, 1] = self.K[1, 1] / scale_h
        self.K_frame[0, 2] = self.K[0, 2] / scale_w - half_crop_w
        self.K_frame[1, 2] = self.K[1, 2] / scale_h - half_crop_h

    def remap(self, img):
        return cv2.remap(img, self.mapx, self.mapy, cv2.INTER_LINEAR)

    @staticmethod
    def from_calib(img_size, W, H, calib, always_undistort=False):
        if not config["use_calib"] and not always_undistort:
            return None
        fx, fy, cx, cy = calib[:4]
        distortion = np.zeros(4)
        if len(calib) > 4:
            distortion = np.array(calib[4:])
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        K_opt = K.copy()
        mapx, mapy = None, None
        center = config["dataset"]["center_principle_point"]
        K_opt, _ = cv2.getOptimalNewCameraMatrix(
            K, distortion, (W, H), 0, (W, H), centerPrincipalPoint=center
        )
        mapx, mapy = cv2.initUndistortRectifyMap(
            K, distortion, None, K_opt, (W, H), cv2.CV_32FC1
        )

        return Intrinsics(img_size, W, H, K, K_opt, distortion, mapx, mapy)


def load_dataset(dataset_path):
    """Route a path to a dataset reader: a video file, or a folder of RGB images."""
    ext = dataset_path.split("/")[-1].split(".")[-1]
    if ext in ["mp4", "avi", "MOV", "mov"]:
        return MP4Dataset(dataset_path)
    return RGBFiles(dataset_path)
