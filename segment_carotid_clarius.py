import argparse
import os
from pathlib import Path

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras import ops
from tqdm import tqdm

from zea import init_device, log
from zea.models.carotid_segmenter import CarotidSegmenter
from zea.visualize import plot_shape_from_mask, set_mpl_style
from zea.tensor_ops import vmap


def load_video_frames(video_path, max_seconds=9):
    """Load frames from video file for the first max_seconds.
    
    Args:
        video_path: Path to the video file.
        max_seconds: Maximum number of seconds to load.
        
    Returns:
        frames: List of frames as numpy arrays (H, W, 3).
        fps: Frames per second of the video.
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    max_frames = int(fps * max_seconds)
    
    frames = []
    frame_count = 0
    
    log.info(f"Loading first {max_seconds} seconds ({max_frames} frames) at {fps:.2f} fps")
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            log.warning(f"Video ended at frame {frame_count}")
            break
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        frame_count += 1
    
    cap.release()
    log.info(f"Loaded {len(frames)} frames")
    
    return frames, fps


def preprocess_frames(frames):
    """Preprocess frames for the carotid segmenter.
    
    Args:
        frames: List of RGB frames (H, W, 3).
        
    Returns:
        Batch tensor of shape (N, H, W, 1) normalized to [0, 1].
    """
    # Convert to grayscale and normalize
    gray_frames = []
    for frame in frames:
        # Convert RGB to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # Normalize to [0, 1]
        gray_normalized = gray.astype(np.float32) / 255.0
        gray_frames.append(gray_normalized)
    
    # Stack and add channel dimension
    batch = np.stack(gray_frames, axis=0)  # (N, H, W)
    batch = batch[..., None]  # (N, H, W, 1)
    batch = batch[:,26:,182:438,:] # crop to center square
    
    return ops.convert_to_tensor(batch)


def segment_frames(model, frames, batch_size=8):
    """Segment frames using the carotid segmenter.
    
    Args:
        model: CarotidSegmenter model.
        frames_batch: Tensor of shape (N, H, W, 1).
        
    Returns:
        masks: Binary masks of shape (N, H, W).
    """
    log.info(f"Segmenting {frames.shape[0]} frames...")
    
    masks = vmap(model, batch_size=batch_size)(frames[:, None, ...])[:, 0, ...]
    
    # Remove channel dimension and apply threshold
    masks = ops.squeeze(masks, axis=-1)
    masks_binary = ops.where(masks > 0.5, 1, 0)
    masks_binary = ops.convert_to_numpy(masks_binary)
    
    return masks_binary


def save_results(frames, masks, preprocessed_frames, output_dir, fps):
    """Save segmentation results as images and video.
    
    Args:
        frames: List of original RGB frames.
        masks: Binary masks of shape (N, H, W).
        preprocessed_frames: Preprocessed grayscale frames tensor of shape (N, H, W, 1).
        output_dir: Directory to save results.
        fps: Frames per second for output video.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert preprocessed frames to numpy
    preprocessed_frames_np = ops.convert_to_numpy(preprocessed_frames)
    preprocessed_frames_np = preprocessed_frames_np.squeeze(-1)  # Remove channel dimension
    
    # Save individual frames with overlays
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    
    log.info(f"Saving {len(frames)} frames to {frames_dir}")
    
    set_mpl_style()
    
    for i, (preprocessed_frame, mask) in enumerate(tqdm(zip(preprocessed_frames_np, masks), total=len(masks))):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Preprocessed frame without overlay
        axes[0].imshow(preprocessed_frame, cmap="gray", vmin=0, vmax=1)
        axes[0].set_title(f"Frame {i}")
        axes[0].axis("off")
        
        # Preprocessed frame with segmentation overlay
        axes[1].imshow(preprocessed_frame, cmap="gray", vmin=0, vmax=1)
        plot_shape_from_mask(axes[1], mask, color="red", alpha=0.5)
        axes[1].set_title(f"Segmentation {i}")
        axes[1].axis("off")
        
        plt.tight_layout()
        plt.savefig(frames_dir / f"frame_{i:04d}.png", dpi=150, bbox_inches="tight")
        plt.close()
    
    # Create output video with side-by-side comparison
    log.info("Creating output video with side-by-side comparison...")
    
    # Get dimensions from preprocessed frames
    height, width = preprocessed_frames_np.shape[1:3]
    
    # Create video writer for side-by-side (double width)
    video_path = output_dir / "segmented_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (width * 2, height))
    
    for preprocessed_frame, mask in tqdm(zip(preprocessed_frames_np, masks), total=len(masks)):
        # Convert grayscale to RGB (0-255 range) for left side (original)
        frame_left = (preprocessed_frame * 255).astype(np.uint8)
        frame_left = cv2.cvtColor(frame_left, cv2.COLOR_GRAY2RGB)
        
        # Create right side with overlay
        frame_right = frame_left.copy()
        
        # Apply red overlay where mask is 1
        overlay = frame_right.copy()
        overlay[mask == 1] = [255, 0, 0]
        
        # Blend
        alpha = 0.5
        frame_right = cv2.addWeighted(frame_right, 1 - alpha, overlay, alpha, 0)
        
        # Concatenate side by side
        side_by_side = np.concatenate([frame_left, frame_right], axis=1)
        
        # Convert back to BGR for video writer
        side_by_side_bgr = cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR)
        
        out.write(side_by_side_bgr)
    
    out.release()
    log.info(f"Saved video to {video_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Segment carotid artery in ultrasound video using CarotidSegmenter"
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default="/mnt/z/Ultrasound-BMd/data/noortje/video_images_4.mp4",
        help="Path to input video file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./segmentation_results",
        help="Directory to save output results"
    )
    parser.add_argument(
        "--max_seconds",
        type=float,
        default=9.0,
        help="Maximum number of seconds to process from the video (default: 9)"
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="carotid-segmenter",
        help="Model preset to use (default: carotid-segmenter)"
    )
    
    args = parser.parse_args()
    
    # Initialize device
    init_device(verbose=True)
    
    # Validate video path
    video_path = Path(args.video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Load video frames
    frames, fps = load_video_frames(video_path, max_seconds=args.max_seconds)
    
    # Load model
    log.info(f"Loading CarotidSegmenter with preset: {args.preset}")
    model = CarotidSegmenter.from_preset(args.preset)
    
    # Preprocess frames
    log.info("Preprocessing frames...")
    preprocessed_frames = preprocess_frames(frames)
    
    # Segment frames
    masks = segment_frames(model, preprocessed_frames)
    
    # Save results
    save_results(frames, masks, preprocessed_frames, args.output_dir, fps)
    
    log.info("Done!")


if __name__ == "__main__":
    main()