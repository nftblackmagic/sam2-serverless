import os
import torch
import numpy as np
import urllib.request
import uuid
import cv2
import logging
import base64
import requests
from sam2.build_sam import build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2

from tqdm import tqdm
import runpod


# Import the image processing functions
from image_processing import (
    show_points, apply_mask,
    load_image_from_url, extract_frame_from_video, encode_image, upload_to_bytescale,
    create_output_video, upload_video_to_bytescale, upload_video  # Add upload_video here
)

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create the static directory if it doesn't exist
os.makedirs('static', exist_ok=True)

# Initialize the model
logger.debug("Initializing the model...")
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

sam2_checkpoint = "./checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
logger.debug("Model initialized successfully.")

def process_video(job):
    job_input = job["input"]
    session_id = job_input.get("session_id")
    points = np.array(job_input["points"], dtype=np.float32)
    labels = np.array(job_input["labels"], dtype=np.int32)
    ann_frame_idx = job_input["ann_frame_idx"]
    ann_obj_id = job_input["ann_obj_id"]
    input_video_url = job_input.get("input_video_url")
    
    # Validate that either session_id or input_video_url is provided
    if session_id is None and input_video_url is None:
        return {"error": "Either session_id or input_video_url must be provided"}
    
    # If both are provided, prioritize session_id
    if session_id is not None and input_video_url is not None:
        logger.warning("Both session_id and input_video_url provided. Using existing session.")
        input_video_url = None
        
    if input_video_url:
        upload_response = upload_video(input_video_url)
        if "error" in upload_response:
            return upload_response
        session_id = upload_response["session_id"]
    
    video_dir = f"./temp_frames_{session_id}"
    
    if not os.path.exists(video_dir):
        return {"error": "Invalid session ID"}

    try:
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        # Load inference_state
        
        if os.environ.get('RUN_ENV') == 'production':
            runpod.serverless.progress_update(job, f"Initializing inference state (1/3)")
        inference_state = predictor.init_state(video_path=video_dir)
    except FileNotFoundError:
        return {"error": "Inference state not found. Please upload the video first."}
    except Exception as e:
        return {"error": str(e)}

    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    # Check if ann_frame_idx is valid
    if ann_frame_idx < 0 or ann_frame_idx >= len(frame_names):
        return {"error": f"Invalid ann_frame_idx. Must be between 0 and {len(frame_names) - 1}"}

    _, out_obj_ids, out_mask_logits = predictor.add_new_points(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )

    # run propagation throughout the video and collect the results in a dict
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
        if os.environ.get('RUN_ENV') == 'production':
            runpod.serverless.progress_update(job, f"Update {out_frame_idx}/{len(frame_names)} (2/3)")

    # Create output videos
    output_video_path = create_output_video(job, session_id, frame_names, video_dir, video_segments)

    # Upload video to Bytescale API
    try:
        bytescale_video_url = upload_video_to_bytescale(output_video_path)
    except Exception as e:
        return {"error": f"Failed to upload video to Bytescale: {str(e)}"}

    return {
        "output_video_url": bytescale_video_url,
        "session_id": session_id,
    }
    
def process_single_image(job):
    job_input = job["input"]
    image_url = job_input.get("input_image_url")
    video_url = job_input.get("input_video_url")
    frame_index = job_input.get("ann_frame_idx")
    points = job_input.get("points")
    labels = job_input.get("labels")

    if not (image_url or (video_url and frame_index is not None)):
        return {"error": "Missing image_url or video_url with frame_index"}
    if points is None or labels is None:
        return {"error": "Missing points or labels parameter"}

    try:
        points = np.array(points, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)
    except ValueError:
        return {"error": "Invalid format for points or labels"}
    
    if video_url and frame_index is not None:
        try:
            image = extract_frame_from_video(video_url, frame_index)
        except Exception as e:
            return {"error": f"Failed to extract frame from video: {str(e)}"}
    else:
        try:
            image = load_image_from_url(image_url)
        except requests.RequestException as e:
            return {"error": f"Failed to download image: {str(e)}"}
        except IOError:
            return {"error": "Failed to open image"}

    if image is None:
        return {"error": "Failed to obtain image"}

    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
    image_predictor = SAM2ImagePredictor(sam2_model)
    logger.debug("image predictor initialized successfully.")

    image_np = np.array(image)
    image_predictor.set_image(image_np)

    try:
        masks, scores, _ = image_predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True,
        )
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    scores = scores[sorted_ind]

    annotated_image = image_np.copy()
    for mask in masks:
        annotated_image = apply_mask(mask, annotated_image.copy(), random_color=True)

    # Add points to the final annotated image
    show_points(points, labels, annotated_image)

    try:
        annotated_buffer = encode_image(annotated_image)
        combined_mask = np.any(masks, axis=0).astype(np.uint8) * 255
        mask_buffer = encode_image(combined_mask)
    except Exception as e:
        return {"error": f"Failed to encode output images: {str(e)}"}

    try:
        bytescale_image_url = upload_to_bytescale(annotated_buffer)
        bytescale_mask_url = upload_to_bytescale(mask_buffer)
    except requests.RequestException as e:
        return {"error": f"Failed to upload images to Bytescale: {str(e)}"}

    return {
        "bytescale_image_url": bytescale_image_url,
        "bytescale_mask_url": bytescale_mask_url,
        "scores": scores.tolist()
    }