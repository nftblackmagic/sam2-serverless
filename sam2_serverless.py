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
from tqdm import tqdm
import runpod


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

def show_mask(mask, image, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 0, 0), (0, 128, 0), (0, 0, 128),
            (128, 128, 0)
        ]
        color_idx = 0 if obj_id is None else obj_id % len(colors)
        color = colors[color_idx] + (153,)  # 153 is roughly 0.6 * 255 for alpha

    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * np.array(color).reshape(1, 1, 4) / 255.0
    mask_image = (mask_image * 255).astype(np.uint8)
    
    # Convert mask_image to BGR for blending
    mask_image_bgr = cv2.cvtColor(mask_image, cv2.COLOR_RGBA2BGR)
    
    # Blend the mask with the original image
    cv2.addWeighted(mask_image_bgr, 0.6, image, 1, 0, image)
    
    return image, mask_image

def show_points(coords, labels, image, marker_size=20):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    
    for point in pos_points:
        cv2.drawMarker(image, tuple(point.astype(int)), (0, 255, 0), cv2.MARKER_STAR, 
                       marker_size, 2)
    for point in neg_points:
        cv2.drawMarker(image, tuple(point.astype(int)), (0, 0, 255), cv2.MARKER_STAR, 
                       marker_size, 2)

def annotate_frame(frame_idx, frame_names, video_dir, masks=None, points=None, labels=None):
    # Load the image
    img_path = os.path.join(video_dir, frame_names[frame_idx])
    img = cv2.imread(img_path)
    mask_image = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
    
    # Display points and masks if provided
    if points is not None and labels is not None:
        show_points(points, labels, img)
    if masks is not None:
        for obj_id, mask in masks.items():
            img, mask_image = show_mask(mask, img, obj_id=obj_id)
    
    # Save the image locally for debugging
    debug_output_dir = "./debug_frames"
    os.makedirs(debug_output_dir, exist_ok=True)
    debug_frame_path = os.path.join(debug_output_dir, f"frame_{frame_idx}.png")
    cv2.imwrite(debug_frame_path, img)
    if masks is not None:
        mask_debug_output_dir = "./debug_frames_mask"
        os.makedirs(mask_debug_output_dir, exist_ok=True)
        mask_debug_frame_path = os.path.join(mask_debug_output_dir, f"frame_{frame_idx}.png")
        cv2.imwrite(mask_debug_frame_path, mask_image)
    
    return img, mask_image

def upload_video(video_url):
    logger.debug("Received request to upload video.")

    if not video_url:
        logger.error("Missing video_url parameter")
        return {"error": "Missing video_url parameter"}

    # Generate a unique ID for this video processing session
    session_id = str(uuid.uuid4())
    video_dir = f"./temp_frames_{session_id}"
    os.makedirs(video_dir, exist_ok=True)
    video_path = os.path.join(video_dir, "input_video.mp4")

    logger.debug(f"Created session with ID: {session_id}")

    try:
        # Download video
        logger.debug(f"Downloading video from URL: {video_url}")
        urllib.request.urlretrieve(video_url, video_path)

        # Extract frames
        logger.debug("Extracting frames from video...")
        vidcap = cv2.VideoCapture(video_path)
        success, image = vidcap.read()
        count = 0
        while success:
            cv2.imwrite(os.path.join(video_dir, f"{count}.jpg"), image)
            success, image = vidcap.read()
            count += 1

        logger.debug(f"Extracted {count} frames from the video.")

    except Exception as e:
        logger.error(f"Error during video upload and frame extraction: {str(e)}")
        # Cleanup in case of an error
        if os.path.exists(video_dir):
            for file in os.listdir(video_dir):
                os.remove(os.path.join(video_dir, file))
            os.rmdir(video_dir)
        return {"error": str(e)}

    logger.debug("Video upload and frame extraction completed successfully.")
    return {
        "message": "Video uploaded, frames extracted, and inference state initialized successfully",
        "session_id": session_id,
        "frame_count": count
    }

def process_image(session_id, frame_idx, points, labels, ann_obj_id):
    logger.debug("Received request to process image.")

    video_dir = f"./temp_frames_{session_id}"
    
    if not os.path.exists(video_dir):
        logger.error(f"Invalid session ID: {session_id}")
        return {"error": "Invalid session ID"}

    # Get all frame names in the directory
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    
    # Check if frame_idx is valid
    if frame_idx < 0 or frame_idx >= len(frame_names):
        logger.error(f"Invalid frame_idx: {frame_idx}")
        return {"error": f"Invalid frame_idx. Must be between 0 and {len(frame_names) - 1}"}
    
    try:
        logger.debug("Initializing inference state...")
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        # Load inference_state
        inference_state = predictor.init_state(video_path=video_dir)
        logger.debug("Inference state initialized successfully.")
    except FileNotFoundError:
        logger.error("Inference state not found. Video may not have been uploaded.")
        return {"error": "Inference state not found. Please upload the video first."}
    except Exception as e:
        logger.error(f"Error initializing inference state: {str(e)}")
        return {"error": str(e)}
    
    try:
        logger.debug("Processing image with predictor...")
        _, out_obj_ids, out_mask_logits = predictor.add_new_points(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
        )

        out_obj_id = (out_mask_logits[0] > 0.0).cpu().numpy()

        logger.debug("Annotating frame...")
        # Create output image
        annotated_frame, mask_image = annotate_frame(frame_idx, frame_names, video_dir, points=points, labels=labels, masks={ann_obj_id: out_obj_id})

        # Convert the annotated frame to a base64 encoded string
        _, buffer = cv2.imencode('.png', annotated_frame)
        img_str = base64.b64encode(buffer).decode('utf-8')

    except Exception as e:
        logger.error(f"Error during image processing: {str(e)}")
        return {"error": str(e)}

    logger.debug("Image processing completed successfully.")
    return {
        "output_image": img_str
    }

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

    # Create output videos
    output_video_path = f"static/segmented_video_{session_id}.mp4"
    mask_video_path = f"static/mask_video_{session_id}.mp4"

    first_frame, _ = annotate_frame(0, frame_names, video_dir)
    height, width, _ = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, 30, (width, height))
    mask_writer = cv2.VideoWriter(mask_video_path, fourcc, 30, (width, height))

    vis_frame_stride = 1
    total_frames = len(range(0, len(frame_names), vis_frame_stride))

    # Create a tqdm progress bar
    with tqdm(total=total_frames, desc="Writing video", unit="frame") as pbar:
        for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
            if out_frame_idx in video_segments:
                annotated_frame, mask_image = annotate_frame(out_frame_idx, frame_names, video_dir, masks=video_segments[out_frame_idx])
            else:
                annotated_frame, mask_image = annotate_frame(out_frame_idx, frame_names, video_dir)
            
            video_writer.write(annotated_frame)
            
            # Convert mask_image to BGR for video writing
            mask_image_bgr = cv2.cvtColor(mask_image, cv2.COLOR_RGBA2BGR)
            mask_writer.write(mask_image_bgr)

            # Update the progress bar
            pbar.update(1)

    video_writer.release()
    mask_writer.release()

    # Upload both videos to Bytescale API
    upload_url = "https://api.bytescale.com/v2/accounts/FW25b7k/uploads/binary"
    headers = {
        "Authorization": "Bearer public_FW25b7k33rVdd9MShz7yH28Z1HWr",
        "Content-Type": "video/mp4"
    }

    bytescale_video_url = ""
    bytescale_mask_video_url = ""

    for video_path in [output_video_path, mask_video_path]:
        with open(video_path, 'rb') as video_file:
            response = requests.post(upload_url, headers=headers, data=video_file)
        
        if response.status_code != 200:
            raise Exception(f"Failed to upload video to Bytescale. Status code: {response.status_code}")
        
        bytescale_response = response.json()
        if video_path == output_video_path:
            bytescale_video_url = bytescale_response.get('fileUrl')
        else:
            bytescale_mask_video_url = bytescale_response.get('fileUrl')

    return {
        "output_video_url": bytescale_video_url,
        "mask_video_url": bytescale_mask_video_url,
        "session_id": session_id,
    }

runpod.serverless.start({"handler": process_video})

