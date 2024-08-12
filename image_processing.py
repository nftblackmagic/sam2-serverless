import cv2
import numpy as np
import os
import base64
from PIL import Image
from io import BytesIO
import requests
import cv2
from tqdm import tqdm
import uuid
import urllib.request
import json
import av
import runpod

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
    
    return image

def draw_single_image(mask, image, random_color=False, borders=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    
    if borders:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
    
    # Convert mask_image to uint8 and ensure it has the same number of channels as the original image
    mask_image = (mask_image * 255).astype(np.uint8)
    if mask_image.shape[-1] == 4 and image.shape[-1] == 3:
        mask_image = cv2.cvtColor(mask_image, cv2.COLOR_RGBA2RGB)
    elif mask_image.shape[-1] == 3 and image.shape[-1] == 4:
        mask_image = cv2.cvtColor(mask_image, cv2.COLOR_RGB2RGBA)
    
    # Ensure mask_image has the same shape as the original image
    mask_image = cv2.resize(mask_image, (image.shape[1], image.shape[0]))
    
    return cv2.addWeighted(image, 1, mask_image, 0.5, 0)

def apply_mask(mask, image, obj_id=0, random_color=False, borders=True):
    # Determine color
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        colors = [
            (70, 130, 180),   # Steel Blue
            (255, 160, 122),  # Light Salmon
            (152, 251, 152),  # Pale Green
            (221, 160, 221),  # Plum
            (176, 196, 222),  # Light Steel Blue
            (255, 182, 193),  # Light Pink
            (240, 230, 140),  # Khaki
            (216, 191, 216),  # Thistle
            (173, 216, 230),  # Light Blue
            (255, 228, 196)   # Bisque
        ]
        color_idx = obj_id % len(colors)
        color = np.array(colors[color_idx] + (153,)) / 255.0  # 153 is roughly 0.6 * 255 for alpha

    # Create mask image
    if mask.ndim == 2:
        h, w = mask.shape
    elif mask.ndim == 3:
        h, w = mask.shape[1:]
        mask = mask.squeeze(0)  # Remove the first dimension if it's (1, h, w)
    else:
        raise ValueError("Unexpected mask shape")
    
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    
    # Draw borders if requested
    if borders:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)

    # Convert mask_image to uint8 and ensure it has the same number of channels as the original image
    mask_image = (mask_image * 255).astype(np.uint8)
    if mask_image.shape[-1] == 4 and image.shape[-1] == 3:
        mask_image = cv2.cvtColor(mask_image, cv2.COLOR_RGBA2RGB)
    elif mask_image.shape[-1] == 3 and image.shape[-1] == 4:
        mask_image = cv2.cvtColor(mask_image, cv2.COLOR_RGB2RGBA)

    # Ensure mask_image has the same shape as the original image
    mask_image = cv2.resize(mask_image, (image.shape[1], image.shape[0]))

    # Blend the mask with the original image
    return cv2.addWeighted(image, 1, mask_image, 0.5, 0)


def show_points(coords, labels, image, marker_size=20):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    
    for point in pos_points:
        cv2.circle(image, tuple(point.astype(int)), marker_size // 2, (0, 255, 0), -1)
        cv2.circle(image, tuple(point.astype(int)), marker_size // 2, (255, 255, 255), 2)
    for point in neg_points:
        cv2.circle(image, tuple(point.astype(int)), marker_size // 2, (0, 0, 255), -1)
        cv2.circle(image, tuple(point.astype(int)), marker_size // 2, (255, 255, 255), 2)

def process_mask(mask, img_shape, color):
    # Process mask
    if mask.ndim == 2:
        h, w = mask.shape
    elif mask.ndim == 3:
        h, w = mask.shape[1:]
        mask = mask.squeeze(0)  # Remove the first dimension if it's (1, h, w)
    else:
        raise ValueError("Unexpected mask shape")
    
    # Convert boolean mask to uint8 before resizing
    mask = mask.astype(np.uint8) * 255
    
    # Ensure mask has the same shape as the image
    mask = cv2.resize(mask, (img_shape[1], img_shape[0]))
    
    # Normalize the mask back to 0-1 range
    mask = mask / 255.0
    
    # Create colored mask
    colored_mask = (mask[:, :, np.newaxis] * color).astype(np.uint8)
    
    # Create alpha channel
    alpha = (mask * 255).astype(np.uint8)
    
    return colored_mask, alpha

def annotate_frame(frame_idx, frame_names, video_dir, mode, masks=None, points=None, labels=None):
    # Load the image
    img_path = os.path.join(video_dir, frame_names[frame_idx])
    img = cv2.imread(img_path)
    
    # Display points if provided
    if points is not None and labels is not None:
        show_points(points, labels, img)
    
    if masks is not None:
        if mode == "overlayer":
            # Current logic: mask applied to image
            for obj_id, mask in masks.items():
                img = apply_mask(mask, img, obj_id=obj_id)
        elif mode == "masked_image":
            # Only show the masked region, other regions are transparent
            result = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
            for obj_id, mask in masks.items():
                color = np.array([255, 255, 255])  # Use white color to preserve original image colors
                colored_mask, alpha = process_mask(mask, img.shape, color)
                # Apply the mask to the original image
                masked_region = cv2.bitwise_and(img, colored_mask)
                result[:, :, :3] += masked_region
                result[:, :, 3] += alpha
            img = result
        elif mode == "mask_only":
            # Only show the mask
            result = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
            for obj_id, mask in masks.items():
                color = np.array(get_color(obj_id))
                colored_mask, alpha = process_mask(mask, img.shape, color)
                result[:, :, :3] += colored_mask
                result[:, :, 3] += alpha
            img = result
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    # Save the image locally for debugging
    debug_output_dir = "./debug_frames"
    os.makedirs(debug_output_dir, exist_ok=True)
    debug_frame_path = os.path.join(debug_output_dir, f"frame_{frame_idx}_{mode}.png")
    cv2.imwrite(debug_frame_path, img)
    
    return img

def get_color(obj_id):
    colors = [
        (70, 130, 180),   # Steel Blue
        (255, 160, 122),  # Light Salmon
        (152, 251, 152),  # Pale Green
        (221, 160, 221),  # Plum
        (176, 196, 222),  # Light Steel Blue
        (255, 182, 193),  # Light Pink
        (240, 230, 140),  # Khaki
        (216, 191, 216),  # Thistle
        (173, 216, 230),  # Light Blue
        (255, 228, 196)   # Bisque
    ]
    return colors[obj_id % len(colors)]

def load_image_from_url(image_url):
    response = requests.get(image_url)
    response.raise_for_status()
    return Image.open(BytesIO(response.content))

def extract_frame_from_video(video_url, frame_index):
    # Download the video to a temporary file
    temp_video_path = f"temp_video_{uuid.uuid4()}.mp4"
    urllib.request.urlretrieve(video_url, temp_video_path)

    # Open the video file
    cap = cv2.VideoCapture(temp_video_path)

    # Set the frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    # Read the frame
    ret, frame = cap.read()

    # Release the video capture object and delete the temporary file
    cap.release()
    os.remove(temp_video_path)

    if not ret:
        raise Exception(f"Failed to extract frame {frame_index} from video")

    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return frame_rgb

def encode_image(image):
    _, buffer = cv2.imencode('.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return buffer

def upload_to_bytescale(image_buffer):
    upload_url = "https://api.bytescale.com/v2/accounts/FW25b7k/uploads/binary"
    headers = {
        "Authorization": "Bearer public_FW25b7k33rVdd9MShz7yH28Z1HWr",
        "Content-Type": "image/png"
    }
    response = requests.post(upload_url, headers=headers, data=image_buffer.tobytes())
    response.raise_for_status()
    return response.json().get('fileUrl')

def create_output_video(job, session_id, frame_names, video_dir, video_segments, mode):

    # Read video information from the JSON file
    video_info_path = os.path.join(video_dir, "video_settings.json")
    with open(video_info_path, 'r') as f:
        video_info = json.load(f)
    
    # Extract video dimensions and fps
    width = video_info['width']
    height = video_info['height']
    fps = video_info['fps']
    
    # Ensure the dimensions are even (required by some codecs)
    width = width if width % 2 == 0 else width + 1
    height = height if height % 2 == 0 else height + 1

    # Create output container
    output_video_path = f"static/segmented_video_{session_id}.{'mov' if mode in ['masked_image'] else 'mp4'}"
    output = av.open(output_video_path, mode='w')
    
    if mode in ['masked_image']:
        # Use Apple ProRes 4444 for masked_image and mask_only modes
        stream = output.add_stream('prores', rate='{0:.4f}'.format(fps))
        stream.codec_tag = 'ap4h'  # ProRes 4444
        stream.pix_fmt = 'yuva444p10le'
    else:
        # Use H.264 for other modes
        stream = output.add_stream('h264', rate='{0:.4f}'.format(fps))
        stream.pix_fmt = 'yuv420p'
        stream.options = {
            'crf': '23',  # Default CRF value, good balance between quality and file size
            'preset': 'medium'  # Default preset, balances encoding speed and compression efficiency
        }
    
    stream.width = width
    stream.height = height

    vis_frame_stride = 1
    total_frames = len(range(0, len(frame_names), vis_frame_stride))

    with tqdm(total=total_frames, desc="Writing video", unit="frame") as pbar:
        for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
            if out_frame_idx in video_segments:
                annotated_frame = annotate_frame(out_frame_idx, frame_names, video_dir, mode, masks=video_segments[out_frame_idx])
            else:
                annotated_frame = annotate_frame(out_frame_idx, frame_names, video_dir, mode)
            
            # Convert BGR to RGBA for masked_image and mask_only modes
            if mode in ['masked_image']:
                if annotated_frame.shape[2] == 3:
                    # If the frame is BGR, convert to BGRA first
                    frame_bgra = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2BGRA)
                    frame_rgba = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2RGBA)
                else:
                    # If the frame is already BGRA, just convert to RGBA
                    frame_rgba = cv2.cvtColor(annotated_frame, cv2.COLOR_BGRA2RGBA)
            else:
                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            # Create PyAV video frame
            if mode in ['masked_image']:
                frame = av.VideoFrame.from_ndarray(frame_rgba, format='rgba')
            else:
                frame = av.VideoFrame.from_ndarray(frame_rgb, format='rgb24')
            
            # Encode and write the frame
            for packet in stream.encode(frame):
                output.mux(packet)
            
            pbar.update(1)
            if os.environ.get('RUN_ENV') == 'production':
                runpod.serverless.progress_update(job, f"Writing video update {out_frame_idx}/{len(frame_names)} (3/3)")

    # Flush the stream
    for packet in stream.encode():
        output.mux(packet)

    # Close the output container
    output.close()

    return output_video_path

def upload_video_to_bytescale(video_path):
    upload_url = "https://api.bytescale.com/v2/accounts/FW25b7k/uploads/binary"
    headers = {
        "Authorization": "Bearer public_FW25b7k33rVdd9MShz7yH28Z1HWr",
        "Content-Type": "video/mp4" if video_path.lower().endswith('.mp4') else "video/quicktime" if video_path.lower().endswith('.mov') else f"video/{os.path.splitext(video_path)[1][1:]}"
    }

    with open(video_path, 'rb') as video_file:
        response = requests.post(upload_url, headers=headers, data=video_file)
    
    if response.status_code != 200:
        raise Exception(f"Failed to upload video to Bytescale. Status code: {response.status_code}")
    
    bytescale_response = response.json()
    return bytescale_response.get('fileUrl')

def upload_video(video_url):
    if not video_url:
        return {"error": "Missing video_url parameter"}

    # Generate a unique ID for this video processing session
    session_id = str(uuid.uuid4())
    video_dir = f"./temp_frames_{session_id}"
    os.makedirs(video_dir, exist_ok=True)
    video_path = os.path.join(video_dir, "input_video.mp4")

    try:
        # Download video
        urllib.request.urlretrieve(video_url, video_path)

        # Extract frames
        vidcap = cv2.VideoCapture(video_path)
        success, image = vidcap.read()
        count = 0
        while success:
            cv2.imwrite(os.path.join(video_dir, f"{count}.jpg"), image)
            success, image = vidcap.read()
            count += 1

        # Extract video information
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = int(vidcap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

        # Create a dictionary with video settings
        video_settings = {
            "fps": fps,
            "width": width,
            "height": height,
            "codec": codec,
            "total_frames": count
        }

        # Write the video settings to a JSON file
        import json
        with open(os.path.join(video_dir, "video_settings.json"), "w") as f:
            json.dump(video_settings, f, indent=4)

        vidcap.release()
    except Exception as e:
        # Cleanup in case of an error
        if os.path.exists(video_dir):
            for file in os.listdir(video_dir):
                os.remove(os.path.join(video_dir, file))
            os.rmdir(video_dir)
        return {"error": str(e)}

    return {
        "message": "Video uploaded, frames extracted, and inference state initialized successfully",
        "session_id": session_id,
        "video_settings": video_settings
    }