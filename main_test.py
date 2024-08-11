import json
import sys
from sam2_processor import process_video  # Assuming the main script is named main_script.py

def process_video_test(input_file):
    # Read input from JSON file
    with open(input_file, 'r') as f:
        input_data = json.load(f)

    # Extract data from input
    session_id = input_data['session_id']
    clicks = input_data['clicks']

    # Process video for each click
    for click in clicks:
        process_data = {
            "session_id": session_id,
            "points": click['points'],
            "labels": click['labels'],
            "ann_frame_idx": click['ann_frame_idx'],
            "ann_obj_id": click['ann_obj_id']
        }
        
        try:
            process_result = process_video(**process_data)

            segmented_video_url = process_result.get('output_video_url')
            mask_only_video_url = process_result.get('mask_video_url')

            if segmented_video_url and mask_only_video_url:
                print(f"Video processed successfully.")
                print(f"Segmented video URL: {segmented_video_url}")
                print(f"Mask-only video URL: {mask_only_video_url}")
            else:
                print("Output video URLs not found in process response")
        except Exception as e:
            print(f"Process request failed: {str(e)}")
            continue

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python script.py <input_json_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    process_video_test(input_file)