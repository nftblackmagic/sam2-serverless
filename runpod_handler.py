import runpod
from sam2_processor import process_video, process_single_image

def handler(event):
    if 'input' not in event:
        return {"error": "No input provided"}

    action = event['input'].get('action', 'process_video')

    if action == 'process_video':
        return process_video(event)
    elif action == 'process_single_image':
        return {"refresh_worker": True, "job_results": process_single_image(event)}
    else:
        return {"error": f"Unknown action: {action}"}
    

runpod.serverless.start({"handler": handler})