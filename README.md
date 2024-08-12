# RunPod Video and Image Processor

This project is a serverless application using RunPod to process videos and single images using the SAM2 (Segment Anything Model 2) processor.

## Features

- Process videos using SAM2
- Process single images using SAM2
- Serverless architecture using RunPod

## Requirements

- Python 3.x
- RunPod SDK
- SAM2 processor (not included in this repository)

## Installation

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install runpod
   ```
3. Ensure you have the SAM2 processor installed and configured

## Usage

The main handler function in `runpod_handler.py` processes incoming events and routes them to the appropriate function based on the `action` parameter.

### Processing a Video

Send an event with the following structure:

```json
{
  "input": {
    "action": "process_video",
    "video_url": "https://example.com/video.mp4",
    "output_bucket": "my-output-bucket",
    "output_key": "processed_video.mp4"
  }
}
```

### Processing a Single Image

Send an event with the following structure:

```json
{
  "input": {
    "action": "process_single_image",
    // Add other required parameters for image processing
  }
}
```

## File Structure

- `runpod_handler.py`: Main handler for RunPod serverless functions
- `sam2_processor.py`: Contains the `process_video` and `process_single_image` functions (not included in this repository)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Add your chosen license here]