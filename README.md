# RunPod Video and Image Processor

This project is a serverless application using RunPod to process videos and single images using the SAM2 (Segment Anything Model 2) processor.

## Features

- Process videos using SAM2
- Process single images using SAM2
- Serverless architecture using RunPod

## Dependencies
- SAM2 
- File uploader [bytescale](https://bytescale.com/) (API key BYTESCALE_API_KEY required) 

## Requirements

- Python 3.x
- RunPod SDK

## Installation

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Download the weights for SAM2 from the [official repository](https://github.com/IDEA-Research/SAM2)
   ```
   ./download_weights.sh
   mv *.pt checkpoints/
   ```
4. Set the BYTESCALE_API_KEY environment variable to your bytescale API key

## Usage

The main handler function in `runpod_handler.py` processes incoming events and routes them to the appropriate function based on the `action` parameter.

To run the handler locally for testing, use the following command:

```
python runpod_handler.py
```

It will load the file test_input.json and process the event.

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
- `sam2_processor.py`: Contains the `process_video` and `process_single_image` functions 

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License