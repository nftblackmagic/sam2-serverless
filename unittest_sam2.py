import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from sam2_processor import process_video

class TestProcessVideo(unittest.TestCase):

    @patch('sam2_processor.upload_video')
    @patch('sam2_processor.predictor')
    @patch('sam2_processor.os.path.exists')
    @patch('sam2_processor.os.listdir')
    @patch('sam2_processor.annotate_frame')
    @patch('sam2_processor.cv2.VideoWriter')
    @patch('sam2_processor.requests.post')
    def test_process_video(self, mock_post, mock_video_writer, mock_annotate_frame, 
                           mock_listdir, mock_exists, mock_predictor, mock_upload_video):
        # Mock the necessary functions and objects
        mock_upload_video.return_value = {"session_id": "test_session"}
        mock_exists.return_value = True
        mock_listdir.return_value = ['0.jpg', '1.jpg', '2.jpg']
        mock_predictor.init_state.return_value = MagicMock()
        mock_predictor.add_new_points.return_value = (None, [1], [np.array([[True, False], [False, True]])])
        mock_predictor.propagate_in_video.return_value = [
            (0, [1], [np.array([[True, False], [False, True]])]),
            (1, [1], [np.array([[False, True], [True, False]])]),
            (2, [1], [np.array([[True, True], [False, False]])])
        ]
        mock_annotate_frame.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_video_writer.return_value = MagicMock()
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"fileUrl": "https://upcdn.io/FW25b7k/raw/uploads/processed_test.mp4"}

        # Use the provided test case
        job = {
            "input": {
                "action": "process_video",
                "input_video_url": "https://upcdn.io/FW25b7k/raw/uploads/test.mp4",
                "points": [[603, 866]],
                "labels": [1],
                "ann_frame_idx": 0,
                "ann_obj_id": 1
            }
        }

        # Call the function
        result = process_video(job)

        # Assert the expected results
        self.assertIn("output_video_url", result)
        # self.assertEqual(result["output_video_url"], "https://upcdn.io/FW25b7k/raw/uploads/processed_test.mp4")
        # self.assertEqual(result["session_id"], "test_session")

        # Verify that the mocked functions were called with correct arguments
        mock_upload_video.assert_called_once_with("https://upcdn.io/FW25b7k/raw/uploads/test.mp4")
        mock_predictor.init_state.assert_called_once()
        mock_predictor.add_new_points.assert_called_once_with(
            inference_state=mock_predictor.init_state.return_value,
            frame_idx=0,
            obj_id=1,
            points=np.array([[603, 866]], dtype=np.float32),
            labels=np.array([1], dtype=np.int32)
        )
        mock_predictor.propagate_in_video.assert_called_once()
        mock_video_writer.assert_called_once()
        mock_post.assert_called_once()

if __name__ == '__main__':
    unittest.main()