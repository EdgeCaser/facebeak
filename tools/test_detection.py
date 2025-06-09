import unittest
import numpy as np
from detection import (
    extract_roi,
    compute_iou,
    merge_overlapping_detections,
    detect_crows_parallel,
    TimeoutError
)

class TestDetection(unittest.TestCase):
    def setUp(self):
        # Create a sample frame for testing
        self.sample_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        # Add a white rectangle to simulate a bird
        self.sample_frame[20:40, 30:50] = 255

    def test_extract_roi(self):
        # Test basic ROI extraction
        bbox = (30, 20, 50, 40)
        roi, (x1, y1, x2, y2) = extract_roi(self.sample_frame, bbox)
        self.assertEqual(roi.shape, (20, 20, 3))
        self.assertEqual((x1, y1, x2, y2), bbox)

        # Test ROI extraction with padding
        roi, (x1, y1, x2, y2) = extract_roi(self.sample_frame, bbox, padding=0.1)
        self.assertEqual(roi.shape, (22, 22, 3))  # 20 + 2 pixels padding
        self.assertEqual(x1, 28)  # 30 - 2
        self.assertEqual(y1, 18)  # 20 - 2
        self.assertEqual(x2, 52)  # 50 + 2
        self.assertEqual(y2, 42)  # 40 + 2

        # Test ROI extraction at image boundaries
        bbox = (0, 0, 10, 10)
        roi, (x1, y1, x2, y2) = extract_roi(self.sample_frame, bbox, padding=0.1)
        self.assertEqual(x1, 0)  # Should not go below 0
        self.assertEqual(y1, 0)  # Should not go below 0

    def test_compute_iou(self):
        # Test overlapping boxes
        bbox1 = (0, 0, 10, 10)
        bbox2 = (5, 5, 15, 15)
        iou = compute_iou(bbox1, bbox2)
        expected_iou = 25 / 175  # 5x5 intersection / (100 + 100 - 25) union
        self.assertAlmostEqual(iou, expected_iou)

        # Test non-overlapping boxes
        bbox1 = (0, 0, 10, 10)
        bbox2 = (20, 20, 30, 30)
        iou = compute_iou(bbox1, bbox2)
        self.assertEqual(iou, 0)

        # Test identical boxes
        bbox1 = (0, 0, 10, 10)
        bbox2 = (0, 0, 10, 10)
        iou = compute_iou(bbox1, bbox2)
        self.assertEqual(iou, 1.0)

    def test_merge_overlapping_detections(self):
        # Test merging of overlapping detections
        detections = [
            {
                'bbox': [0, 0, 10, 10],
                'score': 0.9,
                'class': 'bird',
                'model': 'yolo',
                'view': 'single'
            },
            {
                'bbox': [5, 5, 15, 15],
                'score': 0.8,
                'class': 'bird',
                'model': 'rcnn',
                'view': 'single'
            }
        ]
        merged = merge_overlapping_detections(detections, iou_threshold=0.5)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]['class'], 'bird')
        self.assertGreater(merged[0]['score'], 0.9)  # Should have confidence bonus

        # Test non-overlapping detections
        detections = [
            {
                'bbox': [0, 0, 10, 10],
                'score': 0.9,
                'class': 'bird',
                'model': 'yolo',
                'view': 'single'
            },
            {
                'bbox': [20, 20, 30, 30],
                'score': 0.8,
                'class': 'bird',
                'model': 'rcnn',
                'view': 'single'
            }
        ]
        merged = merge_overlapping_detections(detections, iou_threshold=0.5)
        self.assertEqual(len(merged), 2)  # Should not merge

        # Test multi-view merging
        detections = [
            {
                'bbox': [0, 0, 10, 10],
                'score': 0.9,
                'class': 'bird',
                'model': 'yolo',
                'view': 'multi'
            },
            {
                'bbox': [5, 5, 15, 15],
                'score': 0.8,
                'class': 'bird',
                'model': 'rcnn',
                'view': 'multi'
            }
        ]
        merged = merge_overlapping_detections(detections, iou_threshold=0.5)
        self.assertEqual(len(merged), 1)
        self.assertIn('views', merged[0])
        self.assertEqual(len(merged[0]['views']), 2)

    def test_detect_crows_parallel(self):
        # Test basic detection
        frames = [self.sample_frame]
        detections = detect_crows_parallel(
            frames,
            score_threshold=0.3,
            yolo_threshold=0.2,
            multi_view_yolo=False,
            multi_view_rcnn=False
        )
        self.assertIsInstance(detections, list)
        self.assertEqual(len(detections), 1)
        self.assertIsInstance(detections[0], list)

        # Test with empty frame
        empty_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detections = detect_crows_parallel(
            [empty_frame],
            score_threshold=0.3,
            yolo_threshold=0.2
        )
        self.assertEqual(len(detections), 1)
        self.assertEqual(len(detections[0]), 0)

if __name__ == '__main__':
    unittest.main() 