"""
Image Processing Module for World Model

This module provides comprehensive image processing capabilities including
image enhancement, YOLO object detection, semantic segmentation, and feature extraction.
"""

import numpy as np
import cv2
from typing import Optional, Tuple, Dict, Any, List, Union
from dataclasses import dataclass
import time
import logging
from PIL import Image, ImageEnhance
import torch
import torch.nn as nn
from torchvision import transforms
from sklearn.cluster import KMeans
import threading
import queue

logger = logging.getLogger(__name__)


@dataclass
class ImageConfig:
    """Configuration for image processing parameters."""
    # Image enhancement
    brightness_factor: float = 1.2
    contrast_factor: float = 1.1
    saturation_factor: float = 1.1
    sharpness_factor: float = 1.0

    # Noise reduction
    denoise_h: int = 10
    denoise_template_window_size: int = 7
    denoise_search_window_size: int = 21

    # Object detection
    detection_confidence_threshold: float = 0.5
    detection_nms_threshold: float = 0.4
    yolo_model_path: str = "yolov8n.pt"  # Default to YOLOv8 nano

    # Semantic segmentation
    segmentation_model_path: str = "deeplabv3_resnet101"

    # Feature extraction
    sift_n_features: int = 0  # 0 means keep all
    sift_n_octave_layers: int = 3
    sift_contrast_threshold: float = 0.04
    sift_edge_threshold: float = 10
    sift_sigma: float = 1.6

    # Performance
    use_gpu: bool = True
    batch_size: int = 8
    num_workers: int = 4


class ImageEnhancer:
    """Image enhancement and preprocessing."""

    def __init__(self, config: ImageConfig):
        self.config = config

    def enhance(self, image: np.ndarray) -> np.ndarray:
        """
        Apply image enhancement techniques.

        Args:
            image: Input image (HxWxC)

        Returns:
            Enhanced image
        """
        try:
            # Convert to PIL for enhancement
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)

            pil_image = Image.fromarray(image)

            # Apply brightness adjustment
            if self.config.brightness_factor != 1.0:
                enhancer = ImageEnhance.Brightness(pil_image)
                pil_image = enhancer.enhance(self.config.brightness_factor)

            # Apply contrast adjustment
            if self.config.contrast_factor != 1.0:
                enhancer = ImageEnhance.Contrast(pil_image)
                pil_image = enhancer.enhance(self.config.contrast_factor)

            # Apply saturation adjustment
            if self.config.saturation_factor != 1.0:
                enhancer = ImageEnhance.Color(pil_image)
                pil_image = enhancer.enhance(self.config.saturation_factor)

            # Apply sharpness adjustment
            if self.config.sharpness_factor != 1.0:
                enhancer = ImageEnhance.Sharpness(pil_image)
                pil_image = enhancer.enhance(self.config.sharpness_factor)

            # Convert back to numpy
            enhanced = np.array(pil_image)

            # Apply denoising
            enhanced = cv2.fastNlMeansDenoisingColored(
                enhanced,
                None,
                self.config.denoise_h,
                self.config.denoise_h,
                self.config.denoise_template_window_size,
                self.config.denoise_search_window_size
            )

            return enhanced

        except Exception as e:
            logger.error(f"Error enhancing image: {str(e)}")
            return image

    def histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """Apply histogram equalization for contrast improvement."""
        if len(image.shape) == 3:
            # Convert to YUV and equalize Y channel
            yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
        else:
            return cv2.equalizeHist(image)

    def adaptive_histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        if len(image.shape) == 3:
            # Convert to LAB and equalize L channel
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            return clahe.apply(image)


class YOLODetector:
    """YOLO-based object detection."""

    def __init__(self, config: ImageConfig):
        self.config = config
        self.model = None
        self.device = "cuda" if config.use_gpu and torch.cuda.is_available() else "cpu"
        self.class_names = []

    def load_model(self) -> bool:
        """Load YOLO model."""
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.config.yolo_model_path)
            self.model.to(self.device)
            self.class_names = self.model.names
            logger.info(f"Loaded YOLO model: {self.config.yolo_model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {str(e)}")
            return False

    def detect(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Detect objects in image using YOLO.

        Args:
            image: Input image (HxWxC)

        Returns:
            Dictionary containing detection results
        """
        if self.model is None:
            if not self.load_model():
                return self._empty_detection_result()

        try:
            # Run inference
            results = self.model(
                image,
                conf=self.config.detection_confidence_threshold,
                iou=self.config.detection_nms_threshold,
                device=self.device
            )

            # Process results
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Extract bounding box
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())

                        detection = {
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'confidence': float(confidence),
                            'class_id': class_id,
                            'class_name': self.class_names[class_id] if class_id < len(self.class_names) else 'unknown',
                            'center': [(x1 + x2) / 2, (y1 + y2) / 2],
                            'area': (x2 - x1) * (y2 - y1)
                        }
                        detections.append(detection)

            return {
                'detections': detections,
                'num_detections': len(detections),
                'image_shape': image.shape
            }

        except Exception as e:
            logger.error(f"Error in YOLO detection: {str(e)}")
            return self._empty_detection_result()

    def _empty_detection_result(self) -> Dict[str, Any]:
        """Return empty detection result."""
        return {
            'detections': [],
            'num_detections': 0,
            'image_shape': None
        }

    def detect_batch(self, images: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Batch process multiple images."""
        results = []
        for image in images:
            result = self.detect(image)
            results.append(result)
        return results


class SemanticSegmentator:
    """Semantic segmentation using deep learning models."""

    def __init__(self, config: ImageConfig):
        self.config = config
        self.model = None
        self.device = "cuda" if config.use_gpu and torch.cuda.is_available() else "cpu"
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def load_model(self) -> bool:
        """Load segmentation model."""
        try:
            import torchvision.models as models
            self.model = models.segmentation.deeplabv3_resnet101(
                pretrained=True, progress=True
            )
            self.model.to(self.device)
            self.model.eval()
            logger.info("Loaded DeepLabV3 segmentation model")
            return True
        except Exception as e:
            logger.error(f"Failed to load segmentation model: {str(e)}")
            return False

    def segment(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Perform semantic segmentation.

        Args:
            image: Input image (HxWxC)

        Returns:
            Dictionary containing segmentation results
        """
        if self.model is None:
            if not self.load_model():
                return self._empty_segmentation_result()

        try:
            # Preprocess
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)

            # Convert to tensor
            tensor_image = self.transform(image).unsqueeze(0).to(self.device)

            # Inference
            with torch.no_grad():
                output = self.model(tensor_image)

            # Post-process
            segmentation_map = torch.argmax(output['out'][0], dim=0).cpu().numpy()
            confidence_map = torch.softmax(output['out'][0], dim=0).max(dim=0)[0].cpu().numpy()

            # Compute class statistics
            unique_classes, counts = np.unique(segmentation_map, return_counts=True)
            class_distribution = dict(zip(unique_classes.tolist(), counts.tolist()))

            return {
                'segmentation_map': segmentation_map,
                'confidence_map': confidence_map,
                'class_distribution': class_distribution,
                'num_classes': len(unique_classes),
                'image_shape': image.shape
            }

        except Exception as e:
            logger.error(f"Error in semantic segmentation: {str(e)}")
            return self._empty_segmentation_result()

    def _empty_segmentation_result(self) -> Dict[str, Any]:
        """Return empty segmentation result."""
        return {
            'segmentation_map': None,
            'confidence_map': None,
            'class_distribution': {},
            'num_classes': 0,
            'image_shape': None
        }


class FeatureExtractor:
    """Feature extraction for images."""

    def __init__(self, config: ImageConfig):
        self.config = config
        self.sift = cv2.SIFT_create(
            nfeatures=config.sift_n_features,
            nOctaveLayers=config.sift_n_octave_layers,
            contrastThreshold=config.sift_contrast_threshold,
            edgeThreshold=config.sift_edge_threshold,
            sigma=config.sift_sigma
        )

    def extract_features(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Extract various features from image.

        Args:
            image: Input image (HxWxC)

        Returns:
            Dictionary containing extracted features
        """
        features = {}

        try:
            # Convert to grayscale for some features
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image

            # SIFT features
            features['sift'] = self._extract_sift_features(gray)

            # Color histograms
            features['color_histogram'] = self._extract_color_histograms(image)

            # Texture features
            features['texture'] = self._extract_texture_features(gray)

            # Edge features
            features['edges'] = self._extract_edge_features(gray)

            # Statistical features
            features['statistics'] = self._extract_statistical_features(image)

            # HOG features
            features['hog'] = self._extract_hog_features(gray)

        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")

        return features

    def _extract_sift_features(self, gray: np.ndarray) -> Dict[str, Any]:
        """Extract SIFT keypoints and descriptors."""
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)

        # Convert keypoints to numpy arrays
        kp_array = np.array([[kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response]
                           for kp in keypoints])

        return {
            'keypoints': kp_array,
            'descriptors': descriptors,
            'num_keypoints': len(keypoints)
        }

    def _extract_color_histograms(self, image: np.ndarray, bins: int = 256) -> Dict[str, np.ndarray]:
        """Extract color histograms for each channel."""
        histograms = {}

        if len(image.shape) == 3:
            # RGB histograms
            for i, channel in enumerate(['R', 'G', 'B']):
                hist = cv2.calcHist([image], [i], None, [bins], [0, 256])
                histograms[channel] = hist.flatten()

            # HSV histograms
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            for i, channel in enumerate(['H', 'S', 'V']):
                hist = cv2.calcHist([hsv], [i], None, [bins], [0, 256])
                histograms[f'HSV_{channel}'] = hist.flatten()
        else:
            # Grayscale histogram
            hist = cv2.calcHist([image], [0], None, [bins], [0, 256])
            histograms['gray'] = hist.flatten()

        return histograms

    def _extract_texture_features(self, gray: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract texture features using Local Binary Patterns."""
        try:
            from skimage.feature import local_binary_pattern

            # LBP parameters
            radius = 3
            n_points = 8 * radius

            # Compute LBP
            lbp = local_binary_pattern(gray, n_points, radius, method='uniform')

            # Compute LBP histogram
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2))
            lbp_hist = lbp_hist.astype(float)
            lbp_hist /= (lbp_hist.sum() + 1e-7)

            # GLCM features
            glcm = self._compute_glcm(gray)

            return {
                'lbp': lbp,
                'lbp_histogram': lbp_hist,
                'glcm': glcm
            }

        except ImportError:
            logger.warning("skimage not available, skipping texture features")
            return {}

    def _compute_glcm(self, gray: np.ndarray, distances=[1, 2, 3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
        """Compute Gray-Level Co-occurrence Matrix features."""
        try:
            from skimage.feature import graycomatrix, graycoprops

            # Compute GLCM
            glcm = graycomatrix(gray, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)

            # Compute properties
            properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
            glcm_features = {}

            for prop in properties:
                glcm_features[prop] = graycoprops(glcm, prop).flatten()

            return glcm_features

        except ImportError:
            return {}

    def _extract_edge_features(self, gray: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract edge features."""
        # Canny edges
        edges = cv2.Canny(gray, 50, 150)

        # Edge density
        edge_density = np.sum(edges > 0) / edges.size

        # Gradient features
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        gradient_orientation = np.arctan2(grad_y, grad_x)

        return {
            'edges': edges,
            'edge_density': edge_density,
            'gradient_magnitude': gradient_magnitude,
            'gradient_orientation': gradient_orientation
        }

    def _extract_statistical_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract statistical features."""
        features = {}

        # Basic statistics
        features['mean'] = float(np.mean(image))
        features['std'] = float(np.std(image))
        features['min'] = float(np.min(image))
        features['max'] = float(np.max(image))
        features['median'] = float(np.median(image))

        # Higher order moments
        features['skewness'] = float(self._skewness(image))
        features['kurtosis'] = float(self._kurtosis(image))

        return features

    def _skewness(self, x: np.ndarray) -> float:
        """Calculate skewness."""
        n = x.size
        mean = np.mean(x)
        std = np.std(x)
        return np.sum(((x - mean) / std) ** 3) / n

    def _kurtosis(self, x: np.ndarray) -> float:
        """Calculate kurtosis."""
        n = x.size
        mean = np.mean(x)
        std = np.std(x)
        return np.sum(((x - mean) / std) ** 4) / n - 3

    def _extract_hog_features(self, gray: np.ndarray) -> np.ndarray:
        """Extract Histogram of Oriented Gradients features."""
        try:
            from skimage.feature import hog

            # Compute HOG
            hog_features = hog(
                gray,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                block_norm='L2-Hys',
                visualize=False,
                feature_vector=True
            )

            return hog_features

        except ImportError:
            logger.warning("skimage not available, skipping HOG features")
            return np.array([])


class ImageProcessor:
    """Main image processing class."""

    def __init__(self, config: Optional[ImageConfig] = None):
        self.config = config or ImageConfig()

        # Initialize processing components
        self.enhancer = ImageEnhancer(self.config)
        self.detector = YOLODetector(self.config)
        self.segmentator = SemanticSegmentator(self.config)
        self.feature_extractor = FeatureExtractor(self.config)

        # Thread pool for batch processing
        self.num_workers = self.config.num_workers
        self.processing_queue = queue.Queue()

    def process(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Process image with full pipeline.

        Args:
            image: Input image (HxWxC)

        Returns:
            Dictionary containing processed results
        """
        start_time = time.time()

        try:
            # Validate input
            if image is None or image.size == 0:
                raise ValueError("Invalid input image")

            # Step 1: Image enhancement
            enhanced = self.enhancer.enhance(image)

            # Step 2: Object detection
            detections = self.detector.detect(enhanced)

            # Step 3: Semantic segmentation
            segmentation = self.segmentator.segment(enhanced)

            # Step 4: Feature extraction
            features = self.feature_extractor.extract_features(enhanced)

            # Step 5: Quality assessment
            quality_score = self._assess_quality(enhanced, image)

            processing_time = time.time() - start_time

            result = {
                'original_image': image,
                'enhanced_image': enhanced,
                'detections': detections,
                'segmentation': segmentation,
                'features': features,
                'quality_score': quality_score,
                'processing_time': processing_time,
                'image_shape': image.shape
            }

            logger.info(f"Processed image in {processing_time:.3f}s")
            return result

        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise

    def process_batch(self, images: List[np.ndarray]) -> List[Dict[str, Any]]:
        """
        Process multiple images in batch.

        Args:
            images: List of input images

        Returns:
            List of processing results
        """
        results = []

        for image in images:
            result = self.process(image)
            results.append(result)

        return results

    def _assess_quality(self, enhanced: np.ndarray, original: np.ndarray) -> float:
        """
        Assess quality of processed image.

        Args:
            enhanced: Enhanced image
            original: Original image

        Returns:
            Quality score between 0 and 1
        """
        try:
            # Convert to grayscale for quality assessment
            if len(original.shape) == 3:
                orig_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
                enh_gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
            else:
                orig_gray = original
                enh_gray = enhanced

            # Structural Similarity Index
            from skimage.metrics import structural_similarity as ssim
            ssim_score = ssim(enh_gray, orig_gray, data_range=255)

            # Contrast improvement
            orig_contrast = np.std(orig_gray)
            enh_contrast = np.std(enh_gray)
            contrast_improvement = min(enh_contrast / (orig_contrast + 1e-7), 2.0) / 2.0

            # Sharpness (Laplacian variance)
            orig_sharpness = cv2.Laplacian(orig_gray, cv2.CV_64F).var()
            enh_sharpness = cv2.Laplacian(enh_gray, cv2.CV_64F).var()
            sharpness_improvement = min(enh_sharpness / (orig_sharpness + 1e-7), 2.0) / 2.0

            # Combine metrics
            quality = 0.5 * ssim_score + 0.25 * contrast_improvement + 0.25 * sharpness_improvement

            return float(np.clip(quality, 0.0, 1.0))

        except ImportError:
            logger.warning("skimage not available, using simple quality metrics")
            # Simple fallback: use contrast and sharpness only
            if len(original.shape) == 3:
                orig_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
                enh_gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
            else:
                orig_gray = original
                enh_gray = enhanced

            orig_std = np.std(orig_gray)
            enh_std = np.std(enh_gray)
            contrast_ratio = min(enh_std / (orig_std + 1e-7), 2.0) / 2.0

            return float(np.clip(contrast_ratio, 0.0, 1.0))

    def set_config(self, config: ImageConfig):
        """Update processing configuration."""
        self.config = config

        # Reinitialize components with new config
        self.enhancer = ImageEnhancer(config)
        self.detector = YOLODetector(config)
        self.segmentator = SemanticSegmentator(config)
        self.feature_extractor = FeatureExtractor(config)