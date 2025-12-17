"""
Point Cloud Processing Module for World Model

This module provides comprehensive point cloud processing capabilities including
voxel downsampling, statistical outlier removal, ICP registration, and feature extraction.
"""

import numpy as np
import open3d as o3d
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
import time
import logging
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


@dataclass
class PointCloudConfig:
    """Configuration for point cloud processing parameters."""
    voxel_size: float = 0.05
    outlier_nb_neighbors: int = 20
    outlier_std_ratio: float = 2.0
    icp_max_distance: float = 0.5
    icp_max_iterations: int = 50
    feature_radius: float = 0.1
    normal_radius: float = 0.05
    use_gpu: bool = True
    batch_size: int = 10000


class VoxelGridSampler:
    """Voxel grid downsampling for point clouds."""

    def __init__(self, voxel_size: float = 0.05):
        self.voxel_size = voxel_size

    def sample(self, points: np.ndarray) -> np.ndarray:
        """
        Downsample point cloud using voxel grid.

        Args:
            points: Nx3 array of 3D points

        Returns:
            Downsampled point cloud
        """
        if len(points) == 0:
            return points

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        downsampled = pcd.voxel_down_sample(self.voxel_size)
        return np.asarray(downsampled.points)

    def sample_batch(self, points_list: List[np.ndarray]) -> List[np.ndarray]:
        """Batch process multiple point clouds."""
        return [self.sample(points) for points in points_list]


class StatisticalOutlierRemover:
    """Statistical outlier removal for point clouds."""

    def __init__(self, nb_neighbors: int = 20, std_ratio: float = 2.0):
        self.nb_neighbors = nb_neighbors
        self.std_ratio = std_ratio

    def filter(self, points: np.ndarray) -> np.ndarray:
        """
        Remove statistical outliers from point cloud.

        Args:
            points: Nx3 array of 3D points

        Returns:
            Filtered point cloud
        """
        if len(points) < self.nb_neighbors:
            return points

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        cl, ind = pcd.remove_statistical_outlier(
            self.nb_neighbors, self.std_ratio
        )

        return np.asarray(cl.points)

    def filter_batch(self, points_list: List[np.ndarray]) -> List[np.ndarray]:
        """Batch process multiple point clouds."""
        return [self.filter(points) for points in points_list]


class ICPRegistration:
    """Iterative Closest Point registration for point cloud alignment."""

    def __init__(self, max_distance: float = 0.5, max_iterations: int = 50):
        self.max_distance = max_distance
        self.max_iterations = max_iterations
        self.transformation_matrix = np.eye(4)

    def register(self, source: np.ndarray, target: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Register point cloud to target or global coordinate system.

        Args:
            source: Source point cloud
            target: Target point cloud (optional)

        Returns:
            Registered point cloud
        """
        if len(source) == 0:
            return source

        source_pcd = o3d.geometry.PointCloud()
        source_pcd.points = o3d.utility.Vector3dVector(source)

        if target is not None and len(target) > 0:
            target_pcd = o3d.geometry.PointCloud()
            target_pcd.points = o3d.utility.Vector3dVector(target)

            # Use ICP registration
            reg_p2p = o3d.pipelines.registration.registration_icp(
                source_pcd, target_pcd, self.max_distance,
                np.eye(4),  # Initial transformation
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=self.max_iterations
                )
            )

            source_pcd.transform(reg_p2p.transformation)
            self.transformation_matrix = reg_p2p.transformation

        return np.asarray(source_pcd.points)

    def get_transformation_matrix(self) -> np.ndarray:
        """Get the last computed transformation matrix."""
        return self.transformation_matrix


class FeatureExtractor:
    """Feature extraction for point clouds."""

    def __init__(self, feature_radius: float = 0.1, normal_radius: float = 0.05):
        self.feature_radius = feature_radius
        self.normal_radius = normal_radius

    def extract_features(self, points: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract geometric features from point cloud.

        Args:
            points: Nx3 array of 3D points

        Returns:
            Dictionary containing extracted features
        """
        if len(points) == 0:
            return {}

        features = {}

        # Surface normals
        features['normals'] = self._compute_normals(points)

        # Curvature features
        features['curvature'] = self._compute_curvature(points)

        # Local density
        features['density'] = self._compute_local_density(points)

        # PCA features
        features['pca'] = self._compute_pca_features(points)

        # FPFH features (if enough points)
        if len(points) > 100:
            features['fpfh'] = self._compute_fpfh(points)

        return features

    def _compute_normals(self, points: np.ndarray) -> np.ndarray:
        """Compute surface normals."""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Estimate normals
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30)
        )

        normals = np.asarray(pcd.normals)
        return normals

    def _compute_curvature(self, points: np.ndarray) -> np.ndarray:
        """Compute curvature at each point."""
        normals = self._compute_normals(points)

        # Use change in normal direction as curvature approximation
        tree = cKDTree(points)
        curvature = np.zeros(len(points))

        for i in range(len(points)):
            if len(points) > 30:
                _, idx = tree.query(points[i], k=30)
            else:
                idx = np.arange(len(points))

            if len(idx) > 1:
                normal_variance = np.var(normals[idx], axis=0)
                curvature[i] = np.linalg.norm(normal_variance)

        return curvature

    def _compute_local_density(self, points: np.ndarray, k: int = 20) -> np.ndarray:
        """Compute local point density."""
        tree = cKDTree(points)
        densities = np.zeros(len(points))

        if len(points) > k:
            distances, _ = tree.query(points, k=k+1)  # +1 to include the point itself
            volumes = (4/3) * np.pi * np.power(distances[:, -1], 3)
            densities = k / volumes
        else:
            densities[:] = len(points) / 1.0  # Uniform density for small clouds

        return densities

    def _compute_pca_features(self, points: np.ndarray) -> np.ndarray:
        """Compute PCA-based geometric features."""
        pca_features = []

        # Compute global PCA
        global_pca = PCA(n_components=3)
        global_pca.fit(points)

        # Local PCA for each point
        tree = cKDTree(points)
        k = min(30, len(points))

        for i in range(len(points)):
            if len(points) > k:
                _, idx = tree.query(points[i], k=k)
            else:
                idx = np.arange(len(points))

            local_points = points[idx]
            local_pca = PCA(n_components=3)
            local_pca.fit(local_points)

            # Concatenate eigenvalues and global contribution
            features = np.concatenate([
                local_pca.explained_variance_,
                local_pca.components_.flatten(),
                global_pca.explained_variance_
            ])

            pca_features.append(features)

        return np.array(pca_features)

    def _compute_fpfh(self, points: np.ndarray) -> np.ndarray:
        """Compute Fast Point Feature Histograms (FPFH)."""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Estimate normals first
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30)
        )

        # Compute FPFH features
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd,
            o3d.geometry.KDTreeSearchParamRadius(radius=self.feature_radius)
        )

        return np.asarray(fpfh.data).T


class PointCloudProcessor:
    """Main point cloud processing class."""

    def __init__(self, config: Optional[PointCloudConfig] = None):
        self.config = config or PointCloudConfig()

        # Initialize processing components
        self.downsampler = VoxelGridSampler(self.config.voxel_size)
        self.denoiser = StatisticalOutlierRemover(
            self.config.outlier_nb_neighbors,
            self.config.outlier_std_ratio
        )
        self.registrator = ICPRegistration(
            self.config.icp_max_distance,
            self.config.icp_max_iterations
        )
        self.feature_extractor = FeatureExtractor(
            self.config.feature_radius,
            self.config.normal_radius
        )

        # Check GPU availability
        self.device = "cuda" if self.config.use_gpu and o3d.core.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

    def process(self, points: np.ndarray, target: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Process point cloud with full pipeline.

        Args:
            points: Input point cloud (Nx3)
            target: Optional target point cloud for registration

        Returns:
            Dictionary containing processed results
        """
        start_time = time.time()

        try:
            # Step 1: Voxel downsampling
            downsampled = self.downsampler.sample(points)
            logger.debug(f"Downsampled from {len(points)} to {len(downsampled)} points")

            # Step 2: Statistical outlier removal
            filtered = self.denoiser.filter(downsampled)
            logger.debug(f"Filtered from {len(downsampled)} to {len(filtered)} points")

            # Step 3: ICP registration
            registered = self.registrator.register(filtered, target)

            # Step 4: Feature extraction
            features = self.feature_extractor.extract_features(registered)

            # Step 5: Quality assessment
            quality_score = self._assess_quality(registered, points)

            processing_time = time.time() - start_time

            result = {
                'points': registered,
                'features': features,
                'quality_score': quality_score,
                'processing_time': processing_time,
                'original_count': len(points),
                'final_count': len(registered),
                'transformation_matrix': self.registrator.get_transformation_matrix()
            }

            logger.info(f"Processed point cloud in {processing_time:.3f}s")
            return result

        except Exception as e:
            logger.error(f"Error processing point cloud: {str(e)}")
            raise

    def process_batch(self, points_list: List[np.ndarray],
                     targets: Optional[List[np.ndarray]] = None) -> List[Dict[str, Any]]:
        """
        Process multiple point clouds in batch.

        Args:
            points_list: List of point clouds
            targets: Optional list of target point clouds

        Returns:
            List of processing results
        """
        results = []

        for i, points in enumerate(points_list):
            target = targets[i] if targets and i < len(targets) else None
            result = self.process(points, target)
            results.append(result)

        return results

    def _assess_quality(self, processed_points: np.ndarray,
                       original_points: np.ndarray) -> float:
        """
        Assess quality of processed point cloud.

        Args:
            processed_points: Processed point cloud
            original_points: Original point cloud

        Returns:
            Quality score between 0 and 1
        """
        if len(processed_points) == 0:
            return 0.0

        # Point preservation ratio
        preservation_ratio = len(processed_points) / max(len(original_points), 1)

        # Density uniformity
        tree = cKDTree(processed_points)
        k = min(20, len(processed_points))
        if len(processed_points) > k:
            distances, _ = tree.query(processed_points, k=k)
            density_std = np.std(distances[:, -1])
            density_uniformity = np.exp(-density_std)
        else:
            density_uniformity = 0.5

        # Spatial coverage
        if len(processed_points) > 10:
            pca = PCA(n_components=3)
            pca.fit(processed_points)
            coverage = np.sum(pca.explained_variance_ratio_)
        else:
            coverage = 0.5

        # Combine metrics
        quality = 0.4 * min(preservation_ratio, 1.0) + 0.3 * density_uniformity + 0.3 * coverage

        return float(np.clip(quality, 0.0, 1.0))

    def set_config(self, config: PointCloudConfig):
        """Update processing configuration."""
        self.config = config

        # Reinitialize components with new config
        self.downsampler = VoxelGridSampler(config.voxel_size)
        self.denoiser = StatisticalOutlierRemover(
            config.outlier_nb_neighbors,
            config.outlier_std_ratio
        )
        self.registrator = ICPRegistration(
            config.icp_max_distance,
            config.icp_max_iterations
        )
        self.feature_extractor = FeatureExtractor(
            config.feature_radius,
            config.normal_radius
        )