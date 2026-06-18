"""
Interactive 3D morphometry engine for FOCUS-3D.

This module is designed to be called from the napari UI instead of from a
standalone command-line script. It accepts raw/label arrays directly, computes
only the feature groups selected by the user, and saves CSV + feature-mapped
TIFF outputs.

Main entry point:
    run_morphometry_from_arrays(raw_data, label_data, out_dir, config)

Dependencies:
    pip install numpy pandas scipy scikit-learn tifffile scikit-image
"""

from __future__ import annotations

import json
import traceback
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff
from scipy.spatial import cKDTree
from skimage.segmentation import relabel_sequential
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler

ProgressCallback = Callable[[int, str], None] | None
CancelCallback = Callable[[], bool] | None

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------


@dataclass
class MorphometryConfig:
    """User-controllable morphometry options.

    All voxel sizes are ordered as (Z, Y, X) in physical units, usually um.
    """

    voxel_size_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0)

    # Feature groups selected by the user.
    compute_morphology: bool = True
    compute_intensity: bool = True
    compute_neighborhood: bool = True
    compute_contact: bool = False
    compute_regions: bool = True
    compute_clustering: bool = True
    compute_anomaly: bool = True

    # Shape details.
    compute_pca_axes: bool = True
    max_voxels_for_exact_pca: int = 200_000

    # Neighborhood settings.
    neighbor_mode: str = (
        'knn_radius'  # "knn", "radius", "knn_radius", "contact"
    )
    knn_k_values: tuple[int, ...] = (5, 10, 20)
    radius_values_um: tuple[float, ...] = (20.0, 50.0)
    local_k: int = 10

    # Region statistics.
    grid_shape_zyx: tuple[int, int, int] = (6, 6, 6)

    # Clustering.
    cluster_feature_set: str = 'shape'  # "shape", "shape_neighborhood", "all"
    n_clusters: int = 6

    # Anomaly detection.
    anomaly_contamination: float = 0.02
    random_state: int = 0

    # Output.
    save_csv: bool = True
    save_feature_tifs: bool = True
    feature_map_mode: str = (
        'selected_groups'  # "core", "selected_groups", "all"
    )
    selected_map_features: tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------


def _emit(
    progress_callback: ProgressCallback, value: int, message: str
) -> None:
    if progress_callback is not None:
        progress_callback(int(value), str(message))


def _check_cancel(cancel_callback: CancelCallback) -> None:
    if cancel_callback is not None and cancel_callback():
        raise RuntimeError('__MORPHOMETRY_CANCELLED__')


def _to_numpy_3d(data, name: str) -> np.ndarray:
    """Convert numpy/dask/zarr-like data to a squeezed 3D numpy array."""
    if data is None:
        return None

    arr = data.compute() if hasattr(data, 'compute') else np.asarray(data)

    arr = np.squeeze(arr)

    if arr.ndim != 3:
        raise ValueError(
            f'{name} must be 3D after squeeze, got shape={arr.shape}'
        )

    return np.asarray(arr)


def _safe_int_dtype(label: np.ndarray) -> np.ndarray:
    """Ensure label image is integer-valued."""
    if not np.issubdtype(label.dtype, np.integer):
        label = label.astype(np.int64)
    return label


def relabel_to_contiguous(label: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Relabel arbitrary positive IDs to contiguous 1..N IDs for fast computation."""
    label = _safe_int_dtype(label)
    label_new, _, _ = relabel_sequential(label)
    original_ids = np.unique(label)
    original_ids = original_ids[original_ids > 0]
    return label_new.astype(np.int32, copy=False), original_ids.astype(
        np.int64
    )


def save_json(obj: dict, path: Path) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------
# Basic morphology
# ---------------------------------------------------------------------


def compute_basic_features(
    label: np.ndarray, voxel_size: Sequence[float]
) -> pd.DataFrame:
    """Compute ID, size, centroid, and internal bounding boxes.

    Bounding boxes are used internally for fast per-object operations, but are
    removed from the public cell_features.csv.
    """
    z_size, y_size, x_size = map(float, voxel_size)
    voxel_volume = z_size * y_size * x_size

    n_labels = int(label.max())
    counts = np.bincount(label.ravel(), minlength=n_labels + 1).astype(
        np.float64
    )

    fg = label > 0
    z, y, x = np.nonzero(fg)
    ids = label[fg].astype(np.int64)

    sum_z = np.bincount(ids, weights=z, minlength=n_labels + 1)
    sum_y = np.bincount(ids, weights=y, minlength=n_labels + 1)
    sum_x = np.bincount(ids, weights=x, minlength=n_labels + 1)

    with np.errstate(divide='ignore', invalid='ignore'):
        cz = sum_z / counts
        cy = sum_y / counts
        cx = sum_x / counts

    min_z = np.full(n_labels + 1, np.inf)
    min_y = np.full(n_labels + 1, np.inf)
    min_x = np.full(n_labels + 1, np.inf)
    max_z = np.full(n_labels + 1, -np.inf)
    max_y = np.full(n_labels + 1, -np.inf)
    max_x = np.full(n_labels + 1, -np.inf)

    np.minimum.at(min_z, ids, z)
    np.minimum.at(min_y, ids, y)
    np.minimum.at(min_x, ids, x)
    np.maximum.at(max_z, ids, z)
    np.maximum.at(max_y, ids, y)
    np.maximum.at(max_x, ids, x)

    rows = []
    for lab in range(1, n_labels + 1):
        if counts[lab] <= 0:
            continue

        volume_um3 = counts[lab] * voxel_volume
        rows.append(
            {
                'label_contiguous': lab,
                'voxel_count': int(counts[lab]),
                'volume_um3': float(volume_um3),
                'equivalent_diameter_um': float(
                    (6.0 * volume_um3 / np.pi) ** (1.0 / 3.0)
                ),
                'centroid_z_vox': float(cz[lab]),
                'centroid_y_vox': float(cy[lab]),
                'centroid_x_vox': float(cx[lab]),
                'centroid_z_um': float(cz[lab] * z_size),
                'centroid_y_um': float(cy[lab] * y_size),
                'centroid_x_um': float(cx[lab] * x_size),
                'bbox_z_min': int(min_z[lab]),
                'bbox_y_min': int(min_y[lab]),
                'bbox_x_min': int(min_x[lab]),
                'bbox_z_max': int(max_z[lab]),
                'bbox_y_max': int(max_y[lab]),
                'bbox_x_max': int(max_x[lab]),
                'touch_border': bool(
                    min_z[lab] == 0
                    or min_y[lab] == 0
                    or min_x[lab] == 0
                    or max_z[lab] == label.shape[0] - 1
                    or max_y[lab] == label.shape[1] - 1
                    or max_x[lab] == label.shape[2] - 1
                ),
            }
        )

    return pd.DataFrame(rows)


def compute_surface_and_contact(
    label: np.ndarray,
    voxel_size: Sequence[float],
    compute_contact: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute voxel-face surface area and optional contact edges.

    Surface area is estimated from exposed voxel faces. It uses physical face
    areas derived from voxel_size, but it remains grid-dependent and should be
    interpreted carefully for strongly anisotropic data.
    """
    z_size, y_size, x_size = map(float, voxel_size)
    face_area = {
        0: y_size * x_size,  # face perpendicular to Z
        1: z_size * x_size,  # face perpendicular to Y
        2: z_size * y_size,  # face perpendicular to X
    }

    n_labels = int(label.max())
    surface_area = np.zeros(n_labels + 1, dtype=np.float64)
    contact: dict[tuple[int, int], float] = {}

    # Image-boundary exposed faces.
    for axis in range(3):
        area = face_area[axis]
        sl0 = [slice(None)] * 3
        sl1 = [slice(None)] * 3
        sl0[axis] = 0
        sl1[axis] = -1
        a0 = label[tuple(sl0)]
        a1 = label[tuple(sl1)]
        surface_area += (
            np.bincount(a0[a0 > 0].ravel(), minlength=n_labels + 1) * area
        )
        surface_area += (
            np.bincount(a1[a1 > 0].ravel(), minlength=n_labels + 1) * area
        )

    # Internal faces.
    for axis in range(3):
        area = face_area[axis]
        sla = [slice(None)] * 3
        slb = [slice(None)] * 3
        sla[axis] = slice(0, -1)
        slb[axis] = slice(1, None)

        a = label[tuple(sla)]
        b = label[tuple(slb)]
        diff = a != b

        a_ids = a[diff]
        b_ids = b[diff]

        surface_area += (
            np.bincount(a_ids[a_ids > 0].ravel(), minlength=n_labels + 1)
            * area
        )
        surface_area += (
            np.bincount(b_ids[b_ids > 0].ravel(), minlength=n_labels + 1)
            * area
        )

        if compute_contact:
            both = (a_ids > 0) & (b_ids > 0) & (a_ids != b_ids)
            if np.any(both):
                u = a_ids[both].astype(np.int64)
                v = b_ids[both].astype(np.int64)
                lo = np.minimum(u, v)
                hi = np.maximum(u, v)
                pairs = np.stack([lo, hi], axis=1)
                unique_pairs, counts = np.unique(
                    pairs, axis=0, return_counts=True
                )
                for (p, q), c in zip(unique_pairs, counts, strict=False):
                    key = (int(p), int(q))
                    contact[key] = contact.get(key, 0.0) + float(c) * area

    surface_df = pd.DataFrame(
        {
            'label_contiguous': np.arange(1, n_labels + 1),
            'surface_area_um2': surface_area[1:],
        }
    )

    if len(contact) == 0:
        contact_df = pd.DataFrame(
            columns=['label_a', 'label_b', 'contact_area_um2']
        )
    else:
        contact_df = pd.DataFrame(
            [
                {'label_a': a, 'label_b': b, 'contact_area_um2': area}
                for (a, b), area in contact.items()
            ]
        )

    return surface_df, contact_df


def add_surface_shape_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add sphericity and compactness."""
    df = df.copy()
    if 'surface_area_um2' not in df.columns:
        return df

    V = df['volume_um3'].to_numpy(float)
    A = df['surface_area_um2'].to_numpy(float)

    with np.errstate(divide='ignore', invalid='ignore'):
        df['sphericity'] = (
            (np.pi ** (1.0 / 3.0)) * ((6.0 * V) ** (2.0 / 3.0)) / (A + 1e-12)
        )
        df['compactness'] = (A**1.5) / (V + 1e-12)

    return df


def compute_pca_axis_features(
    label: np.ndarray,
    df: pd.DataFrame,
    voxel_size: Sequence[float],
    max_voxels_for_exact_pca: int = 200_000,
    cancel_callback: CancelCallback = None,
) -> pd.DataFrame:
    """Compute PCA-based axis lengths in physical coordinates."""
    z_size, y_size, x_size = map(float, voxel_size)
    records = []

    rng = np.random.default_rng(0)

    for row in df.itertuples(index=False):
        _check_cancel(cancel_callback)
        lab = int(row.label_contiguous)

        z0, y0, x0 = (
            int(row.bbox_z_min),
            int(row.bbox_y_min),
            int(row.bbox_x_min),
        )
        z1, y1, x1 = (
            int(row.bbox_z_max) + 1,
            int(row.bbox_y_max) + 1,
            int(row.bbox_x_max) + 1,
        )

        crop = label[z0:z1, y0:y1, x0:x1] == lab
        coords = np.argwhere(crop)

        if coords.shape[0] < 5:
            records.append(
                {
                    'label_contiguous': lab,
                    'axis_major_um': np.nan,
                    'axis_intermediate_um': np.nan,
                    'axis_minor_um': np.nan,
                    'elongation': np.nan,
                    'flatness': np.nan,
                    'major_minor_ratio': np.nan,
                    'orientation_z': np.nan,
                    'orientation_y': np.nan,
                    'orientation_x': np.nan,
                }
            )
            continue

        if coords.shape[0] > max_voxels_for_exact_pca:
            keep = rng.choice(
                coords.shape[0], size=max_voxels_for_exact_pca, replace=False
            )
            coords = coords[keep]

        coords = coords.astype(np.float64)
        coords[:, 0] = (coords[:, 0] + z0) * z_size
        coords[:, 1] = (coords[:, 1] + y0) * y_size
        coords[:, 2] = (coords[:, 2] + x0) * x_size

        coords -= coords.mean(axis=0, keepdims=True)
        cov = np.cov(coords, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)

        order = np.argsort(eigvals)[::-1]
        eigvals = np.maximum(eigvals[order], 0.0)
        eigvecs = eigvecs[:, order]

        # For a solid ellipsoid, covariance eigenvalue roughly equals semi_axis^2 / 5.
        axis_lengths = 2.0 * np.sqrt(5.0 * eigvals)
        major, inter, minor = axis_lengths

        orientation = eigvecs[:, 0]
        if orientation[np.argmax(np.abs(orientation))] < 0:
            orientation = -orientation

        records.append(
            {
                'label_contiguous': lab,
                'axis_major_um': float(major),
                'axis_intermediate_um': float(inter),
                'axis_minor_um': float(minor),
                'elongation': float(major / (minor + 1e-12)),
                'flatness': float(inter / (minor + 1e-12)),
                'major_minor_ratio': float(major / (minor + 1e-12)),
                'orientation_z': float(orientation[0]),
                'orientation_y': float(orientation[1]),
                'orientation_x': float(orientation[2]),
            }
        )

    return df.merge(pd.DataFrame(records), on='label_contiguous', how='left')


def add_contact_features(
    df: pd.DataFrame, contact_edges_df: pd.DataFrame
) -> pd.DataFrame:
    """Add per-cell contact graph summary features."""
    df = df.copy()
    n_labels = int(df['label_contiguous'].max())

    degree = np.zeros(n_labels + 1, dtype=np.int64)
    total = np.zeros(n_labels + 1, dtype=np.float64)
    maximum = np.zeros(n_labels + 1, dtype=np.float64)

    if contact_edges_df is not None and len(contact_edges_df) > 0:
        for row in contact_edges_df.itertuples(index=False):
            a, b, area = (
                int(row.label_a),
                int(row.label_b),
                float(row.contact_area_um2),
            )
            degree[a] += 1
            degree[b] += 1
            total[a] += area
            total[b] += area
            maximum[a] = max(maximum[a], area)
            maximum[b] = max(maximum[b], area)

    lab = df['label_contiguous'].to_numpy(int)
    df['contact_neighbor_count'] = degree[lab]
    df['contact_area_total_um2'] = total[lab]
    df['contact_area_mean_um2'] = total[lab] / (degree[lab] + 1e-12)
    df['contact_area_max_um2'] = maximum[lab]
    if 'surface_area_um2' in df.columns:
        df['contact_fraction'] = total[lab] / (
            df['surface_area_um2'].to_numpy(float) + 1e-12
        )
    else:
        df['contact_fraction'] = np.nan
    return df


# ---------------------------------------------------------------------
# Intensity features
# ---------------------------------------------------------------------


def compute_intensity_features(
    raw: np.ndarray, label: np.ndarray
) -> pd.DataFrame:
    """Compute per-instance intensity moment features."""
    if raw.shape != label.shape:
        raise ValueError(
            f'raw and label shapes do not match: raw={raw.shape}, label={label.shape}'
        )

    raw = raw.astype(np.float64, copy=False)
    n_labels = int(label.max())
    fg = label > 0
    ids = label[fg].astype(np.int64)
    vals = raw[fg]

    counts = np.bincount(ids, minlength=n_labels + 1).astype(np.float64)
    s1 = np.bincount(ids, weights=vals, minlength=n_labels + 1)
    s2 = np.bincount(ids, weights=vals**2, minlength=n_labels + 1)
    s3 = np.bincount(ids, weights=vals**3, minlength=n_labels + 1)
    s4 = np.bincount(ids, weights=vals**4, minlength=n_labels + 1)

    with np.errstate(divide='ignore', invalid='ignore'):
        mean = s1 / counts
        e2 = s2 / counts
        e3 = s3 / counts
        e4 = s4 / counts
        var = np.maximum(e2 - mean**2, 0.0)
        std = np.sqrt(var)
        cv = std / (np.abs(mean) + 1e-12)
        m3 = e3 - 3 * mean * e2 + 2 * mean**3
        m4 = e4 - 4 * mean * e3 + 6 * mean**2 * e2 - 3 * mean**4
        skew = m3 / (std**3 + 1e-12)
        kurtosis_excess = m4 / (var**2 + 1e-12) - 3.0

    min_int = np.full(n_labels + 1, np.inf)
    max_int = np.full(n_labels + 1, -np.inf)
    np.minimum.at(min_int, ids, vals)
    np.maximum.at(max_int, ids, vals)

    return pd.DataFrame(
        {
            'label_contiguous': np.arange(1, n_labels + 1),
            'mean_intensity': mean[1:],
            'std_intensity': std[1:],
            'cv_intensity': cv[1:],
            'min_intensity': min_int[1:],
            'max_intensity': max_int[1:],
            'sum_intensity': s1[1:],
            'skewness_intensity': skew[1:],
            'kurtosis_intensity': kurtosis_excess[1:],
        }
    )


# ---------------------------------------------------------------------
# Neighborhood features
# ---------------------------------------------------------------------


def compute_neighborhood_features(
    df: pd.DataFrame,
    neighbor_mode: str = 'knn_radius',
    k_values: Sequence[int] = (5, 10, 20),
    radius_values_um: Sequence[float] = (20.0, 50.0),
    local_k: int = 10,
) -> pd.DataFrame:
    """Compute centroid-based neighborhood features.

    neighbor_mode:
        "knn": compute kNN distance and kNN density.
        "radius": compute radius-based count and density.
        "knn_radius": compute both.
        "contact": skip centroid-based features; contact features are computed separately.
    """
    df = df.copy()
    neighbor_mode = str(neighbor_mode).lower()

    if neighbor_mode == 'contact':
        return df

    coords = df[['centroid_z_um', 'centroid_y_um', 'centroid_x_um']].to_numpy(
        float
    )
    n = coords.shape[0]

    if n <= 1:
        df['nearest_neighbor_distance_um'] = np.nan
        return df

    tree = cKDTree(coords)

    do_knn = neighbor_mode in {'knn', 'knn_radius', 'all'}
    do_radius = neighbor_mode in {'radius', 'knn_radius', 'all'}

    if do_knn:
        k_values = tuple(sorted({int(k) for k in k_values if int(k) > 0}))
        if not k_values:
            k_values = (10,)

        max_k = min(max(max(k_values), int(local_k)) + 1, n)
        dists, idx = tree.query(coords, k=max_k, workers=-1)
        if dists.ndim == 1:
            dists = dists[:, None]
            idx = idx[:, None]

        neighbor_dists = dists[:, 1:]
        neighbor_idx = idx[:, 1:]

        df['nearest_neighbor_distance_um'] = (
            neighbor_dists[:, 0] if neighbor_dists.shape[1] > 0 else np.nan
        )

        for k in k_values:
            kk = min(int(k), neighbor_dists.shape[1])
            if kk <= 0:
                continue
            dk = neighbor_dists[:, :kk]
            r_k = dk[:, -1]
            sphere_volume = (4.0 / 3.0) * np.pi * (r_k**3)
            df[f'knn_mean_distance_k{k}_um'] = np.mean(dk, axis=1)
            df[f'knn_median_distance_k{k}_um'] = np.median(dk, axis=1)
            df[f'local_density_k{k}_per_um3'] = kk / (sphere_volume + 1e-12)

        kk = min(int(local_k), neighbor_idx.shape[1])
        if kk > 0:
            for feature in ['volume_um3', 'sphericity', 'elongation']:
                if feature not in df.columns:
                    continue
                vals = df[feature].to_numpy(float)
                neigh_vals = vals[neighbor_idx[:, :kk]]
                neigh_mean = np.nanmean(neigh_vals, axis=1)
                neigh_std = np.nanstd(neigh_vals, axis=1)
                df[f'neighbor_mean_{feature}'] = neigh_mean
                df[f'neighbor_std_{feature}'] = neigh_std
                df[f'self_vs_neighbor_{feature}_zscore'] = (
                    vals - neigh_mean
                ) / (neigh_std + 1e-12)

    if do_radius:
        radius_values_um = tuple(
            float(r) for r in radius_values_um if float(r) > 0
        )
        for r in radius_values_um:
            counts = np.array(
                [len(v) - 1 for v in tree.query_ball_point(coords, r=r)]
            )
            sphere_volume = (4.0 / 3.0) * np.pi * (r**3)
            r_name = f'{r:g}'
            df[f'radius_neighbor_count_r{r_name}_um'] = counts
            df[f'radius_density_r{r_name}_per_um3'] = counts / (
                sphere_volume + 1e-12
            )

    return df


# ---------------------------------------------------------------------
# Region statistics
# ---------------------------------------------------------------------


def assign_spatial_regions(
    df: pd.DataFrame,
    label_shape: Sequence[int],
    voxel_size: Sequence[float],
    grid_shape: Sequence[int],
) -> pd.DataFrame:
    """Assign cells to a regular 3D grid based on centroid coordinates."""
    df = df.copy()
    gz, gy, gx = [max(1, int(v)) for v in grid_shape]
    z_size, y_size, x_size = map(float, voxel_size)

    ext_z = label_shape[0] * z_size
    ext_y = label_shape[1] * y_size
    ext_x = label_shape[2] * x_size

    bz = np.floor(df['centroid_z_um'].to_numpy(float) / (ext_z / gz)).astype(
        int
    )
    by = np.floor(df['centroid_y_um'].to_numpy(float) / (ext_y / gy)).astype(
        int
    )
    bx = np.floor(df['centroid_x_um'].to_numpy(float) / (ext_x / gx)).astype(
        int
    )

    bz = np.clip(bz, 0, gz - 1)
    by = np.clip(by, 0, gy - 1)
    bx = np.clip(bx, 0, gx - 1)

    df['region_z_bin'] = bz
    df['region_y_bin'] = by
    df['region_x_bin'] = bx
    df['region_id'] = [
        f'z{z:02d}_y{y:02d}_x{x:02d}'
        for z, y, x in zip(bz, by, bx, strict=False)
    ]
    return df


def compute_region_features(
    df: pd.DataFrame,
    label_shape: Sequence[int],
    voxel_size: Sequence[float],
    grid_shape: Sequence[int],
) -> pd.DataFrame:
    """Compute per-region summary statistics."""
    z_size, y_size, x_size = map(float, voxel_size)
    gz, gy, gx = [max(1, int(v)) for v in grid_shape]
    image_volume_um3 = (
        label_shape[0]
        * z_size
        * label_shape[1]
        * y_size
        * label_shape[2]
        * x_size
    )
    region_volume_um3 = image_volume_um3 / (gz * gy * gx)

    summary_features = [
        'volume_um3',
        'equivalent_diameter_um',
        'sphericity',
        'elongation',
        'flatness',
        'nearest_neighbor_distance_um',
        'anomaly_score',
    ]
    summary_features += [
        c
        for c in df.columns
        if c.startswith('local_density_k') and c.endswith('_per_um3')
    ]

    rows = []
    for region_id, g in df.groupby('region_id', observed=True):
        row = {
            'region_id': region_id,
            'region_cell_count': int(len(g)),
            'region_volume_um3': float(region_volume_um3),
            'region_cell_density_per_um3': float(
                len(g) / (region_volume_um3 + 1e-12)
            ),
        }

        for feature in summary_features:
            if feature not in g.columns:
                continue
            vals = g[feature].to_numpy(float)
            row[f'region_mean_{feature}'] = float(np.nanmean(vals))
            row[f'region_median_{feature}'] = float(np.nanmedian(vals))
            row[f'region_std_{feature}'] = float(np.nanstd(vals))

        if 'is_anomaly' in g.columns:
            row['region_outlier_fraction'] = float(
                np.mean(g['is_anomaly'].to_numpy(bool))
            )

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Clustering and anomaly
# ---------------------------------------------------------------------


def robust_zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    return 0.6745 * (x - med) / (mad + 1e-12)


def normalize_score(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    valid = np.isfinite(x)
    out = np.zeros_like(x, dtype=float)
    if not np.any(valid):
        return out
    lo, hi = np.nanpercentile(x[valid], [1, 99])
    out[valid] = np.clip((x[valid] - lo) / (hi - lo + 1e-12), 0.0, 1.0)
    return out


def choose_feature_columns(
    df: pd.DataFrame, feature_set: str, include_intensity: bool
) -> list[str]:
    """Choose feature columns for clustering/anomaly detection."""
    shape_cols = [
        'volume_um3',
        'equivalent_diameter_um',
        'surface_area_um2',
        'sphericity',
        'compactness',
        'elongation',
        'flatness',
        'axis_major_um',
        'axis_intermediate_um',
        'axis_minor_um',
        'major_minor_ratio',
    ]

    neighborhood_cols = [
        'nearest_neighbor_distance_um',
        'neighbor_mean_volume_um3',
        'self_vs_neighbor_volume_um3_zscore',
        'self_vs_neighbor_sphericity_zscore',
        'self_vs_neighbor_elongation_zscore',
    ] + [
        c
        for c in df.columns
        if c.startswith('local_density_k') and c.endswith('_per_um3')
    ]

    contact_cols = [
        'contact_neighbor_count',
        'contact_area_total_um2',
        'contact_area_mean_um2',
        'contact_fraction',
    ]

    intensity_cols = [
        'mean_intensity',
        'std_intensity',
        'cv_intensity',
        'sum_intensity',
        'skewness_intensity',
        'kurtosis_intensity',
    ]

    if feature_set == 'shape':
        candidates = shape_cols
    elif feature_set == 'shape_neighborhood':
        candidates = shape_cols + neighborhood_cols
    elif feature_set == 'all':
        candidates = (
            shape_cols
            + neighborhood_cols
            + contact_cols
            + (intensity_cols if include_intensity else [])
        )
    else:
        raise ValueError(
            "cluster_feature_set must be 'shape', 'shape_neighborhood', or 'all'."
        )

    return [
        c
        for c in candidates
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]


def prepare_feature_matrix(
    df: pd.DataFrame, feature_cols: Sequence[str]
) -> tuple[np.ndarray, list[str]]:
    X_df = df[list(feature_cols)].copy()

    log_like = [
        c
        for c in X_df.columns
        if any(
            key in c
            for key in [
                'volume',
                'diameter',
                'area',
                'axis',
                'distance',
                'density',
                'intensity',
            ]
        )
    ]

    for c in log_like:
        vals = X_df[c].to_numpy(float)
        if np.nanmin(vals) >= 0:
            X_df[c] = np.log1p(vals)

    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    for c in X_df.columns:
        med = X_df[c].median()
        if not np.isfinite(med):
            med = 0.0
        X_df[c] = X_df[c].fillna(med)

    X = RobustScaler().fit_transform(X_df.to_numpy(float))
    return X, list(X_df.columns)


def add_clustering(
    df: pd.DataFrame,
    feature_set: str,
    n_clusters: int,
    include_intensity: bool,
    random_state: int,
) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    feature_cols = choose_feature_columns(
        df, feature_set=feature_set, include_intensity=include_intensity
    )

    if len(feature_cols) < 2 or len(df) < max(3, int(n_clusters)):
        df['cluster_id'] = -1
        return df, feature_cols

    X, used_cols = prepare_feature_matrix(df, feature_cols)
    n_pcs = min(10, X.shape[0], X.shape[1])
    pcs = PCA(n_components=n_pcs, random_state=random_state).fit_transform(X)

    for i in range(min(5, n_pcs)):
        df[f'PC{i + 1}'] = pcs[:, i]

    km = KMeans(
        n_clusters=int(n_clusters), random_state=random_state, n_init='auto'
    )
    df['cluster_id'] = km.fit_predict(pcs[:, : min(5, n_pcs)])
    return df, used_cols


def add_anomaly(
    df: pd.DataFrame,
    contamination: float,
    include_intensity: bool,
    random_state: int,
) -> pd.DataFrame:
    df = df.copy()
    components = []
    reasons = [[] for _ in range(len(df))]

    if 'volume_um3' in df.columns:
        z = robust_zscore(np.log1p(df['volume_um3'].to_numpy(float)))
        df['robust_z_log_volume'] = z
        components.append(normalize_score(np.abs(z)))
        for i, v in enumerate(z):
            if v > 3:
                reasons[i].append('large_volume')
            elif v < -3:
                reasons[i].append('small_volume')

    if 'sphericity' in df.columns:
        z = robust_zscore(df['sphericity'].to_numpy(float))
        df['robust_z_sphericity'] = z
        components.append(normalize_score(np.maximum(-z, 0)))
        for i, v in enumerate(z):
            if v < -3:
                reasons[i].append('low_sphericity')

    if 'elongation' in df.columns:
        z = robust_zscore(df['elongation'].to_numpy(float))
        df['robust_z_elongation'] = z
        components.append(normalize_score(np.maximum(z, 0)))
        for i, v in enumerate(z):
            if v > 3:
                reasons[i].append('high_elongation')

    for col, reason in [
        ('self_vs_neighbor_volume_um3_zscore', 'local_volume_outlier'),
        ('self_vs_neighbor_sphericity_zscore', 'local_sphericity_outlier'),
        ('self_vs_neighbor_elongation_zscore', 'local_elongation_outlier'),
    ]:
        if col in df.columns:
            z = df[col].to_numpy(float)
            components.append(normalize_score(np.abs(z)))
            for i, v in enumerate(z):
                if np.isfinite(v) and abs(v) > 3:
                    reasons[i].append(reason)

    feature_cols = choose_feature_columns(
        df,
        feature_set='all' if include_intensity else 'shape_neighborhood',
        include_intensity=include_intensity,
    )
    if len(feature_cols) >= 2 and len(df) >= 20:
        X, _ = prepare_feature_matrix(df, feature_cols)

        try:
            iso = IsolationForest(
                n_estimators=200,
                contamination=float(contamination),
                random_state=random_state,
                n_jobs=-1,
            )
            iso.fit(X)
            df['isolation_forest_score'] = normalize_score(
                -iso.decision_function(X)
            )
            components.append(df['isolation_forest_score'].to_numpy(float))
        except Exception:
            df['isolation_forest_score'] = np.nan

        try:
            n_neighbors = min(20, len(df) - 1)
            lof = LocalOutlierFactor(
                n_neighbors=n_neighbors,
                contamination=float(contamination),
                novelty=False,
                n_jobs=-1,
            )
            lof.fit_predict(X)
            df['lof_score'] = normalize_score(-lof.negative_outlier_factor_)
            components.append(df['lof_score'].to_numpy(float))
        except Exception:
            df['lof_score'] = np.nan

    if components:
        df['anomaly_score'] = np.nanmean(np.vstack(components), axis=0)
    else:
        df['anomaly_score'] = 0.0

    threshold = np.nanpercentile(
        df['anomaly_score'].to_numpy(float),
        100.0 * (1.0 - float(contamination)),
    )
    df['is_anomaly'] = df['anomaly_score'] >= threshold
    df['anomaly_reason'] = [';'.join(r) if r else 'none' for r in reasons]
    return df


# ---------------------------------------------------------------------
# Output and visualization maps
# ---------------------------------------------------------------------


FEATURE_GROUPS = {
    'morphology': [
        'volume_um3',
        'equivalent_diameter_um',
        'surface_area_um2',
        'sphericity',
        'compactness',
        'axis_major_um',
        'axis_intermediate_um',
        'axis_minor_um',
        'elongation',
        'flatness',
        'major_minor_ratio',
    ],
    'intensity': [
        'mean_intensity',
        'std_intensity',
        'cv_intensity',
        'sum_intensity',
        'skewness_intensity',
        'kurtosis_intensity',
    ],
    'neighborhood': [
        'nearest_neighbor_distance_um',
        'neighbor_mean_volume_um3',
        'self_vs_neighbor_volume_um3_zscore',
        'self_vs_neighbor_sphericity_zscore',
        'self_vs_neighbor_elongation_zscore',
    ],
    'contact': [
        'contact_neighbor_count',
        'contact_area_total_um2',
        'contact_area_mean_um2',
        'contact_fraction',
    ],
    'region': [
        'region_z_bin',
        'region_y_bin',
        'region_x_bin',
    ],
    'anomaly': [
        'anomaly_score',
        'robust_z_log_volume',
        'robust_z_sphericity',
        'robust_z_elongation',
        'isolation_forest_score',
        'lof_score',
    ],
    'cluster': [
        'cluster_id',
    ],
}

CORE_MAP_FEATURES = [
    'volume_um3',
    'equivalent_diameter_um',
    'sphericity',
    'elongation',
    'nearest_neighbor_distance_um',
    'anomaly_score',
    'cluster_id',
]


def make_public_cell_table(df: pd.DataFrame) -> pd.DataFrame:
    """Remove internal labels and bbox fields from the user-facing CSV."""
    out = df.copy()
    if 'original_label_id' in out.columns:
        out = out.rename(columns={'original_label_id': 'label_id'})

    drop_cols = [
        c
        for c in out.columns
        if c == 'label_contiguous' or c.startswith('bbox_')
    ]
    out = out.drop(columns=drop_cols, errors='ignore')

    if 'label_id' in out.columns:
        cols = ['label_id'] + [c for c in out.columns if c != 'label_id']
        out = out[cols]

    return out


def compute_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = [{'metric': 'cell_count_total', 'value': int(len(df))}]
    for feature in [
        'volume_um3',
        'equivalent_diameter_um',
        'surface_area_um2',
        'sphericity',
        'elongation',
        'flatness',
        'nearest_neighbor_distance_um',
        'anomaly_score',
    ]:
        if feature not in df.columns:
            continue
        vals = df[feature].to_numpy(float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        rows.extend(
            [
                {'metric': f'{feature}_mean', 'value': float(np.mean(vals))},
                {
                    'metric': f'{feature}_median',
                    'value': float(np.median(vals)),
                },
                {'metric': f'{feature}_std', 'value': float(np.std(vals))},
                {
                    'metric': f'{feature}_q05',
                    'value': float(np.quantile(vals, 0.05)),
                },
                {
                    'metric': f'{feature}_q95',
                    'value': float(np.quantile(vals, 0.95)),
                },
            ]
        )
    if 'is_anomaly' in df.columns:
        rows.append(
            {
                'metric': 'anomaly_fraction',
                'value': float(np.mean(df['is_anomaly'].to_numpy(bool))),
            }
        )
    return pd.DataFrame(rows)


def normalize_feature_to_uint16(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    out = np.zeros_like(values, dtype=np.uint16)
    valid = np.isfinite(values)
    if not np.any(valid):
        return out
    lo, hi = np.nanpercentile(values[valid], [1, 99])
    scaled = np.clip((values - lo) / (hi - lo + 1e-12), 0.0, 1.0)
    out[valid] = (scaled[valid] * 65535).astype(np.uint16)
    return out


def make_mapped_volume(
    label: np.ndarray, df: pd.DataFrame, feature: str
) -> np.ndarray:
    """Map a per-cell feature back to a 3D uint16 image."""
    n_labels = int(label.max())
    lut = np.zeros(n_labels + 1, dtype=np.uint16)
    labs = df['label_contiguous'].to_numpy(int)

    if feature in {
        'cluster_id',
        'is_anomaly',
        'region_z_bin',
        'region_y_bin',
        'region_x_bin',
    }:
        vals = df[feature].fillna(-1).to_numpy(int)
        vals = np.maximum(vals + 1, 0)
        lut[labs] = np.clip(vals, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    else:
        vals = df[feature].to_numpy(float)
        lut[labs] = normalize_feature_to_uint16(vals)

    return lut[label]


def selected_features_for_maps(
    config: MorphometryConfig, df: pd.DataFrame
) -> list[str]:
    """Decide which feature maps should be saved based on UI choices."""
    if config.selected_map_features:
        candidates = list(config.selected_map_features)
    elif config.feature_map_mode == 'core':
        candidates = CORE_MAP_FEATURES
    elif config.feature_map_mode == 'all':
        candidates = []
        for features in FEATURE_GROUPS.values():
            candidates.extend(features)
        # Include dynamic k/r features.
        candidates.extend(
            [
                c
                for c in df.columns
                if c.startswith(('knn_', 'local_density_k', 'radius_'))
            ]
        )
    else:
        candidates = []
        if config.compute_morphology:
            candidates.extend(FEATURE_GROUPS['morphology'])
        if config.compute_intensity:
            candidates.extend(FEATURE_GROUPS['intensity'])
        if config.compute_neighborhood:
            candidates.extend(FEATURE_GROUPS['neighborhood'])
            candidates.extend(
                [
                    c
                    for c in df.columns
                    if c.startswith(('knn_', 'local_density_k', 'radius_'))
                ]
            )
        if config.compute_contact:
            candidates.extend(FEATURE_GROUPS['contact'])
        if config.compute_regions:
            candidates.extend(FEATURE_GROUPS['region'])
        if config.compute_anomaly:
            candidates.extend(FEATURE_GROUPS['anomaly'])
            candidates.append('is_anomaly')
        if config.compute_clustering:
            candidates.extend(FEATURE_GROUPS['cluster'])

    # Deduplicate while preserving order.
    out = []
    seen = set()
    for c in candidates:
        if c in df.columns and c not in seen:
            out.append(c)
            seen.add(c)
    return out


def save_feature_maps(
    label: np.ndarray,
    df: pd.DataFrame,
    out_dir: Path,
    config: MorphometryConfig,
) -> pd.DataFrame:
    """Save selected feature-mapped TIFFs and return a manifest DataFrame."""
    feature_root = out_dir / 'feature_tifs'
    feature_root.mkdir(parents=True, exist_ok=True)

    rows = []
    features = selected_features_for_maps(config, df)

    for feature in features:
        # Find group name for cleaner folder organization.
        group = 'custom'
        for g, fs in FEATURE_GROUPS.items():
            if feature in fs:
                group = g
                break
        if feature.startswith(('knn_', 'local_density_k', 'radius_')):
            group = 'neighborhood'

        group_dir = feature_root / group
        group_dir.mkdir(parents=True, exist_ok=True)

        mapped = make_mapped_volume(label, df, feature)
        out_path = group_dir / f'{feature}_map_uint16.tif'
        tiff.imwrite(out_path, mapped, bigtiff=True, photometric='minisblack')

        rows.append(
            {
                'group': group,
                'feature': feature,
                'path': str(out_path),
                'note': 'uint16 display map. Exact values are in cell_features.csv.',
            }
        )

    manifest = pd.DataFrame(rows)
    if len(manifest) > 0:
        manifest.to_csv(feature_root / 'feature_tif_manifest.csv', index=False)
    return manifest


def convert_contact_edges_to_original_ids(
    contact_df: pd.DataFrame, original_ids: np.ndarray
) -> pd.DataFrame:
    if contact_df is None or len(contact_df) == 0:
        return pd.DataFrame(columns=['label_a', 'label_b', 'contact_area_um2'])
    id_map = {i + 1: int(original_ids[i]) for i in range(len(original_ids))}
    out = contact_df.copy()
    out['label_a'] = out['label_a'].map(id_map)
    out['label_b'] = out['label_b'].map(id_map)
    return out


# ---------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------


def run_morphometry_from_arrays(
    raw_data,
    label_data,
    out_dir: str | Path,
    config: MorphometryConfig | dict,
    progress_callback: ProgressCallback = None,
    cancel_callback: CancelCallback = None,
) -> dict[str, object]:
    """Run selected morphometry analysis from arrays.

    Returns a dictionary containing output paths and summary information.
    """
    if isinstance(config, dict):
        config = MorphometryConfig(**config)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        _emit(progress_callback, 2, 'Loading labels...')
        label_raw = _to_numpy_3d(label_data, 'label_data')
        label_raw = _safe_int_dtype(label_raw)
        _check_cancel(cancel_callback)

        _emit(
            progress_callback,
            6,
            'Relabeling labels for internal computation...',
        )
        label, original_ids = relabel_to_contiguous(label_raw)
        del label_raw
        n_labels = int(label.max())

        if n_labels == 0:
            raise ValueError('No non-background labels were found.')

        raw = None
        if raw_data is not None and config.compute_intensity:
            _emit(progress_callback, 10, 'Loading raw image...')
            raw = _to_numpy_3d(raw_data, 'raw_data')
            if raw.shape != label.shape:
                raise ValueError(
                    f'Raw and label shapes do not match: raw={raw.shape}, label={label.shape}'
                )

        _check_cancel(cancel_callback)

        _emit(
            progress_callback,
            15,
            'Computing basic size and centroid features...',
        )
        cell_df = compute_basic_features(
            label, voxel_size=config.voxel_size_zyx
        )
        cell_df['original_label_id'] = original_ids[
            cell_df['label_contiguous'].to_numpy(int) - 1
        ]

        contact_df = pd.DataFrame(
            columns=['label_a', 'label_b', 'contact_area_um2']
        )

        need_surface = (
            config.compute_morphology
            or config.compute_contact
            or config.compute_anomaly
            or config.compute_clustering
        )
        if need_surface:
            _emit(
                progress_callback,
                25,
                'Computing surface area and contact graph...',
            )
            contact_needed = bool(
                config.compute_contact or config.neighbor_mode == 'contact'
            )
            surface_df, contact_df = compute_surface_and_contact(
                label,
                voxel_size=config.voxel_size_zyx,
                compute_contact=contact_needed,
            )
            cell_df = cell_df.merge(
                surface_df, on='label_contiguous', how='left'
            )
            cell_df = add_surface_shape_features(cell_df)
            if contact_needed:
                cell_df = add_contact_features(cell_df, contact_df)

        _check_cancel(cancel_callback)

        if config.compute_morphology and config.compute_pca_axes:
            _emit(
                progress_callback, 38, 'Computing PCA-based axis features...'
            )
            cell_df = compute_pca_axis_features(
                label,
                cell_df,
                voxel_size=config.voxel_size_zyx,
                max_voxels_for_exact_pca=int(config.max_voxels_for_exact_pca),
                cancel_callback=cancel_callback,
            )

        if raw is not None and config.compute_intensity:
            _emit(progress_callback, 52, 'Computing intensity features...')
            intensity_df = compute_intensity_features(raw, label)
            cell_df = cell_df.merge(
                intensity_df, on='label_contiguous', how='left'
            )

        _check_cancel(cancel_callback)

        if config.compute_neighborhood:
            _emit(progress_callback, 62, 'Computing neighborhood features...')
            cell_df = compute_neighborhood_features(
                cell_df,
                neighbor_mode=config.neighbor_mode,
                k_values=config.knn_k_values,
                radius_values_um=config.radius_values_um,
                local_k=int(config.local_k),
            )

        _check_cancel(cancel_callback)

        if config.compute_anomaly:
            _emit(progress_callback, 72, 'Computing anomaly scores...')
            cell_df = add_anomaly(
                cell_df,
                contamination=float(config.anomaly_contamination),
                include_intensity=raw is not None and config.compute_intensity,
                random_state=int(config.random_state),
            )

        used_cluster_features = []
        if config.compute_clustering:
            _emit(progress_callback, 80, 'Clustering cells...')
            cell_df, used_cluster_features = add_clustering(
                cell_df,
                feature_set=str(config.cluster_feature_set),
                n_clusters=int(config.n_clusters),
                include_intensity=raw is not None and config.compute_intensity,
                random_state=int(config.random_state),
            )

        region_df = pd.DataFrame()
        if config.compute_regions:
            _emit(
                progress_callback, 86, 'Computing spatial region statistics...'
            )
            cell_df = assign_spatial_regions(
                cell_df,
                label_shape=label.shape,
                voxel_size=config.voxel_size_zyx,
                grid_shape=config.grid_shape_zyx,
            )
            region_df = compute_region_features(
                cell_df,
                label_shape=label.shape,
                voxel_size=config.voxel_size_zyx,
                grid_shape=config.grid_shape_zyx,
            )

        _emit(progress_callback, 90, 'Saving CSV results...')
        public_cell_df = make_public_cell_table(cell_df)
        summary_df = compute_summary_stats(cell_df)

        cell_csv = out_dir / 'cell_features.csv'
        region_csv = out_dir / 'region_features.csv'
        summary_csv = out_dir / 'summary_stats.csv'
        contact_csv = out_dir / 'contact_edges.csv'

        if config.save_csv:
            public_cell_df.to_csv(cell_csv, index=False)
            if len(region_df) > 0:
                region_df.to_csv(region_csv, index=False)
            summary_df.to_csv(summary_csv, index=False)
            convert_contact_edges_to_original_ids(
                contact_df, original_ids
            ).to_csv(contact_csv, index=False)

        manifest = pd.DataFrame()
        if config.save_feature_tifs:
            _emit(progress_callback, 94, 'Saving feature-mapped TIFFs...')
            manifest = save_feature_maps(label, cell_df, out_dir, config)

        params = asdict(config)
        params.update(
            {
                'n_cells': int(n_labels),
                'label_shape_zyx': list(map(int, label.shape)),
                'used_cluster_features': list(used_cluster_features),
                'voxel_anisotropy_ratio': float(
                    max(config.voxel_size_zyx)
                    / (min(config.voxel_size_zyx) + 1e-12)
                ),
                'notes': {
                    'label_id': 'cell_features.csv uses the original user-provided label IDs.',
                    'surface_area': 'Estimated using exposed voxel faces with physical face areas; sensitive to strong anisotropy.',
                    'sphericity': 'pi^(1/3)*(6V)^(2/3)/A, using physical volume and estimated physical surface area.',
                    'axis_lengths': 'Computed from PCA on physical voxel coordinates.',
                    'anomaly_score': 'Ranking score combining robust global outliers, local self-vs-neighbor outliers, Isolation Forest, and LOF when available.',
                    'feature_tifs': 'uint16 display maps. Exact values are stored in cell_features.csv.',
                },
            }
        )
        save_json(params, out_dir / 'analysis_parameters.json')

        _emit(progress_callback, 100, 'Done.')

        return {
            'out_dir': str(out_dir),
            'cell_csv': str(cell_csv),
            'region_csv': str(region_csv) if len(region_df) > 0 else '',
            'summary_csv': str(summary_csv),
            'contact_csv': str(contact_csv),
            'parameter_json': str(out_dir / 'analysis_parameters.json'),
            'feature_manifest': str(
                out_dir / 'feature_tifs' / 'feature_tif_manifest.csv'
            )
            if len(manifest) > 0
            else '',
            'feature_maps': manifest.to_dict('records')
            if len(manifest) > 0
            else [],
            'n_cells': int(n_labels),
            'cell_features_preview': public_cell_df.head(20).to_dict(
                'records'
            ),
            'summary': summary_df.to_dict('records'),
        }

    except RuntimeError as e:
        if str(e) == '__MORPHOMETRY_CANCELLED__':
            return {'cancelled': True, 'error': 'Cancelled'}
        raise
    except Exception as e:
        raise RuntimeError(
            f'Morphometry analysis failed: {e}\n{traceback.format_exc()}'
        ) from e


# ---------------------------------------------------------------------
# New simplified analysis API
# ---------------------------------------------------------------------

BASIC_FEATURE_LABELS = {
    'volume_um3': 'Volume',
    'equivalent_diameter_um': 'Equivalent diameter',
    'surface_area_um2': 'Surface area',
    'centroid': 'Centroid',
    'sphericity': 'Sphericity',
    'compactness': 'Compactness',
    'axis_major_um': 'Axis major',
    'elongation': 'Elongation',
    'flatness': 'Flatness',
    'min_intensity': 'Min intensity',
    'max_intensity': 'Max intensity',
    'mean_intensity': 'Mean intensity',
    'std_intensity': 'Std intensity',
}

BASIC_FEATURE_COLUMN_MAP = {
    'volume_um3': ['volume_um3'],
    'equivalent_diameter_um': ['equivalent_diameter_um'],
    'surface_area_um2': ['surface_area_um2'],
    'centroid': ['centroid_z_um', 'centroid_y_um', 'centroid_x_um'],
    'sphericity': ['sphericity'],
    'compactness': ['compactness'],
    'axis_major_um': ['axis_major_um'],
    'elongation': ['elongation'],
    'flatness': ['flatness'],
    'min_intensity': ['min_intensity'],
    'max_intensity': ['max_intensity'],
    'mean_intensity': ['mean_intensity'],
    'std_intensity': ['std_intensity'],
}

NON_VISUAL_FEATURES = {
    'centroid',
    'axis_major_um',
    'min_intensity',
    'max_intensity',
}

INTENSITY_FEATURES = {
    'min_intensity',
    'max_intensity',
    'mean_intensity',
    'std_intensity',
}

SURFACE_DEPENDENT_FEATURES = {
    'surface_area_um2',
    'sphericity',
    'compactness',
}

PCA_DEPENDENT_FEATURES = {
    'axis_major_um',
    'elongation',
    'flatness',
}

CONTACT_FEATURE_LABELS = {
    'contact_neighbor_count': 'Neighbor count',
    'contact_area_total_um2': 'Total contact area',
    'contact_area_mean_um2': 'Mean contact area',
    'contact_area_max_um2': 'Max contact area',
    'contact_fraction': 'Contact fraction',
}

LOCAL_FEATURE_LABELS = {
    'mean_intensity': 'Mean intensity',
    'sphericity': 'Sphericity',
    'volume_um3': 'Volume',
    'surface_area_um2': 'Surface area',
    'compactness': 'Compactness',
    'elongation': 'Elongation',
    'flatness': 'Flatness',
}


def _feature_requires_raw(feature: str) -> bool:
    return str(feature) in INTENSITY_FEATURES


def _ensure_raw_available(raw, selected_features):
    if (
        any(_feature_requires_raw(f) for f in selected_features)
        and raw is None
    ):
        raise ValueError(
            'Selected intensity features require a raw image layer.'
        )


def _expand_requested_columns(selected_features):
    cols = []
    for feature in selected_features:
        cols.extend(BASIC_FEATURE_COLUMN_MAP.get(feature, [feature]))
    return cols


def _make_visualizable_features(selected_features, label_dict=None):
    rows = []
    label_dict = label_dict or BASIC_FEATURE_LABELS

    for feature in selected_features:
        if feature in NON_VISUAL_FEATURES:
            continue

        # Centroid expands to 3 columns and should not be visualized.
        if feature == 'centroid':
            continue

        rows.append(
            {
                'feature': feature,
                'label': label_dict.get(feature, feature),
            }
        )

    return rows


def _save_task_csv(df: pd.DataFrame, out_dir: Path, filename: str) -> str:
    out_path = out_dir / filename
    df.to_csv(out_path, index=False)
    return str(out_path)


def _summary_for_selected_features(df: pd.DataFrame, selected_columns):
    rows = [{'metric': 'cell_count_total', 'value': int(len(df))}]

    for col in selected_columns:
        if col not in df.columns:
            continue
        if col == 'label_id':
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        vals = df[col].to_numpy(float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue

        rows.extend(
            [
                {'metric': f'{col}_mean', 'value': float(np.mean(vals))},
                {'metric': f'{col}_median', 'value': float(np.median(vals))},
                {'metric': f'{col}_std', 'value': float(np.std(vals))},
                {
                    'metric': f'{col}_q05',
                    'value': float(np.quantile(vals, 0.05)),
                },
                {
                    'metric': f'{col}_q95',
                    'value': float(np.quantile(vals, 0.95)),
                },
            ]
        )

    return pd.DataFrame(rows)


def _prepare_base_table(
    raw_data,
    label_data,
    voxel_size_zyx,
    selected_features,
    progress_callback=None,
    cancel_callback=None,
):
    """
    Compute a cell-level feature table containing exactly the dependencies
    needed by selected_features.

    Returns:
        label: contiguous internal label image
        original_ids: original user label IDs
        cell_df: internal dataframe with label_contiguous + original_label_id
    """
    selected_features = list(selected_features or [])

    _emit(progress_callback, 2, 'Loading labels...')
    label_raw = _to_numpy_3d(label_data, 'label_data')
    label_raw = _safe_int_dtype(label_raw)
    _check_cancel(cancel_callback)

    _emit(progress_callback, 8, 'Relabeling labels...')
    label, original_ids = relabel_to_contiguous(label_raw)
    del label_raw

    if int(label.max()) == 0:
        raise ValueError('No non-background labels were found.')

    raw = None
    if any(_feature_requires_raw(f) for f in selected_features):
        _emit(progress_callback, 12, 'Loading raw image...')
        raw = _to_numpy_3d(raw_data, 'raw_data')
        if raw.shape != label.shape:
            raise ValueError(
                f'Raw and label shapes do not match: raw={raw.shape}, label={label.shape}'
            )

    _emit(
        progress_callback, 20, 'Computing basic size and centroid features...'
    )
    cell_df = compute_basic_features(label, voxel_size=voxel_size_zyx)
    cell_df['original_label_id'] = original_ids[
        cell_df['label_contiguous'].to_numpy(int) - 1
    ]

    need_surface = any(
        f in SURFACE_DEPENDENT_FEATURES for f in selected_features
    )
    need_pca = any(f in PCA_DEPENDENT_FEATURES for f in selected_features)

    # PCA uses bbox and physical coordinates. Surface-derived features are
    # independent but usually requested together with morphology.
    if need_surface:
        _emit(progress_callback, 35, 'Computing surface shape features...')
        surface_df, _ = compute_surface_and_contact(
            label,
            voxel_size=voxel_size_zyx,
            compute_contact=False,
        )
        cell_df = cell_df.merge(surface_df, on='label_contiguous', how='left')
        cell_df = add_surface_shape_features(cell_df)

    if need_pca:
        # PCA-derived elongation/flatness can be requested even if surface
        # features are not requested.
        _emit(progress_callback, 50, 'Computing PCA shape features...')
        cell_df = compute_pca_axis_features(
            label,
            cell_df,
            voxel_size=voxel_size_zyx,
            max_voxels_for_exact_pca=200_000,
            cancel_callback=cancel_callback,
        )

    if raw is not None:
        _emit(progress_callback, 65, 'Computing intensity features...')
        intensity_df = compute_intensity_features(raw, label)
        cell_df = cell_df.merge(
            intensity_df, on='label_contiguous', how='left'
        )

    _check_cancel(cancel_callback)
    return label, original_ids, cell_df


def run_basic_info_from_arrays(
    raw_data,
    label_data,
    out_dir,
    common_config,
    basic_config,
    progress_callback=None,
    cancel_callback=None,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_features = list(basic_config.get('selected_features', []))
    if not selected_features:
        selected_features = ['volume_um3']

    voxel_size = tuple(common_config['voxel_size_zyx'])

    label, original_ids, cell_df = _prepare_base_table(
        raw_data=raw_data,
        label_data=label_data,
        voxel_size_zyx=voxel_size,
        selected_features=selected_features,
        progress_callback=progress_callback,
        cancel_callback=cancel_callback,
    )

    _emit(progress_callback, 82, 'Preparing output table...')

    public_df = make_public_cell_table(cell_df)

    selected_columns = ['label_id']
    selected_columns.extend(_expand_requested_columns(selected_features))
    selected_columns = [c for c in selected_columns if c in public_df.columns]

    out_df = public_df[selected_columns].copy()

    cell_csv = _save_task_csv(out_df, out_dir, 'basic_info_cell_features.csv')

    summary_df = _summary_for_selected_features(out_df, selected_columns)
    summary_csv = _save_task_csv(summary_df, out_dir, 'basic_info_summary.csv')

    visualizable = _make_visualizable_features(selected_features)

    _emit(progress_callback, 100, 'Done.')

    return {
        'analysis_name': 'basic_info',
        'out_dir': str(out_dir),
        'cell_csv': cell_csv,
        'summary_csv': summary_csv,
        'parameter_json': '',
        'feature_manifest': '',
        'feature_maps': [],
        'n_cells': int(len(out_df)),
        'cell_features_preview': out_df.head(20).to_dict('records'),
        'summary': summary_df.to_dict('records'),
        'visualizable_features': visualizable,
    }


def compute_simple_neighborhood_features(
    df: pd.DataFrame,
    mode: str,
    k: int = 10,
    radius_vox: float = 20.0,
) -> pd.DataFrame:
    """
    Compute first-step neighborhood features.

    kNN mode uses physical centroid coordinates.
    Radius mode uses voxel/pixel centroid coordinates for more intuitive tuning.
    """
    df = df.copy()
    mode = str(mode).lower()

    if mode == 'radius':
        coords = df[
            ['centroid_z_vox', 'centroid_y_vox', 'centroid_x_vox']
        ].to_numpy(float)
    else:
        coords = df[
            ['centroid_z_um', 'centroid_y_um', 'centroid_x_um']
        ].to_numpy(float)

    n = coords.shape[0]

    if n <= 1:
        df['nearest_neighbor_distance_um'] = np.nan
        return df

    tree = cKDTree(coords)

    if mode == 'knn':
        k = max(1, int(k))
        max_k = min(k + 1, n)

        dists, idx = tree.query(coords, k=max_k, workers=-1)
        if dists.ndim == 1:
            dists = dists[:, None]

        neighbor_dists = dists[:, 1:]

        if neighbor_dists.shape[1] == 0:
            df['nearest_neighbor_distance_um'] = np.nan
            return df

        kk = neighbor_dists.shape[1]
        r_k = neighbor_dists[:, -1]
        sphere_volume = (4.0 / 3.0) * np.pi * (r_k**3)

        df['nearest_neighbor_distance_um'] = neighbor_dists[:, 0]
        df[f'knn_mean_distance_k{k}_um'] = np.mean(neighbor_dists, axis=1)
        df[f'knn_median_distance_k{k}_um'] = np.median(neighbor_dists, axis=1)
        df[f'local_density_k{k}_per_um3'] = kk / (sphere_volume + 1e-12)

    elif mode == 'radius':
        radius_vox = float(radius_vox)
        if radius_vox <= 0:
            raise ValueError('radius_vox must be positive.')

        counts = np.array(
            [len(v) - 1 for v in tree.query_ball_point(coords, r=radius_vox)]
        )

        sphere_volume_vox = (4.0 / 3.0) * np.pi * (radius_vox**3)
        r_name = f'{radius_vox:g}'

        df[f'radius_neighbor_count_r{r_name}_px'] = counts
        df[f'radius_density_r{r_name}_per_px3'] = counts / (
            sphere_volume_vox + 1e-12
        )

    else:
        raise ValueError("neighbor_mode must be 'knn' or 'radius'.")

    return df


# ---------------------------------------------------------------------
# Decoupled task wrappers
# ---------------------------------------------------------------------


def _rename_task_outputs(result: dict, out_dir, task_name: str) -> dict:
    """
    Rename generic output files to task-specific output files.

    Example:
        cell_features.csv -> morphology_cell_features.csv
    """
    out_dir = Path(out_dir)

    rename_pairs = [
        ('cell_features.csv', f'{task_name}_cell_features.csv', 'cell_csv'),
        ('summary_stats.csv', f'{task_name}_summary.csv', 'summary_csv'),
        (
            'region_features.csv',
            f'{task_name}_region_features.csv',
            'region_csv',
        ),
        ('contact_edges.csv', f'{task_name}_contact_edges.csv', 'contact_csv'),
    ]

    for old_name, new_name, result_key in rename_pairs:
        old_path = out_dir / old_name
        new_path = out_dir / new_name

        if old_path.exists():
            if new_path.exists():
                new_path.unlink()
            old_path.replace(new_path)
            result[result_key] = str(new_path)

    result['analysis_name'] = task_name
    result['out_dir'] = str(out_dir)
    return result


def _bool_from_config(config: dict, key: str, default: bool = True) -> bool:
    return bool(config.get(key, default))


def run_morphology_from_arrays(
    label_data,
    out_dir,
    common_config,
    morphology_config,
    progress_callback=None,
    cancel_callback=None,
):
    selected_maps = morphology_config.get(
        'selected_map_features',
        [
            'volume_um3',
            'equivalent_diameter_um',
            'surface_area_um2',
            'sphericity',
            'elongation',
            'flatness',
        ],
    )

    config = MorphometryConfig(
        voxel_size_zyx=tuple(common_config['voxel_size_zyx']),
        compute_morphology=True,
        compute_intensity=False,
        compute_neighborhood=False,
        compute_contact=False,
        compute_regions=False,
        compute_clustering=False,
        compute_anomaly=False,
        compute_pca_axes=bool(morphology_config.get('compute_pca_axes', True)),
        # Internal performance safeguard.
        # If a single object has more voxels than this, PCA uses random subsampling.
        # This should not be exposed in the UI.
        max_voxels_for_exact_pca=200_000,
        save_csv=True,
        save_feature_tifs=bool(common_config.get('save_feature_tifs', True)),
        feature_map_mode='custom',
        selected_map_features=tuple(selected_maps),
    )

    result = run_morphometry_from_arrays(
        raw_data=None,
        label_data=label_data,
        out_dir=out_dir,
        config=config,
        progress_callback=progress_callback,
        cancel_callback=cancel_callback,
    )

    return _rename_task_outputs(result, out_dir, 'morphology')


def _intensity_selected_map_features(intensity_config: dict):
    features = intensity_config.get('features', {})

    maps = []

    if features.get('mean', True):
        maps.append('mean_intensity')
    if features.get('std', True):
        maps.append('std_intensity')
    if features.get('cv', True):
        maps.append('cv_intensity')
    if features.get('sum', True):
        maps.append('sum_intensity')
    if features.get('skewness', False):
        maps.append('skewness_intensity')
    if features.get('kurtosis', False):
        maps.append('kurtosis_intensity')

    # The current low-level intensity function always computes these;
    # map creation only uses existing columns.
    if features.get('minmax', False):
        maps.extend(['min_intensity', 'max_intensity'])

    return maps


def run_intensity_from_arrays(
    raw_data,
    label_data,
    out_dir,
    common_config,
    intensity_config,
    progress_callback=None,
    cancel_callback=None,
):
    selected_maps = _intensity_selected_map_features(intensity_config)

    if len(selected_maps) == 0:
        selected_maps = ['mean_intensity']

    config = MorphometryConfig(
        voxel_size_zyx=tuple(common_config['voxel_size_zyx']),
        compute_morphology=False,
        compute_intensity=True,
        compute_neighborhood=False,
        compute_contact=False,
        compute_regions=False,
        compute_clustering=False,
        compute_anomaly=False,
        save_csv=True,
        save_feature_tifs=bool(common_config.get('save_feature_tifs', True)),
        feature_map_mode='custom',
        selected_map_features=tuple(selected_maps),
    )

    result = run_morphometry_from_arrays(
        raw_data=raw_data,
        label_data=label_data,
        out_dir=out_dir,
        config=config,
        progress_callback=progress_callback,
        cancel_callback=cancel_callback,
    )

    result['intensity_config'] = intensity_config
    return _rename_task_outputs(result, out_dir, 'intensity')


def _neighborhood_selected_map_features(neighborhood_config: dict):
    maps = [
        'nearest_neighbor_distance_um',
    ]

    for k in neighborhood_config.get('knn_k_values', (10,)):
        maps.append(f'knn_mean_distance_k{int(k)}_um')
        maps.append(f'local_density_k{int(k)}_per_um3')

    for r in neighborhood_config.get('radius_values_um', (20.0,)):
        r_name = f'{float(r):g}'
        maps.append(f'radius_neighbor_count_r{r_name}_um')
        maps.append(f'radius_density_r{r_name}_per_um3')

    maps.extend(
        [
            'self_vs_neighbor_volume_um3_zscore',
            'self_vs_neighbor_sphericity_zscore',
            'self_vs_neighbor_elongation_zscore',
        ]
    )

    return maps


def run_neighborhood_from_arrays(
    label_data,
    out_dir,
    common_config,
    neighborhood_config,
    progress_callback=None,
    cancel_callback=None,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mode = neighborhood_config.get('neighbor_mode', 'knn')
    k = int(neighborhood_config.get('k', 10))
    radius_vox = float(neighborhood_config.get('radius_vox', 20.0))
    voxel_size = tuple(common_config['voxel_size_zyx'])

    _emit(progress_callback, 2, 'Loading labels...')
    label_raw = _to_numpy_3d(label_data, 'label_data')
    label_raw = _safe_int_dtype(label_raw)

    _emit(progress_callback, 8, 'Relabeling labels...')
    label, original_ids = relabel_to_contiguous(label_raw)
    del label_raw

    if int(label.max()) == 0:
        raise ValueError('No non-background labels were found.')

    _emit(progress_callback, 25, 'Computing centroids...')
    cell_df = compute_basic_features(label, voxel_size=voxel_size)
    cell_df['original_label_id'] = original_ids[
        cell_df['label_contiguous'].to_numpy(int) - 1
    ]

    _emit(progress_callback, 55, 'Computing neighborhood features...')
    cell_df = compute_simple_neighborhood_features(
        cell_df,
        mode=mode,
        k=k,
        radius_vox=radius_vox,
    )

    public_df = make_public_cell_table(cell_df)

    if mode == 'knn':
        feature_cols = [
            'nearest_neighbor_distance_um',
            f'knn_mean_distance_k{k}_um',
            f'knn_median_distance_k{k}_um',
            f'local_density_k{k}_per_um3',
        ]
    else:
        r_name = f'{radius_vox:g}'
        feature_cols = [
            f'radius_neighbor_count_r{r_name}_px',
            f'radius_density_r{r_name}_per_px3',
        ]

    selected_columns = ['label_id'] + [
        c for c in feature_cols if c in public_df.columns
    ]
    out_df = public_df[selected_columns].copy()

    cell_csv = _save_task_csv(
        out_df, out_dir, 'neighborhood_cell_features.csv'
    )
    summary_df = _summary_for_selected_features(out_df, selected_columns)
    summary_csv = _save_task_csv(
        summary_df, out_dir, 'neighborhood_summary.csv'
    )

    visualizable = [
        {'feature': c, 'label': c} for c in feature_cols if c in out_df.columns
    ]

    _emit(progress_callback, 100, 'Done.')

    return {
        'analysis_name': 'neighborhood',
        'out_dir': str(out_dir),
        'cell_csv': cell_csv,
        'summary_csv': summary_csv,
        'feature_manifest': '',
        'parameter_json': '',
        'feature_maps': [],
        'n_cells': int(len(out_df)),
        'cell_features_preview': out_df.head(20).to_dict('records'),
        'summary': summary_df.to_dict('records'),
        'visualizable_features': visualizable,
        'neighborhood_config': neighborhood_config,
    }


def compute_local_comparison_zscore(
    df: pd.DataFrame,
    feature: str,
    mode: str,
    k: int = 10,
    radius_vox: float = 20.0,
) -> pd.DataFrame:
    """
    Compare each cell with its local neighbors and compute z-score.

    kNN mode uses physical coordinates.
    Radius mode uses voxel/pixel coordinates.
    """
    df = df.copy()

    if feature not in df.columns:
        raise ValueError(
            f"Feature '{feature}' is not available for local comparison."
        )

    mode = str(mode).lower()

    if mode == 'radius':
        coords = df[
            ['centroid_z_vox', 'centroid_y_vox', 'centroid_x_vox']
        ].to_numpy(float)
    else:
        coords = df[
            ['centroid_z_um', 'centroid_y_um', 'centroid_x_um']
        ].to_numpy(float)

    values = df[feature].to_numpy(float)

    n = coords.shape[0]
    if n <= 1:
        out_name = f'local_zscore_{feature}'
        df[out_name] = np.nan
        return df

    tree = cKDTree(coords)

    neigh_mean = np.full(n, np.nan, dtype=float)
    neigh_std = np.full(n, np.nan, dtype=float)

    if mode == 'knn':
        k = max(1, int(k))
        max_k = min(k + 1, n)
        dists, idx = tree.query(coords, k=max_k, workers=-1)

        if idx.ndim == 1:
            idx = idx[:, None]

        neigh_idx = idx[:, 1:]

        for i in range(n):
            vals = values[neigh_idx[i]]
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            neigh_mean[i] = float(np.nanmean(vals))
            neigh_std[i] = float(np.nanstd(vals))

    elif mode == 'radius':
        radius_vox = float(radius_vox)
        if radius_vox <= 0:
            raise ValueError('radius_vox must be positive.')

        all_idx = tree.query_ball_point(coords, r=radius_vox)

        for i, ids in enumerate(all_idx):
            ids = [j for j in ids if j != i]
            if not ids:
                continue
            vals = values[np.asarray(ids, dtype=int)]
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            neigh_mean[i] = float(np.nanmean(vals))
            neigh_std[i] = float(np.nanstd(vals))

    else:
        raise ValueError("neighbor_mode must be 'knn' or 'radius'.")

    df[f'local_neighbor_mean_{feature}'] = neigh_mean
    df[f'local_neighbor_std_{feature}'] = neigh_std
    df[f'local_zscore_{feature}'] = (values - neigh_mean) / (neigh_std + 1e-12)

    return df


def run_local_comparison_from_arrays(
    raw_data,
    label_data,
    out_dir,
    common_config,
    local_config,
    progress_callback=None,
    cancel_callback=None,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feature = str(local_config.get('feature', 'volume_um3'))
    mode = local_config.get('neighbor_mode', 'knn')
    k = int(local_config.get('k', 10))
    radius_vox = float(local_config.get('radius_vox', 20.0))
    voxel_size = tuple(common_config['voxel_size_zyx'])

    label, original_ids, cell_df = _prepare_base_table(
        raw_data=raw_data,
        label_data=label_data,
        voxel_size_zyx=voxel_size,
        selected_features=[feature],
        progress_callback=progress_callback,
        cancel_callback=cancel_callback,
    )

    _emit(progress_callback, 75, 'Computing local comparison z-score...')
    cell_df = compute_local_comparison_zscore(
        cell_df,
        feature=feature,
        mode=mode,
        k=k,
        radius_vox=radius_vox,
    )

    public_df = make_public_cell_table(cell_df)

    z_col = f'local_zscore_{feature}'
    mean_col = f'local_neighbor_mean_{feature}'
    std_col = f'local_neighbor_std_{feature}'

    selected_columns = [
        'label_id',
        feature,
        mean_col,
        std_col,
        z_col,
    ]
    selected_columns = [c for c in selected_columns if c in public_df.columns]

    out_df = public_df[selected_columns].copy()

    cell_csv = _save_task_csv(
        out_df, out_dir, 'local_comparison_cell_features.csv'
    )
    summary_df = _summary_for_selected_features(out_df, selected_columns)
    summary_csv = _save_task_csv(
        summary_df, out_dir, 'local_comparison_summary.csv'
    )

    _emit(progress_callback, 100, 'Done.')

    return {
        'analysis_name': 'local_comparison',
        'out_dir': str(out_dir),
        'cell_csv': cell_csv,
        'summary_csv': summary_csv,
        'feature_manifest': '',
        'parameter_json': '',
        'feature_maps': [],
        'n_cells': int(len(out_df)),
        'cell_features_preview': out_df.head(20).to_dict('records'),
        'summary': summary_df.to_dict('records'),
        'visualizable_features': [
            {
                'feature': z_col,
                'label': f'Local z-score: {LOCAL_FEATURE_LABELS.get(feature, feature)}',
            }
        ],
        'local_config': local_config,
    }


def run_contact_from_arrays(
    label_data,
    out_dir,
    common_config,
    contact_config,
    progress_callback=None,
    cancel_callback=None,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_features = list(contact_config.get('selected_features', []))
    if not selected_features:
        selected_features = ['contact_neighbor_count']

    voxel_size = tuple(common_config['voxel_size_zyx'])

    _emit(progress_callback, 2, 'Loading labels...')
    label_raw = _to_numpy_3d(label_data, 'label_data')
    label_raw = _safe_int_dtype(label_raw)

    _emit(progress_callback, 8, 'Relabeling labels...')
    label, original_ids = relabel_to_contiguous(label_raw)
    del label_raw

    if int(label.max()) == 0:
        raise ValueError('No non-background labels were found.')

    _emit(progress_callback, 25, 'Computing basic cell features...')
    cell_df = compute_basic_features(label, voxel_size=voxel_size)
    cell_df['original_label_id'] = original_ids[
        cell_df['label_contiguous'].to_numpy(int) - 1
    ]

    _emit(progress_callback, 50, 'Computing surface and contact graph...')
    surface_df, contact_edges_df = compute_surface_and_contact(
        label,
        voxel_size=voxel_size,
        compute_contact=True,
    )

    cell_df = cell_df.merge(surface_df, on='label_contiguous', how='left')
    cell_df = add_surface_shape_features(cell_df)
    cell_df = add_contact_features(cell_df, contact_edges_df)

    public_df = make_public_cell_table(cell_df)

    selected_columns = ['label_id'] + [
        f for f in selected_features if f in public_df.columns
    ]
    out_df = public_df[selected_columns].copy()

    cell_csv = _save_task_csv(out_df, out_dir, 'contact_cell_features.csv')

    contact_csv = out_dir / 'contact_edges.csv'
    convert_contact_edges_to_original_ids(
        contact_edges_df,
        original_ids,
    ).to_csv(contact_csv, index=False)

    summary_df = _summary_for_selected_features(out_df, selected_columns)
    summary_csv = _save_task_csv(summary_df, out_dir, 'contact_summary.csv')

    visualizable = [
        {
            'feature': f,
            'label': CONTACT_FEATURE_LABELS.get(f, f),
        }
        for f in selected_features
        if f in out_df.columns
    ]

    _emit(progress_callback, 100, 'Done.')

    return {
        'analysis_name': 'contact',
        'out_dir': str(out_dir),
        'cell_csv': cell_csv,
        'contact_csv': str(contact_csv),
        'summary_csv': summary_csv,
        'feature_manifest': '',
        'parameter_json': '',
        'feature_maps': [],
        'n_cells': int(len(out_df)),
        'cell_features_preview': out_df.head(20).to_dict('records'),
        'summary': summary_df.to_dict('records'),
        'visualizable_features': visualizable,
        'contact_config': contact_config,
    }


def run_region_from_arrays(
    label_data,
    out_dir,
    common_config,
    region_config,
    progress_callback=None,
    cancel_callback=None,
):
    selected_maps = [
        'region_z_bin',
        'region_y_bin',
        'region_x_bin',
    ]

    config = MorphometryConfig(
        voxel_size_zyx=tuple(common_config['voxel_size_zyx']),
        compute_morphology=True,
        compute_intensity=False,
        compute_neighborhood=True,
        compute_contact=False,
        compute_regions=True,
        compute_clustering=False,
        compute_anomaly=False,
        grid_shape_zyx=tuple(region_config.get('grid_shape_zyx', (6, 6, 6))),
        save_csv=True,
        save_feature_tifs=bool(common_config.get('save_feature_tifs', True)),
        feature_map_mode='custom',
        selected_map_features=tuple(selected_maps),
    )

    result = run_morphometry_from_arrays(
        raw_data=None,
        label_data=label_data,
        out_dir=out_dir,
        config=config,
        progress_callback=progress_callback,
        cancel_callback=cancel_callback,
    )

    result['region_config'] = region_config
    return _rename_task_outputs(result, out_dir, 'region')


def _compute_feature_for_clustering(
    raw_data,
    label_data,
    voxel_size,
    feature,
    progress_callback=None,
    cancel_callback=None,
):
    """
    Compute a single feature table for clustering.
    Supports basic, intensity, and contact features.
    """
    if feature.startswith('contact_'):
        label_raw = _to_numpy_3d(label_data, 'label_data')
        label_raw = _safe_int_dtype(label_raw)
        label, original_ids = relabel_to_contiguous(label_raw)
        del label_raw

        if int(label.max()) == 0:
            raise ValueError('No non-background labels were found.')

        cell_df = compute_basic_features(label, voxel_size=voxel_size)
        cell_df['original_label_id'] = original_ids[
            cell_df['label_contiguous'].to_numpy(int) - 1
        ]

        surface_df, contact_edges_df = compute_surface_and_contact(
            label,
            voxel_size=voxel_size,
            compute_contact=True,
        )
        cell_df = cell_df.merge(surface_df, on='label_contiguous', how='left')
        cell_df = add_surface_shape_features(cell_df)
        cell_df = add_contact_features(cell_df, contact_edges_df)
        return cell_df

    label, original_ids, cell_df = _prepare_base_table(
        raw_data=raw_data,
        label_data=label_data,
        voxel_size_zyx=voxel_size,
        selected_features=[feature],
        progress_callback=progress_callback,
        cancel_callback=cancel_callback,
    )
    return cell_df


def add_single_feature_clustering(
    df: pd.DataFrame,
    feature: str,
    n_clusters: int,
    random_state: int = 0,
) -> pd.DataFrame:
    df = df.copy()

    if feature not in df.columns:
        raise ValueError(
            f"Feature '{feature}' is not available for clustering."
        )

    vals = df[feature].to_numpy(float)
    valid = np.isfinite(vals)

    df['cluster_id'] = -1

    if np.sum(valid) < max(3, int(n_clusters)):
        return df

    x = vals[valid].reshape(-1, 1)

    # Use log1p for non-negative scale-like features.
    if np.nanmin(x) >= 0 and any(
        key in feature
        for key in [
            'volume',
            'diameter',
            'area',
            'axis',
            'distance',
            'density',
            'intensity',
        ]
    ):
        x = np.log1p(x)

    x = RobustScaler().fit_transform(x)

    km = KMeans(
        n_clusters=int(n_clusters),
        random_state=int(random_state),
        n_init='auto',
    )
    labels = km.fit_predict(x)

    df.loc[valid, 'cluster_id'] = labels.astype(int)
    return df


def run_clustering_from_arrays(
    raw_data,
    label_data,
    out_dir,
    common_config,
    clustering_config,
    progress_callback=None,
    cancel_callback=None,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feature = str(clustering_config.get('feature', 'volume_um3'))
    n_clusters = int(clustering_config.get('n_clusters', 6))
    voxel_size = tuple(common_config['voxel_size_zyx'])

    _emit(
        progress_callback,
        10,
        f'Computing feature for clustering: {feature}...',
    )
    cell_df = _compute_feature_for_clustering(
        raw_data=raw_data,
        label_data=label_data,
        voxel_size=voxel_size,
        feature=feature,
        progress_callback=progress_callback,
        cancel_callback=cancel_callback,
    )

    _emit(progress_callback, 70, 'Clustering cells...')
    cell_df = add_single_feature_clustering(
        cell_df,
        feature=feature,
        n_clusters=n_clusters,
        random_state=0,
    )

    public_df = make_public_cell_table(cell_df)

    selected_columns = [
        'label_id',
        feature,
        'cluster_id',
    ]
    selected_columns = [c for c in selected_columns if c in public_df.columns]

    out_df = public_df[selected_columns].copy()

    cell_csv = _save_task_csv(out_df, out_dir, 'clustering_cell_features.csv')
    summary_df = _summary_for_selected_features(out_df, selected_columns)
    summary_csv = _save_task_csv(summary_df, out_dir, 'clustering_summary.csv')

    _emit(progress_callback, 100, 'Done.')

    return {
        'analysis_name': 'clustering',
        'out_dir': str(out_dir),
        'cell_csv': cell_csv,
        'summary_csv': summary_csv,
        'feature_manifest': '',
        'parameter_json': '',
        'feature_maps': [],
        'n_cells': int(len(out_df)),
        'cell_features_preview': out_df.head(20).to_dict('records'),
        'summary': summary_df.to_dict('records'),
        'visualizable_features': [
            {
                'feature': 'cluster_id',
                'label': f'Cluster ID from {feature}',
            }
        ],
        'clustering_config': clustering_config,
    }


def run_anomaly_from_arrays(
    label_data,
    out_dir,
    common_config,
    anomaly_config,
    progress_callback=None,
    cancel_callback=None,
):
    selected_maps = [
        'anomaly_score',
        'is_anomaly',
        'robust_z_log_volume',
        'robust_z_sphericity',
        'robust_z_elongation',
    ]

    config = MorphometryConfig(
        voxel_size_zyx=tuple(common_config['voxel_size_zyx']),
        compute_morphology=True,
        compute_intensity=False,
        compute_neighborhood=True,
        compute_contact=False,
        compute_regions=False,
        compute_clustering=False,
        compute_anomaly=True,
        anomaly_contamination=float(anomaly_config.get('top_fraction', 0.02)),
        save_csv=True,
        save_feature_tifs=bool(common_config.get('save_feature_tifs', True)),
        feature_map_mode='custom',
        selected_map_features=tuple(selected_maps),
    )

    result = run_morphometry_from_arrays(
        raw_data=None,
        label_data=label_data,
        out_dir=out_dir,
        config=config,
        progress_callback=progress_callback,
        cancel_callback=cancel_callback,
    )

    result['anomaly_config'] = anomaly_config
    return _rename_task_outputs(result, out_dir, 'anomaly')
