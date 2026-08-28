#!/usr/bin/env python3
"""
Solid component extraction and 2D diameter measurement for lung tumors.

This script implements the method described in the manuscript:

1. Within an AI-generated tumor segmentation mask, CT attenuation thresholds
   from +200 HU to -600 HU are applied at 50-HU intervals.
2. On each axial slice, voxels/pixels with CT attenuation >= threshold are
   extracted.
3. Two-dimensional connected-component analysis is performed, and only the
   largest connected component is retained.
4. To reduce computation time, analysis is restricted to the central 60% of
   tumor-containing axial slices.
5. The maximum 2D diameter is calculated with the PyRadiomics shape2D
   MaximumDiameter feature.
6. For each threshold, the largest diameter among the analyzed slices is
   recorded.
7. Whole-tumor diameter is measured in the same way using the tumor
   segmentation mask without CT attenuation thresholding.

Example
-------
python solid_component_diameter.py \
    --ct-dir "D:/nnUNet/nnUNet_results/Dataset860_LungTumor/image" \
    --seg-dir "D:/nnUNet/nnUNet_results/Dataset860_LungTumor/seg" \
    --yaml "D:/nnUNet/Params.yaml" \
    --output "D:/nnUNet/nnUNet_results/Dataset860_LungTumor/diameter_summary.csv"

Dependencies
------------
pip install SimpleITK numpy scipy pyradiomics
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.ndimage
import SimpleITK as sitk
from radiomics import featureextractor


DEFAULT_THRESHOLDS = list(range(200, -601, -50))


def build_extractor(yaml_path: Path) -> featureextractor.RadiomicsFeatureExtractor:
    """Create a PyRadiomics extractor that calculates shape2D MaximumDiameter."""
    extractor = featureextractor.RadiomicsFeatureExtractor(str(yaml_path))
    extractor.disableAllFeatures()
    extractor.enableFeaturesByName(shape2D=["MaximumDiameter"])
    return extractor


def central_60_percent_slices(seg_array: np.ndarray) -> np.ndarray:
    """
    Return tumor-containing axial slice indices restricted to the central 60%.

    The first and last 20% of tumor-containing slices are excluded.
    For very small tumors, at least one tumor-containing slice is retained.
    """
    z_indices = np.unique(np.where(seg_array > 0)[0])

    if len(z_indices) == 0:
        return np.array([], dtype=int)

    start = int(np.floor(len(z_indices) * 0.20))
    stop = int(np.ceil(len(z_indices) * 0.80))

    selected = z_indices[start:stop]

    if len(selected) == 0:
        selected = np.array([z_indices[len(z_indices) // 2]], dtype=int)

    return selected


def largest_connected_component(mask: np.ndarray) -> Optional[np.ndarray]:
    """
    Keep the largest 2D connected component.

    scipy.ndimage.label is used with its default 2D connectivity, matching the
    original implementation.
    """
    labeled, num = scipy.ndimage.label(mask)

    if num == 0:
        return None

    component_sizes = np.bincount(labeled.ravel())[1:]
    largest_label = int(np.argmax(component_sizes)) + 1
    return (labeled == largest_label).astype(np.uint8)


def pyradiomics_maximum_diameter(
    image_slice: np.ndarray,
    mask_slice: np.ndarray,
    spacing_xy: tuple[float, float],
    extractor: featureextractor.RadiomicsFeatureExtractor,
) -> float:
    """Calculate PyRadiomics shape2D MaximumDiameter in millimeters."""
    image_2d = sitk.GetImageFromArray(image_slice.astype(np.float32))
    mask_2d = sitk.GetImageFromArray(mask_slice.astype(np.uint8))

    image_2d.SetSpacing(spacing_xy)
    mask_2d.SetSpacing(spacing_xy)

    result = extractor.execute(image_2d, mask_2d)
    return float(result["original_shape2D_MaximumDiameter"])


def measure_max_2d_diameter(
    ct_img: sitk.Image,
    seg_img: sitk.Image,
    extractor: featureextractor.RadiomicsFeatureExtractor,
    threshold: Optional[int],
) -> tuple[float, Optional[int], Optional[float]]:
    """
    Measure the largest axial 2D diameter across the central 60% of tumor slices.

    Parameters
    ----------
    threshold:
        HU threshold. Pixels with CT value >= threshold are retained.
        If None, the whole tumor segmentation mask is used without thresholding.

    Returns
    -------
    max_diameter_mm, slice_index, z_position_mm
    """
    ct_array = sitk.GetArrayFromImage(ct_img)   # Z, Y, X
    seg_array = sitk.GetArrayFromImage(seg_img)

    if ct_array.shape != seg_array.shape:
        raise ValueError(
            f"CT and segmentation shapes differ: {ct_array.shape} vs {seg_array.shape}"
        )

    spacing = ct_img.GetSpacing()  # X, Y, Z
    origin = ct_img.GetOrigin()

    z_range = central_60_percent_slices(seg_array)

    if len(z_range) == 0:
        return float("nan"), None, None

    max_diameter = -np.inf
    max_slice = None

    for z in z_range:
        ct_slice = ct_array[z]
        seg_slice = seg_array[z] > 0

        if threshold is None:
            candidate_mask = seg_slice
        else:
            candidate_mask = seg_slice & (ct_slice >= threshold)

        final_mask = largest_connected_component(candidate_mask)

        if final_mask is None or np.count_nonzero(final_mask) == 0:
            continue

        try:
            diameter = pyradiomics_maximum_diameter(
                image_slice=ct_slice,
                mask_slice=final_mask,
                spacing_xy=(spacing[0], spacing[1]),
                extractor=extractor,
            )
        except Exception:
            # PyRadiomics may fail for extremely small or degenerate components.
            continue

        if diameter > max_diameter:
            max_diameter = diameter
            max_slice = int(z)

    if max_slice is None:
        return float("nan"), None, None

    z_position = origin[2] + max_slice * spacing[2]
    return float(max_diameter), max_slice, float(z_position)


def find_segmentation_for_ct(ct_path: Path, seg_dir: Path) -> Optional[Path]:
    """
    Match CT and segmentation files.

    Expected CT naming examples:
        lung_0001_0000.nrrd
        lung_0001_0000.nii.gz

    Expected segmentation examples:
        lung_0001.nrrd
        lung_0001.nii.gz
    """
    name = ct_path.name

    if name.endswith(".nii.gz"):
        case_id = name[:-7]
    else:
        case_id = ct_path.stem

    if case_id.endswith("_0000"):
        case_id = case_id[:-5]

    candidates = [
        seg_dir / f"{case_id}.nrrd",
        seg_dir / f"{case_id}.nii.gz",
        seg_dir / f"{case_id}.nii",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Fallback for minor filename variations.
    fallback = sorted(
        p for p in seg_dir.iterdir()
        if p.is_file() and case_id in p.name
    )
    return fallback[0] if fallback else None


def collect_ct_files(ct_dir: Path) -> list[Path]:
    """Collect supported CT image files."""
    files = []
    for pattern in ("*.nrrd", "*.nii.gz", "*.nii"):
        files.extend(ct_dir.glob(pattern))
    return sorted(set(files))


def run_batch(
    ct_dir: Path,
    seg_dir: Path,
    yaml_path: Path,
    output_csv: Path,
    thresholds: list[int],
) -> None:
    """Run solid-component and whole-tumor diameter measurements for all cases."""
    extractor = build_extractor(yaml_path)
    ct_files = collect_ct_files(ct_dir)

    if not ct_files:
        raise FileNotFoundError(f"No supported CT files were found in: {ct_dir}")

    pairs: list[tuple[Path, Path]] = []

    for ct_path in ct_files:
        seg_path = find_segmentation_for_ct(ct_path, seg_dir)
        if seg_path is not None:
            pairs.append((ct_path, seg_path))
        else:
            print(f"[WARNING] No segmentation found for: {ct_path.name}")

    if not pairs:
        raise FileNotFoundError("No valid CT/segmentation pairs were found.")

    print(f"Found {len(pairs)} valid CT/segmentation pairs.")

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "Case",
        "Measurement",
        "Threshold_HU",
        "MaximumDiameter_mm",
        "SliceIndex",
        "Zpos_mm",
    ]

    with output_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for ct_path, seg_path in pairs:
            case_name = ct_path.name
            if case_name.endswith(".nii.gz"):
                case_name = case_name[:-7]
            else:
                case_name = Path(case_name).stem

            if case_name.endswith("_0000"):
                case_name = case_name[:-5]

            print(f"\nProcessing: {case_name}")
            print(f"  CT : {ct_path}")
            print(f"  SEG: {seg_path}")

            ct_img = sitk.ReadImage(str(ct_path))
            seg_img = sitk.ReadImage(str(seg_path))

            # Whole-tumor diameter
            diameter, slice_index, zpos = measure_max_2d_diameter(
                ct_img=ct_img,
                seg_img=seg_img,
                extractor=extractor,
                threshold=None,
            )

            writer.writerow({
                "Case": case_name,
                "Measurement": "WholeTumor",
                "Threshold_HU": "",
                "MaximumDiameter_mm": diameter,
                "SliceIndex": "" if slice_index is None else slice_index,
                "Zpos_mm": "" if zpos is None else zpos,
            })

            print(f"  Whole tumor: {diameter:.2f} mm" if np.isfinite(diameter)
                  else "  Whole tumor: not measurable")

            # Threshold-based solid-component diameters
            for threshold in thresholds:
                diameter, slice_index, zpos = measure_max_2d_diameter(
                    ct_img=ct_img,
                    seg_img=seg_img,
                    extractor=extractor,
                    threshold=threshold,
                )

                writer.writerow({
                    "Case": case_name,
                    "Measurement": "SolidComponent",
                    "Threshold_HU": threshold,
                    "MaximumDiameter_mm": diameter,
                    "SliceIndex": "" if slice_index is None else slice_index,
                    "Zpos_mm": "" if zpos is None else zpos,
                })

                if np.isfinite(diameter):
                    print(f"  Threshold {threshold:+d} HU: {diameter:.2f} mm")
                else:
                    print(f"  Threshold {threshold:+d} HU: not measurable")

    print(f"\nCompleted. Results saved to:\n{output_csv}")


def parse_thresholds(text: str) -> list[int]:
    """
    Parse a comma-separated threshold list.

    Example:
        "200,150,100,50,0,-50,...,-600"
    """
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure whole-tumor and threshold-defined solid-component "
                    "maximum 2D diameters from CT and tumor segmentation masks."
    )
    parser.add_argument(
        "--ct-dir",
        type=Path,
        required=True,
        help="Directory containing CT images.",
    )
    parser.add_argument(
        "--seg-dir",
        type=Path,
        required=True,
        help="Directory containing tumor segmentation masks.",
    )
    parser.add_argument(
        "--yaml",
        type=Path,
        required=True,
        help="PyRadiomics parameter YAML file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--thresholds",
        type=parse_thresholds,
        default=DEFAULT_THRESHOLDS,
        help=(
            "Comma-separated HU thresholds. "
            "Default: +200 to -600 HU in 50-HU steps."
        ),
    )

    args = parser.parse_args()

    run_batch(
        ct_dir=args.ct_dir,
        seg_dir=args.seg_dir,
        yaml_path=args.yaml,
        output_csv=args.output,
        thresholds=args.thresholds,
    )


if __name__ == "__main__":
    main()
