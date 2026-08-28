# Lung Tumor Solid Component Analysis

Python code for CT-based solid component extraction and 2D diameter measurement in lung tumors.

## Method

Within an AI-generated tumor segmentation mask, CT attenuation thresholds from +200 HU to -600 HU are applied at 50-HU intervals.

For each axial slice:

- Pixels with CT attenuation values equal to or greater than the specified threshold are extracted.
- Connected-component analysis is performed.
- The largest connected component is retained.
- The maximum 2D diameter is measured using the PyRadiomics shape2D `MaximumDiameter` feature.

Analysis is restricted to the central 60% of tumor-containing axial slices.

For each threshold, the largest diameter among all analyzed slices is recorded.

Whole-tumor diameter is measured using the same tumor segmentation mask without attenuation thresholding.

## Requirements

- Python
- SimpleITK
- NumPy
- SciPy
- PyRadiomics

Install the required packages with:

```bash
pip install SimpleITK numpy scipy pyradiomics
```

## Usage

Run the script as follows:

```bash
python solid_component_diameter.py \
  --ct-dir "/path/to/ct" \
  --seg-dir "/path/to/segmentation" \
  --yaml "/path/to/Params.yaml" \
  --output "/path/to/diameter_summary.csv"
```

The script automatically evaluates CT attenuation thresholds from +200 HU to -600 HU at 50-HU intervals and also measures the whole-tumor diameter without attenuation thresholding.

## Input

* `--ct-dir`: Directory containing CT images.
* `--seg-dir`: Directory containing tumor segmentation masks.
* `--yaml`: Path to the PyRadiomics parameter YAML file.
* `--output`: Path to the output CSV file.

Supported image formats are `.nrrd`, `.nii`, and `.nii.gz`.

## Output

The results are saved as a CSV file containing the following information:

* Case identifier
* Measurement type
* CT attenuation threshold
* Maximum 2D diameter
* Axial slice index
* Z-position

For threshold-based analysis, one result is recorded for each CT attenuation threshold. Whole-tumor diameter is also recorded separately.


