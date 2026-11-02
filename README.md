# Smartphone Screen Inspection AI

An end-to-end computer-vision pipeline for inspecting smartphone images. It detects and crops the phone, localizes and rectifies the screen, segments visible defects, and produces a decision with supporting technical measurements.

## Pipeline

```text
Input image
   |
Image quality assessment
   |
Phone detection and crop
   |
Screen localization and perspective normalization
   |
Defect segmentation (scratch / oil / stain)
   |
Mask analysis (confidence, area, components, zones)
   |
Decision engine and inspection report
```

## Capabilities

- Phone detection and padded cropping.
- Screen segmentation and perspective normalization.
- Scratch, oil, and stain segmentation.
- Confidence filtering and image-quality checks.
- Defect area, connected components, centroids, and screen zones.
- Plain-language inspection decisions with technical details.
- FastAPI deployment with a browser interface.
- Optional gamma and CLAHE enhancement for dark screens.

## Baseline test results

The following values are copied from the committed baseline evaluation reports. They are **test-set metrics**, not training scores.

| Model | Precision | Recall | mAP@50 | mAP@50–95 |
|---|---:|---:|---:|---:|
| Phone detector | 0.709 | 0.650 | 0.557 | 0.326 |
| Screen segmentation | 0.757 | 0.867 | 0.812 | 0.556 |
| Defect segmentation | 0.857 | 0.790 | 0.801 | 0.506 |

### Interpretation and current limitations

- The phone detector's test mAP@50 of **0.557** indicates that phone localization needs further improvement before relying on it in production.
- Defect segmentation is stronger overall, but the baseline stain class is a known weakness: test recall is approximately **0.526** and mAP@50 approximately **0.534**.
- Baseline training and evaluation reports record a CPU-only environment. Performance will vary with hardware, image size, and input quality.
- A separate real-world defect fine-tuning workflow is provided in `src/defect/finetune_defect.py`. Its checkpoint and metrics must be generated locally; no unverified improvement is claimed here.

## Project structure

- `src/phone` — phone detection, training, and evaluation
- `src/screen` — screen labeling, localization, normalization, and training
- `src/defect` — defect inspection, dataset preparation, labeling, training, and real-world fine-tuning
- `src/pipeline` — mask analysis, decision logic, and end-to-end inference
- `src/evaluation` — evaluation summary tools
- `src/api` — FastAPI application and browser interface
- `configs` — model paths, dataset definitions, and decision thresholds
- `reports` — compact evaluation summaries and metrics

## Requirements

Use Python 3.11 or newer.

Install the base dependencies:

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate
pip install -r requirements.txt
```

For the API, install its additional dependencies:

```bash
pip install -r requirements_api.txt
```

## Model weights and data

Model paths are configured in `configs/pipeline.yaml`. The configured phone, screen, and defect checkpoints are intentionally not committed. Place the corresponding weights in the configured local paths or update the YAML file to point to your own checkpoints.

Raw datasets, processed images, training runs, model weights, and large generated visual review images are not included. Obtain datasets and checkpoints separately and keep local copies outside version control.

## Run the pipeline

Process one image:

```bash
python src/pipeline/run_pipeline.py --input path/to/photo.jpg --output-dir runs/pipeline
```

Process a folder:

```bash
python src/pipeline/run_pipeline.py --input-dir path/to/images --output-dir runs/pipeline
```

Add `--save-annotated` to save defect overlays. Each result includes a user-facing decision and a technical section for evaluation.

## Run the API

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`.

- `GET /health` — service status
- `POST /predict` — inspect an image
- `GET /` — browser interface

Use `POST /predict?format=json` for a developer-facing JSON response.

## Data and reproducibility policy

- Raw datasets are never modified by the pipeline.
- Training, evaluation, preprocessing, and inference are kept separate.
- Generated run outputs and visual review artifacts belong in local `runs/` and `reports/visualizations/` folders, not in Git.
- Keep the compact metrics reports, but do not commit raw images, large generated plots, model checkpoints, or machine-specific absolute paths.

## License and usage

Check dataset and pretrained-weight licenses before redistribution or commercial use. This repository is an experimental inspection system and should not be treated as a certified quality-control instrument without further validation.
