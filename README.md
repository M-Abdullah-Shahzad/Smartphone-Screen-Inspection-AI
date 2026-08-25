Smartphone Screen Inspection AI
===============================

An end-to-end computer vision system for inspecting smartphone screens. The system detects the phone, localizes and rectifies the screen, segments visible defects, and returns a simple inspection result with supporting technical details.

Capabilities
------------

- Phone detection and padded cropping
- Screen segmentation and perspective normalization
- Scratch, oil, and stain segmentation
- Confidence filtering and image quality checks
- Defect area, connected components, centroids, and screen zones
- Plain-language decisions for users
- FastAPI deployment with a browser interface
- Optional test-time gamma and CLAHE enhancement for dark screens

Project Structure
-----------------

- `src/phone`: phone detection, training, and evaluation tools
- `src/screen`: screen labeling, localization, normalization, and training tools
- `src/defect`: defect inspection, dataset preparation, labeling, and training tools
- `src/pipeline`: mask analysis, decision logic, and end-to-end inference
- `src/evaluation`: evaluation summary tools
- `src/api`: FastAPI application and browser interface
- `configs`: model paths, dataset definitions, and decision thresholds

Requirements
------------

Use Python 3.11 or newer. Install the base dependencies with:

```powershell
pip install -r requirements.txt
```

Install the API dependencies with:

```powershell
pip install -r requirements_api.txt
```

Model Weights
------------

Model paths are configured in `configs/pipeline.yaml`:

- Phone detector
- Screen segmentation model
- Defect segmentation model

Place the corresponding weights in the configured local paths. Model files are intentionally excluded from the public repository and must be provided separately.

Run the Pipeline
----------------

Process one image:

```powershell
python src/pipeline/run_pipeline.py --input path/to/photo.jpg --output-dir runs/pipeline
```

Process a folder:

```powershell
python src/pipeline/run_pipeline.py --input-dir path/to/images --output-dir runs/pipeline
```

Add `--save-annotated` to save defect overlays. Each result includes a simple user-facing section and a technical section for development and evaluation.

Run the API
-----------

Start the service with:

```powershell
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` in a browser to upload a phone photo. The API provides:

- `GET /health` for service status
- `POST /predict` for image inspection
- `GET /` for the browser interface

Use `POST /predict?format=json` when a developer-facing JSON response is required.

Data Policy
-----------

This repository does not contain raw datasets, processed image data, run outputs, or model checkpoints. These paths are excluded by `.gitignore`. Obtain data and weights separately and keep them local when reproducing experiments.

Development Notes
-----------------

Dataset inspection and preparation scripts are designed to preserve raw inputs. Training scripts are separate from inference and evaluation tools. Decision thresholds can be adjusted in `configs/decision_thresholds.yaml` without changing application code.
