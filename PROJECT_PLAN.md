# Mobile Phone Defect AI — Master Project Plan

## 1. Project Goal

Build a computer vision system that accepts a mobile phone image and:

1. Finds the mobile phone even when it is far away or surrounded by background.
2. Crops and resizes the phone region.
3. Finds/isolate the screen region.
4. Normalizes the screen perspective when needed.
5. Detects and segments:

   * scratch
   * oil
   * stain
6. Calculates defect:

   * type
   * confidence
   * location
   * affected area percentage
7. Supports multiple defects in the same image.
8. Returns:

   * OK
   * Scratch
   * Oil
   * Stain
   * Multiple
   * Uncertain
   * Phone Too Far
   * Image Too Blurry / Poor Quality
9. Must prioritize reliable real-world behavior instead of only high validation mAP.

---

# 2. Datasets

## Dataset A — Phone Detection Dataset

Name:
Mobile Phone Dataset | Smartphone & Feature Phone

Location:

datasets/phone_dataset/

Current size:
Approximately 304.9 MB

Current reported structure includes:

* Mobile_image/
* Annotations/

Approximate image count currently reported:
200 images.

Primary purpose:

TRAIN THE PHONE DETECTOR.

Target class:

phone

Do not mix this dataset with the defect dataset.

---

## Dataset B — Defect Segmentation Dataset

Name:
Mobile Phone Defect Segmentation Dataset

Location:

datasets/defect_dataset/

Approximate size:
2 GB

Target defect classes:

* scratch
* oil
* stain

Also contains good/non-defective examples and ground-truth/mask information.

Primary purpose:

TRAIN THE DEFECT SEGMENTATION MODEL.

Do not assume the exact folder or annotation structure until the dataset inspection phase confirms it.

---

# 3. Final System Architecture

INPUT IMAGE
|
v
IMAGE QUALITY CHECK
|
v
PHONE DETECTOR
|
v
PHONE CROP
|
v
PADDING + RESIZE
|
v
SCREEN LOCALIZATION
|
v
SCREEN CROP
|
v
PERSPECTIVE NORMALIZATION
|
v
DEFECT SEGMENTATION
|
+---- scratch
+---- oil
+---- stain
|
v
MASK ANALYSIS
|
+---- location
+---- area percentage
+---- confidence
|
v
DECISION ENGINE
|
+---- OK
+---- SCRATCH
+---- OIL
+---- STAIN
+---- MULTIPLE
+---- UNCERTAIN
+---- PHONE_TOO_FAR
+---- POOR_IMAGE_QUALITY

---

# 4. Core Design Principles

## Principle 1

Do not merge the two raw datasets.

Dataset A is primarily for phone detection.

Dataset B is primarily for defect segmentation.

## Principle 2

Do not train before inspecting the datasets and annotations.

## Principle 3

Never invent annotation formats.

Inspect the actual files first.

## Principle 4

Never modify raw datasets.

Create processed copies under a separate generated/processed directory if required.

## Principle 5

Keep training, evaluation, preprocessing and inference code separate.

## Principle 6

Prefer small modular scripts instead of one giant Python file.

## Principle 7

Do not permanently generate millions of augmented images unless there is a demonstrated need.

Prefer runtime augmentation.

## Principle 8

Do not trust high mAP from a potentially easy or leaked validation split.

Check duplicates and split correctly.

## Principle 9

The final system must support multiple defects.

An image may contain:

scratch + oil
scratch + stain
oil + stain
scratch + oil + stain

## Principle 10

The system must have an UNCERTAIN result.

Never force an ambiguous prediction into scratch/oil/stain.

---

# 5. Development Phases

## PHASE 0 — Project Setup

Tasks:

* verify project folders
* verify datasets exist
* verify Python environment
* verify requirements
* verify Git
* verify Colab compatibility
* create README

No model training.

---

## PHASE 1 — Dataset Inspection

Tasks:

* inspect Dataset A
* inspect Dataset B
* count images
* inspect image sizes
* inspect annotation formats
* inspect classes
* inspect masks
* detect missing annotations
* detect corrupt images
* detect duplicate/near-duplicate images
* calculate class distribution
* produce dataset reports

Expected reports:

reports/phone_dataset_report.json
reports/defect_dataset_report.json

No training until this phase is complete.

---

## PHASE 2 — Dataset Validation and Split

Tasks:

* validate annotations
* define train/validation/test split
* avoid duplicate leakage
* keep source image and all derived variants in the same split
* generate reproducible split files
* create processed dataset structure

Target starting split:

70% train
15% validation
15% test

Adjust after inspection if necessary.

---

## PHASE 3 — Phone Detection

Dataset:

Dataset A.

Target:

class 0 = phone

Tasks:

* convert annotation format if needed
* create YOLO detection dataset
* train a small YOLO detection model
* evaluate precision
* evaluate recall
* evaluate mAP50
* evaluate mAP50-95
* inspect false positives
* inspect false negatives
* test phone size robustness

Important requirement:

The detector must handle phones that are small/far away in a larger image.

---

## PHASE 4 — Phone Crop Pipeline

Tasks:

* load trained phone detector
* detect phone
* expand bounding box slightly with configurable padding
* crop phone
* resize phone
* save visualization examples
* implement failure handling

Failure conditions:

* phone not detected
* confidence too low
* phone too small

Possible output:

PHONE_TOO_FAR
PHONE_NOT_FOUND
UNCERTAIN

---

## PHASE 5 — Screen Localization

First inspect whether the existing datasets already contain reliable screen information.

Choose the simplest reliable solution.

Priority:

1. Existing screen annotation
2. Reliable derived screen annotation
3. OpenCV geometric/contour approach
4. Separate deep-learning screen model only if required

Do not train a screen model automatically without evidence that it is needed.

---

## PHASE 6 — Perspective Normalization

Tasks:

* find screen corners
* transform screen to a normalized rectangle
* maintain coordinate mapping
* test different phone angles
* verify that defect masks can still be mapped correctly

---

## PHASE 7 — Defect Segmentation

Dataset:

Dataset B.

Classes:

0 = scratch
1 = oil
2 = stain

Tasks:

* preserve original masks
* convert masks to the required segmentation format
* create train/validation/test split
* train YOLO segmentation
* compare multiple input sizes
* evaluate per class
* inspect small scratches
* inspect false classifications

Primary metrics:

precision
recall
mAP50
mAP50-95
per-class performance
small-defect recall

---

## PHASE 8 — Robustness Improvement

Tasks:

* realistic augmentation
* brightness changes
* contrast changes
* blur
* noise
* scale
* rotation
* perspective changes
* far-phone simulation
* difficult examples
* hard negatives

Hard negatives include:

* clean screen
* reflections
* glare
* dust
* screen protector artifacts
* normal screen patterns
* other visual artifacts

Do not use unrealistic augmentations that destroy defect appearance.

---

## PHASE 9 — Multiple Defect Handling

The model must support simultaneous detections.

Example:

scratch + oil

or:

scratch + oil + stain

The pipeline must not force only one defect.

---

## PHASE 10 — Mask Post-Processing

Tasks:

* clean masks
* remove tiny noise where appropriate
* calculate connected components
* calculate defect area
* calculate defect centroid
* calculate location
* map location to screen zones

Area:

defect_pixels / screen_pixels * 100

Total affected area must use the union of defect masks to avoid double-counting overlap.

---

## PHASE 11 — Decision Engine

Possible final states:

OK
SCRATCH
OIL
STAIN
MULTIPLE
UNCERTAIN
PHONE_TOO_FAR
POOR_IMAGE_QUALITY
PHONE_NOT_FOUND

The decision engine must use calibrated thresholds based on validation/test results.

Do not blindly use 0.5 confidence.

---

## PHASE 12 — End-to-End Pipeline

Connect:

image quality
->
phone detector
->
phone crop
->
screen localization
->
screen crop
->
perspective correction
->
defect segmentation
->
mask analysis
->
decision engine

Test complete pipeline on difficult cases.

---

## PHASE 13 — Evaluation

Create separate evaluation groups:

* normal
* clean
* far phone
* close phone
* tilted phone
* low light
* blur
* reflections
* small defects
* multiple defects

Report results separately.

Do not report only overall mAP.

---

## PHASE 14 — Deployment

After the model is stable:

* create inference API
* create FastAPI endpoint
* accept image
* run complete pipeline
* return structured JSON
* optionally save annotated output image

Do not start deployment before model evaluation is complete.

---

# 6. Recommended Model Strategy

Initial phone model:

YOLO small detection model.

Initial defect model:

YOLO small segmentation model.

Do not use a large model immediately.

If the model underfits, test a larger model.

For defect segmentation, evaluate multiple input resolutions because scratches can be very small.

---

# 7. Data Rules

Raw datasets must remain unchanged.

Use:

datasets/

for raw data.

Use a separate generated directory for:

* converted annotations
* splits
* processed images
* generated metadata

Never overwrite raw images or masks.

---

# 8. Code Rules

Use Python.

Use pathlib for paths.

Use YAML configs.

Do not hard-code absolute Windows or Colab paths inside training scripts.

Use relative/configurable paths.

Every training script must support:

* configurable dataset path
* model name
* image size
* batch size
* epochs
* output directory
* device

Do not duplicate code.

Create reusable utility functions where appropriate.

---

# 9. Experiment Rules

Every training run must record:

* model
* dataset version
* image size
* batch size
* epochs
* augmentation settings
* metrics
* weights path

Do not overwrite previous experiments.

Use:

runs/

and unique experiment names.

---

# 10. Copilot Agent Rules

Before every task:

1. Read PROJECT_PLAN.md.
2. Identify the current phase.
3. Inspect existing files.
4. Do not modify unrelated files.
5. Do not skip ahead.
6. Do not invent dataset structure.
7. Do not train before the phase explicitly requires training.
8. Do not create duplicate utilities.
9. Keep implementation modular.
10. Keep responses short.

After each task, report only:

* files created/changed
* what was implemented
* command to run
* result
* next phase

Do not generate large explanations unless requested.

---

# 11. Current Starting Point

The project is currently at:

PHASE 0

The next action is:

Inspect project structure and datasets.

Do NOT:

* train a model
* convert annotations
* modify raw datasets
* create synthetic data
* create inference code

until inspection is complete.

---

# 12. Definition of Success

The final system should be able to:

1. Detect a phone in a full image.
2. Handle a phone that is relatively far away.
3. Crop the phone automatically.
4. Isolate the screen.
5. Normalize screen perspective.
6. Detect scratch/oil/stain.
7. Support multiple defects.
8. Segment defects.
9. Calculate defect location.
10. Calculate defect percentage.
11. Reject poor-quality images.
12. Return UNCERTAIN instead of guessing when appropriate.
13. Produce reliable evaluation results on unseen test data.

The primary goal is real-world reliability, not merely a high training/validation score.
