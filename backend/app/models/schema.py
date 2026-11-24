# `models/schema.py` defines the data structures and validation rules for the
# information moving into and out of the application.
#
# Think of it as the application's data contract: it establishes what valid data
# looks like, which fields are required, what types they must have, and what
# values are acceptable.
#
# Flow:
#
#     Client submits data
#         -> models/schema.py validates structure, types, bounds and permitted values
#         -> Application logic / database / ML processing
#         -> Validated API response
#
# Architectural distinction:
# - `models/` (in the database sense) defines how information is represented and
#   stored in the database.
# - `schema.py` defines how information should be structured and validated when
#   entering or leaving the application.
# They serve different purposes, even though they may describe the same entity.
#
# In one sentence: this file establishes the application's data contracts,
# ensuring that the API, database-facing logic and ML components communicate
# using predictable, validated structures.

# What this file contains
# -----------------------
# 1. Shared enums: `AlertTier` (alert severities) and `SignalQuality` (signal quality).
# 2. Biometric payload schemas: per-modality request and model-input validation.
# 3. Response schemas: predictable API output structures, for example
#
#     {
#       "id": "alert_0001",
#       "tier": "HIGH",
#       "message": "Abnormal cardiac signal detected",
#       "created_at": "2026-11-04T10:30:00Z"
#     }
#
# Biometric payload schemas establish which fields are required, which must be
# strings, the permitted string lengths, which values or formats are acceptable,
# and how missing or malformed data is rejected. This protects downstream
# processing and ML components from receiving unexpected input.

# Layout
# ------
# models/
# └── schema.py
#     ├── SHARED ENUMS          AlertTier, SignalQuality
#     ├── REQUEST SCHEMAS       ECGRequest, PPGRequest, ParkinsonsRequest,
#     │                         FaceRequest, DysarthriaRequest
#     ├── MODEL INPUT SCHEMAS   ECGModelInput, PPGModelInput, ParkinsonsModelInput,
#     │                         FaceModelInput, DysarthriaModelInput
#     └── RESPONSE SCHEMAS      ECGResponse, PPGResponse, ParkinsonsResponse,
#                               FaceResponse, DysarthriaResponse, AlertResponse


# =============================================================================
# SHARED ENUMS
# =============================================================================

from datetime import datetime
from enum import Enum

# 1. AlertTier
# ------------
# Permitted severity levels for alerts. The purpose is to ensure that different
# parts of the application use consistent alert classifications rather than
# inventing arbitrary strings.
#
# Why `class AlertTier(str, Enum)`?
# - `Enum` makes the class a fixed collection of permitted values.
# - `str` makes its members behave like strings, which is useful when exchanging
#   data through JSON APIs.
#
#     tier = AlertTier.HIGH
#     print(tier.value)  # HIGH
#     print(tier.name)   # HIGH
#
# `.name` is the member's identifier in Python; `.value` is the value assigned to
# that member. Here they coincide, but they do not have to.
#
# Pydantic then validates incoming values against the enum, so there is no need
# to manually check every severity:
#
#     class AlertResponse(BaseModel):
#         tier: AlertTier
#
#     AlertResponse(tier="HIGH")     # valid
#     AlertResponse(tier="EXTREME")  # validation error
#
# Senior-engineer insight: this is a small example of domain modelling. You are
# defining a concept that has a fixed set of valid states, rather than allowing
# arbitrary strings to represent it.
class AlertTier(str, Enum):
    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# 2. SignalQuality
# ----------------
# Permitted quality classifications for incoming biometric signals. Gives the
# application a consistent way to describe whether a signal is reliable enough
# to be used for analysis.
class SignalQuality(str, Enum):
    POOR = "POOR"
    FAIR = "FAIR"
    GOOD = "GOOD"
    EXCELLENT = "EXCELLENT"


# =============================================================================
# DESIGN DECISIONS BEHIND THE MODALITY SCHEMAS
# =============================================================================

# Five input modalities
# ---------------------
# Stroke Shield needs to support five input categories:
#
# | Modality            | What the payload represents                              |
# | ------------------- | -------------------------------------------------------- |
# | ECG                 | Electrocardiogram signal data                            |
# | PPG                 | Photoplethysmography signal data                         |
# | Parkinson's data    | Measurements or features used for Parkinson's assessment |
# | MediaPipe face data | Facial landmarks and extracted facial movement features  |
# | Voice data          | Audio input or extracted speech/acoustic features        |
#
# Vocabulary: a "payload" is the actual data sent from one system to another in
# a request or response. When the frontend posts Parkinson's voice measurements
# to the FastAPI backend, those measurements make up the request payload.

# Why separate schemas per modality?
# ----------------------------------
# Each modality has different data structures and validation requirements:
# - ECG and PPG need sampling rates, signal measurements and timestamps.
# - Parkinson's assessment may involve tremor, tapping or gait measurements,
#   depending on the assessment protocol.
# - MediaPipe facial data may contain arrays of landmark coordinates.
# - Voice may contain audio metadata, transcription or acoustic features.
# Forcing all of that into one generic structure would make validation
# complicated and introduce unnecessary optional fields.

# Raw data versus derived features
# --------------------------------
# Keep these two representations distinct in the schemas:
# - Raw ECG: sampled voltage measurements.
#   Derived ECG: heart rate, RR intervals and HRV features.
# - Raw voice: an audio file or audio stream.
#   Derived voice: MFCCs, pitch, speech rate and other acoustic features.
# - Raw face data: landmark coordinates.
#   Derived face data: facial movement measurements.

# Three schemas per modality
# --------------------------
# 1. <Modality>Request    - what the client sends (submission metadata).
# 2. <Modality>ModelInput - the preprocessed input handed to the ML pipeline.
# 3. <Modality>Response   - what the backend returns (prediction and confidence).
#
# Because of that split, request schemas usually carry only `content_type` (and
# sometimes a `source`). The actual bytes are received separately through
# FastAPI's `UploadFile` or a streaming mechanism: the request describes the
# submission, while the processing pipeline handles the recording itself.

# Where the signal lives
# ----------------------
# A model-input schema describes the processed data a model receives, not the
# uploaded file. For ECG, PPG and Dysarthria it currently carries only metadata;
# the samples themselves are NumPy arrays or tensors built in the preprocessing
# and inference layer:
#
#     uploaded file -> validate metadata -> decode & preprocess -> array -> model
#
# Do not add a NumPy array or PyTorch tensor to an API schema just because the
# model needs one: those are internal computational representations, not part of
# the API contract. They get defined with the pipelines.

# Audio: raw audio in, features extracted in the backend
# ------------------------------------------------------
#     audio upload / microphone
#         -> FastAPI receives file -> validate -> decode & preprocess
#         -> extract model-specific features -> inference -> response
#
# Parkinson's: the UCI dataset holds pre-extracted acoustic features, not raw
# audio, so the backend must reproduce those same 16 features with compatible
# extraction methods before inference.
# Dysarthria: TORGO provides recordings, so the preprocessing and feature
# extraction pipeline follows whatever model we train.
#
# Both audio request schemas stay metadata-only; neither carries audio bytes.


# =============================================================================
# SHARED BASE SCHEMA AND PYDANTIC CONVENTIONS
# =============================================================================

from pydantic import BaseModel, ConfigDict, Field

# ImmutableSchema
# ---------------
# Holds the configuration every schema in this file shares, so classes do not
# repeat `model_config = ConfigDict(...)`:
#
# | Setting               | Purpose                                                     |
# | --------------------- | ----------------------------------------------------------- |
# | `frozen=True`         | Prevents reassignment of model fields after creation.       |
# | `extra="forbid"`      | Rejects unexpected fields instead of silently ignoring them. |
# | `allow_inf_nan=False` | Rejects NaN and infinity in numeric fields.                 |
#
# This is inheritance and configuration reuse: a schema that inherits from
# `ImmutableSchema` automatically receives these validation rules. A subclass
# that needs more settings declares only those extra keys - Pydantic merges them
# with the inherited config.
#
# Note: this provides shallow immutability. A field cannot be reassigned, but a
# mutable object it points to could still be changed in place. The facial
# schemas below therefore use tuples for their collection fields; see "Deep
# immutability" in the Face section.
class ImmutableSchema(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        allow_inf_nan=False,
    )


# Conventions used by the fields in this file:
# - `Field(alias=...)`: connects clean Python names to the original dataset
#   column names.
# - `ge` / `gt` / `le`: numerical boundaries (`ge=0` means greater than or equal
#   to zero; `gt=0` means strictly greater than zero; `le=1` means less than or
#   equal to one).
# - `populate_by_name=True`: allows a model to accept either the Python field
#   names or their aliases. Only `ParkinsonsVoiceFeatures` needs it, so that
#   class adds it to the inherited config.
#
# Response schemas share the same two ideas:
# - `prediction`: the predicted class label, e.g. AFIB or NORMAL.
# - `confidence`: the model's confidence score, always between 0 and 1.
#
# Caveat: observed dataset minimums and maximums should not automatically become
# permanent API limits. A real recording may produce valid values outside the
# training dataset's observed range. Treat these constraints as structural
# guards, not as a complete clinical validation system; the dataset's actual
# ranges still need to be verified before enforcing stricter limits.


# =============================================================================
# 1. PARKINSON'S VOICE SCHEMAS (UCI dataset)
# =============================================================================

# ParkinsonsVoiceFeatures
# -----------------------
# Defines the 16 numerical voice measurements the model expects, giving one
# reusable, validated definition for API requests, model preprocessing and tests.
#
# | Field                                                          | Meaning                                    |
# | -------------------------------------------------------------- | ------------------------------------------ |
# | `Jitter(%)`, `Jitter(Abs)`                                     | Variation in vocal frequency               |
# | `Jitter:RAP`, `Jitter:PPQ5`, `Jitter:DDP`                      | Different measures of frequency variation  |
# | `Shimmer`, `Shimmer(dB)`                                       | Variation in vocal amplitude               |
# | `Shimmer:APQ3`, `Shimmer:APQ5`, `Shimmer:APQ11`, `Shimmer:DDA` | Different measures of amplitude variation  |
# | `NHR`, `HNR`                                                   | Noise-to-harmonics measurements            |
# | `RPDE`, `DFA`, `PPE`                                           | Nonlinear characteristics of voice signals |
#
# The dataset's original column names contain `%`, `:` and parentheses, which
# cannot be used in Python attribute names. The fields therefore get clean Python
# names, and `Field(alias=...)` maps them back to the original columns;
# `populate_by_name=True` lets callers use either spelling.
#
# Constraint rationale:
# - RPDE: 0-1 is appropriate for this feature.
# - DFA: no upper bound. Observed values are below 1, but DFA is not universally
#   restricted to that interval.
# - HNR: no nonnegative constraint; HNR can be negative for some signals.
# - Jitter, shimmer, NHR and PPE: nonnegative constraints are reasonable.
class ParkinsonsVoiceFeatures(ImmutableSchema):
    # Merged with the inherited config; frozen/extra/allow_inf_nan still apply.
    model_config = ConfigDict(populate_by_name=True)
    jitter_percent: float = Field(alias="Jitter(%)", ge=0)
    jitter_abs: float = Field(alias="Jitter(Abs)", ge=0)
    jitter_rap: float = Field(alias="Jitter:RAP", ge=0)
    jitter_ppq5: float = Field(alias="Jitter:PPQ5", ge=0)
    jitter_ddp: float = Field(alias="Jitter:DDP", ge=0)
    shimmer: float = Field(ge=0)
    shimmer_db: float = Field(alias="Shimmer(dB)", ge=0)
    shimmer_apq3: float = Field(alias="Shimmer:APQ3", ge=0)
    shimmer_apq5: float = Field(alias="Shimmer:APQ5", ge=0)
    shimmer_apq11: float = Field(alias="Shimmer:APQ11", ge=0)
    shimmer_dda: float = Field(alias="Shimmer:DDA", ge=0)
    nhr: float = Field(ge=0)
    hnr: float
    rpde: float = Field(ge=0, le=1)
    dfa: float = Field(ge=0)
    ppe: float = Field(ge=0)


# AudioInputSource
# ----------------
# Where the audio came from: an uploaded file or a live microphone recording.
class AudioInputSource(str, Enum):
    UPLOAD = "UPLOAD"
    MICROPHONE = "MICROPHONE"


# ParkinsonsRequest
# -----------------
# Wraps the submission metadata for a Parkinson's voice assessment: the audio
# source and its format. The audio bytes themselves arrive via `UploadFile`.
class ParkinsonsRequest(ImmutableSchema):
    source: AudioInputSource
    content_type: str


# ParkinsonsModelInput
# --------------------
# Packages the extracted features for inference. Composing schemas this way
# reuses an existing validated model instead of repeating 16 fields.
class ParkinsonsModelInput(ImmutableSchema):
    voice_features: ParkinsonsVoiceFeatures


# ParkinsonsResponse
# ------------------
# The UCI dataset has two regression targets, so the response represents both.
# `motor_UPDRS` and `total_UPDRS` are prediction targets in this dataset, which
# is why they are not input fields. These are predicted severity scores, not
# probabilities or diagnostic labels.
class ParkinsonsResponse(ImmutableSchema):
    motor_updrs: float
    total_updrs: float

# Parkinson's structure recap
# ---------------------------
# | Schema                  | Responsibility                            |
# | ----------------------- | ----------------------------------------- |
# | ParkinsonsRequest       | Receives audio submission metadata        |
# | ParkinsonsVoiceFeatures | Validates the 16 acoustic features        |
# | ParkinsonsModelInput    | Packages extracted features for inference |
# | ParkinsonsResponse      | Returns predicted UPDRS scores            |
#
# One important caveat: the feature extractor must produce measurements
# compatible with the UCI training features. Otherwise the model receives a
# different feature distribution from the one it learned.


# =============================================================================
# 2. FACE SCHEMAS (MediaPipe Face Landmarker)
# =============================================================================

#     FaceRequest -> MediaPipe processing -> FaceModelInput -> Asymmetry model -> FaceResponse
#
# MediaPipe Face Landmarker detects facial landmarks: points representing
# locations on the face. The standard Face Mesh model produces 468 landmarks
# (478 with refined landmarks). Each landmark has coordinates:
# - `x`: horizontal position.
# - `y`: vertical position.
# - `z`: depth estimate.
# Coordinates are generally normalized, so x and y are typically between 0 and 1,
# while z can be negative or positive.
#
# Final structure:
# - FaceRequest    - what the client sends (`source` plus `content_type`; the
#                    image itself arrives through `UploadFile`).
# - FaceFrame      - one captured or uploaded frame (`timestamp` and `landmarks`).
# - FaceLandmark   - one facial coordinate (`x`, `y`, `z`).
# - FaceModelInput - one or more processed frames supplied to the analysis
#                    pipeline.
# - FaceResponse   - the asymmetry result returned to the client.
#
# A separate `FacePayload` or `FaceAnalysisInput` wrapper would be unnecessary
# duplication; `FaceRequest` and `FaceModelInput` cover those roles.

# FaceInputSource
# ---------------
class FaceInputSource(str, Enum):
    UPLOAD = "UPLOAD"
    CAMERA = "CAMERA"


# FaceRequest
# -----------
class FaceRequest(ImmutableSchema):
    source: FaceInputSource
    content_type: str


# FaceModelInput and its two supporting classes
# ---------------------------------------------
# `FaceModelInput` is the data structure the facial asymmetry model receives
# after MediaPipe has processed the image.
#
# Deep immutability
# -----------------
# `frozen=True` blocks reassignment of a field, but not in-place mutation of the
# object it points to:
#
#     face.frames = []         # blocked by frozen=True
#     face.frames.append(...)  # still possible if `frames` is a list
#
# Because validated data should stay unchanged after creation, collection fields
# use `tuple` instead of `list`:
#
# | `list[FaceFrame]`              | `tuple[FaceFrame, ...]`            |
# | ------------------------------ | ---------------------------------- |
# | Mutable                        | Immutable                          |
# | Items can be added or removed  | Items cannot be added or removed   |
#
# Pydantic still accepts an incoming list and converts it to a tuple during
# validation, so API callers keep sending ordinary JSON arrays, and
# `min_length` / `max_length` continue to constrain the collection's length.
#
# Together with the base class, this makes the facial hierarchy deeply immutable:
# - `FaceModelInput` contains tuples of `FaceFrame`.
# - `FaceFrame` contains tuples of `FaceLandmark`.
# - `FaceLandmark` contains only numeric scalar fields.
# - All three inherit from `ImmutableSchema`.
#
# Caveat: a tuple fixes collection mutability, but it does not make every
# possible nested object deeply immutable. A future field holding another mutable
# object (for example a dict) would need the same treatment.
#
# The other modalities need no conversion today: `ParkinsonsVoiceFeatures` holds
# only scalars, and the ECG, PPG and Dysarthria model inputs hold only metadata.
class FaceLandmark(ImmutableSchema):
    x: float
    y: float
    z: float


class FaceFrame(ImmutableSchema):
    timestamp: float = Field(ge=0)
    landmarks: tuple[FaceLandmark, ...] = Field(min_length=468, max_length=478)


class FaceModelInput(ImmutableSchema):
    frames: tuple[FaceFrame, ...] = Field(min_length=1)

# Why this structure?
# - FaceLandmark: one facial coordinate.
# - FaceFrame: all detected landmarks at one point in time.
# - FaceModelInput: one or more processed frames supplied to the analysis
#   pipeline.
#
# Why only `min_length=1` on `frames`? Because we have not confirmed whether the
# downstream processing uses every frame or selects a subset. Do not impose an
# arbitrary restriction before the model's actual requirements are defined.


# FaceResponse
# ------------
# The result the backend sends back to the client after facial asymmetry
# analysis.
# - asymmetry_score: numerical measure of facial asymmetry, normalised 0-1.
# - prediction: the model's predicted class, such as ASYMMETRIC or SYMMETRIC.
# - confidence: model confidence in its prediction, between 0 and 1.
#
# These are proposed API output fields, not confirmed outputs of the existing
# model. Align them with the model's actual output and evaluation before
# finalising.
class FaceResponse(ImmutableSchema):
    asymmetry_score: float = Field(ge=0, le=1)
    prediction: str
    confidence: float = Field(ge=0, le=1)


# =============================================================================
# 3. DYSARTHRIA SCHEMAS (TORGO recordings)
# =============================================================================

# TORGO contains speech recordings from people with dysarthria and control
# speakers. The audio stays raw; features are extracted in the backend:
# 1. `DysarthriaRequest`    - the incoming audio submission.
# 2. `DysarthriaModelInput` - metadata for the preprocessed audio.
# 3. `DysarthriaResponse`   - the model's prediction.
#
# `source` mirrors the facial input: UPLOAD or MICROPHONE (`AudioInputSource`).
class DysarthriaRequest(ImmutableSchema):
    source: AudioInputSource
    content_type: str

# - `source`: whether the audio came from an uploaded file or a live microphone.
# - `content_type`: the audio format, such as `audio/wav` or `audio/mpeg`.


# DysarthriaModelInput
# --------------------
# TORGO is raw audio; the model may consume features, spectrograms or tensors
# depending on its architecture. Metadata only (see "Where the signal lives").
# - `sample_rate`: samples per second (e.g. 16,000 Hz).
# - `duration_seconds`: length of the recording.
# - `audio_format`: format of the processed audio.
class DysarthriaModelInput(ImmutableSchema):
    sample_rate: int = Field(gt=0)
    duration_seconds: float = Field(gt=0)
    audio_format: str


# DysarthriaResponse
# ------------------
# Distinguishes the model's prediction from its confidence.
# - `prediction`: the predicted class, such as DYSARTHRIC or CONTROL.
# - `confidence`: the model's confidence score, between 0 and 1.
#
# These labels assume a binary classifier; verify them against the actual
# training labels when the model is implemented.
class DysarthriaResponse(ImmutableSchema):
    prediction: str
    confidence: float = Field(ge=0, le=1)


# =============================================================================
# 4. ECG SCHEMAS
# =============================================================================

# ECGRequest
# ----------
# `content_type` identifies the uploaded file format, e.g. CSV or WAV (depending
# on the ECG dataset and preprocessing pipeline). No input-source enum is needed
# because uploads are the only supported submission method.
class ECGRequest(ImmutableSchema):
    content_type: str


# ECGModelInput
# -------------
# Signal metadata; the samples themselves live in the preprocessing layer (see
# "Where the signal lives").
# - `sampling_rate`: samples per second.
# - `num_leads`: number of ECG leads in the recording.
# - `duration_seconds`: length of the recording.
#
# These values must eventually match the model's training configuration.
class ECGModelInput(ImmutableSchema):
    sampling_rate: int = Field(gt=0)
    num_leads: int = Field(gt=0)
    duration_seconds: float = Field(gt=0)


# ECGResponse
# -----------
# The response depends on what the ECG model is trained to predict. For the AFib
# classification model used in Stroke Shield:
# - `prediction`: predicted class, e.g. AFIB or NORMAL.
# - `confidence`: model confidence score, between 0 and 1.
#
# This assumes a binary AFib classifier; update the prediction vocabulary if more
# arrhythmia classes are added later.
class ECGResponse(ImmutableSchema):
    prediction: str
    confidence: float = Field(ge=0, le=1)


# =============================================================================
# 5. PPG SCHEMAS (Photoplethysmography)
# =============================================================================

# PPG measures changes in blood volume using light, commonly through a pulse
# oximeter or wearable sensor. The same three boundaries apply, and recordings
# are uploaded files only.

# PPGRequest
# ----------
# `content_type` identifies the uploaded recording format, such as CSV. No
# input-source enum is necessary because uploads are the only supported method.
class PPGRequest(ImmutableSchema):
    content_type: str


# PPGModelInput
# -------------
# Like ECG, minus `num_leads`: PPG sensors do not use ECG leads. Metadata only.
# - `sampling_rate`: samples per second.
# - `duration_seconds`: length of the recording.
class PPGModelInput(ImmutableSchema):
    sampling_rate: int = Field(gt=0)
    duration_seconds: float = Field(gt=0)


# PPGResponse
# -----------
# Align this with the actual PPG model's prediction task. The dataset has
# balanced classes (395 per class), so a binary classifier yields:
# - `prediction`: predicted class, using the exact labels from the training
#   dataset.
# - `confidence`: model confidence score, between 0 and 1.
#
# Verify the dataset's class labels before hardcoding an enum.
class PPGResponse(ImmutableSchema):
    prediction: str
    confidence: float = Field(ge=0, le=1)


# =============================================================================
# 6. ALERT RESPONSE
# =============================================================================

# One alert as returned by the API. Deliberately minimal: `tier` reuses the
# `AlertTier` enum so severities stay consistent, and fields get added only when
# the alert workflow requires them.
# - `id`: identifies the alert.
# - `tier`: severity, from the `AlertTier` enum.
# - `message`: explains what triggered the alert.
# - `created_at`: when it was generated.
class AlertResponse(ImmutableSchema):
    id: str
    tier: AlertTier
    message: str
    created_at: datetime


# =============================================================================
# CROSS-MODALITY RECAP
# =============================================================================
#
# | Modality    | Request           | Model input           | Response            |
# | ----------- | ----------------- | --------------------- | ------------------- |
# | Parkinson's | ParkinsonsRequest | ParkinsonsModelInput  | ParkinsonsResponse  |
# | Face        | FaceRequest       | FaceModelInput        | FaceResponse        |
# | Dysarthria  | DysarthriaRequest | DysarthriaModelInput  | DysarthriaResponse  |
# | ECG         | ECGRequest        | ECGModelInput         | ECGResponse         |
# | PPG         | PPGRequest        | PPGModelInput         | PPGResponse         |
#
# Every modality follows Request -> ModelInput -> Response. Requests validate
# metadata only, model inputs describe preprocessed data, and responses pair a
# class label with a confidence score. `AlertResponse` is the first
# cross-modality response: an alert can be raised by any modality.
#
# Next step: backend structure and the first functional endpoint. Until then,
# `content_type` stays a free-form string.
