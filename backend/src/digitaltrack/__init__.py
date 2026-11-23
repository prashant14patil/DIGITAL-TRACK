"""DigitalTrack: geometry-first body scanning POC backend."""

__version__ = "0.2.0"

# Dataset contract (session.json, frame_metadata.json, calibration/*.json). Written by input adapters.
DATASET_SCHEMA_VERSION = "0.2.0"
# Older dataset versions that are still read (validated against schemas/v<version>/ and upgraded in memory).
LEGACY_DATASET_SCHEMA_VERSIONS = ("0.1.0",)
SUPPORTED_DATASET_SCHEMA_VERSIONS = LEGACY_DATASET_SCHEMA_VERSIONS + (DATASET_SCHEMA_VERSION,)

# Processing outputs (measurements, quality_report, processing_log). Unchanged by the dataset 0.2.0 bump.
OUTPUT_SCHEMA_VERSION = "0.1.0"
