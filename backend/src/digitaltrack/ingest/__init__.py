"""Capture ingestion: turn an uploaded capture into a dataset-schema-0.2.0 session.

Implemented: the RGB-only fallback path (Camera-app video -> `video_rgb` session).
Not implemented on purpose: an RGB-D (LiDAR) export adapter. No real export from our iPad has been inspected
yet, and no export format is assumed (see CLAUDE.md, open items).
"""
