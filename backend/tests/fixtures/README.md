# fixtures

Tiny committed sample sessions. Everything here is SYNTHETIC (no real person, no real capture).

* `session_v0_1_0_cylinder/`: a **frozen schema v0.1.0 session** (12 views, 48x64 depth, cylinder r=0.15 m,
  h=0.6 m) written on 2026-11-23 by the unmodified v0.1.0 writer, before schema v0.2.0 existed. It is the
  backward-compatibility reference: tests must keep validating, loading, migrating and reconstructing it.
  **Never regenerate or edit it**; its checksums are part of the test.

Other tests render synthetic data on the fly into pytest temp directories.
