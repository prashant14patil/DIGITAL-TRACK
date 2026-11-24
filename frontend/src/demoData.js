// DEMO data for the body-avatar view. These are SAMPLE values, not measurements of anyone.
// Real values will come from the Tanita BCM (body composition) once its import is approved and built.
// Categories are part of each sample reading; no official standard ranges are loaded or computed here.

export const DEMO_PROFILE = { sex: "Male", height_cm: 174, age: 30 };

export const SEGMENTS = [
  { key: "left_arm", label: "Left arm", side: "left", slot: 0 },
  { key: "left_leg", label: "Left leg", side: "left", slot: 2 },
  { key: "right_arm", label: "Right arm", side: "right", slot: 0 },
  { key: "trunk", label: "Trunk", side: "right", slot: 1 },
  { key: "right_leg", label: "Right leg", side: "right", slot: 2 },
];

export const CATEGORIES = [
  { key: "under", label: "Under", color: "#6ea8e8" },
  { key: "standard", label: "Standard", color: "#62c3a4" },
  { key: "upper", label: "Upper standard", color: "#e2c15c" },
  { key: "over", label: "Over", color: "#e79a45" },
];
export const CATEGORY = Object.fromEntries(CATEGORIES.map((c) => [c.key, c]));

const seg = (la, ra, tr, ll, rl) => ({
  left_arm: { fat_pct: la[0], category: la[1] },
  right_arm: { fat_pct: ra[0], category: ra[1] },
  trunk: { fat_pct: tr[0], category: tr[1] },
  left_leg: { fat_pct: ll[0], category: ll[1] },
  right_leg: { fat_pct: rl[0], category: rl[1] },
});

export const DEMO_READINGS = [
  { date: "2026-03-12", weight_kg: 67.4, body_fat_pct: 21.6, muscle_kg: 50.6,
    segments: seg([18.3, "upper"], [18.0, "upper"], [23.6, "upper"], [21.8, "upper"], [21.5, "upper"]) },
  { date: "2026-05-02", weight_kg: 67.2, body_fat_pct: 20.9, muscle_kg: 50.9,
    segments: seg([17.6, "upper"], [17.2, "standard"], [22.8, "upper"], [21.2, "standard"], [21.0, "standard"]) },
  { date: "2026-06-20", weight_kg: 67.0, body_fat_pct: 20.1, muscle_kg: 51.1,
    segments: seg([16.8, "standard"], [16.5, "standard"], [22.0, "upper"], [20.8, "standard"], [20.5, "standard"]) },
  { date: "2026-08-25", weight_kg: 66.8, body_fat_pct: 19.5, muscle_kg: 51.4,
    segments: seg([16.1, "standard"], [15.8, "standard"], [21.3, "standard"], [20.4, "standard"], [20.1, "standard"]) },
];

export const DEMO_TARGET = { date: "target", weight_kg: 66.0, body_fat_pct: 18.0, muscle_kg: 52.0,
  segments: seg([15.0, "standard"], [14.8, "standard"], [19.5, "standard"], [19.0, "standard"], [18.8, "standard"]) };

const lerp = (a, b, t) => a + (b - a) * t;

// Reading at fractional position t in [0, n-1]: numbers interpolate, categories follow the nearer reading.
export function readingAt(readings, t) {
  const i = Math.max(0, Math.min(readings.length - 1, Math.floor(t)));
  const j = Math.min(readings.length - 1, i + 1);
  const f = Math.max(0, Math.min(1, t - i));
  const a = readings[i], b = readings[j];
  // segments are optional on manual readings: interpolate only when both sides have them, else use the nearer one
  let segments = null;
  if (a.segments && b.segments) {
    segments = {};
    for (const k of Object.keys(a.segments)) {
      segments[k] = {
        fat_pct: lerp(a.segments[k].fat_pct, b.segments[k].fat_pct, f),
        category: (f < 0.5 ? a : b).segments[k].category ?? null,
        blend: { from: a.segments[k].category ?? null, to: b.segments[k].category ?? null, f },
      };
    }
  } else segments = (f < 0.5 ? a.segments : b.segments) || a.segments || b.segments || null;
  return {
    date: (f < 0.5 ? a : b).date,
    weight_kg: lerp(a.weight_kg, b.weight_kg, f),
    body_fat_pct: lerp(a.body_fat_pct, b.body_fat_pct, f),
    muscle_kg: lerp(a.muscle_kg, b.muscle_kg, f),
    segments,
  };
}

export function formatDate(iso) {
  if (iso === "target") return "Target";
  return new Date(`${iso}T00:00:00`).toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" });
}
