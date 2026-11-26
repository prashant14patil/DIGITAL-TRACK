// Small shared UI pieces. Presentation only: no data logic lives here.
import { useEffect, useState } from "react";

// Theme: "system" follows the OS; "light"/"dark" pin it via <html data-theme>, which styles.css reads.
// index.html applies the stored choice before first paint so the page never flashes the wrong theme.
const THEME_KEY = "dt-theme";
const THEMES = [
  { id: "system", label: "Match system theme", Icon: MonitorIcon },
  { id: "light", label: "Light theme", Icon: SunIcon },
  { id: "dark", label: "Dark theme", Icon: MoonIcon },
];

function readTheme() {
  try {
    const t = localStorage.getItem(THEME_KEY);
    return t === "light" || t === "dark" ? t : "system";
  } catch {
    return "system";
  }
}

export function ThemeSwitch() {
  const [theme, setTheme] = useState(readTheme);
  useEffect(() => {
    const root = document.documentElement;
    if (theme === "system") root.removeAttribute("data-theme");
    else root.setAttribute("data-theme", theme);
    try {
      if (theme === "system") localStorage.removeItem(THEME_KEY);
      else localStorage.setItem(THEME_KEY, theme);
    } catch { /* storage blocked: the choice lasts for this page only */ }
  }, [theme]);
  return (
    <div className="seg theme-switch" role="radiogroup" aria-label="Theme">
      {THEMES.map(({ id, label, Icon }) => (
        <button key={id} className={theme === id ? "on" : ""} role="radio" aria-checked={theme === id}
          aria-label={label} title={label} onClick={() => setTheme(id)}>
          <Icon />
        </button>
      ))}
    </div>
  );
}

function SunIcon() {
  return (
    <svg viewBox="0 0 16 16" width="15" height="15" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round">
      <circle cx="8" cy="8" r="2.9" />
      <path d="M8 1.5v1.3M8 13.2v1.3M1.5 8h1.3M13.2 8h1.3M3.4 3.4l.9.9M11.7 11.7l.9.9M3.4 12.6l.9-.9M11.7 4.3l.9-.9" />
    </svg>
  );
}

function MoonIcon() {
  return (
    <svg viewBox="0 0 16 16" width="15" height="15" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinejoin="round">
      <path d="M13.5 9.6A5.6 5.6 0 0 1 6.4 2.5a5.6 5.6 0 1 0 7.1 7.1Z" />
    </svg>
  );
}

function MonitorIcon() {
  return (
    <svg viewBox="0 0 16 16" width="15" height="15" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round">
      <rect x="1.8" y="2.6" width="12.4" height="8.4" rx="1.2" />
      <path d="M5.5 13.6h5M8 11v2.6" />
    </svg>
  );
}

export function SectionHead({ step, title, desc, actions }) {
  return (
    <header className="section-head">
      <div>
        {step && <div className="eyebrow">{step}</div>}
        <h2>{title}</h2>
        {desc && <p>{desc}</p>}
      </div>
      {actions && <div className="actions">{actions}</div>}
    </header>
  );
}

// Rotation axis over a platform: the capture rig, reduced to three strokes.
export function BrandMark() {
  return (
    <svg className="brand-mark" viewBox="0 0 20 20" width="18" height="18" aria-hidden="true">
      <ellipse cx="10" cy="15" rx="7" ry="2.5" fill="none" stroke="currentColor" strokeWidth="1.6" />
      <line x1="10" y1="2.5" x2="10" y2="15" stroke="currentColor" strokeWidth="1.6" />
      <circle cx="10" cy="5.5" r="2" fill="currentColor" />
    </svg>
  );
}

export function Chevron() {
  return (
    <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true">
      <path d="M6 3.5 10.5 8 6 12.5" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export function UploadIcon() {
  return (
    <svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true">
      <path d="M12 15V4m0 0L7.5 8.5M12 4l4.5 4.5M4 14.5V18a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3.5" fill="none"
        stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
