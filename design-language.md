# Design Language Document
## bhuv's notebook

This document outlines the complete design system and color scheme for bhuvanesh09.github.io, providing a reference for maintaining visual consistency across the website and data visualizations.

---

## Color Palette

### Primary Background Colors

| Color Name | Hex Code | RGB | Usage |
|------------|----------|-----|-------|
| Linen | `#FAF0E6` | rgb(250, 240, 230) | Main website background (light mode) |
| Cream | `#FAF8F1` | rgb(250, 248, 241) | Plot/chart background |
| Pure Black | `#000000` | rgb(0, 0, 0) | Dark mode background (90% opacity) |
| Transparent | `transparent` | - | Default HTML background |

**Usage Guidelines:**
- Use **Linen** (#FAF0E6) as the primary background for all web pages
- Use **Cream** (#FAF8F1) as the background for plots and data visualizations to maintain consistency
- Dark mode uses pure black with 90% opacity for depth

---

### Data Visualization Palette (Primary)

These are the main pastel colors used in plots and charts. They provide excellent contrast against the cream background while maintaining a soft, professional aesthetic.

| Color Name | Hex Code | RGB | Usage |
|------------|----------|-----|-------|
| Steel Blue | `#4A90C0` | rgb(74, 144, 192) | Primary data series, first line |
| Sandy Orange | `#FFA54F` | rgb(255, 165, 79) | Secondary data series, second line |
| Medium Purple | `#9370DB` | rgb(147, 112, 219) | Tertiary data series, third line |
| Medium Aquamarine | `#66CDAA` | rgb(102, 205, 170) | Quaternary data series, fourth line |
| Indian Red | `#CD5C5C` | rgb(205, 92, 92) | Quinary data series, fifth line |

**Usage Guidelines:**
- Use these colors in sequence for multi-line plots
- Maintain this exact order for consistency across all visualizations
- All colors work harmoniously with the cream background (#FAF8F1)
- Each color has sufficient contrast for accessibility

---

### Typography Colors

#### Light Mode
| Element | Hex Code | RGB | Variable Name |
|---------|----------|-----|---------------|
| Body Text | `#000000` | rgb(0, 0, 0) | Primary |
| Article Text | `#374151` | rgb(55, 65, 81) | `--tw-prose-body` |
| Headings | `#111827` | rgb(17, 24, 39) | `--tw-prose-headings` |
| Links | `#111827` | rgb(17, 24, 39) | `--tw-prose-links` |
| Lead Text | `#4b5563` | rgb(75, 85, 99) | `--tw-prose-lead` |
| Code Text | `#111827` | rgb(17, 24, 39) | `--tw-prose-code` |

#### Dark Mode
| Element | Hex Code | RGB | Variable Name |
|---------|----------|-----|---------------|
| Body Text | `#FFFFFF` | rgb(255, 255, 255) | Primary |
| Article Text | `#d1d5db` | rgb(209, 213, 219) | `--tw-prose-invert-body` |
| Headings | `#FFFFFF` | rgb(255, 255, 255) | `--tw-prose-invert-headings` |
| Lead Text | `#9ca3af` | rgb(156, 163, 175) | `--tw-prose-invert-lead` |

---

### UI Elements

#### Borders & Dividers
| Element | Hex Code | RGB | Usage |
|---------|----------|-----|-------|
| Light Border | `#e5e7eb` | rgb(229, 231, 235) | Default borders, hr elements |
| Medium Border | `#E0E0E0` | rgb(224, 224, 224) | Plot grid lines |
| Dark Border (10% opacity) | `rgba(0, 0, 0, 0.1)` | - | Subtle separators |

#### Grays & Neutrals
| Element | Hex Code | RGB | Usage |
|---------|----------|-----|-------|
| Counters | `#6b7280` | rgb(107, 114, 128) | List numbers, counters |
| Bullets | `#d1d5db` | rgb(209, 213, 219) | List bullet points |
| Captions | `#6b7280` | rgb(107, 114, 128) | Image captions, footnotes |
| Placeholder | `#9ca3af` | rgb(156, 163, 175) | Input placeholders |

#### Interactive Elements
| Element | Hex Code | RGB | Usage |
|---------|----------|-----|-------|
| Button Background (Light) | `#000000` | rgb(0, 0, 0) | Primary buttons |
| Button Text (Light) | `#FFFFFF` | rgb(255, 255, 255) | Button text |
| Button Background (Dark) | `#FFFFFF` | rgb(255, 255, 255) | Dark mode buttons |
| Button Text (Dark) | `#000000` | rgb(0, 0, 0) | Dark mode button text |
| Hover Background | `rgba(0, 0, 0, 0.02)` | - | Light hover state |

#### Code Blocks
| Element | Hex Code | RGB | Usage |
|---------|----------|-----|-------|
| Pre Background (Light) | `#1f2937` | rgb(31, 41, 55) | Code block background |
| Pre Code (Light) | `#e5e7eb` | rgb(229, 231, 235) | Code text |
| Pre Background (Dark) | `rgba(0, 0, 0, 0.5)` | - | Dark mode code blocks |
| Line Numbers | `rgba(255, 255, 255, 0.4)` | - | Code line numbers |

---

### Accent & Highlight Colors

| Color Name | Hex Code | RGB | Usage |
|------------|----------|-----|-------|
| Accent Red | `#ff3b2d` | rgb(255, 59, 45) | Error states, important notices |
| Ring Blue | `rgb(59, 130, 246)` | - | Focus rings (50% opacity) |

---

## Typography

### Font Families

```css
/* Headings & Bold Text */
font-family: 'JunicodeVF'
/* Source: JunicodeVF-Roman.woff2 */
/* Weight: 700 */
/* Stretch: 80% */

/* Body Text & Code */
font-family: 'FiraCode'
/* Source: FiraCode-Regular.woff2 */
/* Stretch: 100% */
```

### Font Sizes & Hierarchy

| Element | Size | Line Height | Usage |
|---------|------|-------------|-------|
| Article Body | 1.0rem (16px) | 1.6rem | Main content |
| H1 | 2.25em (36px) | 1.111 | Page titles |
| H2 | 1.5em (24px) | 1.333 | Section headings |
| H3 | 1.25em (20px) | 1.6 | Subsection headings |
| H4 | 1em (16px) | 1.5 | Minor headings |
| Large Text | 1.125rem (18px) | 1.75rem | Featured text |
| Small Text | 0.875rem (14px) | 1.25rem | Captions, small UI |
| Code | 0.875rem (14px) | 1.5rem | Inline code |

---

## Plot & Visualization Guidelines

### Standard Plot Configuration

```json
{
  "plot_bgcolor": "#FAF8F1",
  "paper_bgcolor": "#FAF8F1",
  "gridcolor": "#E0E0E0",
  "gridwidth": 1,
  "width": 800,
  "height": 600
}
```

### Color Sequence for Data Series

Use these colors in order for multi-series visualizations:

1. **Steel Blue** (`#4A90C0`) - First data series
2. **Sandy Orange** (`#FFA54F`) - Second data series
3. **Medium Purple** (`#9370DB`) - Third data series
4. **Medium Aquamarine** (`#66CDAA`) - Fourth data series
5. **Indian Red** (`#CD5C5C`) - Fifth data series

### Marker Configuration

```json
{
  "marker": {
    "color": "#4A90C0",  // Match line color
    "size": 8
  },
  "line": {
    "color": "#4A90C0",
    "width": 2
  },
  "mode": "lines+markers"
}
```

---

## Spacing & Layout

### Container Widths
- Max content width: `48rem` (768px) - `max-w-3xl`
- Prose width: `65ch`

### Padding & Margins
| Element | Value | Usage |
|---------|-------|-------|
| Section Padding (Horizontal) | 2rem (32px) | `px-8` |
| Large Vertical Margin | 4rem (64px) | `my-16` |
| Medium Vertical Margin | 2.5rem (40px) | `my-10` |
| Small Vertical Margin | 0.5rem (8px) | `my-2` |

---

## Component Styles

### Buttons

```css
.btn {
  border-radius: 9999px;           /* Fully rounded */
  background-color: rgb(0, 0, 0);  /* Black in light mode */
  padding: 1rem 1.75rem;           /* 16px 28px */
  font-size: 0.875rem;             /* 14px */
  color: rgb(255, 255, 255);       /* White text */
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  transition-duration: 100ms;
}

.btn:active {
  transform: scale(0.95);
}

/* Dark mode */
.dark .btn {
  background-color: rgb(255, 255, 255);
  color: rgb(0, 0, 0);
}
```

### Links

```css
.link {
  transition-duration: 200ms;
}

.link:hover {
  color: rgb(0, 0, 0);  /* Black in light mode */
}

.dark .link:hover {
  color: rgb(255, 255, 255);  /* White in dark mode */
}
```

### Blockquotes

```css
article blockquote {
  border-left: 3px solid #000;  /* Light mode */
  padding-left: 1em;
  font-style: italic;
  font-weight: 500;
  margin: 1.6em 0;
}

.dark article blockquote {
  border-left-color: #fff;  /* Dark mode */
}
```

---

## Accessibility

### Contrast Ratios

All color combinations meet WCAG AA standards:

- **Body text (#000000) on Linen (#FAF0E6):** ✓ 18.84:1 (AAA)
- **Plot colors on Cream (#FAF8F1):** ✓ All pass AA (minimum 4.5:1)
- **Dark mode text (#FFFFFF) on Black (#000):** ✓ 21:1 (AAA)

### Focus States

- Focus ring color: `rgb(59, 130, 246)` with 50% opacity
- Focus ring width: 2px
- Focus ring offset: 2px

---

## Dark Mode Specifications

The website defaults to light mode regardless of system preferences, with manual toggle available.

### Background Transitions
- Light: Linen (#FAF0E6)
- Dark: Black (#000000) at 90% opacity `rgba(0, 0, 0, 0.9)`

### Key Differences
- All text inverts (black ↔ white)
- Borders invert
- Code blocks use darker backgrounds
- Plots can maintain light backgrounds or adapt as needed

---

## Usage Examples

### Python (Plotly) Configuration

```python
import plotly.graph_objects as go

# Color palette
COLORS = {
    'steel_blue': '#4A90C0',
    'sandy_orange': '#FFA54F',
    'medium_purple': '#9370DB',
    'medium_aquamarine': '#66CDAA',
    'indian_red': '#CD5C5C'
}

# Standard layout
layout = go.Layout(
    plot_bgcolor='#FAF8F1',
    paper_bgcolor='#FAF8F1',
    width=800,
    height=600,
    xaxis=dict(
        gridcolor='#E0E0E0',
        gridwidth=1
    ),
    yaxis=dict(
        gridcolor='#E0E0E0',
        gridwidth=1
    )
)

# Create trace with standard styling
trace = go.Scatter(
    x=x_data,
    y=y_data,
    mode='lines+markers',
    line=dict(color=COLORS['steel_blue'], width=2),
    marker=dict(color=COLORS['steel_blue'], size=8)
)
```

### CSS Custom Properties

```css
:root {
  /* Backgrounds */
  --bg-linen: #FAF0E6;
  --bg-cream: #FAF8F1;
  --bg-dark: rgba(0, 0, 0, 0.9);

  /* Plot colors */
  --plot-blue: #4A90C0;
  --plot-orange: #FFA54F;
  --plot-purple: #9370DB;
  --plot-aqua: #66CDAA;
  --plot-red: #CD5C5C;

  /* Typography */
  --font-heading: 'JunicodeVF';
  --font-body: 'FiraCode';

  /* Grays */
  --gray-border: #e5e7eb;
  --gray-grid: #E0E0E0;
  --gray-text: #6b7280;
}
```

---

## Brand Identity

### Visual Characteristics
- **Clean & Minimal:** Generous white space, simple layouts
- **Technical & Academic:** Monospace fonts, precise data visualizations
- **Approachable:** Soft pastels, warm backgrounds
- **Professional:** Consistent typography, clear hierarchy

### Design Principles
1. **Clarity First:** Information should be immediately accessible
2. **Consistency:** Use the defined color palette across all visualizations
3. **Readability:** Maintain high contrast ratios and legible font sizes
4. **Simplicity:** Avoid unnecessary decoration or complexity

---

## File References

- Main CSS: `bhuv-webpage/assets/custom.css`
- Theme CSS: `bhuv-webpage/themes/paper/assets/main.css`
- Configuration: `bhuv-webpage/hugo.toml`
- Fonts: `bhuv-webpage/static/fonts/`
  - `JunicodeVF-Roman.woff2`
  - `FiraCode-Regular.woff2`

---

## Version History

- **v1.0** (2025-12-07): Initial design language documentation

---

**Document maintained by:** Bhuvanesh Sridharan
**Website:** https://bhuvanesh09.github.io
**Last Updated:** December 7, 2025
