# CLAUDE.md — Project Instructions

## Overarching Goal

Create an **interactive MyST Markdown book** (online-course level) that explains compressed sensing (CS) in general and for MRI applications, and reproduces and extends key figures from:

> Lustig M, Donoho D, Pauly JM. **Sparse MRI: The Application of Compressed Sensing for Rapid MR Imaging.** *Magnetic Resonance in Medicine*, 58(6):1182–1195, 2007.

The paper lives in `.paper/` as JPG/PDF scans. All interactive figures should faithfully reflect those scans while adding interactivity (zoom, parameter sliders, hover tooltips, etc.) via Plotly.

---

## Project Type

This is a **MyST Markdown book** (using the MyST CLI, `mystmd`) **with interactive Plotly figures**, structured after the template in `.templates/mooc/`. When in doubt about project structure, file naming, config, or build process, refer to that template first.

---

## Style Reference

Visual style (fonts, colours, spacing, component look-and-feel) must match `.templates/note/`.

**On first run, or whenever asked to set up / refresh styles:**
1. Read `.templates/note/content/_static/custom.css` and `.templates/note/content/_config.yml`.
2. Extract: font families, font sizes, colour palette, spacing scale, any custom theme overrides.
3. Update this file under **Extracted Style Tokens** below.
4. Apply those tokens consistently across all new files.

### Extracted Style Tokens
*(Populated 2026-03-12 from `.templates/note/content/_static/custom.css`)*

| Token | Value |
|-------|-------|
| Font family (body) | STIX Two Text |
| Font family (headings) | STIX Two Text |
| Font family (code) | system default monospace |
| Base font size | browser default (captions: 80%) |
| Primary colour (links) | `#870000` (dark red) |
| Secondary colour (visited links) | `#6f42c1` (purple) |
| Heading colour (h1) | `#342727` (very dark brown) |
| Accent colour (hover) | `brick` |
| Line height (captions) | 1.25 |
| Max content width | auto-centred (`margin: 0 auto`) |
| Paragraph alignment | justified |

---

## Workflow Rules

### 1. Always Plan First
Before doing *any* implementation work, write out a numbered plan:
- What files will be created or modified
- What tests will be written
- What the expected outcome is

Then **stop and wait for explicit confirmation** before proceeding. Do not begin implementation until the user says yes (or requests changes to the plan).

### 2. Build the Book Before Showing It
After implementing any change that affects rendered output:
```bash
myst build --html
```
Only present results (links, screenshots, summaries) after a successful build. If the build fails, fix it before reporting back.

### 3. Test-Driven Development (TDD)
For **all code that is not inside a Jupyter notebook**, follow strict TDD:

```
RED   → write a test that defines the desired behaviour → run it → confirm it fails
GREEN → implement the minimum code to make the test pass → run it → confirm it passes
REFACTOR → clean up, then re-run to confirm still green
```

**Test quality bar:**
- ❌ Too simple: `assert function_runs_without_error()` — do not write these
- ❌ Too specific: `assert result == 3.14159` (brittle, magic-number checks) — avoid unless the value is a true invariant
- ✅ Behaviour-focused: test *what* the function should do (shape of output, monotonicity, physical constraints, edge-case handling, etc.)

Tests live in a `tests/` directory mirroring the source structure. Use `pytest`. After any new feature, run the full test suite and show the summary.

### 4. External Code References (`.external/`)
`.external/` contains code snippets that are **reference material only**.
- Do **not** copy them verbatim.
- You may study them for approach, algorithm, or domain logic.
- Always re-implement from scratch, adapting to this project's conventions.
- Note in comments if a function was inspired by a snippet in `.external/`.

### 5. Paper Figures (`.paper/`)
`.paper/` contains JPG/PDF scans of the target paper.
- Read them carefully before implementing any figure.
- Each interactive figure should reproduce the paper figure faithfully, then enhance it.
- Keep a mapping table (in `docs/figure_map.md`) linking paper figure numbers to source files.

---

## Repository Conventions

### File Structure
```
.
├── myst.yml             # MyST CLI config + TOC
├── index.md             # Landing page
├── content/             # .md and .ipynb source files
├── figures/             # Python modules producing Plotly figures
├── tests/               # pytest test suite (mirrors figures/ and any other modules)
├── data/                # raw or processed data files
├── docs/                # project documentation (figure_map.md, etc.)
├── _static/             # CSS and static assets
├── requirements.txt     # Python dependencies
├── .paper/              # JPG/PDF scans (read-only reference)
├── .templates/          # Reference templates (read-only)
├── .external/           # External snippets (read-only reference)
└── CLAUDE.md            # This file
```

### Naming Conventions
- Figure modules: `figures/fig_<number>_<short_description>.py`
- Tests: `tests/test_fig_<number>_<short_description>.py`
- Content pages: `content/<NN>_<short_title>.md` or `.ipynb`

### Plotly Figures
- All figures must be self-contained functions: `def make_figure(...) -> go.Figure`
- Accept keyword arguments for any parameter that a reader might want to explore
- Return the `Figure` object; never call `.show()` inside the module
- Embed in MyST via notebook cells or `{glue}` directives as used in `.templates/mooc/`

### Commits
Write descriptive commit messages in the imperative mood:
> `Add interactive reproduction of Fig 2 with undersampling slider`

---

## Commands Reference

| Action | Command |
|--------|---------|
| Build book (HTML) | `myst build --html` |
| Build book (PDF) | `myst build --pdf` |
| Start dev server | `myst start` |
| Run tests | `pytest tests/ -v` |
| Run single test file | `pytest tests/test_<name>.py -v` |
| View locally | open `http://localhost:3000` after `myst start` |

---

## Things Claude Should Never Do
- Start implementing before the plan is confirmed
- Present output before the book builds successfully
- Write tests that only check whether code runs without error
- Copy code verbatim from `.external/`
- Modify anything inside `.paper/`, `.templates/`, or `.external/`
- Use `.show()` inside figure modules

---

## Proposed Additions (feel free to accept/reject)

- **`docs/figure_map.md`** — living document mapping paper figure → source file → status (planned / in progress / done).
- **`CHANGELOG.md`** — brief notes per session to preserve context across conversations.
- **Data provenance** — if data is digitised from the paper (e.g. WebPlotDigitizer), record source and method in `data/README.md`.
- **Accessibility** — Plotly figures should include axis labels, titles, and hover text sufficient for greyscale / screen reader use.
- **Environment file** — update `requirements.txt` whenever a new dependency is introduced.

---

## Saved Plans (Pending Confirmation)

*These plans were proposed but not yet approved. Kept here so they are not lost between sessions.*

### Plan A — Full book structure + Figures 2 and 1 (proposed 2026-03-12)

Reproduce paper figures in the order: **Figure 2 first, then Figure 1**.

**Figure 2** (paper p3.jpg): "An intuitive reconstruction of a sparse signal from pseudo-random k-space undersampling."
- 1D sparse signal (few non-zero components)
- Panels: (a) original sparse signal, (b) full k-space, (c) pseudo-random undersampled k-space, (d) zero-filled reconstruction (aliasing), (e) detected components above threshold, (f) undetected components, (g) PSF of the undersampling pattern, (h) recovered signal after thresholding
- Module: `figures/fig_02_sparse_signal_reconstruction.py`
- Tests: `tests/test_fig_02_sparse_signal_reconstruction.py`
- **Interactivity**: TBD — awaiting user guidance (e.g. undersampling fraction slider, threshold slider, toggle between recovery steps)

**Figure 1** (paper p2.jpg): "Illustration of the domains and operators used in CS."
- Conceptual diagram: k-space image ↔ sparse-transform domain ↔ partial k-space
- Arrows for operators F (Fourier), Ψ (sparsifying transform), F_u (undersampled Fourier)
- Module: `figures/fig_01_cs_domains.py`
- Tests: `tests/test_fig_01_cs_domains.py`
- **Interactivity**: TBD — awaiting user guidance (e.g. click domain to highlight, selector for different images/transforms)

**Content pages** (educational narrative, online-course level):
- `content/01_introduction.md` — what is compressed sensing? sparsity, incoherence, nonlinear recovery
- `content/02_mri_application.md` — why CS applies to MRI (k-space, Fourier encoding, clinical tradeoffs)
- `content/03_fig2_demo.ipynb` — Figure 2 with narrative
- `content/04_fig1_domains.ipynb` — Figure 1 with narrative

**Supporting files:**
- `docs/figure_map.md`
- `requirements.txt` updated with new deps

**Status**: Awaiting figure interactivity guidance before implementation.
