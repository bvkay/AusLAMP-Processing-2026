# Outputs Directory

Generated figures and reports (git-ignored).

## Structure

```
outputs/
├── figures/          # Plots and maps (PNG, PDF, SVG)
│   ├── site_locations/
│   ├── timeseries/
│   └── qaqc/
│
└── reports/          # Analysis reports (Markdown, HTML, PDF)
    └── qaqc_report.md
```

## Usage

Notebooks and scripts should save outputs here using:

```python
from src.config import DataPaths

# Save figure
fig.savefig(DataPaths.FIGURES_DIR / 'site_map.png', dpi=300, bbox_inches='tight')

# Save report
report_path = DataPaths.REPORTS_DIR / 'analysis_summary.md'
```

All files in this directory are automatically ignored by git.
