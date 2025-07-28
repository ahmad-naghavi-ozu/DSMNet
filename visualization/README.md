# DSM Quality Analysis Visualization

This visualization system provides streamlined analysis of DSM (Digital Surface Model) prediction quality by analyzing test results, organizing samples into performance quartiles, and creating clear visualizations.

## Features

- **Simple Log Parsing**: Extracts key metrics (RMSE, MAE, Delta1, etc.) from test output log files
- **Quartile Analysis**: Divides tiles into performance quartiles based on RMSE
- **Essential Visualizations**: Shows RGB, ground truth DSM, predicted DSM, and difference maps
- **Statistical Summaries**: Basic distribution plots and performance statistics
- **Easy Configuration**: Just change one variable to analyze different datasets

## File Structure

```
visualization/
├── simple_dsm_analysis.ipynb     # Main analysis notebook (simplified)
├── dsm_sample_visualizer.ipynb   # Sample tile visualizer
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── test_system.py               # System compatibility test
```

## Installation

1. **Install Dependencies**:
   ```bash
   cd visualization
   pip install -r requirements.txt
   ```

2. **For conda users**:
   ```bash
   conda install numpy pandas matplotlib seaborn pillow scipy jupyter
   pip install rasterio  # May need conda-forge: conda install -c conda-forge rasterio
   ```

## Usage

### Quick Start

1. **Open the Main Notebook**:
   ```bash
   jupyter notebook simple_dsm_analysis.ipynb
   ```

2. **Configure the Analysis**:
   - Change `DATASET_NAME` variable in the first cell
   - Available datasets: 'DFC2023S', 'DFC2019_crp512_bin', 'Huawei_Contest'

3. **Run All Cells**:
   - The notebook will automatically parse log files
   - Create quartile analysis and visualizations
   - Show best/worst performing tiles with complete visual breakdown

### What You Get

- **Statistical Analysis**: Performance distribution across all tiles
- **Quartile Breakdown**: Best vs worst performing tiles identified
- **Visual Analysis**: RGB + Ground Truth DSM + Predicted DSM + Difference maps
- **Summary Report**: Clear performance overview with actionable insights
   ```python
   # Dataset Configuration
   DATASET_NAME = 'Huawei_Contest'  # Change this for your dataset
   BASE_DATASETS_PATH = '../datasets'  # Path to datasets directory
   BASE_OUTPUT_PATH = './output'  # Path to output directory
   
   # Analysis Configuration
   PRIMARY_METRIC = 'rmse'  # Primary metric for quartile analysis
   SAMPLES_PER_QUARTILE = 5  # Number of samples to visualize per quartile
   ```

3. **Run All Cells**: Execute the notebook cells in sequence

### Expected Directory Structure

The system expects the following directory structure:

```
DSMNet/
├── datasets/
│   └── Huawei_Contest/  # or your dataset name
│       └── test/
│           ├── rgb/     # RGB images (.png, .jpg, .tif)
│           ├── dsm/     # Ground truth DSMs (.tif)
│           └── sem/     # Semantic masks (.png, .tif) [optional]
├── output/
│   └── Huawei_Contest/  # or your dataset name
│       ├── _logs/
│       │   └── Huawei_Contest_dae_test_dsm_output.log
│       └── dsm_110+/    # or similar pattern (dsm_*+)
│           └── *.tif    # Predicted DSM files
└── visualization/
    └── dsm_quality_analysis.ipynb
```

### Configuration for Different Datasets

#### For Huawei Contest Dataset:
```python
DATASET_NAME = 'Huawei_Contest'
PRIMARY_METRIC = 'rmse'  # or 'mae', 'delta1', etc.
```

#### For DFC2023S Dataset:
```python
DATASET_NAME = 'DFC2023S'
PRIMARY_METRIC = 'rmse'
BASE_DATASETS_PATH = '../datasets'  # Adjust if needed
```

#### For Other Datasets:
```python
DATASET_NAME = 'YourDatasetName'
PRIMARY_METRIC = 'rmse'  # Choose appropriate metric
```

## Output

The analysis generates several types of visualizations:

### 1. Individual Tile Visualizations
For each sampled tile, creates a 4-panel visualization showing:
- **RGB Image**: Original satellite/aerial imagery
- **Ground Truth DSM**: True height values with building outlines
- **Predicted DSM**: Model predictions with building outlines  
- **Difference Map**: GT - Predicted (red = overestimation, blue = underestimation)

### 2. Statistical Analysis
- **Quartile Performance Summary**: Statistics for each performance quartile
- **Metrics Distribution Plots**: Histograms showing distribution of error metrics
- **Correlation Heatmap**: Relationships between different metrics

### 3. Saved Files (if enabled)
- Individual tile visualizations saved as PNG files
- Summary plots (distributions, correlations)
- All files saved to `./visualization_outputs/[DATASET_NAME]/`

## Customization

### Adding New Metrics
To analyze additional metrics, modify the log parsing patterns in `utils/log_parser.py`:

```python
metric_patterns = {
    'your_metric': r"Tile Your Metric:\s*([\d.]+)",
    # ... existing patterns
}
```

### Modifying Visualizations
Customize plot appearance in `utils/plotting.py`:
- Colormaps for DSM visualization
- Figure sizes and layouts
- Color schemes and styling

### Supporting New Datasets
Add dataset configuration in `utils/data_loader.py`:

```python
elif dataset_name.startswith('YourDataset'):
    config.update({
        'has_semantic': True,  # Whether dataset has semantic labels
        'binary_semantic': True,  # Whether semantic is binary (0,1)
        'crop_size': 512,  # Image dimensions
    })
```

## Troubleshooting

### Common Issues

1. **"Log file not found"**:
   - Check that the log file exists in `output/[DATASET]//_logs/`
   - Verify the log file name pattern matches expected format
   - The system will try to auto-detect log files if the exact name isn't found

2. **"No DSM prediction folders found"**:
   - Ensure prediction folders follow the pattern `dsm_*+` (e.g., `dsm_110+`)
   - Check that the output directory structure is correct

3. **"Dataset path not found"**:
   - Verify the `BASE_DATASETS_PATH` configuration
   - Ensure the dataset directory structure follows expected format
   - You can still run metric analysis without the dataset files

4. **Missing dependencies**:
   ```bash
   pip install rasterio  # For TIFF file reading
   conda install -c conda-forge rasterio  # Alternative for conda
   ```

### Performance Tips

- **Memory Usage**: Large datasets may require closing figures after display (`plt.close(fig)`)
- **Processing Time**: Reduce `SAMPLES_PER_QUARTILE` for faster analysis
- **File Size**: Saved figures are high-resolution (300 DPI) - adjust in plotting utilities if needed

## Advanced Usage

### Batch Processing Multiple Datasets
```python
datasets = ['Huawei_Contest', 'DFC2023S', 'YourDataset']
for dataset in datasets:
    DATASET_NAME = dataset
    # Run analysis...
```

### Custom Metric Analysis
```python
# Analyze different metrics
for metric in ['rmse', 'mae', 'delta1']:
    PRIMARY_METRIC = metric
    # Run quartile analysis...
```

### Export Results
```python
# Save quartile summary to CSV
quartile_summary.to_csv(f'{OUTPUT_DIR}/quartile_summary.csv')

# Save full metrics to Excel
metrics_df.to_excel(f'{OUTPUT_DIR}/all_metrics.xlsx', index=False)
```

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify your directory structure matches expectations
3. Ensure all dependencies are installed correctly
4. Check that log files contain the expected metric patterns

The system is designed to be robust and will attempt to auto-detect files and configurations when possible.
