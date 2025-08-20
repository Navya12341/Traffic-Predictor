# User Guide

## Basic Usage

1. Initialize the pipeline:
```python
from src.model_training2 import MaxAccuracyPipeline

pipeline = MaxAccuracyPipeline("combined_data.csv")
```

2. Run the complete pipeline:
```python
model, accuracy = pipeline.run_max_accuracy_pipeline()
```

## Advanced Usage

### Testing with Sample Data
```python
# Use 10% of data for testing
pipeline = MaxAccuracyPipeline("your_data.csv", sample_size=0.1)
```

### Running with Sleep Prevention (Mac)
```bash
caffeinate -i python src/model_training2.py > training_log.txt 2>&1
```

### Monitoring Progress
- Check `training_log.txt` for real-time updates
- View checkpoints in `checkpoints/` directory
- Monitor memory usage in output
