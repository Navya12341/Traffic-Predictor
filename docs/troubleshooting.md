# Troubleshooting Guide

## Common Issues

### Memory Errors
```
MemoryError: Unable to allocate array
```
**Solution:**
- Reduce sample_size parameter
- Free up system memory
- Use smaller feature set

### Runtime Issues
**Problem:** Long processing times
**Solutions:**
1. Reduce number of features:
```python
max_features = min(20, self.X.shape[1])
```
2. Use fewer cross-validation folds:
```python
n_splits = 3 if len(self.X) > 50000 else 5
```

### NaN/Infinity Errors
**Solution:**
Check data cleaning in create_ensemble method:
```python
X_clean = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
```

## Checkpoint Recovery
If process interrupts, restart using latest checkpoint:
```python
checkpoint = joblib.load('checkpoints/checkpoint_latest.pkl')
```
