# Pipeline Technical Details

## 1. Preprocessing (`advanced_preprocess`)
- Handles missing values
- Encodes categorical features
- Removes duplicates
- Class distribution balancing

## 2. Feature Engineering (`advanced_feature_engineering`)
- Parallel feature processing
- Outlier detection using IQR
- Polynomial feature creation
- Statistical feature generation
- Feature selection using mutual information

## 3. Model Training
- Cross-validation with stratification
- Hyperparameter optimization
- RandomForest base model
- Parameter grid:
  ```python
  param_grid = {
      'n_estimators': [200, 500],
      'max_depth': [15, None],
      'min_samples_split': [5],
      'min_samples_leaf': [2],
      'max_features': ['sqrt'],
      'class_weight': ['balanced'],
      'criterion': ['gini']
  }
  ```

## 4. Ensemble Creation
- Combines RandomForest and HistGradientBoosting
- Soft voting classifier
- Handles NaN values
- Memory-efficient processing

## 5. Performance Optimization
- Parallel processing using joblib
- Progress tracking with tqdm
- Checkpoint saving
- Memory usage monitoring
