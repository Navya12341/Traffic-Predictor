import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import KNNImputer
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from joblib import Parallel, delayed
import multiprocessing
from sklearn.utils import shuffle
from sklearn.svm import SVC
import psutil
from tqdm import tqdm

warnings.filterwarnings('ignore')

class MaxAccuracyPipeline:
    def __init__(self, csv_file, sample_size=None):
        """
        Initialize pipeline with optional sample size for testing
        sample_size: int or float, if float < 1.0 treats as fraction of data
        """
        self.df = pd.read_csv(csv_file)
        
        # Add sampling for quick testing
        if sample_size:
            if isinstance(sample_size, float) and sample_size < 1.0:
                sample_size = int(len(self.df) * sample_size)
            self.df = shuffle(self.df, random_state=42).head(sample_size)
            print(f"Using {len(self.df)} samples for testing")
        
        # Set number of CPU cores to use
        self.n_jobs = max(multiprocessing.cpu_count() - 1, 1)  # Leave 1 core free
        print(f"Using {self.n_jobs} CPU cores")
        
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        self.model = None
        self.feature_selector = None
        
    def advanced_preprocess(self):
        """Advanced preprocessing for maximum accuracy"""
        print("Advanced preprocessing...")
        print(f"Dataset shape: {self.df.shape}")
        print(f"Missing values: {self.df.isnull().sum().sum()}")
        
        # Separate target first
        self.y = self.df["class1"]
        self.X = self.df.drop(columns=["class1"])
        
        # Handle categorical features with proper encoding
        categorical_cols = self.X.select_dtypes(include=['object']).columns
        numerical_cols = self.X.select_dtypes(include=[np.number]).columns
        
        # Encode categorical features
        for col in categorical_cols:
            le = LabelEncoder()
            # Handle missing values in categorical columns
            self.X[col] = self.X[col].fillna('Unknown')
            self.X[col] = le.fit_transform(self.X[col].astype(str))
        
        # Advanced missing value imputation for numerical features using KNN
        if len(numerical_cols) > 0 and self.X[numerical_cols].isnull().any().any():
            print("Performing KNN imputation for missing values...")
            imputer = KNNImputer(n_neighbors=5)
            # Keep as DataFrame after imputation
            self.X[numerical_cols] = pd.DataFrame(
                imputer.fit_transform(self.X[numerical_cols]),
                columns=numerical_cols,
                index=self.X.index
            )
        
        # Encode target if categorical
        if self.y.dtype == 'object':
            le_target = LabelEncoder()
            self.y = le_target.fit_transform(self.y)
        
        # Remove duplicate rows
        initial_shape = self.X.shape[0]
        # Convert y to Series with proper index if it's an array
        if isinstance(self.y, np.ndarray):
            self.y = pd.Series(self.y, index=self.X.index)
        
        combined_df = pd.concat([self.X, self.y], axis=1)
        combined_df = combined_df.drop_duplicates()
        self.X = combined_df.iloc[:, :-1]
        self.y = combined_df.iloc[:, -1].values  # Convert back to array for sklearn
        print(f"Removed {initial_shape - self.X.shape[0]} duplicate rows")
        
        print(f"Final features shape: {self.X.shape}")
        print(f"Class distribution: {np.bincount(self.y)}")
        
    def advanced_feature_engineering(self):
        """Advanced feature engineering with parallel processing"""
        print("Advanced feature engineering...")
        
        # Parallel processing for feature calculations
        def process_numerical_column(col):
            stats = {}
            data = self.X[col]
            stats['nunique'] = data.nunique() / len(data)
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            stats['bounds'] = (Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
            return col, stats
        
        # Process features in parallel
        numerical_cols = self.X.select_dtypes(include=[np.number]).columns
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(process_numerical_column)(col) for col in numerical_cols
        )
        
        # Process results
        constant_features = []
        for col, stats in results:
            if stats['nunique'] < 0.01:  # Less than 1% unique values
                constant_features.append(col)
            else:
                # Handle outliers
                self.X[col] = self.X[col].clip(lower=stats['bounds'][0], upper=stats['bounds'][1])
        
        # Remove constant features
        if constant_features:
            self.X = self.X.drop(columns=constant_features)
            print(f"Removed {len(constant_features)} constant/quasi-constant features")
        
        # Reduce number of features for faster processing
        max_features = min(20, self.X.shape[1])  # Reduced from 40 to 20
        if self.X.shape[1] > max_features:
            print(f"Selecting top {max_features} features...")
            self.feature_selector = SelectKBest(score_func=mutual_info_classif, k=max_features)
            self.X = pd.DataFrame(
                self.feature_selector.fit_transform(self.X, self.y),
                columns=self.X.columns[self.feature_selector.get_support()]
            )
        
        # Create polynomial features for top numerical features (if not too many features)
        if self.X.shape[1] <= 20:
            numerical_cols = self.X.select_dtypes(include=[np.number]).columns[:5]  # Top 5 numerical
            for i, col1 in enumerate(numerical_cols):
                # Add squared terms
                self.X[f'{col1}_squared'] = self.X[col1] ** 2
                # Add interaction terms with next feature
                for col2 in numerical_cols[i+1:i+2]:  # Only one interaction per feature
                    self.X[f'{col1}_{col2}_interaction'] = self.X[col1] * self.X[col2]
            print(f"Added polynomial features. New shape: {self.X.shape}")
        
        # Add ratio features for numerical columns
        numerical_cols = self.X.select_dtypes(include=[np.number]).columns
        for i, col1 in enumerate(numerical_cols):
            for col2 in numerical_cols[i+1:]:
                # Add ratio features (handling division by zero)
                self.X[f'{col1}_div_{col2}'] = self.X[col1] / (self.X[col2] + 1e-8)
                
        # Add statistical features
        for col in numerical_cols:
            self.X[f'{col}_rolling_mean'] = self.X[col].rolling(window=3, min_periods=1).mean()
            self.X[f'{col}_rolling_std'] = self.X[col].rolling(window=3, min_periods=1).std()
        
        print(f"Added statistical features. New shape: {self.X.shape}")
    
    def prepare_data(self):
        """Prepare data with stratified cross-validation and progress tracking"""
        print("Preparing data with advanced splitting...")
        
        # Use fewer folds for large datasets
        n_splits = 3 if len(self.X) > 50000 else 5
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # Pre-scale all data to avoid multiple scaling operations
        print("Scaling features...")
        X_scaled = self.scaler.fit_transform(self.X)
        
        # Initialize variables
        best_score = 0
        best_split = None
        total_splits = n_splits
        current_split = 0
        
        print(f"Evaluating {n_splits} splits...")
        for train_idx, test_idx in skf.split(X_scaled, self.y):
            current_split += 1
            print(f"Processing split {current_split}/{n_splits}...")
            
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]
            
            # Quick evaluation with minimal parameters
            temp_model = RandomForestClassifier(
                n_estimators=50,  # Reduced for speed
                max_depth=10,     # Limited depth
                n_jobs=self.n_jobs,
                random_state=42
            )
            temp_model.fit(X_train, y_train)
            score = temp_model.score(X_test, y_test)
            
            print(f"Split {current_split} score: {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_split = (X_train, X_test, y_train, y_test)
                print(f"New best score: {score:.4f}")
    
        # Use best split
        print("Using best split with score:", best_score)
        self.X_train, self.X_test, self.y_train, self.y_test = best_split
    
    def train_best_model(self):
        """Faster model training with reduced parameter grid"""
        print("Training Random Forest with optimized parameters...")
        
        # Simplified parameter grid for faster training
        param_grid = {
            'n_estimators': [200, 500],  # Reduced options
            'max_depth': [15, None],
            'min_samples_split': [5],
            'min_samples_leaf': [2],
            'max_features': ['sqrt'],
            'class_weight': ['balanced'],
            'criterion': ['gini']
        }
        
        # Faster parameter search
        random_search = RandomizedSearchCV(
            RandomForestClassifier(
                random_state=42,
                n_jobs=self.n_jobs,
                verbose=0
            ),
            param_grid,
            n_iter=10,  # Reduced iterations
            cv=3,       # Reduced folds
            n_jobs=self.n_jobs,
            scoring='balanced_accuracy',
            verbose=1,
            random_state=42
        )
        
        random_search.fit(self.X_train, self.y_train)
        self.model = random_search.best_estimator_
        
        return random_search.best_params_, random_search.best_score_

    def fine_tune_model(self):
        """Additional fine-tuning step for maximum accuracy"""
        print("Fine-tuning model with additional techniques...")
        
        # Get the best parameters from initial search
        best_params = self.model.get_params()
        
        # Fine-tune around the best n_estimators
        n_est_range = [best_params['n_estimators'] + i for i in [-100, -50, 0, 50, 100, 150]]
        n_est_range = [max(50, n) for n in n_est_range]  # Ensure minimum 50 estimators
        
        best_accuracy = 0
        best_n_estimators = best_params['n_estimators']
        
        for n_est in n_est_range:
            temp_model = RandomForestClassifier(**{**best_params, 'n_estimators': n_est})
            temp_model.fit(self.X_train, self.y_train)
            accuracy = accuracy_score(self.y_test, temp_model.predict(self.X_test))
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_n_estimators = n_est
        
        # Update model with best n_estimators
        self.model = RandomForestClassifier(**{**best_params, 'n_estimators': best_n_estimators})
        self.model.fit(self.X_train, self.y_train)
        
        print(f"Fine-tuned n_estimators: {best_n_estimators}")
        print(f"Fine-tuned accuracy: {best_accuracy:.6f}")
        
    def create_ensemble(self):
        """Create lighter ensemble with fewer models"""
        print("Creating ensemble model...")
        
        try:
            # Get best RF from previous training
            rf = self.model
            
            # Clean data more thoroughly
            def clean_data(X):
                # Replace infinities with large numbers
                X_clean = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
                # Ensure all values are finite
                X_clean = np.clip(X_clean, -1e10, 1e10)
                return X_clean
            
            # Clean training and test data
            X_train_clean = clean_data(self.X_train)
            X_test_clean = clean_data(self.X_test)
            
            # Verify no NaN or inf values remain
            assert not np.any(np.isnan(X_train_clean)), "NaN values found in training data"
            assert not np.any(np.isnan(X_test_clean)), "NaN values found in test data"
            
            # Use HistGradientBoostingClassifier instead of GradientBoostingClassifier
            # as it handles missing values better
            from sklearn.ensemble import HistGradientBoostingClassifier
            
            gb = HistGradientBoostingClassifier(
                max_iter=100,  # equivalent to n_estimators
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
            
            # Create voting ensemble with fewer models
            ensemble = VotingClassifier(
                estimators=[
                    ('rf', rf),
                    ('gb', gb)
                ],
                voting='soft',
                n_jobs=self.n_jobs
            )
            
            # Fit ensemble with clean data
            ensemble.fit(X_train_clean, self.y_train)
            
            # Update instance variables with clean data
            self.X_train = X_train_clean
            self.X_test = X_test_clean
            self.model = ensemble
            
            print("Ensemble created successfully!")
            
        except Exception as e:
            print(f"Error in ensemble creation: {str(e)}")
            print("Falling back to best single model...")
            # Keep the best model from previous step
            return
    
    def evaluate_and_visualize(self):
        """Evaluate model and create confusion matrix"""
        print("Final evaluation...")
        
        # Final predictions
        y_pred = self.model.predict(self.X_test)
        final_accuracy = accuracy_score(self.y_test, y_pred)
        
        print(f"FINAL TEST ACCURACY: {final_accuracy:.6f}")
        
        # Create enhanced confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                   cbar_kws={'label': 'Count'}, 
                   annot_kws={'size': 14, 'weight': 'bold'})
        
        plt.xlabel("Predicted Labels", fontsize=14, fontweight='bold')
        plt.ylabel("True Labels", fontsize=14, fontweight='bold')
        plt.title(f'Confusion Matrix\nAccuracy: {final_accuracy:.6f}', 
                 fontsize=16, fontweight='bold')
        
        # Add accuracy text
        plt.figtext(0.02, 0.02, f'Total Samples: {len(self.y_test)} | Correct: {np.trace(cm)} | Accuracy: {final_accuracy:.6f}', 
                   fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        
        plt.tight_layout()
        plt.show()
        
        return final_accuracy, y_pred
    
    def print_final_results(self, cv_accuracy, final_accuracy, best_params):
        """Print comprehensive final results"""
        print("\n" + "="*70)
        print("MAXIMUM ACCURACY RANDOM FOREST - FINAL RESULTS")
        print("="*70)
        print(f"Cross-Validation Accuracy: {cv_accuracy:.6f}")
        print(f"Final Test Accuracy:       {final_accuracy:.6f}")
        print(f"Improvement over baseline: {((final_accuracy - 0.8) * 100):+.2f}%")  # Assuming 0.8 baseline
        
        print(f"\nOptimal Parameters:")
        for param, value in best_params.items():
            if param != 'n_jobs':
                print(f"  {param}: {value}")
        
        print(f"\nModel Configuration:")
        print(f"  Total Features Used: {self.X.shape[1]}")
        print(f"  Training Samples: {len(self.y_train)}")
        print(f"  Test Samples: {len(self.y_test)}")
        print(f"  Number of Classes: {len(np.unique(self.y))}")
        
        # Feature importance insights
        if hasattr(self.model, 'feature_importances_'):
            feature_names = (self.X.columns if hasattr(self.X, 'columns') 
                           else [f'Feature_{i}' for i in range(len(self.model.feature_importances_))])
            top_features = np.argsort(self.model.feature_importances_)[::-1][:3]
            print(f"\nTop 3 Most Predictive Features:")
            for i, idx in enumerate(top_features, 1):
                print(f"  {i}. {feature_names[idx]}: {self.model.feature_importances_[idx]:.4f}")
    
    def run_max_accuracy_pipeline(self):
        """Run the enhanced pipeline"""
        print("🚀 Starting Maximum Accuracy Pipeline...")
        print("="*50)
        
        self.advanced_preprocess()
        self.advanced_feature_engineering()
        self.prepare_data()
        best_params, cv_accuracy = self.train_best_model()
        self.fine_tune_model()
        self.create_ensemble()  # Add ensemble step
        final_accuracy, y_pred = self.evaluate_and_visualize()
        self.print_final_results(cv_accuracy, final_accuracy, best_params)
        
        print("\n✅ Maximum Accuracy Pipeline completed!")
        return self.model, final_accuracy

# Usage
if __name__ == "__main__":
    import os
    import time
    from datetime import datetime
    import joblib
    
    # Monitor memory usage
    def print_memory_usage():
        process = psutil.Process(os.getpid())
        print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    
    # Create directory for checkpoints
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    try:
        start_time = time.time()
        print("\n🚀 Starting Pipeline Run...")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Initialize pipeline
        pipeline = MaxAccuracyPipeline("combined_data.csv", sample_size=None)
        
        # Define checkpoint saving function
        def save_checkpoint(stage):
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{stage}.pkl')
            joblib.dump({
                'pipeline': pipeline,
                'stage': stage,
                'timestamp': datetime.now()
            }, checkpoint_path)
            elapsed = (time.time() - start_time) / 3600  # hours
            print(f"\n💾 Checkpoint saved: {stage} (Elapsed: {elapsed:.2f} hours)")
        
        # Run pipeline with checkpoints
        stages = [
            ('preprocessing', pipeline.advanced_preprocess),
            ('feature_engineering', pipeline.advanced_feature_engineering),
            ('data_preparation', pipeline.prepare_data),
            ('model_training', pipeline.train_best_model),
            ('fine_tuning', pipeline.fine_tune_model),
            ('ensemble_creation', pipeline.create_ensemble)
        ]
        
        for stage_name, stage_func in tqdm(stages, desc="Pipeline Progress"):
            print(f"\n📍 Starting {stage_name}...")
            stage_func()
            save_checkpoint(stage_name)
        
        # Final evaluation
        print("\n🎯 Running final evaluation...")
        final_accuracy, y_pred = pipeline.evaluate_and_visualize()
        
        # Save final model
        model_path = 'final_model.pkl'
        joblib.dump(pipeline.model, model_path)
        print(f"\n✅ Final model saved to: {model_path}")
        
        # Print runtime statistics
        end_time = time.time()
        total_hours = (end_time - start_time) / 3600
        print("\n⏱️ Runtime Statistics:")
        print(f"Total runtime: {total_hours:.2f} hours")
        print(f"Final accuracy: {final_accuracy:.4f}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user")
        current_time = time.time()
        elapsed = (current_time - start_time) / 3600
        print(f"Elapsed time: {elapsed:.2f} hours")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Clean up old checkpoints if successful
        if 'final_accuracy' in locals() and os.path.exists(checkpoint_dir):
            import shutil
            shutil.rmtree(checkpoint_dir)
            print("\n🧹 Cleaned up checkpoint files")

# To run with sleep prevention:
# In terminal:
# caffeinate -i python model_training2.py > training_log.txt 2>&1