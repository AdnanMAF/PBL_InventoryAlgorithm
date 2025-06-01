import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
from pmdarima import auto_arima
import warnings
warnings.filterwarnings('ignore')

class SARIMAXForecaster:
    def __init__(self, data_path, family=None, model_dir='models'):
        self.data_path = data_path
        self.family = family  # Can be None for multi-family processing
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, f'sarimax_{family}.pkl') if family else None
        os.makedirs(model_dir, exist_ok=True)
        
        self.df = None
        self.ts_data = None
        self.model = None
        self.fit = None
        self.order = None
        self.seasonal_order = None
        self.available_families = None
        
    def load_and_preprocess(self):
        """Load and preprocess the data"""
        # Load data
        self.df = pd.read_csv(self.data_path, parse_dates=['date'])
        self.df.drop(columns=['id'], inplace=True)
        
        # Get available families
        self.available_families = sorted(self.df['family'].unique())
        print(f"Available families: {self.available_families}")
        
        if self.family:
            print(f"Loading data for {self.family}...")
            self.df = self.df[self.df['family'] == self.family]
            print(f"Data shape after filtering: {self.df.shape}")
        else:
            print("Loading data for all families combined...")
            
        print(f"Date range: {self.df['date'].min()} to {self.df['date'].max()}")
        
        # Aggregate daily across all stores
        self.ts_data = (
            self.df
            .groupby('date')[['sales', 'onpromotion']]
            .sum()
            .reset_index()
        )
        
        # Set date as index and convert to period
        self.ts_data.set_index('date', inplace=True)
        self.ts_data.index = pd.DatetimeIndex(self.ts_data.index).to_period('D')
        
        # Handle missing dates
        full_range = pd.period_range(
            start=self.ts_data.index.min(),
            end=self.ts_data.index.max(),
            freq='D'
        )
        self.ts_data = self.ts_data.reindex(full_range, fill_value=0)
        
        # Create target variable with log transformation
        # Add small constant to handle zeros
        self.ts_data['y'] = np.log1p(self.ts_data['sales'] + 1)
        
        # Create additional features
        self._create_features()
        
        print(f"Final time series shape: {self.ts_data.shape}")
        print(f"Sales statistics:")
        print(self.ts_data['sales'].describe())
        
    def _create_features(self):
        """Create additional features for the model"""
        # Day of week
        self.ts_data['dow'] = self.ts_data.index.to_timestamp().dayofweek
        
        # Month
        self.ts_data['month'] = self.ts_data.index.to_timestamp().month
        
        # Year
        self.ts_data['year'] = self.ts_data.index.to_timestamp().year
        
        # Promotion lag features
        self.ts_data['onpromotion_lag1'] = self.ts_data['onpromotion'].shift(1)
        self.ts_data['onpromotion_lag7'] = self.ts_data['onpromotion'].shift(7)
        
        # Rolling averages for promotions
        self.ts_data['onpromotion_ma7'] = self.ts_data['onpromotion'].rolling(7).mean()
        self.ts_data['onpromotion_ma30'] = self.ts_data['onpromotion'].rolling(30).mean()
        
        # Fill NaN values
        self.ts_data.fillna(method='bfill', inplace=True)
        self.ts_data.fillna(0, inplace=True)
        
    def eda_and_diagnostics(self, plot=True):
        """Perform exploratory data analysis and diagnostics"""
        print("\n" + "="*50)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*50)
        
        # Basic statistics
        print(f"\nZero sales days: {(self.ts_data['sales'] == 0).sum()}")
        print(f"Promotion days: {(self.ts_data['onpromotion'] > 0).sum()}")
        
        # Stationarity test
        adf_result = adfuller(self.ts_data['y'])
        print(f"\nStationarity Test (ADF):")
        print(f"ADF Statistic: {adf_result[0]:.4f}")
        print(f"p-value: {adf_result[1]:.4f}")
        print(f"Is stationary: {adf_result[1] < 0.05}")
        
        if plot:
            self._plot_eda()
            
    def _plot_eda(self):
        """Create EDA plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Original and log-transformed sales
        recent_data = self.ts_data.loc[self.ts_data.index.to_timestamp() >= '2016-01-01']
        
        axes[0,0].plot(recent_data.index.to_timestamp(), recent_data['sales'])
        axes[0,0].set_title('Sales (Recent 2 Years)')
        axes[0,0].grid(True)
        
        axes[0,1].plot(recent_data.index.to_timestamp(), recent_data['y'])
        axes[0,1].set_title('Log-transformed Sales')
        axes[0,1].grid(True)
        
        # Seasonal decomposition (sample recent data)
        sample_data = recent_data['y'].iloc[-365:]  # Last year
        decomposition = seasonal_decompose(sample_data.values, model='additive', period=7)
        
        axes[1,0].plot(decomposition.trend)
        axes[1,0].set_title('Trend Component')
        axes[1,0].grid(True)
        
        axes[1,1].plot(decomposition.seasonal[:21])  # Show 3 weeks of seasonality
        axes[1,1].set_title('Seasonal Component (3 weeks)')
        axes[1,1].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Correlation plot
        plt.figure(figsize=(10, 6))
        corr_data = self.ts_data[['sales', 'onpromotion', 'onpromotion_lag1', 
                                 'onpromotion_ma7', 'dow', 'month']].corr()
        sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.show()
        
    def find_optimal_parameters(self):
        """Find optimal SARIMAX parameters using auto_arima"""
        print("\n" + "="*50)
        print("PARAMETER OPTIMIZATION")
        print("="*50)
        
        # Prepare exogenous variables
        exog_vars = ['onpromotion', 'onpromotion_lag1', 'onpromotion_ma7']
        exog_data = self.ts_data[exog_vars]
        
        print("Running auto_arima optimization...")
        stepwise = auto_arima(
            self.ts_data['y'],
            exogenous=exog_data,
            seasonal=True,
            m=7,  # Weekly seasonality
            start_p=0, start_q=0, max_p=3, max_q=3,
            start_P=0, start_Q=0, max_P=2, max_Q=2,
            d=None, D=None,
            trace=True,
            suppress_warnings=True,
            error_action='ignore',
            stepwise=True,
            n_jobs=-1
        )
        
        self.order = stepwise.order
        self.seasonal_order = stepwise.seasonal_order
        
        print(f"\nOptimal parameters:")
        print(f"ARIMA order: {self.order}")
        print(f"Seasonal order: {self.seasonal_order}")
        print(f"AIC: {stepwise.aic():.2f}")
        
    def train_model(self, force_retrain=False):
        """Train the SARIMAX model"""
        print("\n" + "="*50)
        print("MODEL TRAINING")
        print("="*50)
        
        if os.path.exists(self.model_path) and not force_retrain:
            print("Loading saved model...")
            self.fit = SARIMAXResults.load(self.model_path)
            # Extract parameters from saved model
            self.order = self.fit.model.order
            self.seasonal_order = self.fit.model.seasonal_order
        else:
            print("Training new model...")
            
            # Prepare exogenous variables
            exog_vars = ['onpromotion', 'onpromotion_lag1', 'onpromotion_ma7']
            exog_data = self.ts_data[exog_vars]
            
            # Create and fit model
            self.model = SARIMAX(
                self.ts_data['y'],
                exog=exog_data,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            
            self.fit = self.model.fit(disp=False, maxiter=100)
            
            # Save model
            self.fit.save(self.model_path)
            print(f"Model saved to {self.model_path}")
            
        print(f"\nModel Summary:")
        print(f"AIC: {self.fit.aic:.2f}")
        print(f"BIC: {self.fit.bic:.2f}")
        print(f"Log-likelihood: {self.fit.llf:.2f}")
        
    def validate_model(self, test_size=30, plot=True):
        """Perform time series cross-validation"""
        print("\n" + "="*50)
        print("MODEL VALIDATION")
        print("="*50)
        
        # Prepare data
        n_test = test_size
        train_data = self.ts_data.iloc[:-n_test]
        test_data = self.ts_data.iloc[-n_test:]
        
        # Prepare exogenous variables
        exog_vars = ['onpromotion', 'onpromotion_lag1', 'onpromotion_ma7']
        exog_train = train_data[exog_vars]
        exog_test = test_data[exog_vars]
        
        # Fit model on training data
        model_val = SARIMAX(
            train_data['y'],
            exog=exog_train,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        fit_val = model_val.fit(disp=False)
        
        # Generate forecasts
        forecast_result = fit_val.get_forecast(steps=n_test, exog=exog_test)
        pred_log = forecast_result.predicted_mean
        pred_ci = forecast_result.conf_int()
        
        # Transform back to original scale
        pred = np.expm1(pred_log) - 1  # Reverse the +1 from log1p
        pred_ci_orig = np.expm1(pred_ci) - 1
        
        # Calculate metrics
        actual = test_data['sales']
        metrics = self._calculate_metrics(actual, pred)
        
        print(f"Validation Results ({n_test} days):")
        for metric, value in metrics.items():
            print(f"{metric}: {value}")
            
        # Residual diagnostics
        self._residual_diagnostics(fit_val)
        
        if plot:
            self._plot_validation(train_data, test_data, pred, pred_ci_orig, actual)
            
    def get_family_list(self):
        """Get list of all available families"""
        if self.available_families is None:
            df_temp = pd.read_csv(self.data_path)
            self.available_families = sorted(df_temp['family'].unique())
        return self.available_families
    
    def train_multiple_families(self, families=None, validation_days=30):
        """Train models for multiple families"""
        if families is None:
            families = self.get_family_list()
            
        results = {}
        
        for family in families:
            print(f"\n{'='*80}")
            print(f"PROCESSING FAMILY: {family}")
            print(f"{'='*80}")
            
            try:
                # Create forecaster for this family
                family_forecaster = SARIMAXForecaster(
                    data_path=self.data_path,
                    family=family,
                    model_dir=self.model_dir
                )
                
                # Run pipeline
                metrics = family_forecaster.run_full_pipeline(validation_days=validation_days)
                results[family] = {
                    'metrics': metrics,
                    'forecaster': family_forecaster,
                    'status': 'success'
                }
                
            except Exception as e:
                print(f"Error processing {family}: {str(e)}")
                results[family] = {
                    'error': str(e),
                    'status': 'failed'
                }
                
    def compare_families(self, families=None, validation_days=30):
        """Compare performance across multiple families"""
        if families is None:
            families = self.get_family_list()
            
        print("Training models for family comparison...")
        results = self.train_multiple_families(families, validation_days)
        
        # Create comparison dataframe
        comparison_data = []
        for family, result in results.items():
            if result['status'] == 'success':
                metrics = result['metrics']
                comparison_data.append({
                    'Family': family,
                    'RMSE': float(metrics['RMSE']),
                    'MAE': float(metrics['MAE']),
                    'MAPE': float(metrics['MAPE'].replace('%', '')),
                    'R²': float(metrics['R²']),
                    'Relative_RMSE': float(metrics['Relative RMSE'].replace('%', '')),
                    'Mean_Sales': float(metrics['Mean Sales'])
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('MAPE')
        
        print(f"\n{'='*80}")
        print("FAMILY PERFORMANCE COMPARISON")
        print(f"{'='*80}")
        print(comparison_df.round(3))
        
        # Plot comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        axes[0,0].barh(comparison_df['Family'], comparison_df['MAPE'])
        axes[0,0].set_title('MAPE by Family')
        axes[0,0].set_xlabel('MAPE (%)')
        
        axes[0,1].barh(comparison_df['Family'], comparison_df['R²'])
        axes[0,1].set_title('R² by Family')
        axes[0,1].set_xlabel('R²')
        
        axes[1,0].scatter(comparison_df['Mean_Sales'], comparison_df['RMSE'])
        axes[1,0].set_title('RMSE vs Mean Sales')
        axes[1,0].set_xlabel('Mean Sales')
        axes[1,0].set_ylabel('RMSE')
        for i, family in enumerate(comparison_df['Family']):
            axes[1,0].annotate(family[:8], (comparison_df['Mean_Sales'].iloc[i], 
                                          comparison_df['RMSE'].iloc[i]), 
                             rotation=45, fontsize=8)
        
        axes[1,1].barh(comparison_df['Family'], comparison_df['Relative_RMSE'])
        axes[1,1].set_title('Relative RMSE by Family')
        axes[1,1].set_xlabel('Relative RMSE (%)')
        
        plt.tight_layout()
        plt.show()
        
        return comparison_df, results
    
    def generate_submission_file(self, test_data_path, families=None, output_file='submission.csv'):
        """Generate submission file for all families"""
        if families is None:
            families = self.get_family_list()
            
        # Load test data
        test_df = pd.read_csv(test_data_path, parse_dates=['date'])
        
        print(f"Test data shape: {test_df.shape}")
        print(f"Test date range: {test_df['date'].min()} to {test_df['date'].max()}")
        
        # Initialize results
        predictions = []
        
        for family in families:
            print(f"Generating predictions for {family}...")
            
            # Load trained model
            model_path = os.path.join(self.model_dir, f'sarimax_{family}.pkl')
            
            if not os.path.exists(model_path):
                print(f"No trained model found for {family}. Skipping...")
                continue
                
            try:
                # Filter test data for this family
                family_test = test_df[test_df['family'] == family].copy()
                
                if family_test.empty:
                    print(f"No test data for {family}")
                    continue
                
                # Load model
                fit = SARIMAXResults.load(model_path)
                
                # Prepare test data similar to training
                family_test_agg = (
                    family_test
                    .groupby('date')[['onpromotion']]
                    .sum()
                    .reset_index()
                )
                
                family_test_agg.set_index('date', inplace=True)
                family_test_agg.index = pd.DatetimeIndex(family_test_agg.index).to_period('D')
                
                # Create features (simplified for test data)
                family_test_agg['onpromotion_lag1'] = 0  # Placeholder
                family_test_agg['onpromotion_ma7'] = family_test_agg['onpromotion'].rolling(7, min_periods=1).mean()
                family_test_agg.fillna(0, inplace=True)
                
                # Generate predictions
                exog_vars = ['onpromotion', 'onpromotion_lag1', 'onpromotion_ma7']
                exog_test = family_test_agg[exog_vars]
                
                pred_log = fit.get_forecast(steps=len(exog_test), exog=exog_test).predicted_mean
                pred_sales = np.expm1(pred_log) - 1
                pred_sales = np.maximum(pred_sales, 0)  # Ensure non-negative
                
                # Map back to original test structure
                for i, (idx, row) in enumerate(family_test.iterrows()):
                    date_idx = pd.Period(row['date'], freq='D')
                    if date_idx in family_test_agg.index:
                        pred_idx = list(family_test_agg.index).index(date_idx)
                        predicted_sales = pred_sales.iloc[pred_idx]
                    else:
                        predicted_sales = 0
                        
                    predictions.append({
                        'id': row['id'] if 'id' in row else len(predictions),
                        'date': row['date'],
                        'store_nbr': row['store_nbr'],
                        'family': row['family'],
                        'sales': predicted_sales
                    })
                    
            except Exception as e:
                print(f"Error predicting for {family}: {str(e)}")
                # Add zero predictions as fallback
                for idx, row in family_test.iterrows():
                    predictions.append({
                        'id': row['id'] if 'id' in row else len(predictions),
                        'date': row['date'],
                        'store_nbr': row['store_nbr'],
                        'family': row['family'],
                        'sales': 0
                    })
        
        # Create submission dataframe
        submission_df = pd.DataFrame(predictions)
        
        # Save submission
        submission_df.to_csv(output_file, index=False)
        print(f"Submission saved to {output_file}")
        print(f"Submission shape: {submission_df.shape}")
        
        return submission_df
        
    def _calculate_metrics(self, actual, pred):
        """Calculate various forecasting metrics"""
        # Handle negative predictions
        pred = np.maximum(pred, 0)
        
        mse = mean_squared_error(actual, pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual, pred)
        
        # Avoid division by zero in MAPE
        mask = actual != 0
        mape = np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100 if mask.any() else np.inf
        
        r2 = r2_score(actual, pred)
        mean_sales = actual.mean()
        rel_rmse_pct = 100 * rmse / mean_sales if mean_sales > 0 else np.inf
        
        return {
            'RMSE': f'{rmse:.2f}',
            'MAE': f'{mae:.2f}',
            'MAPE': f'{mape:.2f}%',
            'R²': f'{r2:.3f}',
            'Relative RMSE': f'{rel_rmse_pct:.2f}%',
            'Mean Sales': f'{mean_sales:.2f}'
        }
        
    def _residual_diagnostics(self, fit_model):
        """Perform residual diagnostics"""
        print(f"\nResidual Diagnostics:")
        
        # Ljung-Box test for autocorrelation
        lb_stat, lb_pvalue = acorr_ljungbox(fit_model.resid, lags=10, return_df=False)
        print(f"Ljung-Box test p-value: {lb_pvalue:.4f}")
        print(f"Residuals are {'not ' if lb_pvalue < 0.05 else ''}white noise")
        
        # Plot residuals
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Residuals plot
        axes[0].plot(fit_model.resid)
        axes[0].set_title('Residuals')
        axes[0].grid(True)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(fit_model.resid, dist="norm", plot=axes[1])
        axes[1].set_title('Q-Q Plot')
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.show()
        
    def _plot_validation(self, train_data, test_data, pred, pred_ci, actual):
        """Plot validation results"""
        plt.figure(figsize=(12, 6))
        
        # Plot recent training data
        recent_train = train_data.iloc[-60:]  # Last 60 days of training
        plt.plot(recent_train.index.to_timestamp(), recent_train['sales'], 
                label='Training', alpha=0.7)
        
        # Plot actual test data
        plt.plot(test_data.index.to_timestamp(), actual, 
                label='Actual', color='red', linewidth=2)
        
        # Plot predictions
        plt.plot(test_data.index.to_timestamp(), pred, 
                label='Forecast', color='blue', linewidth=2)
        
        # Plot confidence intervals
        plt.fill_between(test_data.index.to_timestamp(), 
                        pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], 
                        alpha=0.3, label='95% CI')
        
        plt.title(f'{self.family} Sales - SARIMAX Validation')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
    def forecast_future(self, steps=30, exog_future=None):
        """Generate future forecasts"""
        print(f"\nGenerating {steps}-day forecast...")
        
        # If no future exogenous data provided, use last known values
        if exog_future is None:
            exog_vars = ['onpromotion', 'onpromotion_lag1', 'onpromotion_ma7']
            last_values = self.ts_data[exog_vars].iloc[-1]
            exog_future = pd.DataFrame([last_values] * steps, 
                                     columns=exog_vars)
        
        # Generate forecast
        forecast_result = self.fit.get_forecast(steps=steps, exog=exog_future)
        pred_log = forecast_result.predicted_mean
        pred_ci = forecast_result.conf_int()
        
        # Transform back to original scale
        pred = np.expm1(pred_log) - 1
        pred_ci_orig = np.expm1(pred_ci) - 1
        
        # Create forecast dates
        last_date = self.ts_data.index[-1].to_timestamp()
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                     periods=steps, freq='D')
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'forecast': pred,
            'lower_ci': pred_ci_orig.iloc[:, 0],
            'upper_ci': pred_ci_orig.iloc[:, 1]
        })
        
        return forecast_df
        
    def run_full_pipeline(self, validation_days=30):
        """Run the complete forecasting pipeline"""
        print("="*60)
        print(f"SARIMAX FORECASTING PIPELINE - {self.family}")
        print("="*60)
        
        # Step 1: Load and preprocess data
        self.load_and_preprocess()
        
        # Step 2: EDA and diagnostics
        self.eda_and_diagnostics()
        
        # Step 3: Find optimal parameters
        self.find_optimal_parameters()
        
        # Step 4: Train model
        self.train_model()
        
        # Step 5: Validate model
        metrics = self.validate_model(test_size=validation_days)
        
        return metrics

# Usage examples
if __name__ == "__main__":
    
    # Option 1: Single family analysis
    print("="*60)
    print("SINGLE FAMILY ANALYSIS")
    print("="*60)
    
    forecaster = SARIMAXForecaster(
        data_path='../data/train.csv',
        family='AUTOMOTIVE',
        model_dir='models'
    )
    
    metrics = forecaster.run_full_pipeline(validation_days=30)
    
    # Option 2: Multiple family comparison
    print("\n" + "="*60)
    print("MULTIPLE FAMILY COMPARISON")
    print("="*60)
    
    multi_forecaster = SARIMAXForecaster(
        data_path='../data/train.csv',
        model_dir='models'
    )
    
    # Compare top 5 families (you can specify which ones)
    top_families = ['AUTOMOTIVE', 'BABY CARE', 'BEAUTY', 'BEVERAGES', 'BOOKS']
    comparison_df, results = multi_forecaster.compare_families(
        families=top_families, 
        validation_days=30
    )
    
    # Option 3: Train all families
    print("\n" + "="*60)
    print("TRAINING ALL FAMILIES")
    print("="*60)
    
    # Uncomment to train all families (this will take a while!)
    # all_results = multi_forecaster.train_multiple_families(validation_days=30)
    
    # Option 4: Generate submission file
    print("\n" + "="*60)
    print("GENERATING SUBMISSION")
    print("="*60)
    
    # Uncomment when you have test data and trained models
    # submission = multi_forecaster.generate_submission_file(
    #     test_data_path='../data/test.csv',
    #     families=top_families,
    #     output_file='sarimax_submission.csv'
    # )