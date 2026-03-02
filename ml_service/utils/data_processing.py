import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

# Configuration constants
SKEWNESS_THRESHOLDS = {
    'normal': 0.5,
    'moderate': 1.0
}

ZSCORE_THRESHOLD = 3.0
IQR_MULTIPLIER = 1.5


class DataAnalyzer:
    """
    Comprehensive data analysis utility for DataFrames.
    Provides analysis of missing values, outliers, distributions, and basic statistics.
    """
    
    def __init__(self, df: pd.DataFrame, deep_memory_check: bool = False):
        """
        Initialize DataAnalyzer.
        
        Args:
            df: pandas DataFrame to analyze
            deep_memory_check: Use deep=True for memory calculation (slower, more accurate)
        
        Raises:
            ValueError: If DataFrame is None or empty
            TypeError: If input is not a DataFrame
        """
        self._validate_input(df)
        self.df = df
        self.deep_memory_check = deep_memory_check
        self._cache = {}
    
    @staticmethod
    def _validate_input(df: Any) -> None:
        """Validate DataFrame input."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(df)}")
        if df.empty:
            raise ValueError("DataFrame is empty")
        if len(df.columns) == 0:
            raise ValueError("DataFrame has no columns")
    
    def analyze_missing_values(self) -> Dict[str, Any]:
        """
        Analyze missing values in the dataset.
        
        Returns:
            Dictionary containing:
                - columns: List of columns with missing values
                - missing_count: Count of missing values per column
                - missing_percentage: Percentage of missing values per column
                - total_cells: Total number of cells in DataFrame
                - total_missing: Total number of missing cells
        """
        if 'missing_values' in self._cache:
            return self._cache['missing_values']
        
        total_missing = int(self.df.isnull().sum().sum())
        total_cells = len(self.df) * len(self.df.columns)
        
        missing_data = {
            'columns': [],
            'missing_count': {},
            'missing_percentage': {},
            'total_cells': total_cells,
            'total_missing': total_missing,
            'total_missing_percentage': float((total_missing / total_cells * 100) if total_cells > 0 else 0)
        }
        
        for column in self.df.columns:
            missing_count = self.df[column].isnull().sum()
            safe_len = len(self.df) if len(self.df) > 0 else 1
            missing_pct = (missing_count / safe_len) * 100
            
            if missing_count > 0:
                missing_data['columns'].append(column)
                missing_data['missing_count'][column] = int(missing_count)
                missing_data['missing_percentage'][column] = float(missing_pct)
        
        self._cache['missing_values'] = missing_data
        return missing_data
    
    def detect_outliers(self, method: str = 'iqr', 
                       z_threshold: float = ZSCORE_THRESHOLD,
                       iqr_multiplier: float = IQR_MULTIPLIER) -> Dict[str, Any]:
        """
        Detect outliers using IQR or Z-Score method.
        
        Args:
            method: Detection method ('iqr' or 'zscore')
            z_threshold: Z-score threshold for outlier detection (default: 3.0)
            iqr_multiplier: IQR multiplier for bounds calculation (default: 1.5)
        
        Returns:
            Dictionary containing outlier information per column
        
        Raises:
            ValueError: If invalid method specified
        """
        if method not in ['iqr', 'zscore']:
            raise ValueError(f"Method must be 'iqr' or 'zscore', got {method}")
        
        cache_key = f'outliers_{method}_{z_threshold}_{iqr_multiplier}'
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        outliers = {
            'method': method,
            'columns': {},
            'total_outliers': 0
        }
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            logger.warning("No numeric columns found for outlier detection")
            self._cache[cache_key] = outliers
            return outliers
        
        for col in numeric_cols:
            data = self.df[col].dropna()
            
            if len(data) == 0:
                continue
            
            try:
                if method == 'iqr':
                    outliers_info = self._detect_outliers_iqr(col, data, iqr_multiplier)
                else:  # zscore
                    outliers_info = self._detect_outliers_zscore(col, data, z_threshold)
                
                if outliers_info:
                    outliers['columns'][col] = outliers_info
            except Exception as e:
                logger.warning(f"Error detecting outliers in column {col}: {e}")
        
        outliers['total_outliers'] = sum(v['count'] for v in outliers['columns'].values())
        self._cache[cache_key] = outliers
        return outliers
    
    @staticmethod
    def _detect_outliers_iqr(col: str, data: pd.Series, 
                             multiplier: float = IQR_MULTIPLIER) -> Optional[Dict[str, Any]]:
        """Helper method for IQR-based outlier detection."""
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        # Handle case where IQR is 0 (no variation in data)
        if IQR == 0:
            return None
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        outlier_mask = (data < lower_bound) | (data > upper_bound)
        outlier_count = outlier_mask.sum()
        
        if outlier_count > 0:
            return {
                'count': int(outlier_count),
                'percentage': float((outlier_count / len(data)) * 100),
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound),
                'iqr': float(IQR),
                'method': 'IQR'
            }
        return None
    
    @staticmethod
    def _detect_outliers_zscore(col: str, data: pd.Series, 
                               threshold: float = ZSCORE_THRESHOLD) -> Optional[Dict[str, Any]]:
        """Helper method for Z-Score-based outlier detection."""
        try:
            z_scores = np.abs(stats.zscore(data))
        except Exception as e:
            logger.warning(f"Could not compute z-scores for {col}: {e}")
            return None
        
        outlier_mask = z_scores > threshold
        outlier_count = outlier_mask.sum()
        
        if outlier_count > 0:
            return {
                'count': int(outlier_count),
                'percentage': float((outlier_count / len(data)) * 100),
                'threshold': float(threshold),
                'method': 'Z-Score'
            }
        return None
    
    def analyze_distribution(self) -> Dict[str, Any]:
        """
        Analyze distribution of numeric columns.
        
        Returns:
            Dictionary with distribution statistics per numeric column
        """
        if 'distributions' in self._cache:
            return self._cache['distributions']
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        distributions = {}
        
        if len(numeric_cols) == 0:
            logger.warning("No numeric columns found for distribution analysis")
            self._cache['distributions'] = distributions
            return distributions
        
        for col in numeric_cols:
            data = self.df[col].dropna()
            
            if len(data) > 0:
                try:
                    dist_info = self._calculate_distribution_stats(data, col)
                    distributions[col] = dist_info
                except Exception as e:
                    logger.warning(f"Error analyzing distribution for {col}: {e}")
        
        self._cache['distributions'] = distributions
        return distributions
    
    @staticmethod
    def _calculate_distribution_stats(data: pd.Series, col: str) -> Dict[str, Any]:
        """Helper method to calculate distribution statistics."""
        skewness = float(stats.skew(data))
        kurtosis = float(stats.kurtosis(data))
        
        # Categorize skewness
        if abs(skewness) < SKEWNESS_THRESHOLDS['normal']:
            skew_category = 'Normal'
        elif abs(skewness) < SKEWNESS_THRESHOLDS['moderate']:
            skew_category = 'Moderately Skewed'
        else:
            skew_category = 'Highly Skewed'
        
        return {
            'mean': float(data.mean()),
            'median': float(data.median()),
            'std': float(data.std()),
            'min': float(data.min()),
            'max': float(data.max()),
            'q25': float(data.quantile(0.25)),
            'q75': float(data.quantile(0.75)),
            'skewness': skewness,
            'skew_category': skew_category,
            'kurtosis': kurtosis,
            'count': int(len(data))
        }
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get overall data summary.
        
        Returns:
            Dictionary with high-level dataset statistics
        """
        if 'summary' in self._cache:
            return self._cache['summary']
        
        total_missing = int(self.df.isnull().sum().sum())
        total_cells = len(self.df) * len(self.df.columns)
        total_duplicates = int(self.df.duplicated().sum())
        
        summary = {
            'rows': len(self.df),
            'columns': len(self.df.columns),
            'column_names': list(self.df.columns),
            'column_types': {col: str(self.df[col].dtype) for col in self.df.columns},
            'memory_usage_mb': float(self.df.memory_usage(deep=self.deep_memory_check).sum() / 1024 / 1024),
            'missing_values': total_missing,
            'missing_values_percentage': float((total_missing / total_cells * 100) if total_cells > 0 else 0),
            'duplicates': total_duplicates,
            'duplicate_percentage': float((total_duplicates / len(self.df) * 100) if len(self.df) > 0 else 0)
        }
        
        self._cache['summary'] = summary
        return summary
    
    def analyze_cardinality(self) -> Dict[str, Any]:
        """
        Analyze cardinality (unique values) for all columns.
        
        Returns:
            Dictionary with unique value counts and percentages per column
        """
        if 'cardinality' in self._cache:
            return self._cache['cardinality']
        
        cardinality = {}
        safe_len = len(self.df) if len(self.df) > 0 else 1
        
        for col in self.df.columns:
            unique_count = self.df[col].nunique()
            cardinality[col] = {
                'unique_count': int(unique_count),
                'unique_percentage': float((unique_count / safe_len) * 100),
                'is_categorical': unique_count < safe_len * 0.1  # Heuristic: <10% unique is categorical
            }
        
        self._cache['cardinality'] = cardinality
        return cardinality
    
    def get_correlation_matrix(self) -> Optional[pd.DataFrame]:
        """
        Get correlation matrix for numeric columns.
        
        Returns:
            Correlation DataFrame or None if no numeric columns
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            logger.warning("No numeric columns found for correlation analysis")
            return None
        
        try:
            return self.df[numeric_cols].corr()
        except Exception as e:
            logger.warning(f"Error calculating correlations: {e}")
            return None
    
    def analyze_all(self) -> Dict[str, Any]:
        """
        Run all analyses and return comprehensive report.
        
        Returns:
            Dictionary containing all analysis results
        """
        return {
            'summary': self.get_data_summary(),
            'missing_values': self.analyze_missing_values(),
            'outliers': self.detect_outliers(),
            'distributions': self.analyze_distribution(),
            'cardinality': self.analyze_cardinality()
        }
    
    def clear_cache(self) -> None:
        """Clear internal cache of analysis results."""
        self._cache.clear()


# Legacy function wrappers for backward compatibility
def analyze_missing_values(df: pd.DataFrame) -> Dict[str, Any]:
    """Legacy function wrapper. Use DataAnalyzer class instead."""
    analyzer = DataAnalyzer(df)
    return analyzer.analyze_missing_values()


def detect_outliers(df: pd.DataFrame, method: str = 'iqr') -> Dict[str, Any]:
    """Legacy function wrapper. Use DataAnalyzer class instead."""
    analyzer = DataAnalyzer(df)
    return analyzer.detect_outliers(method=method)


def analyze_distribution(df: pd.DataFrame) -> Dict[str, Any]:
    """Legacy function wrapper. Use DataAnalyzer class instead."""
    analyzer = DataAnalyzer(df)
    return analyzer.analyze_distribution()


def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Legacy function wrapper. Use DataAnalyzer class instead."""
    analyzer = DataAnalyzer(df)
    return analyzer.get_data_summary()
