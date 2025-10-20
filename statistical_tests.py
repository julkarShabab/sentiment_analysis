"""
Statistical Significance Testing for Gender Bias in MH Chatbots
Tests whether observed differences are statistically significant
"""

import pandas as pd
import numpy as np
from scipy import stats
import json

class BiasStatisticalAnalyzer:
    def __init__(self):
        pass
    
    def cohens_d(self, group1, group2):
        """
        Calculate Cohen's d effect size
        Small: 0.2, Medium: 0.5, Large: 0.8
        """
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
        
        # Cohen's d
        d = (np.mean(group1) - np.mean(group2)) / pooled_std
        return d
    
    def interpret_effect_size(self, d):
        """Interpret Cohen's d effect size"""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def interpret_pvalue(self, p):
        """Interpret p-value"""
        if p < 0.001:
            return "highly significant (***)"
        elif p < 0.01:
            return "very significant (**)"
        elif p < 0.05:
            return "significant (*)"
        else:
            return "not significant (ns)"
    
    def test_metric(self, df, metric_col, gender_col='Gender', gender1='Male', gender2='Female'):
        """
        Run statistical tests for a single metric
        """
        try:
            group1 = df[df[gender_col] == gender1][metric_col].dropna()
            group2 = df[df[gender_col] == gender2][metric_col].dropna()
            
            if len(group1) == 0 or len(group2) == 0:
                return {
                    'error': f'No data found for one or both groups',
                    f'{gender1}_n': int(len(group1)),
                    f'{gender2}_n': int(len(group2))
                }
            
            # Descriptive statistics
            desc_stats = {
                f'{gender1.lower()}_mean': float(group1.mean()),
                f'{gender1.lower()}_std': float(group1.std()),
                f'{gender1.lower()}_n': int(len(group1)),
                f'{gender2.lower()}_mean': float(group2.mean()),
                f'{gender2.lower()}_std': float(group2.std()),
                f'{gender2.lower()}_n': int(len(group2)),
                'mean_difference': float(group1.mean() - group2.mean())
            }
            
            # Check normality (Shapiro-Wilk test)
            if len(group1) >= 3 and len(group2) >= 3:
                _, p_norm1 = stats.shapiro(group1)
                _, p_norm2 = stats.shapiro(group2)
                is_normal = (p_norm1 > 0.05) and (p_norm2 > 0.05)
            else:
                is_normal = False
            
            # Choose appropriate test
            if is_normal and len(group1) >= 30 and len(group2) >= 30:
                # Use independent t-test for normal distributions
                t_stat, p_value = stats.ttest_ind(group1, group2)
                test_used = "Independent t-test"
            else:
                # Use Mann-Whitney U test for non-normal or small samples
                u_stat, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
                test_used = "Mann-Whitney U test"
            
            # Effect size (Cohen's d)
            effect_size = self.cohens_d(group1, group2)
            
            results = {
                **desc_stats,
                'test_used': test_used,
                'is_normal_distribution': is_normal,
                'p_value': float(p_value),
                'significance': self.interpret_pvalue(p_value),
                'cohens_d': float(effect_size),
                'effect_size_interpretation': self.interpret_effect_size(effect_size),
                'is_significant': bool(p_value < 0.05)
            }
            
            return results
            
        except Exception as e:
            return {
                'error': str(e),
                'metric': metric_col
            }
    
    def test_proportions(self, df, binary_col, gender_col='Gender', gender1='Male', gender2='Female'):
        """
        Test for significant difference in proportions (e.g., gendered language presence)
        Uses Chi-square test
        """
        try:
            # Create contingency table
            contingency = pd.crosstab(df[gender_col], df[binary_col])
            
            # Chi-square test
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
            
            # Calculate proportions
            group1_data = df[df[gender_col] == gender1]
            group2_data = df[df[gender_col] == gender2]
            
            prop1 = group1_data[binary_col].sum() / len(group1_data) * 100
            prop2 = group2_data[binary_col].sum() / len(group2_data) * 100
            
            results = {
                f'{gender1.lower()}_proportion': float(prop1),
                f'{gender2.lower()}_proportion': float(prop2),
                'difference': float(prop1 - prop2),
                'chi_square': float(chi2),
                'p_value': float(p_value),
                'significance': self.interpret_pvalue(p_value),
                'is_significant': bool(p_value < 0.05)
            }
            
            return results
            
        except Exception as e:
            return {
                'error': str(e),
                'column': binary_col
            }
    
    def full_analysis(self, csv_path, gender_col='Gender', gender1='Male', gender2='Female'):
        """
        Run complete statistical analysis on scored dataset
        """
        df = pd.read_csv(csv_path)
        
        print(f"Running statistical analysis...")
        print(f"Comparing: {gender1} (n={len(df[df[gender_col]==gender1])}) vs {gender2} (n={len(df[df[gender_col]==gender2])})")
        print("="*70)
        
        results = {}
        
        # Test continuous metrics
        # First try automated column names, then fall back to manual column names
        continuous_metrics = {
            'empathy_score': 'Empathy Score',
            'sentiment_score': 'Sentiment Score',
            'advice_quality_score': 'Advice Quality Score',
            'word_count': 'Word Count',
            'Empathy': 'Empathy Score (Manual)',
            'Sentiment': 'Sentiment Score (Manual)',
            'AdviceQuality': 'Advice Quality (Manual)'
        }
        
        for col, name in continuous_metrics.items():
            if col in df.columns:
                print(f"\nTesting: {name}")
                results[col] = self.test_metric(df, col, gender_col, gender1, gender2)
                
                # Print summary
                r = results[col]
                print(f"  {gender1}: {r[f'{gender1}_mean']:.2f} ± {r[f'{gender1}_std']:.2f}")
                print(f"  {gender2}: {r[f'{gender2}_mean']:.2f} ± {r[f'{gender2}_std']:.2f}")
                print(f"  Difference: {r['mean_difference']:+.2f}")
                print(f"  Test: {r['test_used']}")
                print(f"  p-value: {r['p_value']:.4f} - {r['significance']}")
                print(f"  Effect size (Cohen's d): {r['cohens_d']:.3f} ({r['effect_size_interpretation']})")
        
        # Test binary metric (gendered language)
        gendered_cols = ['gendered_language_flag', 'GenderedLang']
        gendered_col_found = None
        
        for col in gendered_cols:
            if col in df.columns:
                gendered_col_found = col
                break
        
        if gendered_col_found:
            print(f"\nTesting: Gendered Language (Proportion)")
            result = self.test_proportions(df, gendered_col_found, gender_col, gender1, gender2)
            
            if 'error' in result:
                print(f"  Error: {result['error']}")
            else:
                results['gendered_language'] = result
                
                r = results['gendered_language']
                print(f"  {gender1}: {r[f'{gender1.lower()}_proportion']:.1f}%")
                print(f"  {gender2}: {r[f'{gender2.lower()}_proportion']:.1f}%")
                print(f"  Difference: {r['difference']:+.1f}%")
                print(f"  Chi-square: {r['chi_square']:.2f}")
                print(f"  p-value: {r['p_value']:.4f} - {r['significance']}")
        
        print("\n" + "="*70)
        print("SUMMARY OF SIGNIFICANT FINDINGS:")
        print("="*70)
        
        significant_findings = []
        for metric, data in results.items():
            if data['is_significant']:
                metric_name = continuous_metrics.get(metric, metric.replace('_', ' ').title())
                significant_findings.append(f"✓ {metric_name}: {data['significance']}")
        
        if significant_findings:
            for finding in significant_findings:
                print(finding)
        else:
            print("No statistically significant differences found.")
        
        return results
    
    def save_results(self, results, output_path='statistical_results.json'):
        """Save results to JSON file"""
        # Convert numpy/pandas types to native Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_to_serializable(results)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"\nStatistical results saved to: {output_path}")


# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_CSV = "scored_responses.csv"  # Your scored CSV from previous step
OUTPUT_JSON = "statistical_results.json"
GENDER_COLUMN = "Gender"
GENDER_1 = "male"
GENDER_2 = "female"

# ============================================================================
# RUN ANALYSIS
# ============================================================================

if __name__ == "__main__":
    analyzer = BiasStatisticalAnalyzer()
    
    try:
        results = analyzer.full_analysis(
            INPUT_CSV,
            gender_col=GENDER_COLUMN,
            gender1=GENDER_1,
            gender2=GENDER_2
        )
        
        analyzer.save_results(results, OUTPUT_JSON)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()