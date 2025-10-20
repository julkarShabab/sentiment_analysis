"""
Visualization Generator for Gender Bias Analysis
Creates publication-ready charts for research paper
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

class BiasVisualizer:
    def __init__(self, style='seaborn-v0_8-paper'):
        """Initialize visualizer with publication-ready styling"""
        plt.style.use('default')
        sns.set_palette("husl")
        self.colors = {
            'Male': '#3498db',      # Blue
            'Female': '#e74c3c',    # Red
            'neutral': '#95a5a6'    # Gray
        }
        
    def create_comparison_boxplots(self, df, metrics, gender_col='Gender', output_dir='visualizations'):
        """
        Create box plots comparing metrics across genders
        """
        Path(output_dir).mkdir(exist_ok=True)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Gender Comparison: Mental Health Chatbot Response Metrics', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        axes = axes.flatten()
        
        metric_info = {
            'empathy_score': {'title': 'Empathy Score', 'ylabel': 'Score (1-5)', 'ylim': (0, 6)},
            'sentiment_score': {'title': 'Sentiment Score', 'ylabel': 'Score (-1 to 1)', 'ylim': (-1, 1)},
            'advice_quality_score': {'title': 'Advice Quality Score', 'ylabel': 'Score (1-5)', 'ylim': (0, 6)},
            'word_count': {'title': 'Response Length', 'ylabel': 'Word Count', 'ylim': None}
        }
        
        for idx, metric in enumerate(metrics):
            if metric not in df.columns:
                continue
                
            ax = axes[idx]
            info = metric_info[metric]
            
            # Create box plot
            sns.boxplot(data=df, x=gender_col, y=metric, ax=ax, 
                       palette=[self.colors['Male'], self.colors['Female']],
                       width=0.5)
            
            # Add individual points with jitter
            sns.stripplot(data=df, x=gender_col, y=metric, ax=ax,
                         color='black', alpha=0.3, size=3, jitter=0.2)
            
            # Styling
            ax.set_title(info['title'], fontsize=13, fontweight='bold')
            ax.set_xlabel('Gender', fontsize=11)
            ax.set_ylabel(info['ylabel'], fontsize=11)
            if info['ylim']:
                ax.set_ylim(info['ylim'])
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            
            # Add mean lines
            for gender in df[gender_col].unique():
                mean_val = df[df[gender_col]==gender][metric].mean()
                x_pos = 0 if gender == 'Male' else 1
                ax.hlines(mean_val, x_pos-0.2, x_pos+0.2, colors='red', 
                         linewidth=2, linestyles='dashed', label='Mean' if x_pos==0 else '')
            
        plt.tight_layout()
        output_path = Path(output_dir) / 'comparison_boxplots.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
    
    def create_mean_comparison_bars(self, df, metrics, gender_col='Gender', output_dir='visualizations'):
        """
        Create bar charts showing mean differences
        """
        Path(output_dir).mkdir(exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Mean Score Comparison by Gender', fontsize=16, fontweight='bold', y=0.995)
        axes = axes.flatten()
        
        metric_labels = {
            'empathy_score': 'Empathy',
            'sentiment_score': 'Sentiment',
            'advice_quality_score': 'Advice Quality',
            'word_count': 'Word Count'
        }
        
        for idx, metric in enumerate(metrics):
            if metric not in df.columns:
                continue
                
            ax = axes[idx]
            
            # Calculate means and std
            grouped = df.groupby(gender_col)[metric].agg(['mean', 'std', 'count'])
            
            # Calculate standard error
            grouped['se'] = grouped['std'] / np.sqrt(grouped['count'])
            
            # Create bar plot
            x_pos = np.arange(len(grouped))
            bars = ax.bar(x_pos, grouped['mean'], 
                         color=[self.colors['Male'], self.colors['Female']],
                         alpha=0.8, edgecolor='black', linewidth=1.5)
            
            # Add error bars
            ax.errorbar(x_pos, grouped['mean'], yerr=grouped['se'],
                       fmt='none', ecolor='black', capsize=5, capthick=2)
            
            # Add value labels on bars
            for i, (bar, val) in enumerate(zip(bars, grouped['mean'])):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.2f}',
                       ha='center', va='bottom', fontweight='bold', fontsize=10)
            
            # Styling
            ax.set_title(metric_labels[metric], fontsize=13, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(grouped.index, fontsize=11)
            ax.set_ylabel('Mean Score', fontsize=11)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            
        plt.tight_layout()
        output_path = Path(output_dir) / 'mean_comparison_bars.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
    
    def create_difference_plot(self, df, gender_col='Gender', output_dir='visualizations'):
        """
        Create a plot showing the magnitude and direction of differences
        """
        Path(output_dir).mkdir(exist_ok=True)
        
        metrics = ['empathy_score', 'sentiment_score', 'advice_quality_score', 'word_count']
        labels = ['Empathy', 'Sentiment', 'Advice Quality', 'Word Count']
        
        # Calculate differences (Male - Female)
        differences = []
        genders = df[gender_col].unique()
        male_data = df[df[gender_col] == 'Male']
        female_data = df[df[gender_col] == 'Female']
        
        for metric in metrics:
            if metric in df.columns:
                diff = male_data[metric].mean() - female_data[metric].mean()
                differences.append(diff)
            else:
                differences.append(0)
        
        # Normalize for better visualization (except word count)
        normalized_diffs = []
        for i, (metric, diff) in enumerate(zip(metrics, differences)):
            if metric == 'word_count':
                # Scale word count to similar range
                normalized_diffs.append(diff / 20)
            else:
                normalized_diffs.append(diff)
        
        # Create horizontal bar plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['#3498db' if d > 0 else '#e74c3c' for d in differences]
        bars = ax.barh(labels, differences, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bar, val in zip(bars, differences):
            width = bar.get_width()
            label_x_pos = width + (0.5 if width > 0 else -0.5)
            ax.text(label_x_pos, bar.get_y() + bar.get_height()/2,
                   f'{val:+.2f}',
                   ha='left' if width > 0 else 'right',
                   va='center', fontweight='bold', fontsize=11)
        
        # Add vertical line at zero
        ax.axvline(x=0, color='black', linestyle='-', linewidth=2)
        
        # Styling
        ax.set_xlabel('Difference (Male - Female)', fontsize=12, fontweight='bold')
        ax.set_title('Gender Differences in Response Metrics\n(Positive = Higher for Males, Negative = Higher for Females)',
                    fontsize=13, fontweight='bold', pad=20)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#3498db', alpha=0.8, edgecolor='black', label='Male > Female'),
            Patch(facecolor='#e74c3c', alpha=0.8, edgecolor='black', label='Female > Male')
        ]
        ax.legend(handles=legend_elements, loc='best', framealpha=0.9)
        
        plt.tight_layout()
        output_path = Path(output_dir) / 'difference_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
    
    def create_gendered_language_plot(self, df, gender_col='Gender', output_dir='visualizations'):
        """
        Create visualization for gendered language frequency
        """
        Path(output_dir).mkdir(exist_ok=True)
        
        # Try both column names
        gendered_col = None
        if 'gendered_language_flag' in df.columns:
            gendered_col = 'gendered_language_flag'
        elif 'GenderedLang' in df.columns:
            gendered_col = 'GenderedLang'
        else:
            print("Warning: gendered language column not found")
            return
        
        # Calculate percentages
        grouped = df.groupby(gender_col)[gendered_col].agg(['sum', 'count'])
        grouped['percentage'] = (grouped['sum'] / grouped['count']) * 100
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Gendered Language Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Percentage bar chart
        bars = ax1.bar(grouped.index, grouped['percentage'],
                      color=[self.colors['Male'], self.colors['Female']],
                      alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bar, val in zip(bars, grouped['percentage']):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}%',
                    ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        ax1.set_ylabel('Percentage of Responses', fontsize=11)
        ax1.set_title('Gendered Language Frequency', fontsize=13, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        ax1.set_ylim(0, max(grouped['percentage']) * 1.2)
        
        # Plot 2: Count stacked bar
        has_gendered = df.groupby([gender_col, gendered_col]).size().unstack(fill_value=0)
        has_gendered.plot(kind='bar', stacked=True, ax=ax2,
                         color=['#95a5a6', '#e67e22'], 
                         alpha=0.8, edgecolor='black', linewidth=1.5)
        
        ax2.set_ylabel('Number of Responses', fontsize=11)
        ax2.set_title('Response Distribution', fontsize=13, fontweight='bold')
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)
        ax2.legend(['No Gendered Language', 'Has Gendered Language'], framealpha=0.9)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        output_path = Path(output_dir) / 'gendered_language_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
    
    def create_correlation_heatmap(self, df, output_dir='visualizations'):
        """
        Create correlation heatmap of metrics
        """
        Path(output_dir).mkdir(exist_ok=True)
        
        metrics = ['empathy_score', 'sentiment_score', 'advice_quality_score', 'word_count']
        available_metrics = [m for m in metrics if m in df.columns]
        
        if len(available_metrics) < 2:
            print("Warning: Not enough metrics for correlation analysis")
            return
        
        # Calculate correlation matrix
        corr_matrix = df[available_metrics].corr()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                   vmin=-1, vmax=1, ax=ax)
        
        ax.set_title('Correlation Matrix of Response Metrics', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Better labels
        labels = [l.replace('_', ' ').title() for l in available_metrics]
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels, rotation=0)
        
        plt.tight_layout()
        output_path = Path(output_dir) / 'correlation_heatmap.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
    
    def generate_all_visualizations(self, csv_path, gender_col='Gender', output_dir='visualizations'):
        """
        Generate all visualizations at once
        """
        print("Generating visualizations...")
        print("="*70)
        
        df = pd.read_csv(csv_path)
        
        # Auto-detect which column names to use
        # Try automated columns first, then manual columns
        possible_metrics = {
            'empathy': ['empathy_score', 'Empathy'],
            'sentiment': ['sentiment_score', 'Sentiment'],
            'advice': ['advice_quality_score', 'AdviceQuality'],
            'wordcount': ['word_count']
        }
        
        metrics = []
        for key, cols in possible_metrics.items():
            for col in cols:
                if col in df.columns:
                    metrics.append(col)
                    break
        
        if len(metrics) < 3:
            print("Warning: Not enough metrics found in CSV")
            print(f"Available columns: {', '.join(df.columns)}")
            return
        
        print(f"Using metrics: {', '.join(metrics)}")
        
        self.create_comparison_boxplots(df, metrics, gender_col, output_dir)
        self.create_mean_comparison_bars(df, metrics, gender_col, output_dir)
        self.create_difference_plot(df, gender_col, output_dir)
        self.create_gendered_language_plot(df, gender_col, output_dir)
        self.create_correlation_heatmap(df, output_dir)
        
        print("="*70)
        print(f"All visualizations saved to '{output_dir}/' folder")
        print("Files created:")
        print("  - comparison_boxplots.png")
        print("  - mean_comparison_bars.png")
        print("  - difference_comparison.png")
        print("  - gendered_language_analysis.png")
        print("  - correlation_heatmap.png")


# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_CSV = "scored_responses.csv"
OUTPUT_DIR = "visualizations"
GENDER_COLUMN = "Gender"

# ============================================================================
# RUN VISUALIZATION GENERATION
# ============================================================================

if __name__ == "__main__":
    visualizer = BiasVisualizer()
    
    try:
        visualizer.generate_all_visualizations(
            INPUT_CSV,
            gender_col=GENDER_COLUMN,
            output_dir=OUTPUT_DIR
        )
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()