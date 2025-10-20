"""
Qualitative Example Extractor for Gender Bias Analysis
Finds and extracts the most biased responses for case study analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path

class QualitativeExtractor:
    def __init__(self):
        pass
    
    def find_matched_pairs(self, df, prompt_col='question', gender_col='Gender', 
                          response_col='response'):
        """
        Find male-female response pairs for the same prompt
        """
        # Group by prompt
        grouped = df.groupby(prompt_col)
        
        matched_pairs = []
        
        for prompt, group in grouped:
            genders = group[gender_col].unique()
            
            # Check if we have both male and female responses
            if len(genders) >= 2 and 'Male' in genders and 'Female' in genders:
                male_response = group[group[gender_col] == 'Male'].iloc[0]
                female_response = group[group[gender_col] == 'Female'].iloc[0]
                
                matched_pairs.append({
                    'prompt': prompt,
                    'male_response': male_response[response_col],
                    'female_response': female_response[response_col],
                    'male_empathy': male_response.get('empathy_score', male_response.get('Empathy', 0)),
                    'female_empathy': female_response.get('empathy_score', female_response.get('Empathy', 0)),
                    'male_advice': male_response.get('advice_quality_score', male_response.get('AdviceQuality', 0)),
                    'female_advice': female_response.get('advice_quality_score', female_response.get('AdviceQuality', 0)),
                    'male_gendered': male_response.get('gendered_language_flag', male_response.get('GenderedLang', 0)),
                    'female_gendered': female_response.get('gendered_language_flag', female_response.get('GenderedLang', 0)),
                    'male_wordcount': male_response.get('word_count', len(str(male_response[response_col]).split())),
                    'female_wordcount': female_response.get('word_count', len(str(female_response[response_col]).split())),
                    'empathy_diff': male_response.get('empathy_score', male_response.get('Empathy', 0)) - 
                                   female_response.get('empathy_score', female_response.get('Empathy', 0)),
                    'advice_diff': male_response.get('advice_quality_score', male_response.get('AdviceQuality', 0)) - 
                                  female_response.get('advice_quality_score', female_response.get('AdviceQuality', 0)),
                    'wordcount_diff': male_response.get('word_count', len(str(male_response[response_col]).split())) - 
                                     female_response.get('word_count', len(str(female_response[response_col]).split()))
                })
        
        return pd.DataFrame(matched_pairs)
    
    def find_most_biased_examples(self, df, response_col='response', gender_col='Gender', n=5):
        """
        Find the most biased responses across different dimensions
        """
        examples = {
            'highest_gendered_language_female': [],
            'lowest_advice_quality_female': [],
            'highest_advice_quality_male': [],
            'shortest_responses_female': [],
            'longest_responses_male': [],
            'patronizing_language_female': []
        }
        
        # Get gendered language column name
        gendered_col = 'gendered_language_flag' if 'gendered_language_flag' in df.columns else 'GenderedLang'
        advice_col = 'advice_quality_score' if 'advice_quality_score' in df.columns else 'AdviceQuality'
        wordcount_col = 'word_count' if 'word_count' in df.columns else None
        prompt_col = 'question' if 'question' in df.columns else 'prompt'
        
        # 1. Highest gendered language in female responses
        female_df = df[df[gender_col] == 'Female']
        if gendered_col in female_df.columns:
            gendered_female = female_df[female_df[gendered_col] == 1].sort_values(
                by=advice_col if advice_col in female_df.columns else gendered_col, 
                ascending=False
            ).head(n)
            
            for _, row in gendered_female.iterrows():
                examples['highest_gendered_language_female'].append({
                    'prompt': row.get(prompt_col, 'N/A'),
                    'response': row[response_col],
                    'gendered_terms': row.get('gendered_terms', 'N/A'),
                    'advice_quality': row.get(advice_col, 'N/A')
                })
        
        # 2. Lowest advice quality for females
        if advice_col in female_df.columns:
            low_advice_female = female_df.nsmallest(n, advice_col)
            
            for _, row in low_advice_female.iterrows():
                examples['lowest_advice_quality_female'].append({
                    'prompt': row.get(prompt_col, 'N/A'),
                    'response': row[response_col],
                    'advice_quality': row[advice_col],
                    'word_count': row.get(wordcount_col, len(str(row[response_col]).split()))
                })
        
        # 3. Highest advice quality for males
        male_df = df[df[gender_col] == 'Male']
        if advice_col in male_df.columns:
            high_advice_male = male_df.nlargest(n, advice_col)
            
            for _, row in high_advice_male.iterrows():
                examples['highest_advice_quality_male'].append({
                    'prompt': row.get(prompt_col, 'N/A'),
                    'response': row[response_col],
                    'advice_quality': row[advice_col],
                    'word_count': row.get(wordcount_col, len(str(row[response_col]).split()))
                })
        
        # 4. Shortest responses for females
        if wordcount_col:
            shortest_female = female_df.nsmallest(n, wordcount_col)
        else:
            female_df_copy = female_df.copy()
            female_df_copy['temp_wc'] = female_df_copy[response_col].apply(lambda x: len(str(x).split()))
            shortest_female = female_df_copy.nsmallest(n, 'temp_wc')
        
        for _, row in shortest_female.iterrows():
            examples['shortest_responses_female'].append({
                'prompt': row.get(prompt_col, 'N/A'),
                'response': row[response_col],
                'word_count': row.get(wordcount_col, len(str(row[response_col]).split()))
            })
        
        # 5. Longest responses for males
        if wordcount_col:
            longest_male = male_df.nlargest(n, wordcount_col)
        else:
            male_df_copy = male_df.copy()
            male_df_copy['temp_wc'] = male_df_copy[response_col].apply(lambda x: len(str(x).split()))
            longest_male = male_df_copy.nlargest(n, 'temp_wc')
        
        for _, row in longest_male.iterrows():
            examples['longest_responses_male'].append({
                'prompt': row.get(prompt_col, 'N/A'),
                'response': row[response_col],
                'word_count': row.get(wordcount_col, len(str(row[response_col]).split()))
            })
        
        # 6. Patronizing language patterns (female responses with specific terms)
        patronizing_terms = ['dear', 'honey', 'sweetie', 'darling', "don't worry", 
                            'everything will be fine', 'you\'ll be fine']
        
        for _, row in female_df.iterrows():
            response_lower = str(row[response_col]).lower()
            found_terms = [term for term in patronizing_terms if term in response_lower]
            
            if found_terms:
                examples['patronizing_language_female'].append({
                    'prompt': row.get(prompt_col, 'N/A'),
                    'response': row[response_col],
                    'patronizing_terms': ', '.join(found_terms),
                    'advice_quality': row.get(advice_col, 'N/A')
                })
        
        # Limit patronizing examples to top n
        examples['patronizing_language_female'] = examples['patronizing_language_female'][:n]
        
        return examples
    
    def generate_comparison_table(self, matched_pairs_df, n=10, sort_by='advice_diff'):
        """
        Generate side-by-side comparison table for matched pairs
        """
        # Sort by the specified metric
        if sort_by in matched_pairs_df.columns:
            sorted_pairs = matched_pairs_df.sort_values(by=sort_by, ascending=False).head(n)
        else:
            sorted_pairs = matched_pairs_df.head(n)
        
        return sorted_pairs
    
    def save_examples_to_file(self, examples, matched_pairs, output_dir='qualitative_examples'):
        """
        Save examples to readable text files
        """
        Path(output_dir).mkdir(exist_ok=True)
        
        # Save individual biased examples
        with open(Path(output_dir) / 'biased_examples.txt', 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("QUALITATIVE EXAMPLES OF GENDER BIAS IN MH CHATBOT RESPONSES\n")
            f.write("="*80 + "\n\n")
            
            for category, items in examples.items():
                if not items:
                    continue
                    
                f.write("\n" + "="*80 + "\n")
                f.write(f"{category.replace('_', ' ').upper()}\n")
                f.write("="*80 + "\n\n")
                
                for i, item in enumerate(items, 1):
                    f.write(f"Example {i}:\n")
                    f.write("-" * 80 + "\n")
                    f.write(f"Prompt: {item['prompt']}\n\n")
                    f.write(f"Response: {item['response']}\n\n")
                    
                    for key, value in item.items():
                        if key not in ['prompt', 'response']:
                            f.write(f"{key.replace('_', ' ').title()}: {value}\n")
                    
                    f.write("\n")
        
        print(f"✓ Saved biased examples to: {output_dir}/biased_examples.txt")
        
        # Save matched pairs comparison
        if matched_pairs is not None and len(matched_pairs) > 0:
            with open(Path(output_dir) / 'matched_pairs_comparison.txt', 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("SIDE-BY-SIDE COMPARISON: MALE vs FEMALE RESPONSES TO SAME PROMPTS\n")
                f.write("="*80 + "\n\n")
                
                # Sort by advice quality difference
                sorted_pairs = matched_pairs.sort_values(by='advice_diff', ascending=False).head(10)
                
                for i, (_, row) in enumerate(sorted_pairs.iterrows(), 1):
                    f.write(f"\n{'='*80}\n")
                    f.write(f"COMPARISON {i}\n")
                    f.write(f"{'='*80}\n\n")
                    f.write(f"PROMPT: {row['prompt']}\n\n")
                    
                    f.write("-" * 80 + "\n")
                    f.write("MALE RESPONSE:\n")
                    f.write("-" * 80 + "\n")
                    f.write(f"{row['male_response']}\n\n")
                    f.write(f"Empathy: {row['male_empathy']:.2f} | ")
                    f.write(f"Advice Quality: {row['male_advice']:.2f} | ")
                    f.write(f"Word Count: {row['male_wordcount']} | ")
                    f.write(f"Gendered Language: {'Yes' if row['male_gendered'] else 'No'}\n\n")
                    
                    f.write("-" * 80 + "\n")
                    f.write("FEMALE RESPONSE:\n")
                    f.write("-" * 80 + "\n")
                    f.write(f"{row['female_response']}\n\n")
                    f.write(f"Empathy: {row['female_empathy']:.2f} | ")
                    f.write(f"Advice Quality: {row['female_advice']:.2f} | ")
                    f.write(f"Word Count: {row['female_wordcount']} | ")
                    f.write(f"Gendered Language: {'Yes' if row['female_gendered'] else 'No'}\n\n")
                    
                    f.write("-" * 80 + "\n")
                    f.write("DIFFERENCES (Male - Female):\n")
                    f.write("-" * 80 + "\n")
                    f.write(f"Empathy Difference: {row['empathy_diff']:+.2f}\n")
                    f.write(f"Advice Quality Difference: {row['advice_diff']:+.2f}\n")
                    f.write(f"Word Count Difference: {row['wordcount_diff']:+d}\n\n")
            
            print(f"✓ Saved matched pairs comparison to: {output_dir}/matched_pairs_comparison.txt")
            
            # Save CSV for further analysis
            matched_pairs.to_csv(Path(output_dir) / 'matched_pairs.csv', index=False)
            print(f"✓ Saved matched pairs CSV to: {output_dir}/matched_pairs.csv")
    
    def extract_all(self, csv_path, response_col='response', gender_col='Gender', 
                   prompt_col='question', output_dir='qualitative_examples'):
        """
        Run complete qualitative extraction
        """
        print("Extracting qualitative examples...")
        print("="*70)
        
        df = pd.read_csv(csv_path)
        
        # Find matched pairs
        print("\nFinding matched prompt pairs...")
        matched_pairs = self.find_matched_pairs(df, prompt_col, gender_col, response_col)
        print(f"Found {len(matched_pairs)} matched pairs")
        
        # Find most biased examples
        print("\nExtracting most biased examples...")
        examples = self.find_most_biased_examples(df, response_col, gender_col, n=5)
        
        print(f"Found {len(examples['highest_gendered_language_female'])} gendered language examples")
        print(f"Found {len(examples['patronizing_language_female'])} patronizing language examples")
        print(f"Found {len(examples['lowest_advice_quality_female'])} low advice quality examples")
        
        # Save everything
        print("\nSaving examples...")
        self.save_examples_to_file(examples, matched_pairs, output_dir)
        
        print("\n" + "="*70)
        print("Qualitative extraction complete!")
        print(f"Check the '{output_dir}/' folder for:")
        print("  - biased_examples.txt (categorized examples)")
        print("  - matched_pairs_comparison.txt (side-by-side comparisons)")
        print("  - matched_pairs.csv (data for further analysis)")


# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_CSV = "scored_responses.csv"  # or "all_responses_clean.csv"
OUTPUT_DIR = "qualitative_examples"
RESPONSE_COLUMN = "response"  # or "Response_Clean"
GENDER_COLUMN = "Gender"
PROMPT_COLUMN = "question"  # or "prompt"

# ============================================================================
# RUN EXTRACTION
# ============================================================================

if __name__ == "__main__":
    extractor = QualitativeExtractor()
    
    try:
        extractor.extract_all(
            INPUT_CSV,
            response_col=RESPONSE_COLUMN,
            gender_col=GENDER_COLUMN,
            prompt_col=PROMPT_COLUMN,
            output_dir=OUTPUT_DIR
        )
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()