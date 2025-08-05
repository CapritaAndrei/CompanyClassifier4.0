#!/usr/bin/env python3
"""
Retry Failed NAICS Classifications
==================================

Re-processes insurance labels that didn't get any valid NAICS mappings
from the initial classification run.
"""

import pandas as pd
import json
import requests
import time
import re
from typing import List, Dict, Any
import os
from datetime import datetime
import glob

class NAICSRetryClassifier:
    def __init__(self, api_key: str = None):
        """Initialize the retry classifier"""
        self.api_key = api_key or "sk-2ea639422e244745a823a88dded278ad"
        self.api_url = "https://api.deepseek.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Load NAICS validation data
        self.load_naics_validation_data()
        
    def load_naics_validation_data(self):
        """Load 2022 NAICS data for validation"""
        try:
            self.naics_codes = pd.read_excel('6-digit_2022_Codes.xlsx')
            self.naics_codes = self.naics_codes.dropna(subset=['2022 NAICS Code'])
            self.naics_codes['2022 NAICS Code'] = self.naics_codes['2022 NAICS Code'].astype(int).astype(str)
            
            self.valid_codes = set(self.naics_codes['2022 NAICS Code'].tolist())
            self.code_to_description = dict(zip(self.naics_codes['2022 NAICS Code'], self.naics_codes['2022 NAICS Title']))
            
            print(f"✅ Loaded {len(self.valid_codes)} valid 2022 NAICS codes for validation")
            
        except Exception as e:
            print(f"❌ Error loading NAICS validation data: {e}")
            self.valid_codes = set()
            self.code_to_description = {}

    def find_latest_results(self):
        """Find the most recent classification results"""
        pattern = "data/processed/deepseek_naics_classification_raw_*.json"
        files = glob.glob(pattern)
        
        if not files:
            print("❌ No previous classification results found!")
            return None
            
        latest_file = max(files, key=os.path.getctime)
        print(f"📂 Found latest results: {latest_file}")
        return latest_file

    def load_previous_results(self, results_file: str):
        """Load and analyze previous classification results"""
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            print(f"📊 Loaded {len(results)} previous classification results")
            
            # Analyze results
            failed_labels = []
            successful_labels = []
            
            for result in results:
                label = result['label']
                if result['status'] == 'success' and result['validation']:
                    valid_count = len(result['validation']['valid_mappings'])
                    if valid_count == 0:
                        failed_labels.append(label)
                    else:
                        successful_labels.append(label)
                else:
                    failed_labels.append(label)
            
            print(f"✅ Successful labels: {len(successful_labels)}")
            print(f"❌ Failed labels (need retry): {len(failed_labels)}")
            
            return failed_labels
            
        except Exception as e:
            print(f"❌ Error loading results: {e}")
            return None

    def create_standard_prompt(self, label: str) -> str:
        """Create the standard classification prompt (same as original)"""
        return f"""**NAICS 2022 Business Code Expert**
You are an expert in NAICS 2022 business coding system. Help me find the appropriate NAICS codes for this business activity.

**Task:** Find NAICS codes that match "{label}"

**Rules:**
- Return only official NAICS 2022 codes and descriptions
- Match codes with ≥80% relevance to the label
- Maximum 7 codes per response
- Output as JSON array only

**Required JSON format:**
[
  {{
    "naics_code": "6-digit code",
    "description": "Official NAICS description",
    "search_term": "{label}",
    "relevance": "high/medium"
  }}
]

**Business activity to code:** "{label}"
"""

    def create_flexible_prompt(self, label: str) -> str:
        """Create a more flexible prompt for difficult labels"""
        return f"""**NAICS 2022 Business Expert**
I need to find NAICS codes for "{label}". This might be a specialized or niche business activity.

**Instructions:**
- Find the closest matching NAICS 2022 codes, even if not perfect matches
- Include broader category codes if specific ones don't exist
- Consider related or similar business activities
- Accept ≥70% relevance (lower threshold)
- Maximum 7 codes

**Output as JSON only:**
[
  {{
    "naics_code": "6-digit code",
    "description": "Official NAICS description",
    "search_term": "{label}",
    "relevance": "high/medium/moderate"
  }}
]

**Business to code:** "{label}"
"""

    def call_deepseek_api(self, prompt: str, max_retries: int = 3) -> tuple:
        """Call DeepSeek API with retry logic"""
        payload = {
            "model": "deepseek-reasoner",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 32000
        }
        
        for attempt in range(max_retries):
            try:
                print(f"   🔄 API call attempt {attempt + 1}/{max_retries}...")
                response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=60)
                
                if response.status_code == 200:
                    result = response.json()
                    content = result['choices'][0]['message']['content'].strip()
                    if len(content) == 0:
                        return None, "Empty content returned from API"
                    return content, None
                else:
                    error_msg = f"API Error {response.status_code}: {response.text}"
                    print(f"   ⚠️  {error_msg}")
                    
            except Exception as e:
                error_msg = f"API call failed: {e}"
                print(f"   ❌ {error_msg}")
                
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return None, "All API attempts failed"

    def validate_naics_response(self, response: List[Dict], label: str) -> Dict:
        """Validate DeepSeek response against NAICS database"""
        validation_result = {
            'valid_mappings': [],
            'invalid_mappings': [],
            'validation_summary': {
                'total_provided': len(response),
                'valid_codes': 0,
                'invalid_codes': 0,
                'hallucinated_descriptions': 0
            }
        }
        
        for mapping in response:
            naics_code = mapping.get('naics_code', '').strip()
            provided_description = mapping.get('description', '').strip()
            
            if not re.match(r'^\d{6}$', naics_code):
                validation_result['invalid_mappings'].append({
                    **mapping,
                    'error': f'Invalid NAICS code format: {naics_code}'
                })
                validation_result['validation_summary']['invalid_codes'] += 1
                continue
            
            if naics_code not in self.valid_codes:
                validation_result['invalid_mappings'].append({
                    **mapping,
                    'error': f'NAICS code {naics_code} not found in 2022 database'
                })
                validation_result['validation_summary']['invalid_codes'] += 1
                continue
            
            actual_description = self.code_to_description.get(naics_code, '')
            description_match = actual_description.lower() in provided_description.lower() or provided_description.lower() in actual_description.lower()
            
            valid_mapping = {
                **mapping,
                'actual_description': actual_description,
                'description_match': description_match
            }
            
            if description_match:
                validation_result['valid_mappings'].append(valid_mapping)
                validation_result['validation_summary']['valid_codes'] += 1
            else:
                validation_result['invalid_mappings'].append({
                    **valid_mapping,
                    'error': f'Description mismatch. Provided: "{provided_description}" | Actual: "{actual_description}"'
                })
                validation_result['validation_summary']['hallucinated_descriptions'] += 1
        
        return validation_result

    def retry_single_label(self, label: str, use_flexible: bool = False) -> Dict:
        """Retry classification for a single label"""
        prompt_type = "flexible" if use_flexible else "standard"
        print(f"🔍 Retrying ({prompt_type}): {label}")
        
        # Create appropriate prompt
        if use_flexible:
            prompt = self.create_flexible_prompt(label)
        else:
            prompt = self.create_standard_prompt(label)
        
        # Call API
        response_text, api_error = self.call_deepseek_api(prompt)
        
        if not response_text:
            return {
                'label': label,
                'retry_type': prompt_type,
                'status': 'api_error',
                'error': api_error
            }
        
        # Parse JSON response
        try:
            json_start = response_text.find('[')
            json_end = response_text.rfind(']') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON array found in response")
            
            json_text = response_text[json_start:json_end]
            parsed_response = json.loads(json_text)
            
            if not isinstance(parsed_response, list):
                raise ValueError("Response is not a JSON array")
                
        except Exception as e:
            return {
                'label': label,
                'retry_type': prompt_type,
                'status': 'parse_error',
                'error': str(e)
            }
        
        # Validate response
        validation = self.validate_naics_response(parsed_response, label)
        
        result = {
            'label': label,
            'retry_type': prompt_type,
            'status': 'success',
            'validation': validation
        }
        
        # Print validation summary
        val_summary = validation['validation_summary']
        print(f"   ✅ Valid mappings: {val_summary['valid_codes']}")
        print(f"   ❌ Invalid mappings: {val_summary['invalid_codes']}")
        
        return result

    def retry_failed_labels(self, failed_labels: List[str]) -> List[Dict]:
        """Retry classification for all failed labels"""
        print(f"\n🚀 RETRYING {len(failed_labels)} FAILED LABELS")
        print("=" * 50)
        
        results = []
        still_failed = []
        
        # First attempt: Use standard prompt
        print(f"\n📋 PHASE 1: Standard prompt retry")
        print("-" * 30)
        
        for i, label in enumerate(failed_labels):
            print(f"[{i+1:2d}/{len(failed_labels)}] ", end="")
            result = self.retry_single_label(label, use_flexible=False)
            results.append(result)
            
            # Check if still failed
            if (result['status'] != 'success' or 
                not result.get('validation') or 
                len(result['validation']['valid_mappings']) == 0):
                still_failed.append(label)
            
            time.sleep(1)  # Rate limiting
        
        # Second attempt: Use flexible prompt for remaining failures
        if still_failed:
            print(f"\n📋 PHASE 2: Flexible prompt retry ({len(still_failed)} labels)")
            print("-" * 30)
            
            for i, label in enumerate(still_failed):
                print(f"[{i+1:2d}/{len(still_failed)}] ", end="")
                result = self.retry_single_label(label, use_flexible=True)
                results.append(result)
                time.sleep(1)  # Rate limiting
        
        return results

    def save_retry_results(self, results: List[Dict], original_file: str):
        """Save retry results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save retry results
        retry_filename = f"data/processed/naics_retry_results_{timestamp}.json"
        with open(retry_filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create summary of successful retries
        successful_retries = {}
        for result in results:
            if result['status'] == 'success' and result.get('validation'):
                valid_maps = result['validation']['valid_mappings']
                if valid_maps:
                    successful_retries[result['label']] = [
                        {
                            'naics_code': m['naics_code'],
                            'description': m['actual_description'],
                            'retry_type': result['retry_type']
                        }
                        for m in valid_maps
                    ]
        
        success_filename = f"data/processed/successful_retries_{timestamp}.json"
        with open(success_filename, 'w') as f:
            json.dump(successful_retries, f, indent=2)
        
        print(f"\n💾 Retry results saved:")
        print(f"   • {retry_filename} (complete retry results)")
        print(f"   • {success_filename} (successful retries only)")
        
        # Print summary
        total_retries = len(results)
        successful = len(successful_retries)
        print(f"\n🎯 RETRY SUMMARY:")
        print(f"   • Labels retried: {total_retries}")
        print(f"   • Now successful: {successful}")
        print(f"   • Still failed: {total_retries - successful}")
        
        return retry_filename, success_filename

def main():
    """Main function"""
    print("🔄 NAICS CLASSIFICATION RETRY SYSTEM")
    print("=" * 50)
    
    # Initialize classifier
    classifier = NAICSRetryClassifier()
    
    # Find latest results
    latest_file = classifier.find_latest_results()
    if not latest_file:
        print("❌ No previous results found. Run the main classification first.")
        return
    
    # Load previous results and identify failures
    failed_labels = classifier.load_previous_results(latest_file)
    if not failed_labels:
        print("❌ No failed labels found or error loading results.")
        return
    
    if len(failed_labels) == 0:
        print("🎉 No failed labels! All labels already have valid NAICS mappings.")
        return
    
    # Retry failed labels
    retry_results = classifier.retry_failed_labels(failed_labels)
    
    # Save results
    classifier.save_retry_results(retry_results, latest_file)

if __name__ == "__main__":
    main() 