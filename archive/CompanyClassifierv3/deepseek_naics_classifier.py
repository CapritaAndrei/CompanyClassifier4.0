#!/usr/bin/env python3
"""
DeepSeek NAICS Classifier
=========================

Uses DeepSeek reasoner API to classify insurance labels to NAICS codes
with batching and validation against 2022 NAICS database.
"""

import pandas as pd
import json
import requests
import time
import re
from typing import List, Dict, Any
import os
from datetime import datetime

class DeepSeekNAICSClassifier:
    def __init__(self, api_key: str = None):
        """Initialize the classifier with DeepSeek API"""
        self.api_key = api_key or "sk-2ea639422e244745a823a88dded278ad"
        if not self.api_key:
            print("⚠️  Warning: No DeepSeek API key found.")
        
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
            # Load 2022 NAICS 6-digit codes (official titles)
            self.naics_codes = pd.read_excel('6-digit_2022_Codes.xlsx')
            self.naics_codes = self.naics_codes.dropna(subset=['2022 NAICS Code'])
            self.naics_codes['2022 NAICS Code'] = self.naics_codes['2022 NAICS Code'].astype(int).astype(str)
            
            # Create validation dictionaries
            self.valid_codes = set(self.naics_codes['2022 NAICS Code'].tolist())
            self.code_to_description = dict(zip(self.naics_codes['2022 NAICS Code'], self.naics_codes['2022 NAICS Title']))
            
            print(f"✅ Loaded {len(self.valid_codes)} valid 2022 NAICS codes for validation")
            
        except Exception as e:
            print(f"❌ Error loading NAICS validation data: {e}")
            self.valid_codes = set()
            self.code_to_description = {}
    
    def create_classification_prompt(self, label: str) -> str:
        """Create the specialized prompt for NAICS classification"""
        
        prompt = f"""**NAICS 2022 Business Code Expert**
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
        return prompt
    
    def call_deepseek_api(self, prompt: str, max_retries: int = 3) -> tuple:
        """Call DeepSeek API with retry logic - returns (response_text, error_message)"""
        
        payload = {
            "model": "deepseek-reasoner",
            "messages": [
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "temperature": 0.1,
            "max_tokens": 32000
        }
        
        last_error = None
        
        for attempt in range(max_retries):
            try:
                print(f"   🔄 API call attempt {attempt + 1}/{max_retries}...")
                response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=60)
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"   ✅ API call successful")
                    
                    # Extract content from response
                    try:
                        content = result['choices'][0]['message']['content'].strip()
                        if len(content) == 0:
                            return None, "Empty content returned from API"
                        return content, None
                    except KeyError as e:
                        return None, f"Missing key in response: {e}"
                    except IndexError:
                        return None, "No choices in response"
                else:
                    error_msg = f"API Error {response.status_code}: {response.text}"
                    print(f"   ⚠️  {error_msg}")
                    last_error = error_msg
                    
            except requests.exceptions.Timeout:
                error_msg = f"Timeout on attempt {attempt + 1} (60s limit)"
                print(f"   ⏰ {error_msg}")
                last_error = error_msg
            except requests.exceptions.ConnectionError:
                error_msg = f"Connection error on attempt {attempt + 1}"
                print(f"   🔌 {error_msg}")
                last_error = error_msg
            except Exception as e:
                error_msg = f"API call attempt {attempt + 1} failed: {e}"
                print(f"   ❌ {error_msg}")
                last_error = error_msg
                
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"   ⏸️  Waiting {wait_time}s before retry...")
                time.sleep(wait_time)  # Exponential backoff
        
        print(f"   ❌ All {max_retries} API attempts failed")
        return None, last_error or "All API attempts failed"
    
    def validate_naics_response(self, response: List[Dict], label: str) -> Dict:
        """Validate DeepSeek response against our NAICS database"""
        
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
            
            # Validate NAICS code format
            if not re.match(r'^\d{6}$', naics_code):
                validation_result['invalid_mappings'].append({
                    **mapping,
                    'error': f'Invalid NAICS code format: {naics_code}'
                })
                validation_result['validation_summary']['invalid_codes'] += 1
                continue
            
            # Check if code exists in our database
            if naics_code not in self.valid_codes:
                validation_result['invalid_mappings'].append({
                    **mapping,
                    'error': f'NAICS code {naics_code} not found in 2022 database'
                })
                validation_result['validation_summary']['invalid_codes'] += 1
                continue
            
            # Check description accuracy
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
    
    def classify_single_label(self, label: str) -> Dict:
        """Classify a single insurance label"""
        
        print(f"🔍 Classifying: {label}")
        
        # Create prompt
        prompt = self.create_classification_prompt(label)
        
        # Call API
        response_text, api_error = self.call_deepseek_api(prompt)
        
        if not response_text:
            return {
                'label': label,
                'status': 'api_error',
                'raw_response': None,
                'parsed_response': None,
                'validation': None,
                'error': api_error
            }
        
        # Parse JSON response
        try:
            # Extract JSON from response (in case there's extra text)
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
                'status': 'parse_error',
                'raw_response': response_text,
                'parsed_response': None,
                'validation': None,
                'error': str(e)
            }
        
        # Validate response
        validation = self.validate_naics_response(parsed_response, label)
        
        result = {
            'label': label,
            'status': 'success',
            'raw_response': response_text,
            'parsed_response': parsed_response,
            'validation': validation
        }
        
        # Print validation summary
        val_summary = validation['validation_summary']
        print(f"   ✅ Valid mappings: {val_summary['valid_codes']}")
        print(f"   ❌ Invalid mappings: {val_summary['invalid_codes']}")
        if val_summary['hallucinated_descriptions'] > 0:
            print(f"   🚨 Hallucinated descriptions: {val_summary['hallucinated_descriptions']}")
        
        return result
    
    def classify_batch(self, labels: List[str], batch_size: int = 10) -> List[Dict]:
        """Classify a batch of labels with rate limiting"""
        
        results = []
        total_labels = len(labels)
        
        print(f"🚀 Starting batch classification of {total_labels} labels")
        print(f"📦 Batch size: {batch_size}")
        print()
        
        for i in range(0, total_labels, batch_size):
            batch = labels[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total_labels + batch_size - 1) // batch_size
            
            print(f"📦 Processing Batch {batch_num}/{total_batches} ({len(batch)} labels)")
            print("-" * 50)
            
            for j, label in enumerate(batch):
                print(f"[{i + j + 1:3d}/{total_labels}] ", end="")
                result = self.classify_single_label(label)
                results.append(result)
                
                # Rate limiting - small delay between calls
                time.sleep(1)
            
            # Longer delay between batches
            if i + batch_size < total_labels:
                print(f"\n⏸️  Batch complete. Waiting 5 seconds before next batch...\n")
                time.sleep(5)
        
        return results
    
    def save_results(self, results: List[Dict], filename_prefix: str = "deepseek_naics_classification"):
        """Save classification results"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save raw results
        raw_filename = f"data/processed/{filename_prefix}_raw_{timestamp}.json"
        with open(raw_filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create summary of valid mappings
        valid_mappings = {}
        for result in results:
            if result['status'] == 'success' and result['validation']:
                label = result['label']
                valid_maps = result['validation']['valid_mappings']
                if valid_maps:
                    valid_mappings[label] = [
                        {
                            'naics_code': m['naics_code'],
                            'description': m['actual_description'],
                            'provided_description': m['description']
                        }
                        for m in valid_maps
                    ]
        
        summary_filename = f"data/processed/{filename_prefix}_valid_mappings_{timestamp}.json"
        with open(summary_filename, 'w') as f:
            json.dump(valid_mappings, f, indent=2)
        
        print(f"\n💾 Results saved:")
        print(f"   • {raw_filename} (complete results)")
        print(f"   • {summary_filename} (valid mappings only)")
        
        return raw_filename, summary_filename

def test_single_label():
    """Test the classifier with a single label"""
    
    print("🧪 TESTING DEEPSEEK NAICS CLASSIFIER")
    print("=" * 50)
    
    # Initialize classifier
    classifier = DeepSeekNAICSClassifier()
    
    # Test with a single label
    test_label = "Digital Marketing Services"
    print(f"🧪 Testing with label: {test_label}")
    
    result = classifier.classify_single_label(test_label)
    
    print(f"\n📊 TEST RESULTS:")
    print(f"   Status: {result['status']}")
    
    if result['status'] == 'success':
        validation = result['validation']
        print(f"   Valid mappings: {len(validation['valid_mappings'])}")
        print(f"   Invalid mappings: {len(validation['invalid_mappings'])}")
        
        if validation['valid_mappings']:
            print(f"\n✅ VALID MAPPINGS:")
            for mapping in validation['valid_mappings']:
                print(f"      {mapping['naics_code']}: {mapping['actual_description']}")
        
        if validation['invalid_mappings']:
            print(f"\n❌ INVALID MAPPINGS:")
            for mapping in validation['invalid_mappings']:
                print(f"      {mapping.get('naics_code', 'N/A')}: {mapping.get('error', 'Unknown error')}")
    
    else:
        print(f"   Error: {result.get('error', 'Unknown error')}")
    
    return result

def run_full_classification():
    """Run classification on all 220 insurance labels"""
    
    print("🚀 RUNNING FULL NAICS CLASSIFICATION")
    print("=" * 50)
    
    # Load all insurance labels
    insurance_df = pd.read_csv('data/input/insurance_taxonomy - insurance_taxonomy.csv')
    all_labels = insurance_df['label'].tolist()
    
    print(f"📊 Total labels to classify: {len(all_labels)}")
    
    # Initialize classifier
    classifier = DeepSeekNAICSClassifier()
    
    # Run batch classification
    results = classifier.classify_batch(all_labels, batch_size=10)
    
    # Save results
    raw_file, summary_file = classifier.save_results(results)
    
    # Print final summary
    successful = sum(1 for r in results if r['status'] == 'success')
    total_valid_mappings = sum(len(r['validation']['valid_mappings']) for r in results if r['status'] == 'success' and r['validation'])
    
    print(f"\n🎯 FINAL SUMMARY:")
    print(f"   • Labels processed: {len(results)}")
    print(f"   • Successful API calls: {successful}")
    print(f"   • Total valid mappings found: {total_valid_mappings}")
    print(f"   • Average mappings per label: {total_valid_mappings / successful:.1f}")
    
    return results

def main():
    """Main function"""
    
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        # Test mode
        test_single_label()
    elif len(sys.argv) > 1 and sys.argv[1] == 'full':
        # Full classification mode
        run_full_classification()
    else:
        print("Usage:")
        print("  python3 deepseek_naics_classifier.py test    # Test with single label")
        print("  python3 deepseek_naics_classifier.py full    # Run full classification")

if __name__ == "__main__":
    main() 