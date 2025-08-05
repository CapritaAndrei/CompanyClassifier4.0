#!/usr/bin/env python3
"""
Single Label NAICS Classifier
=============================

Classify a single insurance label with multiple prompt strategies
until we get at least one valid NAICS mapping.
"""

import pandas as pd
import json
import requests
import time
import re
from typing import List, Dict, Any
from datetime import datetime

class SingleLabelClassifier:
    def __init__(self):
        """Initialize the classifier"""
        self.api_key = "sk-2ea639422e244745a823a88dded278ad"
        self.api_url = "https://api.deepseek.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Load NAICS validation data
        self.load_naics_validation_data()
        
    def load_naics_validation_data(self):
        """Load 2022 NAICS data for validation"""
        self.naics_codes = pd.read_excel('6-digit_2022_Codes.xlsx')
        self.naics_codes = self.naics_codes.dropna(subset=['2022 NAICS Code'])
        self.naics_codes['2022 NAICS Code'] = self.naics_codes['2022 NAICS Code'].astype(int).astype(str)
        
        self.valid_codes = set(self.naics_codes['2022 NAICS Code'].tolist())
        self.code_to_description = dict(zip(self.naics_codes['2022 NAICS Code'], self.naics_codes['2022 NAICS Title']))
        
        print(f"✅ Loaded {len(self.valid_codes)} valid 2022 NAICS codes")

    def create_prompts(self, label: str) -> List[Dict]:
        """Create multiple prompt strategies for the label"""
        
        prompts = [
            {
                "name": "Standard",
                "prompt": f"""**NAICS 2022 Business Code Expert**
Find NAICS codes that match "{label}".

Rules:
- Return only official NAICS 2022 codes and descriptions
- Match codes with ≥80% relevance
- Maximum 7 codes
- Output as JSON array only

Format:
[
  {{
    "naics_code": "6-digit code",
    "description": "Official NAICS description",
    "search_term": "{label}",
    "relevance": "high/medium"
  }}
]

Business activity: "{label}"
"""
            },
            {
                "name": "Flexible",
                "prompt": f"""**NAICS 2022 Expert**
Find the best NAICS codes for "{label}". Accept broader matches if needed.

Instructions:
- Find closest matching NAICS 2022 codes (≥70% relevance)
- Include broader category codes if specific ones don't exist
- Consider related business activities
- Maximum 7 codes

JSON output:
[
  {{
    "naics_code": "6-digit code",
    "description": "Official NAICS description",
    "search_term": "{label}",
    "relevance": "high/medium/moderate"
  }}
]

Business: "{label}"
"""
            },
            {
                "name": "Broad_Categories",
                "prompt": f"""**NAICS Classification Expert**
"{label}" needs NAICS codes. If no exact match exists, find the broader industry category it belongs to.

Task: Map "{label}" to NAICS codes
- Start with broad categories, then specific if available
- Include parent industry codes if needed
- Any relevance level acceptable
- Maximum 7 codes

Output as JSON:
[
  {{
    "naics_code": "6-digit code", 
    "description": "Official NAICS description",
    "search_term": "{label}",
    "relevance": "high/medium/low"
  }}
]

What industry category does "{label}" belong to?
"""
            },
            {
                "name": "Related_Services",
                "prompt": f"""**Business Classification Expert**
I need NAICS codes for "{label}". If this exact service doesn't have a code, what similar or related services would be closest?

Approach:
- Think about what industry sector this belongs to
- Consider related or similar business activities
- Include supporting services or broader categories
- Any reasonable match is acceptable

JSON format:
[
  {{
    "naics_code": "6-digit code",
    "description": "Official NAICS description", 
    "search_term": "{label}",
    "relevance": "any"
  }}
]

Related business activities for "{label}":
"""
            }
        ]
        
        return prompts

    def call_api(self, prompt: str) -> tuple:
        """Call DeepSeek API"""
        payload = {
            "model": "deepseek-reasoner",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 32000
        }
        
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=60)
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content'].strip()
                return content, None
            else:
                return None, f"API Error {response.status_code}"
        except Exception as e:
            return None, str(e)

    def validate_response(self, response_text: str, label: str) -> Dict:
        """Parse and validate API response"""
        try:
            # Extract JSON
            json_start = response_text.find('[')
            json_end = response_text.rfind(']') + 1
            
            if json_start == -1 or json_end == 0:
                return {'valid_mappings': [], 'error': 'No JSON found'}
            
            json_text = response_text[json_start:json_end]
            parsed_response = json.loads(json_text)
            
            if not isinstance(parsed_response, list):
                return {'valid_mappings': [], 'error': 'Not a JSON array'}
                
        except Exception as e:
            return {'valid_mappings': [], 'error': f'Parse error: {e}'}
        
        # Validate NAICS codes
        valid_mappings = []
        for mapping in parsed_response:
            naics_code = mapping.get('naics_code', '').strip()
            provided_description = mapping.get('description', '').strip()
            
            if not re.match(r'^\d{6}$', naics_code):
                continue
                
            if naics_code not in self.valid_codes:
                continue
            
            actual_description = self.code_to_description.get(naics_code, '')
            description_match = (actual_description.lower() in provided_description.lower() or 
                               provided_description.lower() in actual_description.lower())
            
            if description_match:
                valid_mappings.append({
                    'naics_code': naics_code,
                    'description': actual_description,
                    'provided_description': provided_description
                })
        
        return {'valid_mappings': valid_mappings, 'total_found': len(valid_mappings)}

    def classify_label(self, label: str):
        """Try multiple prompt strategies until we get valid mappings"""
        print(f"🎯 CLASSIFYING: {label}")
        print("=" * 50)
        
        prompts = self.create_prompts(label)
        
        for i, prompt_info in enumerate(prompts):
            print(f"\n📋 Strategy {i+1}: {prompt_info['name']}")
            print("-" * 30)
            
            response_text, error = self.call_api(prompt_info['prompt'])
            
            if error:
                print(f"   ❌ API Error: {error}")
                continue
            
            validation = self.validate_response(response_text, label)
            
            if 'error' in validation:
                print(f"   ❌ Parse Error: {validation['error']}")
                continue
            
            valid_count = len(validation['valid_mappings'])
            print(f"   📊 Valid mappings found: {valid_count}")
            
            if valid_count > 0:
                print(f"   🎉 SUCCESS! Found {valid_count} valid mappings:")
                for mapping in validation['valid_mappings']:
                    print(f"      • {mapping['naics_code']}: {mapping['description']}")
                
                # Save result
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                result_file = f"data/processed/single_label_result_{timestamp}.json"
                
                result = {
                    'label': label,
                    'strategy_used': prompt_info['name'],
                    'valid_mappings': validation['valid_mappings'],
                    'timestamp': timestamp
                }
                
                with open(result_file, 'w') as f:
                    json.dump(result, f, indent=2)
                
                print(f"   💾 Result saved to: {result_file}")
                return result
            else:
                print(f"   ⚠️  No valid mappings found")
            
            time.sleep(2)  # Rate limiting between attempts
        
        print(f"\n❌ All strategies failed for: {label}")
        return None

def main():
    """Main function"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python3 classify_single_label.py 'Label Name'")
        print("Example: python3 classify_single_label.py 'Publishing Services'")
        return
    
    label = sys.argv[1]
    classifier = SingleLabelClassifier()
    result = classifier.classify_label(label)
    
    if result:
        print(f"\n✅ Successfully classified: {label}")
    else:
        print(f"\n❌ Failed to classify: {label}")

if __name__ == "__main__":
    main() 