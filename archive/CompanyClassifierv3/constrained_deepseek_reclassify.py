#!/usr/bin/env python3
"""
Constrained DeepSeek Re-classification: Force selection from 220 taxonomy labels only
"""

import pandas as pd
import json
import requests
import time
from typing import List, Dict, Optional
import sys


class ConstrainedDeepSeekClassifier:
    """DeepSeek classifier that ONLY picks from the 220 taxonomy labels"""
    
    def __init__(self, api_key: str = "sk-e15ca800c3ba4473860c3697f84052ba"):
        self.api_key = api_key
        self.api_url = "https://api.deepseek.com/v1/chat/completions"
        self.taxonomy_labels = self.load_taxonomy()
        self.taxonomy_labels_lower = {label.lower(): label for label in self.taxonomy_labels}
        
    def load_taxonomy(self) -> List[str]:
        """Load the 220 taxonomy labels"""
        taxonomy_df = pd.read_csv('data/input/insurance_taxonomy - insurance_taxonomy.csv')
        return list(taxonomy_df['label'].str.strip())
    
    def create_classification_prompt(self, company_description: str, business_tags: List[str], 
                                   sector: str = None, category: str = None, niche: str = None) -> str:
        """Create a constrained prompt that forces selection from taxonomy"""
        
        # Format business tags
        tags_text = ', '.join(business_tags) if business_tags else "None provided"
        
        # Add sector/category/niche information
        additional_context = []
        if sector:
            additional_context.append(f"Sector: {sector}")
        if category:
            additional_context.append(f"Category: {category}")
        if niche:
            additional_context.append(f"Niche: {niche}")
        
        context_text = '\n'.join(additional_context) if additional_context else "Additional context: Not provided"
        
        # Create the taxonomy list for the prompt
        taxonomy_text = '\n'.join([f"{i+1}. {label}" for i, label in enumerate(self.taxonomy_labels)])
        
        prompt = f"""You are an industry classification expert. You must classify a company by selecting ONLY from the provided taxonomy list.

COMPANY TO CLASSIFY:
Description: {company_description}
Business Tags: {tags_text}
{context_text}

TAXONOMY LIST (CHOOSE ONLY FROM THESE 220 OPTIONS):
{taxonomy_text}

CRITICAL REQUIREMENTS:
1. You MUST select labels ONLY from the numbered taxonomy list above
2. Use the EXACT label names as they appear (do not modify spelling, capitalization, or wording)
3. Select 1-3 labels that best represent the company's primary business activities
4. DO NOT create new categories, modify existing ones, or reference numbers/brackets

CLASSIFICATION GUIDELINES:
- If a company MANUFACTURES/PRODUCES products → Choose "Manufacturing" labels
- If a company INSTALLS/SERVICES equipment → Choose "Installation/Services" labels  
- If a company DISTRIBUTES/WHOLESALES → Choose "Distribution/Wholesale" labels
- For specialized products (bike seats, optical tools, etc.) → Consider "Accessory Manufacturing" or relevant manufacturing category
- For testing/quality services → "Testing and Inspection Services" can be secondary
- Match the company's PRIMARY business activity, not just any activity they do

CLASSIFICATION TASK:
1. Analyze what this company actually does as their main business (manufacture, install, distribute, service?)
2. Match their activities to the most appropriate taxonomy labels
3. Prioritize manufacturing labels for companies that make products
4. Select the labels by their EXACT text (not numbers)

REQUIRED OUTPUT FORMAT (follow exactly):
SELECTED_LABELS: Label Name 1, Label Name 2, Label Name 3
CONFIDENCE: High/Medium/Low

EXAMPLE:
SELECTED_LABELS: Software Development Services, Information Technology Consulting Services
CONFIDENCE: High

Now classify this company using the EXACT label names from the taxonomy list:"""

        return prompt
    
    def create_correction_prompt(self, original_prompt: str, invalid_response: str, 
                               invalid_labels: List[str]) -> str:
        """Create a correction prompt when labels are invalid"""
        
        invalid_labels_text = ', '.join([f"'{label}'" for label in invalid_labels])
        
        correction_prompt = f"""{original_prompt}

IMPORTANT CORRECTION NEEDED:
Your previous response contained invalid labels that are NOT in the taxonomy list:
{invalid_labels_text}

Your previous response was:
{invalid_response}

ERRORS IN YOUR RESPONSE:
1. You used labels that don't exist in the taxonomy
2. You may have referenced numbers (like [147]) instead of label names
3. You may have modified the exact wording of taxonomy labels

CRITICAL INSTRUCTIONS FOR CORRECTION:
1. Look ONLY at the numbered taxonomy list above (items 1-220)
2. Find labels by reading their EXACT TEXT (ignore the numbers)
3. Copy the label names EXACTLY as written (do not change spelling/wording)
4. Do NOT reference numbers like [147] or **[147]**
5. Use this exact format: "SELECTED_LABELS: Exact Label Name 1, Exact Label Name 2"

EXAMPLE OF CORRECT FORMAT:
SELECTED_LABELS: Software Development Services, Information Technology Consulting Services
CONFIDENCE: High

Try again with EXACT label names from the taxonomy list:"""

        return correction_prompt
    
    def call_deepseek_api(self, prompt: str) -> Optional[Dict]:
        """Call the DeepSeek API with the classification prompt"""
        
        if not self.api_key:
            print("⚠️ No API key provided")
            return None
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "deepseek-reasoner",  # Use reasoning model for complex task
            "messages": [
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "max_tokens": 32000  # Higher limit for reasoning model
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=data, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            message = result['choices'][0]['message']
            
            # Handle reasoning model response format
            reasoning_content = message.get('reasoning_content', '')
            final_content = message['content']
            
            return {
                'reasoning': reasoning_content,
                'content': final_content
            }
            
        except Exception as e:
            print(f"API Error: {e}")
            return None
    
    def parse_response(self, response_data: Dict) -> Dict:
        """Parse the DeepSeek reasoning model response to extract structured data"""
        
        if not response_data or 'content' not in response_data:
            return {}
        
        content = response_data['content']
        reasoning = response_data.get('reasoning', '')
        
        response_dict = {'reasoning': reasoning}
        
        # Extract selected labels - try multiple formats
        labels = []
        
        # Format 1: SELECTED_LABELS: [label1, label2, label3]
        if "SELECTED_LABELS:" in content:
            labels_start = content.find("SELECTED_LABELS:") + len("SELECTED_LABELS:")
            labels_end = content.find("CONFIDENCE:")
            if labels_end == -1:
                # Look for other potential end markers
                for marker in ["\n\n", "REASONING:", "EXPLANATION:"]:
                    marker_pos = content.find(marker, labels_start)
                    if marker_pos != -1:
                        labels_end = marker_pos
                        break
                else:
                    labels_end = len(content)
            
            labels_text = content[labels_start:labels_end].strip()
            
            # Clean up the labels text
            labels_text = labels_text.strip('[]').strip()
            
            # Handle different formats
            if labels_text:
                # Split by comma and clean each label
                raw_labels = labels_text.split(',')
                for label in raw_labels:
                    clean_label = label.strip().strip('"').strip("'").strip('[]').strip('*').strip()
                    # Remove any bracket references like [147] or **[147]**
                    import re
                    clean_label = re.sub(r'\*?\[?\d+\]?\*?', '', clean_label).strip()
                    if clean_label and len(clean_label) > 3:  # Only keep meaningful labels
                        labels.append(clean_label)
        
        # Format 2: Look for numbered list format in reasoning or content
        if not labels:
            # Try to find labels in a numbered format
            import re
            # Look for patterns like "1. Label Name" or "- Label Name"
            patterns = [
                r'(?:^|\n)\s*\d+\.\s*([A-Za-z][^.\n]+)',
                r'(?:^|\n)\s*-\s*([A-Za-z][^.\n]+)',
                r'(?:^|\n)\s*•\s*([A-Za-z][^.\n]+)'
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, content, re.MULTILINE)
                if matches:
                    for match in matches:
                        clean_match = match.strip()
                        # Check if it might be a taxonomy label
                        if len(clean_match) > 5 and any(word in clean_match.lower() for word in ['services', 'manufacturing', 'equipment', 'installation', 'dealer']):
                            labels.append(clean_match)
                    if labels:
                        break
        
        response_dict["SELECTED_LABELS"] = labels
        
        # Extract confidence
        confidence = "Unknown"
        if "CONFIDENCE:" in content:
            conf_start = content.find("CONFIDENCE:") + len("CONFIDENCE:")
            conf_end = content.find("\n", conf_start)
            if conf_end == -1:
                conf_end = len(content)
            confidence = content[conf_start:conf_end].strip().split()[0] if content[conf_start:conf_end].strip() else "Unknown"
        
        response_dict["CONFIDENCE"] = confidence
        
        return response_dict
    
    def validate_labels(self, labels: List[str]) -> Dict:
        """Validate that all labels exist in the taxonomy"""
        
        valid_labels = []
        invalid_labels = []
        
        for label in labels:
            # First try exact match
            if label in self.taxonomy_labels:
                valid_labels.append(label)
            # Then try case-insensitive match
            elif label.lower() in self.taxonomy_labels_lower:
                valid_labels.append(self.taxonomy_labels_lower[label.lower()])
            else:
                # Try partial matching
                found_match = False
                for tax_label in self.taxonomy_labels:
                    # Check if the label is contained in taxonomy label or vice versa
                    if (label.lower() in tax_label.lower() and len(label) > 5) or \
                       (tax_label.lower() in label.lower() and len(tax_label) > 5):
                        valid_labels.append(tax_label)
                        found_match = True
                        break
                
                if not found_match:
                    invalid_labels.append(label)
        
        return {
            'valid_labels': valid_labels,
            'invalid_labels': invalid_labels,
            'all_valid': len(invalid_labels) == 0
        }
    
    def classify_company(self, company_data: Dict, max_retries: int = 2) -> Dict:
        """Classify a single company with validation and retry logic"""
        
        description = company_data.get('description', '')
        business_tags = company_data.get('business_tags', [])
        sector = company_data.get('sector', '')
        category = company_data.get('category', '')
        niche = company_data.get('niche', '')
        
        # Handle business_tags if they're in string format
        if isinstance(business_tags, str):
            try:
                business_tags = eval(business_tags) if business_tags.startswith('[') else [business_tags]
            except:
                business_tags = [business_tags] if business_tags else []
        
        # Create initial prompt
        prompt = self.create_classification_prompt(description, business_tags, sector, category, niche)
        
        for attempt in range(max_retries + 1):
            print(f"🤖 API Call attempt {attempt + 1}/{max_retries + 1}")
            
            # Call API
            response_data = self.call_deepseek_api(prompt)
            
            if not response_data:
                print("❌ API call failed")
                continue
            
            # Parse response
            parsed_response = self.parse_response(response_data)
            
            if "SELECTED_LABELS" not in parsed_response or not parsed_response["SELECTED_LABELS"]:
                print("⚠️ No labels found in response")
                if attempt == max_retries:
                    print(f"Raw response: {response_data.get('content', '')[:200]}...")
                continue
            
            # Validate labels
            validation_result = self.validate_labels(parsed_response["SELECTED_LABELS"])
            
            if validation_result['all_valid'] and len(validation_result['valid_labels']) > 0:
                # Success - all labels are valid
                print(f"✅ Valid labels found: {validation_result['valid_labels']}")
                return {
                    'original_label': company_data.get('primary_label', ''),
                    'new_labels': validation_result['valid_labels'],
                    'primary_label': validation_result['valid_labels'][0],
                    'confidence': parsed_response.get("CONFIDENCE", "Unknown"),
                    'success': True,
                    'attempts': attempt + 1,
                    'reasoning': parsed_response.get('reasoning', ''),
                    'raw_response': response_data
                }
            else:
                # Some labels are invalid - need to retry
                valid_count = len(validation_result['valid_labels'])
                invalid_count = len(validation_result['invalid_labels'])
                print(f"⚠️ Found {valid_count} valid, {invalid_count} invalid labels")
                print(f"   Valid: {validation_result['valid_labels']}")
                print(f"   Invalid: {validation_result['invalid_labels']}")
                
                if attempt < max_retries:
                    # Create correction prompt for next attempt
                    prompt = self.create_correction_prompt(prompt, response_data['content'], validation_result['invalid_labels'])
                    time.sleep(1)  # Brief pause before retry
                else:
                    # Max retries reached - return what we can salvage if we have any valid labels
                    if len(validation_result['valid_labels']) > 0:
                        print(f"⚠️ Returning partial success with valid labels: {validation_result['valid_labels']}")
                        return {
                            'original_label': company_data.get('primary_label', ''),
                            'new_labels': validation_result['valid_labels'],
                            'primary_label': validation_result['valid_labels'][0],
                            'confidence': parsed_response.get("CONFIDENCE", "Unknown"),
                            'success': True,  # Partial success
                            'attempts': attempt + 1,
                            'invalid_labels': validation_result['invalid_labels'],
                            'reasoning': parsed_response.get('reasoning', ''),
                            'raw_response': response_data
                        }
                    else:
                        print(f"❌ No valid labels found after all attempts")
                        return {
                            'original_label': company_data.get('primary_label', ''),
                            'new_labels': [],
                            'primary_label': None,
                            'confidence': parsed_response.get("CONFIDENCE", "Unknown"),
                            'success': False,
                            'attempts': attempt + 1,
                            'invalid_labels': validation_result['invalid_labels'],
                            'reasoning': parsed_response.get('reasoning', ''),
                            'raw_response': response_data
                        }
        
        # All attempts failed
        return {
            'original_label': company_data.get('primary_label', ''),
            'new_labels': [],
            'primary_label': None,
            'confidence': "None",
            'success': False,
            'attempts': max_retries + 1,
            'error': "All API attempts failed"
        }


def test_on_sample_companies(classifier: ConstrainedDeepSeekClassifier, num_samples: int = 5):
    """Test the classifier on a few sample companies first"""
    print(f"🧪 TESTING ON {num_samples} SAMPLE COMPANIES")
    print("=" * 60)
    
    # Load problematic companies
    training_df = pd.read_csv('data/processed/training_data_auto_fixed.csv')
    
    # Load taxonomy to identify remaining problematic labels
    taxonomy_df = pd.read_csv('data/input/insurance_taxonomy - insurance_taxonomy.csv')
    taxonomy_labels = set(taxonomy_df['label'].str.strip())
    
    # Find companies with problematic labels
    problematic_mask = ~training_df['primary_label'].isin(taxonomy_labels)
    problematic_companies = training_df[problematic_mask]
    
    if len(problematic_companies) == 0:
        print("✅ No problematic companies found!")
        return []
    
    # Sample a few companies
    sample_companies = problematic_companies.sample(n=min(num_samples, len(problematic_companies)), random_state=42)
    
    results = []
    
    for idx, row in sample_companies.iterrows():
        print(f"\n📊 Testing Company {len(results)+1}/{num_samples}")
        print(f"Original Label: '{row['primary_label']}'")
        print(f"Sector: {row.get('sector', 'N/A')}")
        print(f"Category: {row.get('category', 'N/A')}")
        print(f"Business Tags: {row['business_tags']}")
        print(f"\n📄 FULL DESCRIPTION:")
        print("-" * 40)
        print(f"{row['description']}")
        print("-" * 40)
        
        # Classify
        company_data = {
            'description': row['description'],
            'business_tags': row['business_tags'],
            'primary_label': row['primary_label'],
            'sector': row.get('sector', ''),
            'category': row.get('category', ''),
            'niche': row.get('niche', '')
        }
        
        result = classifier.classify_company(company_data)
        results.append(result)
        
        print(f"\n✅ CLASSIFICATION RESULT:")
        print(f"  🏷️  New Labels: {result['new_labels']}")
        print(f"  🎯 Primary Label: {result['primary_label']}")
        print(f"  📊 Confidence: {result['confidence']}")
        print(f"  ✅ Success: {result['success']}")
        print(f"  🔄 Attempts: {result['attempts']}")
        if 'invalid_labels' in result and result['invalid_labels']:
            print(f"  ❌ Invalid Labels: {result['invalid_labels']}")
        if 'reasoning' in result and result['reasoning']:
            print(f"\n🧠 AI REASONING:")
            print(f"  {result['reasoning'][:500]}...")
        
        # Ask for user validation
        print(f"\n🤔 QUALITY CHECK:")
        print(f"  Original problematic label: '{row['primary_label']}'")
        print(f"  New selected labels: {result['new_labels']}")
        
        validation = input(f"  Do these labels seem appropriate? (y/n/s to skip): ").strip().lower()
        result['manual_validation'] = validation
        
        if validation == 's':
            break
        elif validation == 'n':
            print("  📝 What would be better labels for this company?")
            better_labels = input("  Enter better labels (comma-separated): ").strip()
            result['suggested_better_labels'] = better_labels.split(',') if better_labels else []
        
        # Add delay to be nice to the API
        time.sleep(2)
    
    # Summary
    successful = sum(1 for r in results if r['success'])
    manually_approved = sum(1 for r in results if r.get('manual_validation') == 'y')
    
    print(f"\n📊 TEST SUMMARY:")
    print(f"Successful classifications: {successful}/{len(results)}")
    print(f"Success rate: {successful/len(results):.1%}")
    print(f"Manually approved: {manually_approved}/{len(results)}")
    if len(results) > 0:
        print(f"Manual approval rate: {manually_approved/len(results):.1%}")
    
    return results


def process_all_problematic_companies(classifier: ConstrainedDeepSeekClassifier):
    """Process all problematic companies and save results"""
    print("🚀 PROCESSING ALL PROBLEMATIC COMPANIES")
    print("=" * 60)
    
    # Load data
    training_df = pd.read_csv('data/processed/training_data_auto_fixed.csv')
    taxonomy_df = pd.read_csv('data/input/insurance_taxonomy - insurance_taxonomy.csv')
    taxonomy_labels = set(taxonomy_df['label'].str.strip())
    
    # Find problematic companies
    problematic_mask = ~training_df['primary_label'].isin(taxonomy_labels)
    problematic_companies = training_df[problematic_mask].copy()
    
    print(f"Found {len(problematic_companies)} companies with problematic labels")
    
    if len(problematic_companies) == 0:
        print("✅ No problematic companies found!")
        return
    
    # Process each company
    results = []
    successful = 0
    
    for idx, row in problematic_companies.iterrows():
        print(f"\n📊 Processing {len(results)+1}/{len(problematic_companies)}: {row['primary_label']}")
        
        company_data = {
            'description': row['description'],
            'business_tags': row['business_tags'],
            'primary_label': row['primary_label'],
            'sector': row.get('sector', ''),
            'category': row.get('category', ''),
            'niche': row.get('niche', '')
        }
        
        result = classifier.classify_company(company_data)
        result['original_index'] = idx
        results.append(result)
        
        if result['success']:
            successful += 1
            print(f"  ✅ {result['primary_label']} (confidence: {result['confidence']})")
        else:
            print(f"  ❌ Failed after {result['attempts']} attempts")
        
        # Rate limiting
        time.sleep(1.5)
    
    # Save results
    results_df = pd.DataFrame(results)
    output_file = 'data/processed/deepseek_reclassification_results.csv'
    results_df.to_csv(output_file, index=False)
    
    print(f"\n📊 FINAL SUMMARY:")
    print(f"Total processed: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Success rate: {successful/len(results):.1%}")
    print(f"Results saved to: {output_file}")
    
    return results


def process_all_problematic_companies_auto(classifier: ConstrainedDeepSeekClassifier):
    """Process all problematic companies automatically (for overnight processing)"""
    print("🌙 STARTING AUTOMATIC OVERNIGHT PROCESSING")
    print("=" * 80)
    
    # Load data
    training_df = pd.read_csv('data/processed/training_data_auto_fixed.csv')
    taxonomy_df = pd.read_csv('data/input/insurance_taxonomy - insurance_taxonomy.csv')
    taxonomy_labels = set(taxonomy_df['label'].str.strip())
    
    # Find problematic companies
    problematic_mask = ~training_df['primary_label'].isin(taxonomy_labels)
    problematic_companies = training_df[problematic_mask].copy()
    
    print(f"📊 PROCESSING SCOPE:")
    print(f"   Total companies in dataset: {len(training_df):,}")
    print(f"   Companies with valid labels: {len(training_df) - len(problematic_companies):,}")
    print(f"   Companies needing reclassification: {len(problematic_companies):,}")
    print(f"")
    print(f"🤖 ESTIMATED API CALLS: {len(problematic_companies):,} - {len(problematic_companies) * 1.2:.0f}")
    print(f"⏱️  ESTIMATED TIME: {(len(problematic_companies) * 1.5) / 60:.1f} - {(len(problematic_companies) * 1.8) / 60:.1f} minutes")
    print(f"")
    
    if len(problematic_companies) == 0:
        print("✅ No problematic companies found!")
        return
    
    # Start processing
    results = []
    successful = 0
    failed = 0
    start_time = time.time()
    
    # Progress tracking
    checkpoint_interval = 50  # Save progress every 50 companies
    
    for idx, (row_idx, row) in enumerate(problematic_companies.iterrows()):
        current_progress = idx + 1
        percentage = (current_progress / len(problematic_companies)) * 100
        
        print(f"\n📊 Processing {current_progress}/{len(problematic_companies)} ({percentage:.1f}%)")
        print(f"   Company: {row['primary_label']}")
        print(f"   Sector: {row.get('sector', 'N/A')} | Category: {row.get('category', 'N/A')}")
        
        company_data = {
            'description': row['description'],
            'business_tags': row['business_tags'],
            'primary_label': row['primary_label'],
            'sector': row.get('sector', ''),
            'category': row.get('category', ''),
            'niche': row.get('niche', '')
        }
        
        result = classifier.classify_company(company_data)
        result['original_index'] = row_idx
        result['processed_at'] = time.time()
        results.append(result)
        
        if result['success'] and len(result['new_labels']) > 0:
            successful += 1
            print(f"   ✅ SUCCESS: {result['primary_label']} (confidence: {result['confidence']})")
            if len(result['new_labels']) > 1:
                print(f"      Additional: {', '.join(result['new_labels'][1:])}")
        else:
            failed += 1
            print(f"   ❌ FAILED after {result['attempts']} attempts")
            if 'invalid_labels' in result and result['invalid_labels']:
                print(f"      Invalid labels: {result['invalid_labels']}")
        
        # Progress checkpoint - save intermediate results
        if current_progress % checkpoint_interval == 0 or current_progress == len(problematic_companies):
            elapsed_time = time.time() - start_time
            avg_time_per_company = elapsed_time / current_progress
            estimated_remaining = (len(problematic_companies) - current_progress) * avg_time_per_company
            
            print(f"\n📈 PROGRESS CHECKPOINT:")
            print(f"   Processed: {current_progress}/{len(problematic_companies)} ({percentage:.1f}%)")
            print(f"   Successful: {successful} ({successful/current_progress:.1%})")
            print(f"   Failed: {failed}")
            print(f"   Elapsed time: {elapsed_time/60:.1f} minutes")
            print(f"   Estimated remaining: {estimated_remaining/60:.1f} minutes")
            
            # Save intermediate results
            intermediate_df = pd.DataFrame(results)
            intermediate_file = f'data/processed/deepseek_reclassification_progress_{current_progress}.csv'
            intermediate_df.to_csv(intermediate_file, index=False)
            print(f"   💾 Progress saved to: {intermediate_file}")
        
        # Rate limiting - slightly faster for overnight processing
        time.sleep(1.2)
    
    # Final results
    total_time = time.time() - start_time
    results_df = pd.DataFrame(results)
    output_file = 'data/processed/deepseek_reclassification_results_final.csv'
    results_df.to_csv(output_file, index=False)
    
    print(f"\n🎉 OVERNIGHT PROCESSING COMPLETE!")
    print("=" * 60)
    print(f"📊 FINAL STATISTICS:")
    print(f"   Total processed: {len(results):,}")
    print(f"   Successful: {successful:,} ({successful/len(results):.1%})")
    print(f"   Failed: {failed:,} ({failed/len(results):.1%})")
    print(f"   Total time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
    print(f"   Average time per company: {total_time/len(results):.1f} seconds")
    print(f"   💾 Final results saved to: {output_file}")
    
    # Analysis of results
    if successful > 0:
        successful_results = [r for r in results if r['success']]
        confidence_counts = {}
        for result in successful_results:
            conf = result.get('confidence', 'Unknown')
            confidence_counts[conf] = confidence_counts.get(conf, 0) + 1
        
        print(f"\n📈 CONFIDENCE BREAKDOWN:")
        for conf, count in sorted(confidence_counts.items()):
            print(f"   {conf}: {count} ({count/successful:.1%})")
        
        # Most common new labels
        all_new_labels = []
        for result in successful_results:
            all_new_labels.extend(result.get('new_labels', []))
        
        from collections import Counter
        label_counts = Counter(all_new_labels)
        
        print(f"\n🏷️  TOP 10 ASSIGNED LABELS:")
        for label, count in label_counts.most_common(10):
            print(f"   {label}: {count} companies")
    
    return results


def main():
    """Main workflow for automatic overnight processing"""
    print("🌙 CONSTRAINED DEEPSEEK RE-CLASSIFICATION - OVERNIGHT MODE")
    print("=" * 80)
    
    # Initialize classifier
    print("Initializing classifier...")
    classifier = ConstrainedDeepSeekClassifier()
    print(f"✅ Loaded {len(classifier.taxonomy_labels)} taxonomy labels")
    
    # Show scope
    training_df = pd.read_csv('data/processed/training_data_auto_fixed.csv')
    taxonomy_df = pd.read_csv('data/input/insurance_taxonomy - insurance_taxonomy.csv')
    taxonomy_labels = set(taxonomy_df['label'].str.strip())
    problematic_mask = ~training_df['primary_label'].isin(taxonomy_labels)
    problematic_count = problematic_mask.sum()
    
    print(f"\n📊 OVERNIGHT PROCESSING SCOPE:")
    print(f"   Companies needing reclassification: {problematic_count:,}")
    print(f"   Estimated time: {(problematic_count * 1.2) / 60:.1f} minutes")
    print(f"   Rate: ~1.2 seconds per company (with validation retries)")
    
    print(f"\n🚀 Starting automatic processing in 5 seconds...")
    print(f"   Progress will be saved every 50 companies")
    print(f"   You can monitor progress by checking the checkpoint files")
    time.sleep(5)
    
    # Run automatic processing
    try:
        results = process_all_problematic_companies_auto(classifier)
        print(f"\n✅ OVERNIGHT PROCESSING COMPLETED SUCCESSFULLY!")
        return results
    except KeyboardInterrupt:
        print(f"\n⚠️ Processing interrupted by user")
        return None
    except Exception as e:
        print(f"\n❌ ERROR during processing: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Test mode - run interactive testing
        print("🧪 CONSTRAINED DEEPSEEK RE-CLASSIFICATION - TEST MODE")
        print("=" * 80)
        
        classifier = ConstrainedDeepSeekClassifier()
        print(f"✅ Loaded {len(classifier.taxonomy_labels)} taxonomy labels")
        
        test_results = test_on_sample_companies(classifier, num_samples=3)
        
        if test_results:
            successful = sum(1 for r in test_results if r['success'])
            approved = sum(1 for r in test_results if r.get('manual_validation') == 'y')
            print(f"\n📊 TEST RESULTS:")
            print(f"   Technical success: {successful}/{len(test_results)} ({successful/len(test_results):.1%})")
            print(f"   Manual approval: {approved}/{len(test_results)} ({approved/len(test_results):.1%})")
            
            if approved >= len(test_results) * 0.6:
                print(f"\n✅ Quality looks good! Ready for overnight processing.")
                print(f"Run without --test flag for automatic processing.")
            else:
                print(f"\n⚠️ Consider adjusting the prompt before overnight processing.")
    else:
        # Automatic overnight mode
        main() 