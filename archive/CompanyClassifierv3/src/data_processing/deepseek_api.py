"""
DeepSeek API Integration for Insurance Classification
Sends company data + top 10 suggestions to DeepSeek for reasoning-based classification
"""

import json
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import time
import pandas as pd


class DeepSeekClassifier:
    """
    Integration with DeepSeek API for insurance company classification
    """
    
    def __init__(self, api_key: str, output_file: str = "data/DeepSeek_validations.json"):
        """
        Initialize DeepSeek API classifier
        
        Args:
            api_key: DeepSeek API key
            output_file: Path to save validation results
        """
        self.api_key = api_key
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(exist_ok=True)
        
        self.api_url = "https://api.deepseek.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Load existing validations
        self.validations = self._load_existing_validations()
        
    def _load_existing_validations(self) -> Dict:
        """Load existing DeepSeek validations if file exists"""
        if self.output_file.exists():
            try:
                with open(self.output_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Could not load existing validations: {e}")
                
        return {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "total_classifications": 0,
                "api_calls": 0
            },
            "validations": []
        }
    
    def _save_validations(self):
        """Save validations to JSON file"""
        with open(self.output_file, 'w') as f:
            json.dump(self.validations, f, indent=2)
    
    def create_classification_prompt(self, company_data: Dict, suggestions: List[Tuple[str, float]], stage: str = "final") -> str:
        """
        Create a structured prompt for DeepSeek API
        
        Args:
            company_data: Company information dictionary
            suggestions: List of (label, confidence) tuples or all labels
            stage: "filtering" for 220→20, "final" for 20→1
            
        Returns:
            Formatted prompt string
        """
        
        # Format business tags
        business_tags = company_data.get('business_tags', [])
        if isinstance(business_tags, str):
            try:
                business_tags = eval(business_tags)
            except:
                business_tags = [business_tags]
        
        # Format suggestions with rankings
        suggestions_text = ""
        for i, (label, confidence) in enumerate(suggestions, 1):
            if stage == "filtering":
                suggestions_text += f"   {i}. {label}\n"  # No scores for filtering stage
            else:
                suggestions_text += f"   {i}. {label} (similarity score: {confidence:.3f})\n"
        
        if stage == "filtering":
            # Stage 1: Filter from all 220 labels to top 20
            description = company_data.get('description', 'N/A')
            # Handle NaN/float values in description
            if pd.isna(description) or not isinstance(description, str):
                description = 'N/A'
            description_truncated = description[:500] + '...' if len(description) > 500 else description
            
            prompt = f"""You are an insurance industry expert tasked with filtering insurance classification options.

COMPANY INFORMATION:
===================
Description: {description_truncated}

Business Tags: {', '.join(business_tags) if business_tags else 'N/A'}

Industry Context:
- Sector: {company_data.get('sector', 'N/A')}
- Category: {company_data.get('category', 'N/A')}
- Niche: {company_data.get('niche', 'N/A')}

ALL AVAILABLE INSURANCE LABELS:
==============================
{suggestions_text}

TASK - STAGE 1 FILTERING:
========================
From ALL {len(suggestions)} labels above, identify ANY labels that might be applicable to this company's business activity.

Be INCLUSIVE in this filtering stage - if there's any possibility a label could match, include it.

Focus on:
1. What is the company's MAIN business activity / value proposition?
2. What do they primarily get PAID for?
3. What is their PRIMARY line of business?

RESPONSE FORMAT:
===============
Return up to 30 labels as a JSON array (include any that might be relevant):

["Label 1", "Label 2", "Label 3", ..., "Label N"]

Important: Only respond with the JSON array, no additional text."""

        else:
            # Stage 2: Final reasoning through filtered labels
            description = company_data.get('description', 'N/A')
            # Handle NaN/float values in description
            if pd.isna(description) or not isinstance(description, str):
                description = 'N/A'
            description_truncated = description[:200] + '...' if len(description) > 200 else description
            
            prompt = f"""You are an insurance industry expert making the FINAL classification decision.

COMPANY INFORMATION:
===================
Description: {description_truncated}

Business Tags: {', '.join(business_tags) if business_tags else 'N/A'}

Industry Context:
- Sector: {company_data.get('sector', 'N/A')}
- Category: {company_data.get('category', 'N/A')}
- Niche: {company_data.get('niche', 'N/A')}

PRE-FILTERED INSURANCE LABELS:
=============================
{suggestions_text}

TASK - STAGE 2 FINAL DECISION:
=============================
From the pre-filtered labels above, carefully reason through each one and select the label(s) that best represent this company's main business activity.

Focus on:
1. What is the company's MAIN business activity / value proposition?
2. What do they primarily get PAID for?
3. What is their PRIMARY line of business?

Select the most appropriate label(s) that represent their main business activity. There's no strict limit - select as many as genuinely apply, but focus on accuracy.

RESPONSE FORMAT:
===============
Provide a COMPACT JSON response with this structure: (make sure to stay within the 8192 token limit)

{{
    "selected_labels": ["Primary Label", "Secondary Label", "..."],
    "primary_label": "Most important single label",
    "confidence": "HIGH/MEDIUM/LOW",
    "reasoning": "Detailed explanation of why these labels represent the business",
    "label_explanations": {{
        "Primary Label": "why this is the main activity",
        "Secondary Label": "why this is also relevant"
    }},
    "primary_activity_keywords": ["key phrases from description that indicate main business"],
    "rejected_alternatives": ["labels considered but rejected with brief reason"]
}}

CRITICAL: Keep the JSON response COMPACT. Only respond with the JSON object, no additional text."""

        return prompt
    
    def call_deepseek_api(self, prompt: str, max_retries: int = 3, use_reasoning_model: bool = True) -> Optional[Dict]:
        """
        Make API call to DeepSeek
        
        Args:
            prompt: The classification prompt
            max_retries: Maximum number of retry attempts
            
        Returns:
            API response or None if failed
        """
        
        # Choose model based on task complexity
        if use_reasoning_model:
            model_name = "deepseek-reasoner"
            max_tokens = 64000  # High limit for reasoning
            temperature_setting = {}  # Temperature not supported by reasoner
        else:
            model_name = "deepseek-chat"
            max_tokens = 8192  # Lower limit for simple filtering
            temperature_setting = {"temperature": 0.1}  # Low temperature for consistency
        
        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": max_tokens,
            **temperature_setting
        }
        
        for attempt in range(max_retries):
            try:
                print(f"🤖 Calling DeepSeek API (attempt {attempt + 1}/{max_retries})...")
                
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=120  # Increased timeout for reasoning model
                )
                
                if response.status_code == 200:
                    result = response.json()
                    message = result['choices'][0]['message']
                    
                    # Handle reasoning model vs chat model responses
                    if use_reasoning_model:
                        # Extract both reasoning and final content (reasoning model)
                        reasoning_content = message.get('reasoning_content', '')
                        final_content = message['content'].strip()
                        
                        print(f"🧠 DeepSeek Reasoning: {reasoning_content[:200]}..." if reasoning_content else "🧠 No reasoning provided")
                        
                        # Try to parse JSON response from final content
                        try:
                            classification_result = json.loads(final_content)
                            # Add reasoning to the result for later analysis
                            classification_result['deepseek_reasoning'] = reasoning_content
                            return classification_result
                        except json.JSONDecodeError:
                            print(f"⚠️ Failed to parse JSON response: {final_content}")
                            if attempt < max_retries - 1:
                                time.sleep(2)
                                continue
                            return None
                    else:
                        # Simple chat model response (Stage 1 filtering)
                        content = message['content'].strip()
                        
                        # Try to parse JSON response
                        try:
                            return json.loads(content)
                        except json.JSONDecodeError:
                            print(f"⚠️ Failed to parse JSON response: {content}")
                            if attempt < max_retries - 1:
                                time.sleep(2)
                                continue
                            return None
                        
                else:
                    print(f"❌ API call failed with status {response.status_code}: {response.text}")
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                        
            except Exception as e:
                print(f"❌ Exception during API call: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                    
        return None
    
    def classify_company_hierarchical(self, company_data: Dict, all_labels: List[str], company_index: int = -1) -> Optional[Dict]:
        """
        Two-stage hierarchical classification: 220 labels → 20 labels → final decision
        
        Args:
            company_data: Company information
            all_labels: All 220 insurance labels
            company_index: Optional company index for tracking
            
        Returns:
            Classification result or None if failed
        """
        
        print(f"🎯 Stage 1: Filtering {len(all_labels)} labels (inclusive filtering, max 30)...")
        
        # Stage 1: Filter to up to 30 labels (inclusive) - use fast chat model
        all_labels_with_scores = [(label, 1.0) for label in all_labels]  # Dummy scores for formatting
        filtering_prompt = self.create_classification_prompt(company_data, all_labels_with_scores, stage="filtering")
        
        filtering_result = self.call_deepseek_api(filtering_prompt, use_reasoning_model=False)
        
        if not filtering_result:
            print("❌ Stage 1 filtering failed")
            return None
            
        # Parse filtering result (should be a list of up to 30 labels)
        try:
            if isinstance(filtering_result, list):
                filtered_labels = filtering_result[:30]  # Ensure max 30
            else:
                print("❌ Stage 1 returned unexpected format")
                return None
        except Exception as e:
            print(f"❌ Failed to parse Stage 1 result: {e}")
            return None
            
        print(f"✅ Stage 1: Filtered to {len(filtered_labels)} labels")
        print(f"   Top 5: {', '.join(filtered_labels[:5])}")
        
        # Stage 2: Final reasoning through filtered labels - use reasoning model
        print(f"🧠 Stage 2: Final reasoning through {len(filtered_labels)} labels...")
        
        filtered_with_scores = [(label, 1.0) for label in filtered_labels]  # Dummy scores
        final_prompt = self.create_classification_prompt(company_data, filtered_with_scores, stage="final")
        
        final_result = self.call_deepseek_api(final_prompt, use_reasoning_model=True)
        
        if final_result:
            # Create validation entry
            validation_entry = {
                "timestamp": datetime.now().isoformat(),
                "company_index": company_index,
                "company_data": {
                    "description": company_data.get('description', ''),
                    "business_tags": company_data.get('business_tags', []),
                    "sector": company_data.get('sector', ''),
                    "category": company_data.get('category', ''),
                    "niche": company_data.get('niche', '')
                },
                "stage_1_filtered_labels": filtered_labels,
                "stage_2_classification": final_result,
                "method": "hierarchical_deepseek",
                "status": "success"
            }
            
            # Add to validations
            self.validations["validations"].append(validation_entry)
            self.validations["metadata"]["total_classifications"] += 1
            self.validations["metadata"]["api_calls"] += 2  # Two API calls
            
            # Save to file
            self._save_validations()
            
            selected_labels = final_result.get('selected_labels', [])
            primary_label = final_result.get('primary_label', 'N/A')
            
            print(f"✅ Final Classification:")
            print(f"   Selected Labels: {selected_labels}")
            print(f"   Primary Label: {primary_label}")
            print(f"   Confidence: {final_result.get('confidence', 'UNKNOWN')}")
            print(f"   Reasoning: {final_result.get('reasoning', 'No reasoning provided')[:100]}...")
            
            return validation_entry
            
        else:
            print("❌ Stage 2 final classification failed")
            return None

    def classify_company(self, company_data: Dict, suggestions: List[Tuple[str, float]], company_index: int = -1) -> Optional[Dict]:
        """
        Classify a single company using DeepSeek API (single-stage for backward compatibility)
        
        Args:
            company_data: Company information
            suggestions: Top N suggestions from similarity system
            company_index: Optional company index for tracking
            
        Returns:
            Classification result or None if failed
        """
        
        # Create prompt
        prompt = self.create_classification_prompt(company_data, suggestions, stage="final")
        
        # Call API
        api_result = self.call_deepseek_api(prompt)
        
        if api_result:
            # Create validation entry
            validation_entry = {
                "timestamp": datetime.now().isoformat(),
                "company_index": company_index,
                "company_data": {
                    "description": company_data.get('description', ''),  # Full description for analysis
                    "business_tags": company_data.get('business_tags', []),
                    "sector": company_data.get('sector', ''),
                    "category": company_data.get('category', ''),
                    "niche": company_data.get('niche', '')
                },
                "similarity_suggestions": suggestions,
                "deepseek_classification": api_result,
                "method": "single_stage_deepseek",
                "status": "success"
            }
            
            # Add to validations
            self.validations["validations"].append(validation_entry)
            self.validations["metadata"]["total_classifications"] += 1
            self.validations["metadata"]["api_calls"] += 1
            
            # Save to file
            self._save_validations()
            
            print(f"✅ DeepSeek classified: {api_result.get('selected_label', 'NONE')}")
            print(f"   Confidence: {api_result.get('confidence', 'UNKNOWN')}")
            print(f"   Reasoning: {api_result.get('reasoning', 'No reasoning provided')[:100]}...")
            
            return validation_entry
            
        else:
            # Record failed attempt
            failed_entry = {
                "timestamp": datetime.now().isoformat(),
                "company_index": company_index,
                "company_data": {
                    "description": company_data.get('description', ''),
                },
                "similarity_suggestions": suggestions[:3],  # Just top 3 for failed attempts
                "deepseek_classification": None,
                "status": "failed"
            }
            
            self.validations["validations"].append(failed_entry)
            self.validations["metadata"]["api_calls"] += 1
            self._save_validations()
            
            print("❌ DeepSeek classification failed")
            return None
    
    def batch_classify(self, companies_data: List[Dict], suggestions_list: List[List[Tuple[str, float]]], delay: float = 1.0):
        """
        Classify multiple companies with API rate limiting
        
        Args:
            companies_data: List of company data dictionaries
            suggestions_list: List of suggestion lists for each company
            delay: Delay between API calls in seconds
        """
        
        print(f"🚀 Starting batch classification of {len(companies_data)} companies")
        print(f"   Delay between calls: {delay} seconds")
        
        successful = 0
        failed = 0
        
        for i, (company_data, suggestions) in enumerate(zip(companies_data, suggestions_list)):
            print(f"\n📊 Processing company {i + 1}/{len(companies_data)}")
            
            result = self.classify_company(company_data, suggestions, company_index=i)
            
            if result:
                successful += 1
            else:
                failed += 1
                
            # Rate limiting delay
            if i < len(companies_data) - 1:  # Don't delay after last company
                time.sleep(delay)
                
        print(f"\n📊 Batch Classification Complete:")
        print(f"   Successful: {successful}")
        print(f"   Failed: {failed}")
        print(f"   Results saved to: {self.output_file}")
    
    def get_statistics(self) -> Dict:
        """Get classification statistics"""
        validations = self.validations["validations"]
        
        successful = len([v for v in validations if v["status"] == "success"])
        failed = len([v for v in validations if v["status"] == "failed"])
        
        # Get label distribution
        labels = [v["deepseek_classification"]["selected_label"] 
                 for v in validations 
                 if v["status"] == "success" and v["deepseek_classification"]]
        
        from collections import Counter
        label_counts = Counter(labels)
        
        return {
            "total_processed": len(validations),
            "successful": successful,
            "failed": failed,
            "success_rate": successful / len(validations) if validations else 0,
            "top_labels": dict(label_counts.most_common(10)),
            "total_api_calls": self.validations["metadata"]["api_calls"]
        }


def test_deepseek_integration():
    """Test function for DeepSeek integration"""
    
    # This would be used for testing with a sample company
    sample_company = {
        "description": "Welchcivils is a civil engineering company specializing in utility network connections...",
        "business_tags": ["Construction Services", "Multi-utilities", "Utility Network Connections"],
        "sector": "Services",
        "category": "Civil Engineering Services",
        "niche": "Other Heavy and Civil Engineering Construction"
    }
    
    sample_suggestions = [
        ("Multi-Family Construction Services", 0.432),
        ("Commercial Construction Services", 0.372),
        ("Pipeline Construction Services", 0.358),
        ("Cable Installation Services", 0.311),
        ("Gas Installation Services", 0.287)
    ]
    
    # Note: Would need actual API key to test
    print("Sample prompt that would be sent to DeepSeek:")
    print("=" * 60)
    
    classifier = DeepSeekClassifier("dummy_key")
    prompt = classifier.create_classification_prompt(sample_company, sample_suggestions[:5])
    print(prompt)


if __name__ == "__main__":
    test_deepseek_integration() 