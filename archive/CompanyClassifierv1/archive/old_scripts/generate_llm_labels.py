# generate_llm_labels.py

import pandas as pd
import requests
import json
import time
import re
from tqdm.auto import tqdm
import os

# --- Configuration ---
SAMPLE_SIZE = 100
COMPANY_FILE = 'ml_insurance_challenge.csv'
TAXONOMY_FILE = 'insurance_taxonomy.xlsx'
OUTPUT_FILE = f'llm_generated_labels_sample{SAMPLE_SIZE}.json' # Store raw LLM outputs
API_KEY = "sk-9bb2066ba85a4980a357b6aaad6e40bc" # ** SECURITY WARNING: Hardcoding keys is risky **
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
REQUEST_DELAY_SECONDS = 2 # Add a small delay between API calls to avoid rate limits

# --- Load Data ---
def load_data(company_file, taxonomy_file):
    print("Loading data...")
    try:
        companies_df = pd.read_csv(company_file)
        taxonomy_df = pd.read_excel(taxonomy_file)
        print(f"Loaded {len(companies_df)} companies and {len(taxonomy_df)} taxonomy labels.")
        # Basic cleaning
        for col in ['description', 'business_tags', 'sector', 'category', 'niche']:
            if col in companies_df.columns:
                companies_df[col] = companies_df[col].fillna('N/A')
        return companies_df, taxonomy_df
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return None, None

# --- Prepare Taxonomy List for Prompt ---
def format_taxonomy_list(taxonomy_df):
    return "\n".join([f"- {label}" for label in taxonomy_df['label']])

# --- Prompt Template ---
PROMPT_TEMPLATE = """
**Task:** You are an expert insurance analyst. Your goal is to identify the most relevant insurance labels from the provided taxonomy that a given company would likely need based on its business profile. Focus on the core activities and risks implied by the profile.

**Company Profile:**
*   **Sector:** {sector}
*   **Category:** {category}
*   **Niche:** {niche}
*   **Business Tags:** {tags}
*   **Description:** {description}

**Insurance Taxonomy (Select from this list):**
{taxonomy_list}

**Instructions:**
1.  Carefully analyze the Company Profile provided above. Consider the sector, category, niche, tags, and description to understand the company's primary operations and potential liabilities.
2.  Identify **up to 5** insurance labels from the Taxonomy list that are MOST relevant to this company's likely operational risks and insurance needs. Prioritize labels that directly address the core business activities.
3.  For EACH label you select, provide a **brief justification** (1-2 sentences) explaining WHY it is relevant based *specifically* on the details in the company's profile (e.g., mention specific tags, description keywords, or the niche).
4.  If you believe NO labels from the list are clearly relevant or if the profile is too ambiguous, state "No relevant labels identified".

**Output Format:**
Return your answer as a list of selected labels, each followed by its justification. Use the following format precisely:

- [Selected Label 1]: [Justification 1]
- [Selected Label 2]: [Justification 2]
...

**Please provide your analysis for the company profile above.**
"""

# --- DeepSeek API Call Function ---
def get_deepseek_completion(prompt, api_key):
    """Calls the DeepSeek API and returns the response content."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": "deepseek-chat", # Or "deepseek-coder" if preferred, might need prompt adjustment
        "messages": [
            {"role": "system", "content": "You are an expert insurance analyst."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1024, # Adjust as needed
        "temperature": 0.2, # Lower temperature for more deterministic output
    }
    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=120) # Increased timeout
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        data = response.json()
        if 'choices' in data and len(data['choices']) > 0:
            return data['choices'][0]['message']['content']
        else:
            print(f"Warning: Unexpected API response format: {data}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error calling DeepSeek API: {e}")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding API response: {response.text}")
        return None

# --- Main Execution ---
if __name__ == "__main__":
    print("Starting LLM Label Generation...")
    companies_df, taxonomy_df = load_data(COMPANY_FILE, TAXONOMY_FILE)

    if companies_df is None or taxonomy_df is None:
        print("Exiting due to data loading failure.")
        exit()

    taxonomy_list_str = format_taxonomy_list(taxonomy_df)
    companies_sample_df = companies_df.head(SAMPLE_SIZE).copy()

    llm_results = []
    print(f"Generating labels for {SAMPLE_SIZE} companies...")

    for index, row in tqdm(companies_sample_df.iterrows(), total=SAMPLE_SIZE):
        # Format the prompt for the current company
        prompt = PROMPT_TEMPLATE.format(
            sector=row.get('sector', 'N/A'),
            category=row.get('category', 'N/A'),
            niche=row.get('niche', 'N/A'),
            tags=row.get('business_tags', 'N/A'),
            description=row.get('description', 'N/A'),
            taxonomy_list=taxonomy_list_str
        )

        # Call the API
        raw_response = get_deepseek_completion(prompt, API_KEY)

        # Store the result along with company identifier
        llm_results.append({
            'company_index': index, # Use the original DataFrame index
            'description': row.get('description', 'N/A'), # Include description for context during review
            'llm_response': raw_response if raw_response else "API_ERROR"
        })

        # Respect API rate limits
        time.sleep(REQUEST_DELAY_SECONDS)

    # Save the raw results
    print(f"\nSaving raw LLM responses to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(llm_results, f, indent=4, ensure_ascii=False)

    print("Label generation complete. Please review the contents of", OUTPUT_FILE)
    print("Next steps: Manually review the 'llm_response' field in the JSON file, correct/confirm the labels, and create your fine-tuning dataset.")
