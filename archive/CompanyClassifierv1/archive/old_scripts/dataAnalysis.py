import pandas as pd
import numpy as np

# Read data
companies = pd.read_csv('ml_insurance_challenge.csv')
taxonomy = pd.read_excel('insurance_taxonomy.xlsx')

# Basic stats
print(f"Companies: {len(companies)}")
print(f"Taxonomy labels: {len(taxonomy)}")

# Check for missing values
print("\nMissing values in company data:")
print(companies.isnull().sum())

# Examine description length
companies['desc_length'] = companies['description'].str.len()
print(f"\nDescription length stats:")
print(companies['desc_length'].describe())

# Examine business tags structure
sample_tags = companies['business_tags'].iloc[:5]
print("\nSample business tags:")
for i, tags in enumerate(sample_tags):
    print(f"{i}: {tags}")

# Examine taxonomy distribution (top sectors)
print("\nTop sectors:")
print(companies['sector'].value_counts().head(10))

# Print sample taxonomy labels
print("\nSample taxonomy labels:")
print(taxonomy['label'].iloc[:10].tolist())

# Let's look at some company examples and taxonomy examples
print("\nSample company descriptions:")
for i in range(3):
    print(f"\nCompany {i+1}:")
    print(f"Description: {companies['description'].iloc[i][:150]}...")
    print(f"Tags: {companies['business_tags'].iloc[i]}")
    print(f"Sector/Category/Niche: {companies['sector'].iloc[i]} / {companies['category'].iloc[i]} / {companies['niche'].iloc[i]}")

print("\nMore taxonomy examples:")
print(taxonomy['label'].iloc[10:20].tolist())