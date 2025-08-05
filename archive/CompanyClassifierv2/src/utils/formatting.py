"""
Formatting utilities for SIC classification
"""


def format_sic_code(division, major_group, industry_group, sic=None):
    """
    Format SIC codes with consistent leading zeros
    
    Args:
        division: Division code (e.g., 'C')
        major_group: Major group code (e.g., 16)
        industry_group: Industry group code (e.g., 162)
        sic: SIC code (e.g., 1623)
    
    Returns:
        str: Formatted SIC code string
    """
    if major_group is None:
        return f"{division}"
    elif industry_group is None:
        return f"{division}-{str(major_group).zfill(2)}"
    elif sic is None:
        return f"{division}-{str(major_group).zfill(2)}-{str(industry_group).zfill(3)}"
    else:
        return f"{division}-{str(major_group).zfill(2)}-{str(industry_group).zfill(3)}-{str(sic).zfill(4)}"


def get_division_name(code):
    """
    Get division name from division code
    
    Args:
        code: Division code (e.g., 'C')
    
    Returns:
        str: Division name
    """
    names = {
        'A': 'Agriculture, Forestry, and Fishing',
        'B': 'Mining', 
        'C': 'Construction', 
        'D': 'Manufacturing',
        'E': 'Transportation, Communications, Electric, Gas, and Sanitary Services',
        'F': 'Wholesale Trade', 
        'G': 'Retail Trade',
        'H': 'Finance, Insurance, and Real Estate',
        'I': 'Services', 
        'J': 'Public Administration'
    }
    return names.get(code, f'Division {code}') 