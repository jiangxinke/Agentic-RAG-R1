import random
import numpy as np
import re

def extract_answer_from_model_output(text):
    """
    Extracts the value from the last <answer> tag in the text.
    Returns None if no valid answer is found.
    """
    # Split on <answer> and take everything after the last occurrence
    parts = text.split("<answer>")
    if len(parts) < 2:  # No <answer> tag found
        return None

    last_part = parts[-1]

    # Extract content up to </answer>
    if "</answer>" not in last_part:
        return None

    answer = last_part.split("</answer>")[0].strip()
    return None if answer == "..." else answer

def extract_answer_from_dataset(text):
    """
    Extracts the answer from the dataset.
    The dataset separates the answer using the '####' delimiter.
    """
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def extract_last_number(text):
    """
    Extracts the last number from text if it's properly separated.
    
    Args:
        text (str): The text to extract a number from.
        
    Returns:
        float or None: The extracted number as a float, or None if no valid number is found.
        
    Explanation:
        1. First removes $ and % signs from the text.
        2. Uses regex to find numbers that are:
           - Preceded by space, equals sign, or start of string
           - Followed by end of string or space
        3. Returns the first matching number as a float, or None if no match is found.
    """
    import re

    # Remove $ and % signs
    text = text.replace('$', '').replace('%', '')

    # HERE
    # Look for numbers that are:
    # - preceded by space or = or start of string (via \b or ^)
    # - followed by end of string or space
    pattern = r'(?:^|\s|=)\s*(-?\d*\.?\d+)\s*$'
    match = re.search(pattern, text)
    return float(match.group(1)) if match else None


def extract_single_number(text):
    """
    Extracts a single number from text if exactly one exists.
    
    Args:
        text (str): The text to extract a number from.
        
    Returns:
        float or None: The extracted number as a float if exactly one number exists,
                      otherwise None.
        
    Explanation:
        1. Uses regex to find all numbers in the text.
        2. Returns the first number as a float if exactly one number is found.
        3. Returns None if zero or multiple numbers are found.
    """
    import re
    numbers = re.findall(r'-?\d*\.?\d+', text)
    return float(numbers[0]) if len(numbers) == 1 else None

def extract_observation_from_text(text):
    """
    Extracts the content between <observation> and </observation> tags.
    
    Args:
        content (str): The string content from which to extract the observation.
        
    Returns:
        str: The content between <observation> and </observation> tags.
    """
    # Use regular expression to extract the content between <observation>...</observation>
    match = re.search(r'<observation>(.*?)</observation>', text, re.DOTALL)
    if match:
        return match.group(1).strip()  # Return the extracted observation content
    return "No observation"  # If no observation tag is found, return an empty string
