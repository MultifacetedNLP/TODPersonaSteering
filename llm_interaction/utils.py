"""Shared helpers for the LLM-driven dialogue simulators.

Sections, in order:

1. Schema readers and method/parameter resolution (against the dataset schema).
2. API-call response parsing (free-form text → method + parameters).
3. CSV-backed "service result" lookups for the dataset's simulated database.
4. CSV I/O for conversation logs and per-domain entry tables.
"""

import os
import re
import ast
import csv
import yaml
import difflib
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz import process


# ---------------------------------------------------------------------------
# Section 5 — small I/O and matching helpers
# ---------------------------------------------------------------------------


def load_model_data(yaml_path):
    """Load model data from YAML."""
    with open(yaml_path, 'r') as file:
        model_data = yaml.safe_load(file)
    return model_data

def get_best_match(query, reference_list, threshold=80):
    match = process.extractOne(query, reference_list)
    if match:
        matched_element, score, _ = match
        return matched_element if score >= threshold else None
    return None
    
# Load schema file
# ---------------------------------------------------------------------------
# Section 1 — schema readers and method/parameter resolution
# ---------------------------------------------------------------------------


def load_schema(schema_file, domain):
    with open(schema_file, 'r') as file:
        schema_data = file.read()
        domains = [line.split(':')[1].strip() for line in schema_data.splitlines() if line.startswith('service_name')]
        closest_match = difflib.get_close_matches(domain, domains, n=1, cutoff=0.7)

        if not closest_match:
            raise ValueError(f"Domain '{domain}' not found in schema. Did you mean: {', '.join(domains)}?")

        domain_corrected = closest_match[0]
        domain_start = schema_data.find(f"service_name: {domain_corrected}")
        domain_schema = schema_data[domain_start:]
        next_domain = domain_schema.find("service_name:", 1)
        if next_domain != -1:
            domain_schema = domain_schema[:next_domain]
        
        return domain_schema.strip()

def check_is_transactional(schema_file, domain, method_name):
    # Read the schema file
    with open(schema_file, 'r', encoding='utf-8') as f:
        schema_content = f.read()
    
    # Split the schema into domains
    domains = schema_content.split('service_name:')[1:]
    for dom in domains:
        dom = dom.strip()
        if dom.startswith(domain):
            # Found the correct domain
            # Now find the method within this domain
            intents = dom.split('intents_')
            for intent in intents:
                if f"name: {method_name}" in intent:
                    # Found the method
                    # Check if it's transactional
                    lines = intent.strip().split('\n')
                    for line in lines:
                        if line.strip().startswith('is_transactional:'):
                            type_line = line.strip()
                            # Extract the type value
                            type_value = type_line.replace('is_transactional: ', '').strip()
                            is_transactional = type_value.lower() == 'true'
                            return is_transactional
                    # If 'is_transactional:' line not found, default to False
                    return False
            # Method not found in this domain
            break  # Exit after searching the correct domain
    # Domain or method not found
    print(f"Domain '{domain}' or method '{method_name}' not found in schema.")
    return False

# ---------------------------------------------------------------------------
# Section 2 — parsing API-call text emitted by the system LLM
# ---------------------------------------------------------------------------


def extract_method_name(llm_response):
    # Extract the method name from the llm_response string
    match = re.search(r"APICall\s*\(\s*method\s*=\s*'([^']+)'\s*,", llm_response)
    if match:
        return match.group(1)
    else:
        return None

def domain_name_schema(schema_file, domain):
    with open(schema_file, 'r') as file:
        schema_data = file.read()
        domains = [line.split(':')[1].strip() for line in schema_data.splitlines() if line.startswith('service_name')]
        closest_match = difflib.get_close_matches(domain, domains, n=1, cutoff=0.7)

        if not closest_match:
            raise ValueError(f"Domain '{domain}' not found in schema. Did you mean: {', '.join(domains)}?")

        domain_corrected = closest_match[0]
        domain_start = schema_data.find(f"service_name: {domain_corrected}")
        domain_schema = schema_data[domain_start:]
        next_domain = domain_schema.find("service_name:", 1)
        if next_domain != -1:
            domain_schema = domain_schema[:next_domain]
        return domain_corrected
    
def extract_key_value_pairs(api_call_text):
    # Extract key-value pairs from the parameters in the APICall
    match = re.search(r"parameters\s*=\s*\{([^}]+)\}", api_call_text)
    if match:
        params_str = match.group(1)
        pairs = re.findall(r"(\w+):\s*([^,}]+)", params_str)
        params = {}
        for key, value in pairs:
            # Remove quotes from values if present
            value = value.strip().strip("'\"")
            params[key.strip()] = value
        if not params:
            pairs = re.findall(r"(['\"])(\w+)\1\s*:\s*(['\"])(.*?)\3", params_str)
            params = {}
            for match in pairs:
                key = match[1]
                value = match[3]
                params[key] = value
        return params
    else:
        return {}

# ---------------------------------------------------------------------------
# Section 3 — CSV-backed "service result" lookups (simulated database)
# ---------------------------------------------------------------------------


def format_service_results(service_results):
    formatted_results = []
    
    columns_to_drop = ['unnamed: 0', 'Unnamed: 0']
        
    formatted_results = [
        row.dropna().drop(columns_to_drop, errors='ignore').to_dict() 
        for _, row in service_results.iterrows()
    ]
    
    return formatted_results

def fetch_service_results_from_csv_no_params(csv_path, limit=3):
    # Fetch random service results from the CSV file
    if not os.path.exists(csv_path):
        print(f"CSV file {csv_path} does not exist.")
        return None

    df = pd.read_csv(csv_path)
    
    # If the dataframe has fewer rows than the limit, return all rows
    if len(df) <= limit:
        return df
    
    # Randomly sample 'limit' number of rows
    return df.sample(n=limit)


def _is_empty_or_dontcare(value):
    if value is None:
        return True
    normalized = str(value).strip().strip("'\"").lower()
    return normalized in {"", "dontcare", "don't care", "dont care", "any", "no preference"}


def fetch_service_results_from_csv(csv_path, query_params, limit=3):
    # Fetch service results from the CSV file based on query_params, ignoring 'date' parameter
    if not os.path.exists(csv_path):
        print(f"CSV file {csv_path} does not exist.")
        return None
    
    nullable_int_columns = {
        'party_size': 'Int64',
        'number_of_seats': 'Int64',
        'number_of_tickets': 'Int64',
        'stay_length': 'Int64',
        'amount': 'Int64',
        'transfer_amount': 'Int64'
    }

    df = pd.read_csv(csv_path, dtype=nullable_int_columns)

    # Convert column names and query_params keys to lowercase for consistent comparison
    df.columns = df.columns.str.lower()
    query_params = {k.lower(): v for k, v in query_params.items()}
    
    # get the date keys
    query_date_keys = [key for key in query_params.keys() if 'date' in key.lower()]
    db_date_keys = [col for col in df.columns if 'date' in col.lower()]

    # Filter the DataFrame based on query_params, ignoring 'date' parameter
    for key, value in query_params.items():
        if key in query_date_keys:
            continue  # Ignore the 'date' parameter during filtering
        # Empty and dontcare values mean "do not constrain this slot".
        if _is_empty_or_dontcare(value):
            continue
        value = str(value).strip().strip("'\"")
        if key in df.columns:
            
            # Use fuzzy matching for comparison
            def fuzzy_match(row_value):
                if pd.isna(row_value):
                    return False
                
                if key in nullable_int_columns:
                    row_value = int(row_value)
                    
                if isinstance(row_value, str):
                    return fuzz.partial_ratio(row_value.lower(), str(value).lower()) >= 90  # Set a threshold, e.g., 90
                else: # Non-string comparison (assume exact match for numbers)
                    return str(row_value).lower() == str(value).lower() 
            
            df = df[df[key].apply(fuzzy_match)]
            
            # df = df[df[key].astype(str).str.lower() == value.lower()]
            
            if df.empty:
                break  # No need to continue filtering if no rows are left
        else:
            # If the key is not in the columns, we can't filter on it
            continue
    if df.empty:
        return None
    else:
        # Replace the 'date' column in the fetched results with the date from query_params
        if len(query_date_keys)>0 and len(db_date_keys)>0:
            for date_key in query_date_keys:
                if date_key in db_date_keys:
                    # query_params[date_key] may be non-string
                    new_date = str(query_params[date_key]).strip("'\"")
                    df[date_key] = new_date  # Replace the 'date' values with the new date
        return df.head(limit)


def api_params_check(method_name, query_params, schema_file, domain):
    # Check for missing required parameters
    missing_required_params = []
    provided_query_params = list(query_params.keys())

    if method_name is None:
        return {
            'missing_required_params': ['method name wrong pattern'],
            'provided_query_params': provided_query_params
        }
    
    required_slots = get_required_slots(schema_file, domain, method_name)
    
    if required_slots == ['None']:
        missing_required_params = []
    elif required_slots == ["Not found"]:
        missing_required_params = ['method name not found']
    else:
        for param in required_slots:
            if param not in query_params or _is_empty_or_dontcare(query_params.get(param)):
                missing_required_params.append(param)

    return {
        'missing_required_params': missing_required_params,
        'provided_query_params': provided_query_params
    }


# ---------------------------------------------------------------------------
# Section 4 — CSV I/O for conversation logs and per-domain entry tables
# ---------------------------------------------------------------------------


def append_to_csv(csv_path, entry_dict, fieldnames):

    file_exists = os.path.exists(csv_path)

    # Open the CSV file in append mode
    with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header if file doesn't exist
        if not file_exists:
            writer.writeheader()

        # Write the entry
        writer.writerow(entry_dict)

    print(f"Appended new entry to {csv_path}")

def get_result_slots(schema_file, domain, method_name):
    # Read the schema file and return the result slots for the given method
    with open(schema_file, 'r', encoding='utf-8') as f:
        schema_content = f.read()

    # Split the schema into domains
    domains = schema_content.split('service_name:')[1:]
    for dom in domains:
        dom = dom.strip()
        if dom.startswith(domain):
            # Now find the intents within this domain
            # Split the schema into intents
            intents = dom.split('intents_')
            for intent in intents:
                if f"name: {method_name}" in intent:
                    # Find the line with result_slots
                    lines = intent.strip().split('\n')
                    for line in lines:
                        if line.strip().startswith('result_slots:'):
                            result_slots_line = line.strip()
                            # Remove 'result_slots:' and split the slots
                            slots = result_slots_line.replace('result_slots:', '').strip().split(',')
                            slots = [slot.strip() for slot in slots]
                            return slots
                    break

    return []

def get_required_slots(schema_file, domain, method_name):
    # Read the schema file and return the required slots for the given method
    with open(schema_file, 'r', encoding='utf-8') as f:
        schema_content = f.read()

    # Split the schema into domains
    domains = schema_content.split('service_name:')[1:]
    for dom in domains:
        dom = dom.strip()
        if dom.startswith(domain):
            # Now find the intents within this domain
            # Split the schema into intents
            intents = dom.split('intents_')
            for intent in intents:
                if f"name: {method_name}" in intent:
                    # Find the line with required_slots
                    lines = intent.strip().split('\n')
                    for line in lines:
                        if line.strip().startswith('required_slots:'):
                            required_slots_line = line.strip()
                            # Remove 'required_slots:' and split the slots
                            slots = required_slots_line.replace('required_slots:', '').strip().split(',')
                            slots = [slot.strip() for slot in slots]
                            return slots
                    break

    return ["Not found"]

# Save the conversation log to a CSV file
def save_conversation_to_csv(conversation_log, output_csv_file):
    fieldnames = ['Speaker', 'Utterance']
    with open(output_csv_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        # Process each conversation log entry before saving
        for entry in conversation_log:
            if ("service_result_" in entry['Utterance']):
                writer.writerow({'Speaker': 'Tool', 'Utterance': entry['Utterance']})
            else:
                writer.writerow(entry)
    
    print(f"Conversation saved to {output_csv_file}")
