import os
import json
import pandas as pd
import argparse

# Create an argument parser
parser = argparse.ArgumentParser(description="Generate conversations from JSON dialogues and save to CSV.")
parser.add_argument('dataset_path', type=str, help="Path to the JSON dataset file (e.g., train/dialogues_002.json)")

# Parse the arguments
args = parser.parse_args()

# Extract the dataset path from the arguments
dataset_path = args.dataset_path

# Change directory to where the JSON file is located
os.chdir(os.path.dirname(dataset_path))

# Load JSON data
with open(os.path.basename(dataset_path), 'r') as f:
    dialogues = json.load(f)

# Extract relevant information including service calls and service results
rows = []

for dialogue in dialogues:
    dialogue_id = dialogue['dialogue_id']  # Get the dialogue_id from the JSON file
    rows.append({'Speaker': 'dialogue_id:', 'Utterance': f' {dialogue_id}'})
    
    for turn in dialogue['turns']:
        speaker = turn['speaker']
        utterance = turn['utterance']
        
        # Append the speaker's utterance
        rows.append({
            'Speaker': speaker,
            'Utterance': f'"{utterance}"'  # Enclose utterance in quotes
        })
        
        # Process frames after the utterance
        for frame in turn.get('frames', []):
            # Check for service calls
            if 'service_call' in frame:
                service_call = frame['service_call']
                intent = service_call['method']
                parameters = service_call['parameters']
                parameters_str = ', '.join([f"{k}: {v}" for k, v in parameters.items()])
                
                # Add the ApiCall entry with speaker 'SYSTEM'
                rows.append({
                    'Speaker': 'SYSTEM',
                    'Utterance': f'"APICall(method=\'{intent}\', parameters={{ {parameters_str} }})"'
                })
            
            # Check for service results and limit to the first two
            if 'service_results' in frame:
                service_results = frame['service_results'][:2]  # Limit to the first two results
                
                result_strings = []
                for result in service_results:
                    result_str = ', '.join([f"{k}: {v}" for k, v in result.items()])
                    result_strings.append(f"{{ {result_str} }}")
                
                # Combine all results into one string
                all_results_str = ', '.join(result_strings)
                formatted_results = f'"Search Results\\n[{all_results_str}]\\nEnd Search Results"'
                
                rows.append({
                    'Speaker': 'Tool',
                    'Utterance': formatted_results
                })
    
    # After the dialogue is complete, add two empty rows for spacing between conversations
    rows.append({'Speaker': '', 'Utterance': ''})
    rows.append({'Speaker': '', 'Utterance': ''})

# Convert rows into a DataFrame
df = pd.DataFrame(rows)

# Save DataFrame to CSV
output_csv_path = f"{os.path.basename(dataset_path).split('.')[0]}_conversations.csv"
df.to_csv(output_csv_path, index=False)

print(f"Conversations have been saved to {output_csv_path}")
