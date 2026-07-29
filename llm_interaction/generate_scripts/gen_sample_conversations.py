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

# Collect all services used in the dataset
all_services = set()
for dialogue in dialogues:
    services_in_dialogue = dialogue.get('services', [])
    all_services.update(services_in_dialogue)

if len(all_services) == 1:
    # All dialogues have the same service_name, save just the first dialogue to CSV
    service_name = list(all_services)[0]
    # Get the first dialogue
    dialogue = dialogues[0]
    # Process the dialogue and save to CSV
    rows = []
    dialogue_id = dialogue['dialogue_id']
    rows.append({'Speaker': 'dialogue_id:', 'Utterance': f' {dialogue_id}'})
    
    for turn in dialogue['turns']:
        speaker = turn['speaker']
        utterance = turn['utterance']
        rows.append({'Speaker': speaker, 'Utterance': f'"{utterance}"'})
        
        for frame in turn.get('frames', []):
            # Process service calls
            if 'service_call' in frame:
                service_call = frame['service_call']
                intent = service_call['method']
                parameters = service_call['parameters']
                parameters_str = ', '.join([f"{k}: {v}" for k, v in parameters.items()])
                rows.append({
                    'Speaker': 'SYSTEM',
                    'Utterance': f'"APICall(method=\'{intent}\', parameters={{ {parameters_str} }})"'
                })
            # Process service results
            if 'service_results' in frame:
                service_results = frame['service_results'][:2]  # Limit to first two results
                result_strings = []
                for result in service_results:
                    result_str = ', '.join([f"{k}: {v}" for k, v in result.items()])
                    result_strings.append(f"{{ {result_str} }}")
                all_results_str = ', '.join(result_strings)
                formatted_results = f'"Search Results\\n[{all_results_str}]\\nEnd Search Results"'
                rows.append({
                    'Speaker': 'Tool',
                    'Utterance': formatted_results
                })
    # Convert rows into a DataFrame
    df = pd.DataFrame(rows)
    # Save DataFrame to CSV
    output_csv_path = f"{service_name}_conversation.csv"
    df.to_csv(output_csv_path, index=False)
    print(f"Conversation has been saved to {output_csv_path}")

else:
    # Multiple services, save a new CSV file for each service_name
    services_processed = set()
    for dialogue in dialogues:
        services_in_dialogue = dialogue.get('services', [])
        for service_name in services_in_dialogue:
            if service_name not in services_processed:
                # Process and save the dialogue
                rows = []
                dialogue_id = dialogue['dialogue_id']
                rows.append({'Speaker': 'dialogue_id:', 'Utterance': f' {dialogue_id}'})
                
                for turn in dialogue['turns']:
                    speaker = turn['speaker']
                    utterance = turn['utterance']

                    
                    for frame in turn.get('frames', []):
                        # Process only frames for the current service
                        if frame.get('service') == service_name:
                            # Process service calls
                            if 'service_call' in frame:
                                service_call = frame['service_call']
                                intent = service_call['method']
                                parameters = service_call['parameters']
                                parameters_str = ', '.join([f"{k}: {v}" for k, v in parameters.items()])
                                rows.append({
                                    'Speaker': 'SYSTEM',
                                    'Utterance': f'"APICall(method=\'{intent}\', parameters={{ {parameters_str} }})"'
                                })
                            # Process service results
                            if 'service_results' in frame:
                                service_results = frame['service_results'][:2]  # Limit to first two results
                                result_strings = []
                                for result in service_results:
                                    result_str = ', '.join([f"{k}: {v}" for k, v in result.items()])
                                    result_strings.append(f"{{ {result_str} }}")
                                all_results_str = ', '.join(result_strings)
                                formatted_results = f'"Search Results\\n[{all_results_str}]\\nEnd Search Results"'
                                rows.append({
                                    'Speaker': 'Tool',
                                    'Utterance': formatted_results
                                })

                    rows.append({'Speaker': speaker, 'Utterance': f'"{utterance}"'})                                
                # Convert rows into a DataFrame
                df = pd.DataFrame(rows)
                # Save DataFrame to CSV
                output_csv_path = f"{service_name}_conversation.csv"
                df.to_csv(output_csv_path, index=False)
                print(f"Conversation for service '{service_name}' has been saved to {output_csv_path}")
                # Mark this service as processed
                services_processed.add(service_name)
                # Break out of the loop since we have saved a dialogue for this service
                break
        # If we have processed all services, break out of the loop
        if len(services_processed) == len(all_services):
            break

print("Conversations have been saved.")
