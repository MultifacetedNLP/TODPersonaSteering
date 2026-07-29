import os
import json

#os.chdir('../datasets/dstc8-schema-guided-dialogue')
# Directories containing Schema.json files
input_directories = [
    '../datasets/dstc8-schema-guided-dialogue/train',
    '../datasets/dstc8-schema-guided-dialogue/test',
    '../datasets/dstc8-schema-guided-dialogue/dev'
]
output_file = 'schema.txt'

def extract_service_data(service):
    service_name = service["service_name"]
    intents = service["intents"]
    slots = service["slots"]
    intent_lines = []
    slot_lines = []

    # Loop through intents with numbering
    for idx, intent in enumerate(intents, start=1):
        required_slots = ', '.join(intent["required_slots"]) if intent["required_slots"] else "None"
        optional_slots = ', '.join(f"{key}: {value}" for key, value in intent["optional_slots"].items()) if intent["optional_slots"] else "None"
        result_slots = ', '.join(intent["result_slots"]) if intent["result_slots"] else "None"
        is_transactional = intent["is_transactional"]

        intent_lines.append(
            f'intents_{idx} \n\tname: {intent["name"]}\n\tis_transactional: {is_transactional}\n'
            f'\trequired_slots: {required_slots}\n\toptional_slots: {optional_slots}\n\tresult_slots: {result_slots}'
        )

    # Loop through slots to extract their possible values if present
    for slot in slots:
        if slot["possible_values"]:  # Only include slots with possible_values
            slot_name = slot["name"]
            possible_values = ', '.join(slot["possible_values"])
            slot_lines.append(f'\tslot_name: {slot_name}\n\t\tpossible_values: {possible_values}')

    intents_str = "\n\n".join(intent_lines)
    slots_str = "\n\n".join(slot_lines)
    return f'service_name: {service_name}\n\nIntents:\n{intents_str}\n\nSlots:\n{slots_str}\n'

# Combine schema.json files from all directories
with open(output_file, 'w') as output:
    for input_directory in input_directories:
        schema_files = [
            os.path.join(input_directory, file)
            for file in os.listdir(input_directory)
            if file.endswith("schema.json")
        ]

        for schema_file in schema_files:
            with open(schema_file, 'r') as f:
                data = json.load(f)

            for i, service in enumerate(data):
                service_info = extract_service_data(service)
                output.write(service_info)
                
                # Add a separator between services, except for the last service in the last file
                if not (schema_file == schema_files[-1] and i == len(data) - 1 and input_directory == input_directories[-1]):
                    output.write("\n\n")

print(f"Data extracted and combined into {output_file}")
