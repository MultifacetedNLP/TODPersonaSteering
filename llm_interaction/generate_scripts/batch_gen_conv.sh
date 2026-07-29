#!/bin/bash

# Check if the user provided a path argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <path_to_json_files>"
    exit 1
fi

# Assign the first argument to the variable JSON_PATH
JSON_PATH=$1

# Check if the provided path exists
if [ ! -d "$JSON_PATH" ]; then
    echo "The directory $JSON_PATH does not exist."
    exit 1
fi

# Loop through all JSON files in the provided directory
for json_file in "$JSON_PATH"/*.json; do
    # Get the base filename without extension
    base_filename=$(basename "$json_file" .csv)

    # Call the Python script for each JSON file and generate conversations
    python3 gen_conversations.py "$json_file"
    
    # Save the output with the same naming convention
    output_file="${base_filename}_conversations.csv"

    # Assuming the Python script is saving the output in the same directory
    if [ -f "$output_file" ]; then
        echo "Generated: $output_file"
    else
        echo "Failed to generate: $output_file"
    fi
done


#example run: chmod +x generate_conversations.sh then ./generate_conversations.sh /path/to/csv/files