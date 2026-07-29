import os
import csv
import yaml
import openai
import argparse
from groq import Groq
import pandas as pd
from dotenv import load_dotenv
from .utils import extract_method_name, get_result_slots, append_to_csv, domain_name_schema, check_is_transactional, load_schema, api_params_check, save_conversation_to_csv, fetch_service_results_from_csv, format_service_results, extract_key_value_pairs

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
openai.api_key = openai_api_key

# Initialize Groq client
client = Groq(api_key=groq_api_key)

# Load instruction prompt from YAML file and replace domain name
def load_instruction(instruction_file, domain, api_call_example):
    with open(instruction_file, 'r') as file:
        instruction_data = yaml.safe_load(file)
        instruction_prompt = instruction_data.get('instruction_prompt', 'No instruction prompt found.')

        # Replace {domain_name} and API call example in the instruction prompt
        instruction_prompt = instruction_prompt.replace("{domain_name}", domain)
        instruction_prompt = instruction_prompt.replace(
            "APICall (method='intent_name', parameters= {required_slot_1: value, required_slot_2: value})",
            api_call_example)

        return instruction_prompt

# Load the full conversation from the user-provided CSV file and search for an API call
def load_full_conversation(csv_file):
    full_conversation = []
    api_call_example = None

    with open(csv_file, newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            full_conversation.append(row)

            # Search for API call in the conversation
            if "APICall" in row['Utterance']:  # Adjust the condition if API calls have a specific format in your CSV
                api_call_example = row['Utterance'].replace("SYSTEM,", "").strip()

    if not full_conversation:
        raise ValueError(f"No conversations found in the CSV file.")

    if not api_call_example:
        api_call_example = "No API call found."

    return full_conversation, api_call_example

def create_chat_gpt_input(instruction_prompt, schema, conversation):
    lines = [
        instruction_prompt,
        "",
        "Schema:",
        "",
        schema,
        "end of schema",
        "",
        "Here is a sample Dialog conversation between a SYSTEM and a USER:",
        ""
    ]

    for row in conversation:
        lines.append(f"{row['Speaker']}: {row['Utterance']}")

    lines.extend([
        "End of Conversation",
        "",
        "Understand the above structure of conversation between a USER and a SYSTEM. Learn how to interact with the USER and generate the most human-like conversational response to the user's intent. You may need to make API calls and use the API call results.",
        "Based on the above instructions and examples from the Restaurant domain, learn how to interact with a user to generate the most human-like conversational response to the user's current intent.",
        "End of Instructions.",
        "",
        "Here are a few general guidelines to follow:",
        "- Don't overwhelm the USER with too many choices.",
        "- Confirm the slot values with the USER before finalizing the call.",
        "- Follow the structure of API call from the above example whenever you are making an API Call.",
        "- If you're unsure about something, it's always better to ask or confirm with the user.",
        "- If you feel the user is confused, guide the user with relevant suggestions and ensure it's relevant to their current intent.",
        "",
        "Here is a Conversation history:"
    ])

    chat_GPT_input = "\n".join(lines)
    return chat_GPT_input

# Function to send the prompt to Groq's LLM and get the response
def call_groq_llm(prompt):
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-groq-70b-8192-tool-use-preview"
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        raise Exception(f"Error while calling Groq LLM: {e}")

# Function to send the prompt to OpenAI's API and get the response
def call_openai_llm(prompt):
    try:
        response = openai.Completion.create(
            engine="GPT-4o",  
            prompt=prompt,
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.7
        )
        return response.choices[0].text.strip()
    except Exception as e:
        raise Exception(f"Error while calling OpenAI API: {e}")

def user_simulator(user_inputs):
    """
    Simulates a user by providing a list of string inputs one by one.

    Args:
        user_inputs (list): A list of user input strings to simulate the conversation.
    
    Yields:
        str: The next user input string in the sequence.
    """
    for user_input in user_inputs:
        yield user_input

def my_system_response_handler(response):
    # Process the SYSTEM response here
    # For example, log it, analyze it, or send it to another service
    print(f"Processed SYSTEM response: {response}")

def interactive_chat_with_llm_user(
    conversation_log,
    domain,
    schema_file,
    api_type,
    response_source,
    user_simulator=None,
    dialog_csv=None,
    system_response_handler=None
):

    domain_name = domain_name_schema(schema_file, domain)
    service_results_path = os.path.join("service_results/", f"{domain_name}.csv")

    if not os.path.exists(service_results_path):
        raise FileNotFoundError(f"Service results CSV for domain '{domain_name}' not found at {service_results_path}")

    print(f"Using service results from: {service_results_path}")

    if user_simulator is not None:
        input_generator = user_simulator
    else:
        raise ValueError("user_simulator cannot be None when using simulated input")

    # Load the dialog CSV (if response_source is 'csv')
    conversation_entries = []
    if dialog_csv and response_source == 'csv':
        with open(dialog_csv, newline='', encoding='utf-8') as dialog_file:
            reader = csv.reader(dialog_file)
            for row in reader:
                if not row or not row[0].strip():  # Skip empty lines
                    continue
                if row[0].startswith('dialogue_id:'):
                    continue  # Skip dialogue_id lines
                speaker = row[0].strip()
                utterance = row[1].strip() if len(row) > 1 else ''
                if speaker and utterance:
                    conversation_entries.append({'Speaker': speaker, 'Utterance': utterance})

    index = 0  # Initialize index to 0

    while True:
        # Process USER input
        if input_generator:
            try:
                user_input = next(input_generator).strip()
                print(f"\nUSER: {user_input}")
            except StopIteration:
                print("Simulation finished.")
                break
        else:
            user_input = input("\nUSER: ").strip()

        if user_input.lower() == 'exit':
            print("Exiting the chat.")
            break

        # Add the user input to the conversation log
        conversation_log.append({'Speaker': 'USER', 'Utterance': user_input})

        # Generate SYSTEM response
        if response_source == 'llm':
            # Call the appropriate LLM API based on api_type
            try:
                if api_type == 'groq':
                    llm_response = call_groq_llm(user_input)
                elif api_type == 'openai':
                    llm_response = call_openai_llm(user_input)
                else:
                    raise ValueError(f"Unsupported api_type: {api_type}")
            except Exception as e:
                print(f"Error during LLM call: {e}")
                break

            if llm_response:
                # Send the SYSTEM response to the handler if provided
                if system_response_handler:
                    system_response_handler(llm_response)
                else:
                    print(f"SYSTEM: {llm_response}")

                conversation_log.append({'Speaker': 'SYSTEM', 'Utterance': llm_response})

                # Handle APICall in the LLM response
                if "APICall" in llm_response:
                    process_api_call(
                        llm_response,
                        conversation_log,
                        service_results_path,
                        schema_file,
                        domain,
                        system_response_handler  # Pass the handler
                    )
        elif response_source == 'csv':
            # Use the next SYSTEM response from the CSV
            while index < len(conversation_entries):
                current_entry = conversation_entries[index]
                index += 1  # Move to next entry
                if current_entry['Speaker'] == 'SYSTEM':
                    llm_response = current_entry['Utterance']
                    # Send the SYSTEM response to the handler if provided
                    if system_response_handler:
                        system_response_handler(llm_response)
                    else:
                        print(f"SYSTEM: {llm_response}")

                    conversation_log.append({'Speaker': 'SYSTEM', 'Utterance': llm_response})

                    # Handle APICall in the CSV response
                    if "APICall" in llm_response:
                        process_api_call(
                            llm_response,
                            conversation_log,
                            service_results_path,
                            schema_file,
                            domain,
                            system_response_handler  # Pass the handler
                        )
                    break  # Move to the next USER input
        else:
            raise ValueError(f"Unsupported response_source: {response_source}")

    print("Conversation ended.")

def process_api_call(
    llm_response,
    conversation_log,
    service_results_path,
    schema_file,
    domain,
    system_response_handler=None
):
    query_params = extract_key_value_pairs(llm_response)
    api_check = api_params_check(llm_response, query_params, schema_file, domain)

    if api_check['missing_required_params']:
        print(f"Missing required parameters in the API call: {api_check['missing_required_params']}")
        return

    if api_check['provided_query_params']:
        print(f"Provided query parameters: {query_params}")

        # Extract the method name from the APICall
        method_name = extract_method_name(llm_response)
        if not method_name:
            print("Could not extract method name from the APICall.")
            return

        # Check if the method is transactional
        is_transactional = check_is_transactional(schema_file, domain, method_name)
        print(f"Method '{method_name}' is transactional: {is_transactional}")

        print(f"Fetching service results for {llm_response}")
        conversation_log.append({'Speaker': 'SYSTEM', 'Utterance': llm_response})

        # Fetch the service results after the API call
        service_results = fetch_service_results_from_csv(service_results_path, query_params, limit=3)

        if service_results is not None and not service_results.empty:
            # Format the service results for display
            formatted_service_results = format_service_results(service_results)
            for service_result in formatted_service_results:
                # Send the service result to the handler if provided
                if system_response_handler:
                    system_response_handler(service_result)
                else:
                    print(f"SYSTEM: {service_result}")
                conversation_log.append({'Speaker': 'SYSTEM', 'Utterance': service_result})
        else:
            no_results_message = "No service results found."
            if system_response_handler:
                system_response_handler(no_results_message)
            else:
                print(f"SYSTEM: {no_results_message}")
            conversation_log.append({'Speaker': 'SYSTEM', 'Utterance': no_results_message})

            if is_transactional:
                # Make a service entry to the CSV file using values from APICall parameters
                print("Making a service entry to the CSV file because the method is transactional and no results were found.")
                # Construct the new service entry from the query_params
                new_entry = query_params.copy()

                # Add missing result slots with empty values
                result_slots = get_result_slots(schema_file, domain, method_name)
                for slot in result_slots:
                    if slot not in new_entry:
                        new_entry[slot] = ''

                # Ensure all columns are present in the new entry
                # Load existing service results to get all columns
                if os.path.exists(service_results_path):
                    df_existing = pd.read_csv(service_results_path)
                    all_columns = df_existing.columns.tolist()
                else:
                    # If CSV doesn't exist, create columns based on result_slots
                    all_columns = result_slots

                # Reorder new_entry to match the columns
                ordered_entry = {col: new_entry.get(col, '') for col in all_columns}

                # Append the new entry to the CSV file
                append_to_csv(service_results_path, ordered_entry, all_columns)

                # Inform ChatGPT that the record has been inserted
                insertion_message = "The new record has been inserted into the database."
                if system_response_handler:
                    system_response_handler(insertion_message)
                else:
                    print(f"SYSTEM: {insertion_message}")
                conversation_log.append({'Speaker': 'SYSTEM', 'Utterance': insertion_message})

                # No need to generate ChatGPT's response here; it will be handled in the main loop
    else:
        print("No provided query parameters to process.")


def main():
    # Define arguments and default values using argparse
    parser = argparse.ArgumentParser(description="Interactive LLM Chat Script")
    parser.add_argument("--domain", default="Restaurants", help="The domain for the conversation (default: Restaurant)")
    parser.add_argument("--conversation_csv", default="sample_conversation.csv")
    parser.add_argument("--dialog_csv", default="data/sgd/train/sample_dialogues_002_conversations.csv",
                        help="Path to the dialog input CSV file")
    parser.add_argument("--input_type", choices=["csv", "manual", "simulated"], default="simulated")
    parser.add_argument("--api_type", choices=["groq", "openai"], default="groq")
    parser.add_argument("--response_source", choices=["llm", "csv"], default="csv",
                        help="Choose whether to use the LLM or SYSTEM inputs from the dialog CSV (default: llm)")
    parser.add_argument("--simulated_inputs", nargs="*", default=[
                                                                "Let's search places to eat please.",
                                                                "Search in Albany for take-out food please.",
                                                                "Find out if they have live music. Also how expensive is this restaurant?",
                                                                "Yeah, that will do just fine.",
                                                                "Yes go ahead and make reservations.",
                                                                "Please make it for seven pm.",
                                                                "No wait. This is for 1 person. Also update it to the 13th of March.",
                                                                "No, Make it for four people on the 10th please.",
                                                                "Yep. What's the phone number of the restaurant please?",
                                                                "Lets try a different time to book the reservation. Try for five in the evening.",
                                                                "Yes, perfect. Go ahead and book it.",
                                                                "Thanks. that will be all.",
                                                                "No. thats it!"], help="Simulated user inputs")

    # Parse arguments
    args = parser.parse_args()

    domain = args.domain
    conversation_csv = args.conversation_csv
    dialog_csv = args.dialog_csv
    input_type = args.input_type
    api_type = args.api_type
    response_source = args.response_source
    simulated_inputs = args.simulated_inputs

    schema_file = 'schema.txt'
    instruction_file = 'instruction_pr.yml'

    if args.input_type == 'csv':
        output_csv_file = f'output_conversations/chat_out_{dialog_csv.split("/")[-1]}'
    if args.input_type == 'manual':
        output_csv_file = f'output_conversations/chat_{api_type}_manual_user.csv'
    if args.input_type == 'simulated':
        output_csv_file = f'output_conversations/chat_{api_type}_simulated_user.csv'

    # Load schema
    schema = load_schema(schema_file, domain)

    # Load the full conversation and extract an API call (if applicable)
    conversation, api_call_example = load_full_conversation(conversation_csv)

    # Load instruction prompt and replace placeholders
    instruction_prompt = load_instruction(instruction_file, domain, api_call_example)

    # Create chat_GPT_input variable
    chat_GPT_input = create_chat_gpt_input(instruction_prompt, schema, conversation)
    print("\n")
    print(chat_GPT_input)
    print("\n")

    # Initialize the conversation log
    conversation_log = []

    simulated_user_inputs = user_simulator(simulated_inputs)

    # Perform interactive chat with LLM using the chosen input method

    if input_type == 'manual':
        interactive_chat_with_llm_user(conversation_log, domain, schema_file, api_type, response_source, user_simulator=None)
    elif input_type == 'simulated':
        interactive_chat_with_llm_user(conversation_log, domain, schema_file, api_type, response_source, simulated_user_inputs, dialog_csv, system_response_handler=my_system_response_handler)
    else:
        print("Invalid input type selected.")
        return

    # Save the entire conversation log to a CSV file
    save_conversation_to_csv(conversation_log, output_csv_file)

if __name__ == "__main__":
    main()