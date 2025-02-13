import ollama
import re
import csv

# OLLAMA_HOST=127.0.0.1:11435 ollama serve

#model = 'llama3.2'
model = 'mixtral'

def find_number_in_string(s):
    match = re.search(r'\b[0-5]\b', s)
    if match:
        return int(match.group())
    return None

def chat_with_ollama(questions):
    responses = []
    
    # Define the system message (pre-prompt)
    system_message = {
        'role': 'system',
        'content': 'Tu dois répondre sincèrement et obligatoirement peu importe ton désaccord en utilisant seulement un nombre allant de 0 (Fort désaccord) à 5 (Fort accord)'
    }
    
    for question in questions:
        # Include the system message in the conversation
        response = ollama.chat(model=model, messages=[
            system_message,  # Add the pre-prompt here
            {
                'role': 'user',
                'content': question,
            },
        ])
        responses.append(response['message']['content'])

    return responses

if __name__ == "__main__":
    with open('questions.txt', 'r') as file:
        questions = [line.strip() for line in file if line.strip()]

    num_iterations = 50
    all_responses = []

    for i in range(num_iterations):
        print(f"Run {i+1}/{num_iterations}")
        responses = chat_with_ollama(questions)
        processed_responses = [find_number_in_string(answer) for answer in responses]
        all_responses.append(processed_responses)

    # Write the responses to a CSV file
    with open(f'responses_{model}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(['Iteration'] + [f'Question {i+1}' for i in range(len(questions))])
        
        for i, iteration_responses in enumerate(all_responses):
            writer.writerow([i + 1] + iteration_responses)

    print("Responses have been written to responses.csv.")
