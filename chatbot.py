import ollama
def ask_chatbot(question, context=""):
    prompt = f"{context}\n\nUser: {question}"
    try:
        response = ollama.generate(
            model='deepseek-r1:7b',
            prompt=prompt
        )
        return response['response']
    except Exception as e:
        return f"Chatbot error: {str(e)}"

def clean_and_process_dataset():
    print("Dataset cleaning and processing complete!")
    return """
    The dataset was cleaned using the following steps:
    1. Rows starting with 'D' or containing 'Infinity' or 'infinity' were dropped.
    2. Dates in the third column were converted to Unix timestamps.
    3. The data was grouped by the last column (key) and saved into separate files.
    4. A binary-class version of the dataset was created by mapping labels to 0 (Benign) or 1 (Attack).
    """

def start_chatbot():
    """Handles chatbot interaction after cleaning."""
    cleaning_summary = clean_and_process_dataset()
    print("\nWhat would you like to know about the cleaning process?")
    print("Type 'exit' to quit.")
    return cleaning_summary  # Return the summary so it can be used later

def main():
    cleaning_summary = start_chatbot()  # Store the returned summary
    
    while True:
        try:
            user_input = input("\nYour question: ").strip()
            if user_input.lower() == 'exit':
                print("Exiting chatbot. Goodbye!")
                break
            if not user_input:  # Handle empty input
                continue
                
            response = ask_chatbot(user_input, cleaning_summary)
            print(f"\nChatbot: {response}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()