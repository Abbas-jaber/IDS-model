from subprocess import call
import subprocess
import sys
import os
# --- Configuration: Define paths to your scripts ---
# Adjust these paths if the scripts are not in the same directory as this launcher script
CLEANING_SCRIPT_PATH = "clean.py"
COMBINING_SCRIPT_PATH = "combining.py"
CLEANING_SCRIPT_V2_PATH = "clean2.py"
TRAINING_DNN_SCRIPT = "Model_training_DNN.py"
TRAINING_LSTM_SCRIPT = "Model_training_LSTM.py"
# Add paths for other scripts here later (e.g., training, evaluation)

def call_script(script_path, *args):
    """
    Calls an external Python script using subprocess.run, passing arguments.
    Uses the same Python interpreter that's running this launcher.
    Allows the called script's output to print directly to the terminal.
    """
    if not os.path.exists(script_path):
        print(f"\nError: Script not found at '{script_path}'. Cannot execute.")
        return

    command = [sys.executable, script_path] + list(args)
    print(f"\n--- Running {os.path.basename(script_path)} ---")
    print(f"Command: {' '.join(command)}")

    try:
        result = subprocess.run(command, check=True)
        print(f"\n--- {os.path.basename(script_path)} finished ---")
        print(f"Return code: {result.returncode}")

    except FileNotFoundError:
        print(f"Error: Python interpreter not found at '{sys.executable}'.")
    except subprocess.CalledProcessError as e:
        print(f"\n--- Error running {os.path.basename(script_path)} ---")
        print(f"Return code: {e.returncode}")
        print("Check the terminal output above for error messages from the script.")
    except Exception as e:
        print(f"An unexpected error occurred while trying to run {script_path}: {e}")


def main():
  print("Welcome to Cortex Guard - A machine learning Framework built to produce IDS trained models")
  while True: # Main loop to keep asking the user [3, 4, 5, 6]
        print("\nPlease choose the operation you want the framework to perform:")
        print("  [1] Clean Data")
        print("  [2] Combine Data")
        print("  [3] Clean Data (Version 2 - Epoch Check)")
        print("  [4] Train Detection Model")
        # Add more options here as you integrate other scripts
        print("  [q] Quit")

        choice = input("Enter your choice: ").strip().lower() # Get user input [7, 8, 9, 10]

        if choice == '1':
            print("\nSelected: Clean Data")
            # --- Argument Handling for cleaning.py ---
            # Ask the user how they want to run the clean script
            # (This assumes cleaning.py accepts arguments like 'all' or specific file paths)
            clean_mode = input("Clean (a)ll files or a (s)pecific file? (a/s): ").strip().lower()
            if clean_mode == 'a':
                # If cleaning.py expects 'all' as an argument to clean everything
                call_script(CLEANING_SCRIPT_PATH, "all")
                # If cleaning.py needs input/output dirs even for 'all', prompt for them:
                # input_dir = input("Enter base directory containing raw CSVs: ")
                # output_dir = input("Enter base directory for cleaned output: ")
                # call_script(CLEANING_SCRIPT_PATH, input_dir, output_dir, "--all") # Example if it uses options
            elif clean_mode == 's':
                input_file = input("Enter the path to the specific CSV file to clean: ")
                output_file_base = input("Enter the desired base name for the output file (e.g., 'cleaned_data'): ")
                # Assuming cleaning.py takes input file and output base name as arguments
                call_script(CLEANING_SCRIPT_PATH, input_file, output_file_base)
            else:
                print("Invalid cleaning mode selected.")
            # --- End Argument Handling ---

        elif choice == '2':
            print("\nSelected: Combine Data")
            # --- Argument Handling for combining.py ---
            # Assuming combining.py now takes arguments like input files and output dir/name
            print("Note: The combine script expects specific input files and an output directory.")
            call_script(COMBINING_SCRIPT_PATH) # Call without args if combining.py handles its inputs internally
            # --- End Argument Handling ---

        elif choice == '3': # Added block for clean2.py
            print("\nSelected: Clean Data (Version 2 - Epoch Check)")
            # --- Argument Handling for clean2.py ---
            clean_mode = input("Clean a (s)pecific file (s): ").strip().lower()
            if clean_mode == 's':
                input_file = input("Enter the path to the specific CSV file to clean: ")
                output_file_base = input("Enter the desired base name for the output file (e.g., 'cleaned_data_v2'): ")
                output_directory = input("Enter the desired output directory name for the output file (e.g., 'Binary-class'): ")
                # Assuming clean2.py takes input file, output base name, and output directory
                call_script(CLEANING_SCRIPT_V2_PATH, input_file, output_file_base, output_directory)
            else:
                print("Invalid cleaning mode selected.")
            # --- End Argument Handling ---
        elif choice == '4':
            print("\nSelected: Model Training")
            print("Available Architectures:")
            print("  [1] Deep Neural Network (DNN)")
            print("  [2] Long Short-Term Memory (LSTM)")
            
            model_choice = input("Select architecture: ").strip()
            
            if model_choice in ['1', '2']:
                input_file = input("Training data filename (e.g., cleaned_data.csv): ")
                if model_choice == '1':
                    call_script(TRAINING_DNN_SCRIPT, input_file)
                else:
                    call_script(TRAINING_LSTM_SCRIPT, input_file)
            else:
                print("Invalid architecture selection")

        elif choice == 'q':
            print("\nExiting Cortex Guard. Goodbye!")
            break # Exit the while loop [3, 4]
        else:
            print("\nInvalid choice. Please try again.")

        # Pause briefly before showing the menu again
        input("\nPress Enter to return to the main menu...")

if __name__ == "__main__":
    # This ensures the main() function runs only when the script is executed directly
    main()
