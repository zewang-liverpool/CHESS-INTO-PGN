import os
import subprocess
import sys

def print_menu():
    print("\n" + "="*65)
    print(" Automated Chess Video to PGN Extraction Pipeline (Pro Edition) ")
    print("="*65)
    print(" Select an execution module:\n")
    
    print(" [Phase 1: Data Acquisition & Preparation]")
    print("   1. Automated Data Collection (For standard opening configurations)")
    print("   2. Manual Assisted Collection (For non-standard or partial configurations)")
    print("")
    
    print(" [Phase 2: Model Training]")
    print("   3. Train Neural Network (Generates serialized .pth weights)")
    print("")
    
    print(" [Phase 3: Inference & Extraction]")
    print("   4. Static Camera Extraction (Recommended for stable, unmoving footage)")
    print("   5. Dynamic Camera Extraction (Fallback for footage with panning or jitter)")
    print("")
    
    print(" [System]")
    print("   0. Terminate Pipeline")
    print("="*65)

def run_script(script_name):
    """Engine for securely executing sub-scripts as distinct processes."""
    if not os.path.exists(script_name):
        print(f"\nFatal Error: Target file '{script_name}' not found.")
        print("Please verify that the file exists in the current working directory and matches the expected naming convention.")
        return
        
    print(f"\nInitializing execution engine for: {script_name}...\n")
    print("-" * 50)
    
    try:
        # sys.executable references the absolute path of the current Python interpreter binary
        subprocess.run([sys.executable, script_name], check=True)
    except KeyboardInterrupt:
        print("\n\nExecution manually terminated by the user (KeyboardInterrupt).")
    except subprocess.CalledProcessError as e:
        print(f"\nExecution aborted due to an internal exception (Exit code: {e.returncode}).")
        
    print("-" * 50)
    print(f"Module '{script_name}' execution completed.")

def main():
    while True:
        print_menu()
        choice = input("Enter module sequence number (0-5) and press Return: ").strip()

        if choice == '1':
            run_script("01_auto_data_collector.py")
        elif choice == '2':
            run_script("01_manual_data_collector.py")
        elif choice == '3':
            run_script("02_model_trainer.py")
        elif choice == '4':
            run_script("03_extract_static_camera.py")
        elif choice == '5':
            run_script("03_extract_dynamic_camera.py")
        elif choice == '0':
            print("\nTerminating the extraction pipeline. Goodbye.\n")
            break
        else:
            print("\nInvalid input exception. Please enter an integer between 0 and 5.")

if __name__ == "__main__":
    # Clear terminal buffer (Cross-platform compatibility for Windows/POSIX architecture)
    os.system('cls' if os.name == 'nt' else 'clear')
    main()