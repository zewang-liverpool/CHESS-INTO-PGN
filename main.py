import subprocess
import sys
import time

def run_script(script_name):
    """
    Executes a script as a subprocess.
    Uses sys.executable to ensure the subprocess shares the same Python virtual environment as main.py.
    """
    print(f"\n[{time.strftime('%H:%M:%S')}] 🚀 [System] Launching module: {script_name}...")
    try:
        # check=True ensures that if the subprocess fails, an exception is raised in the main process
        subprocess.run([sys.executable, script_name], check=True)
        print(f"[{time.strftime('%H:%M:%S')}] ✅ [System] Module {script_name} executed successfully.\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[{time.strftime('%H:%M:%S')}] ❌ [Error] Module {script_name} crashed or was interrupted.")
        return False
    except FileNotFoundError:
        print(f"[{time.strftime('%H:%M:%S')}] ❌ [Error] File not found: {script_name}. Verify the working directory.")
        return False

def path_auto_pipeline():
    """Pipeline 1: Automated Data Ingestion + Cyclic Training"""
    print("\n" + "-"*40)
    print(" 🛠️  Activating Pipeline 1: [Automated Data Flow & Cyclic Training]")
    print("-"*40)
    
    # 1. Execute automated data collection
    if not run_script("01_auto_data_collector.py"):
        return
    
    # 2. Retrieve user-defined epoch cycles
    try:
        train_count = int(input("👉 Data ingestion complete. Enter the number of training cycles (e.g., 3): "))
        if train_count <= 0:
            print("⚠️ Cycle count must be > 0. Operation aborted.")
            return
    except ValueError:
        print("❌ Invalid input. Integer required. Operation aborted.")
        return
        
    # 3. Iteratively execute the training script
    for i in range(train_count):
        print(f"\n>>> 🔄 Executing Training Cycle {i+1}/{train_count} <<<")
        if not run_script("02_model_trainer.py"):
            print("❌ Training interrupted. Halting subsequent cycles.")
            break
            
    print("\n🎉 [Success] Pipeline 1 execution completed!")

def path_manual_pipeline():
    """Pipeline 2: Manual Data Collection + Single-pass Training"""
    print("\n" + "-"*40)
    print(" 🖐️  Activating Pipeline 2: [Human-in-the-Loop Annotation & Training]")
    print("-"*40)
    
    if not run_script("01_manual_data_collector.py"):
        return
        
    print("\n>>> 🚀 Initiating model training on manually annotated data <<<")
    run_script("02_model_trainer.py")
    
    print("\n🎉 [Success] Pipeline 2 execution completed!")

def main():
    while True:
        print("\n" + "="*55)
        print("      🤖 Chess AI End-to-End CLI (ML Pipeline UI)")
        print("="*55)
        print(" [1] Automated Pipeline: 01_auto_data_collector -> 02_model_trainer (Iterative)")
        print(" [2] Manual Pipeline: 01_manual_data_collector -> 02_model_trainer (Single-pass)")
        print(" [0] Terminate System (Exit)")
        print("="*55)
        
        choice = input("Select workflow (0/1/2): ")
        
        if choice == '1':
            path_auto_pipeline()
        elif choice == '2':
            path_manual_pipeline()
        elif choice == '0':
            print("👋 [System] CLI terminated. Good luck with your presentation!")
            break
        else:
            print("⚠️ Invalid selection. Please try again.")

if __name__ == "__main__":
    main()