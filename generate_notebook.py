import json
import os

# Core modules to be integrated into the Jupyter Notebook
files_to_convert = [
    "01_auto_data_collector.py",
    "02_model_trainer.py",
    "03_extract_static_camera.py"
]

# Base JSON schema for the Jupyter Notebook
notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Automated Chess Video to PGN Extraction Pipeline\n",
                "\n",
                "This notebook provides an automated end-to-end machine learning workflow. Execute the cells sequentially to perform the following stages: **Data Acquisition -> Model Training -> Game State Extraction**.\n",
                "\n",
                "> **OpenCV GUI Warning:** Executing cells containing `cv2.imshow` will initialize an external GUI window. Upon completion of the required interactions, strictly use the `Esc` key to terminate the window. **Do not force-close the window using the OS close button**, as this may result in a kernel deadlock."
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Iteratively read Python scripts and parse them into Notebook code cells
for py_file in files_to_convert:
    if os.path.exists(py_file):
        # Append a Markdown cell for section titling
        notebook["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [f"## Execute Module: `{py_file}`"]
        })
        
        # Ingest script content and append as a Code cell
        with open(py_file, "r", encoding="utf-8") as f:
            code_lines = f.readlines()
            
        notebook["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": code_lines
        })
        print(f"Success: Module '{py_file}' successfully loaded into the Notebook structure.")
    else:
        print(f"Warning: File '{py_file}' not found. Skipping integration.")

# Serialize and export the final .ipynb file
output_file = "Chess_AI_Pipeline.ipynb"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=2)

print("\nProcess completed. Notebook successfully generated: " + output_file)