import os
from processor import InsulatorInspector

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(project_root, "model", "best.pt")
input_folder = os.path.join(project_root, "test_data", "input")
output_folder = os.path.join(project_root, "test_data", "output")

inspector = InsulatorInspector(model_path)

path_to_report = inspector.process_folder(
    input_folder=input_folder,
    output_folder=output_folder
)
