from roboflow import Roboflow
import os

# get roboflow api key
ROBOFLOW_KEY = os.getenv("ROBOFLOW_KEY")

if not ROBOFLOW_KEY:
    raise ValueError("Api Key must be set as ROBOFLOW_KEY")

def get_data_roboflow(user_id, project_id, version=1):
    """ Get coco dataset from Roboflow """
    rf = Roboflow(api_key=ROBOFLOW_KEY)
    project = rf.workspace(user_id).project(project_id)
    version = project.version(version)
    dataset = version.download("coco")
    
    print(f"Dataset downloaded to {dataset}")
    return dataset.location