## Check sample pics
import os
import threading
from imjoy_elfinder.app import main

root_dir = "~/sd-train"

def start_file_explorer(root_dir=root_dir, port=8765):
    try:
        main(["--root-dir=" + root_dir, "--port=" + str(port)])
    except Exception as e:
        print("Error starting file explorer:", str(e))


def open_file_explorer(root_dir=root_dir, port=8765):
    thread = threading.Thread(target=start_file_explorer, args=[root_dir, port])
    thread.start()


# Example usage
sample_dir = os.path.join(root_dir, "fine_tune/output/sample")
open_file_explorer(root_dir=sample_dir, port=8765)

input("Press the Enter key to continue: ")


## Visualize loss graph
import subprocess

training_logs_path = os.path.join(root_dir, "fine_tune/logs")

repo_dir = os.path.join(root_dir, "sd-scripts")
os.chdir(repo_dir)
subprocess.Popen(f"tensorboard --logdir {training_logs_path}", shell=True)

input("Press the Enter key to continue: ")
