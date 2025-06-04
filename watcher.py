import os
import shutil
import time

class Watcher:
    def __init__(self, watch_dir, target_dir="/server_storage"):
        self.watch_dir = watch_dir
        self.target_dir = target_dir
        os.makedirs(self.target_dir, exist_ok=True)

    def sync_models(self):
        print(f"[Watcher] Looking for models in: {self.watch_dir}")
        for file in os.listdir(self.watch_dir):
            if file.endswith(".json") or file.endswith(".csv"):
                src = os.path.join(self.watch_dir, file)
                dst = os.path.join(self.target_dir, file)
                shutil.copy2(src, dst)
                print(f"[Watcher] Copied {file} to {self.target_dir}")
