from trainer import XGBoostTrainer
from watcher import Watcher
import os

def main():
    client_id = os.environ.get("CLIENT_ID", "client1")
    data_path = os.environ.get("DATA_PATH", "/data/client1.csv")
    output_dir = os.environ.get("OUTPUT_DIR", "/output")
    server_dir = os.environ.get("SERVER_DIR", "/server_storage")

    print(f"[{client_id}] Starting training with data from {data_path}")

    trainer = XGBoostTrainer(data_path)
    results, _, _, _ = trainer.train_and_evaluate()
    print(f"[{client_id}] Training complete. MSEs: {results}")

    trainer.save_models(output_dir)

    watcher = Watcher(output_dir, server_dir)
    watcher.sync_models()

if __name__ == "__main__":
    main()
