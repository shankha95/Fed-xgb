from trainer import XGBoostTrainer
import os

class Agent:
    def __init__(self, data_path, client_id):
        self.data_path = data_path
        self.client_id = client_id
        self.output_dir = "/output"  # Host-mounted dir in Docker

    def run(self):
        trainer = XGBoostTrainer(self.data_path)
        results, models, preds, y_test = trainer.train_and_evaluate()

        print(f"[{self.client_id}] Training complete. MSEs: {results}")
        trainer.save_models(self.output_dir)

        # Optional: save predictions for inspection
        result_df = preds.copy()
        result_df[['sum_avg_cpu', 'sum_avg_memory', 'sum_max_cpu', 'sum_max_memory']] = y_test
        result_df.to_csv(f"{self.output_dir}/{self.client_id}_predictions.csv", index=False)
