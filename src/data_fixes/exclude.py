import pandas as pd
import torch
import os


class Excluder:
    def __call__(
        self, features: dict, outcomes: dict, k: int = 2, dir: str = ""
    ) -> pd.DataFrame:
        # Exclude patients with few concepts
        features, outcomes = self.exclude_short_sequences(
            features, outcomes, k=k, dir=dir
        )
        return features, outcomes

    @staticmethod  # Currently unused
    def exclude_rare_concepts(
        features: dict, k: int = 2, dir: str = ""
    ) -> pd.DataFrame:
        unique_codes = {}
        for patient in features["concept"]:
            for code in patient:
                unique_codes[code] = unique_codes.get(code, 0) + 1
        exclude = {code for code, count in unique_codes.items() if count < k}

        for i, patient in enumerate(features["concept"]):
            kept_indices = [
                idx for idx, code in enumerate(patient) if not code in exclude
            ]
            for key, values in features.items():
                features[key][i] = [values[i][j] for j in kept_indices]

        return features

    @staticmethod
    def exclude_short_sequences(
        features: dict, outcomes: dict, k: int = 2, dir: str = ""
    ) -> tuple:
        kept_indices = []
        for i, concepts in enumerate(features["concept"]):
            unique_codes = set([code for code in concepts if not code.startswith("[")])
            if len(unique_codes) >= k:
                kept_indices.append(i)
        torch.save(kept_indices, os.path.join(dir, "excluder_kept_indices.pt"))

        for key, values in features.items():
            features[key] = [values[i] for i in kept_indices]

        for key, values in outcomes.items():
            outcomes[key] = [values[i] for i in kept_indices]

        return features, outcomes

    @staticmethod
    def exclude_covid_negative(features: dict, outcomes: dict):
        kept_indices = []
        for i, result in enumerate(outcomes["COVID"]):
            if pd.notna(result):
                kept_indices.append(i)

        for key, values in features.items():
            features[key] = [values[i] for i in kept_indices]

        for key, values in outcomes.items():
            outcomes[key] = [values[i] for i in kept_indices]

        return features, outcomes
