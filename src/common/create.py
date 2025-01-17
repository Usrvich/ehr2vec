import pandas as pd
import src.common.loading as loading
from src.data.dataset import MLMDataset, HierarchicalDataset, CensorDataset
from transformers import BertConfig
from torch.optim import AdamW
from torch.utils.data import WeightedRandomSampler


def mlm_dataset(cfg):
    train_encoded, val_encoded = loading.encoded(cfg)
    vocabulary = loading.vocabulary(cfg)

    for enc in [train_encoded, val_encoded]:
        yield MLMDataset(
            enc,
            vocabulary=vocabulary,
            ignore_special_tokens=cfg.ignore_special_tokens,
        )


def hierarchical_dataset(cfg):
    train_encoded, val_encoded = loading.encoded(cfg)
    vocabulary = loading.vocabulary(cfg)
    tree = loading.tree(cfg)

    for enc in [train_encoded, val_encoded]:
        yield HierarchicalDataset(
            enc,
            tree=tree,
            vocabulary=vocabulary,
            ignore_special_tokens=cfg.ignore_special_tokens,
        )


def censor_dataset(cfg):
    train_encoded, val_encoded = loading.encoded(cfg)
    train_outcomes, val_outcomes = loading.outcomes(cfg)

    n_hours, outcome_type, censor_type = (
        cfg.outcome.n_hours,
        cfg.outcome.type,
        cfg.outcome.censor_type,
    )

    for enc, out in zip([train_encoded, val_encoded], [train_outcomes, val_outcomes]):
        yield CensorDataset(
            enc,
            outcomes=out[outcome_type],
            censor_outcomes=out[censor_type],
            n_hours=n_hours,
        )


def model(model_class, cfg, vocabulary=None, tree=None, **model_kwargs):
    # Automatically calculate model params if not given
    if (
        vocabulary and cfg.model.vocab_size is None
    ):  # Calculates vocab_size if not given
        cfg.model.vocab_size = len(vocabulary)
    if cfg.model.type_vocab_size is None:  # Max number of segments if SEP tokens
        cfg.model.type_vocab_size = cfg.model.max_position_embeddings // 2
    if (
        tree and cfg.model.leaf_size is None
    ):  # Calculate leaf_size if Tree and not given
        tree = loading.tree(cfg)
        cfg.model.leaf_size = tree.num_children_leaves()

    model = model_class(BertConfig(**cfg.model, **model_kwargs))
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
        eps=cfg.optimizer.epsilon,
    )

    return model, optimizer


def sampler(cfg, train_dataset):
    sampler, pos_weight = None, None
    if cfg.trainer_args["sampler"]:
        labels = pd.Series(train_dataset.outcomes).notna().astype(int)
        label_weight = 1 / labels.value_counts()
        weights = labels.map(label_weight).values
        sampler = WeightedRandomSampler(
            weights=weights, num_samples=len(train_dataset), replacement=True
        )
    elif cfg.trainer_args["pos_weight"]:
        pos_weight = sum(pd.isna(train_dataset.outcomes)) / sum(
            pd.notna(train_dataset.outcomes)
        )

    return sampler, pos_weight
