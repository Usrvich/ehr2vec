import torch.nn as nn
from embeddings.time2vec import Time2Vec
import torch


class EhrEmbeddings(nn.Module):
    """
        EHR Embeddings

        Forward inputs:
            input_ids: torch.LongTensor             - (batch_size, sequence_length)
            token_type_ids: torch.LongTensor        - (batch_size, sequence_length)
            position_ids: dict(str, torch.Tensor)   - (batch_size, sequence_length)

        Config:
            vocab_size: int                         - size of the vocabulary
            hidden_size: int                        - size of the hidden layer
            type_vocab_size: int                    - size of max segments
            layer_norm_eps: float                   - epsilon for layer normalization
            hidden_dropout_prob: float              - dropout probability
            linear: bool                            - whether to linearly scale embeddings (a: concept, b: age, c: abspos, d: segment)
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.concept_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.age_embeddings = Time2Vec(1, config.hidden_size)
        self.abspos_embeddings = Time2Vec(1, config.hidden_size)
        self.segment_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if self.config.linear is not None:
            self.a = nn.Parameter(torch.ones(1))
            self.b = nn.Parameter(torch.zeros(1))
            self.c = nn.Parameter(torch.zeros(1))
            self.d = nn.Parameter(torch.zeros(1))
        else:
            self.a = self.b = self.c = self.d = 1

    def forward(
        self,
        input_ids: torch.LongTensor,                  # concepts
        token_type_ids: torch.LongTensor = None,      # segments
        position_ids: dict = None, # age and abspos
        inputs_embeds: torch.Tensor = None,
        **kwargs
    ):
        if inputs_embeds is not None:
            return inputs_embeds

        embeddings = self.a * self.concept_embeddings(input_ids)
        
        if token_type_ids is not None:
            segments_embedded = self.segment_embeddings(token_type_ids)
            embeddings += self.b * segments_embedded

        if position_ids is not None:
            if 'age' in position_ids:
                ages_embedded = self.age_embeddings(position_ids['age'])
                embeddings += self.c * ages_embedded
            if 'abspos' in position_ids:
                abspos_embedded = self.abspos_embeddings(position_ids['abspos'])
                embeddings += self.d * abspos_embedded
        
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings

class EhrEmbeddings_separate_value_embedding(nn.Module):
    """
        EHR Embeddings

        Forward inputs:
            input_ids: torch.LongTensor             - (batch_size, sequence_length)
            token_type_ids: torch.LongTensor        - (batch_size, sequence_length)
            position_ids: dict(str, torch.Tensor)   - (batch_size, sequence_length)

        Config:
            vocab_size: int                         - size of the vocabulary
            hidden_size: int                        - size of the hidden layer
            type_vocab_size: int                    - size of max segments
            layer_norm_eps: float                   - epsilon for layer normalization
            hidden_dropout_prob: float              - dropout probability
            linear: bool                            - whether to linearly scale embeddings (a: concept, b: age, c: abspos, d: segment)
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        if self.config.method == 'separate_value_embedding':
            # print("vocab_size: ", config.vocab_size)
            # print("vocab_type_size: ", config.type_vocab_size)
            self.concept_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
            self.age_embeddings = Time2Vec(1, config.hidden_size)
            self.abspos_embeddings = Time2Vec(1, config.hidden_size)
            self.segment_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
            # Since the number of different medicines, value and unit are relatively small, so directly apply nn.Embedding
            self.value_embeddings = nn.Embedding(config.vocab_size, config.hidden_size) # vocab_size or type_vocab_size?
            self.unit_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        elif self.config.get('method') == None:
            self.concept_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
            self.age_embeddings = Time2Vec(1, config.hidden_size)
            self.abspos_embeddings = Time2Vec(1, config.hidden_size)
            self.segment_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if self.config.linear is not None:
            self.a = nn.Parameter(torch.ones(1))
            self.b = nn.Parameter(torch.zeros(1))
            self.c = nn.Parameter(torch.zeros(1))
            self.d = nn.Parameter(torch.zeros(1))

            if self.config.method == 'separate_value_embedding':
                self.e = nn.Parameter(torch.zeros(1))
                self.f = nn.Parameter(torch.zeros(1))
        else:
            self.a = self.b = self.c = self.d = 1
            
            if self.config.method == 'separate_value_embedding':
                self.e = self.f = 1

    def forward(
        self,
        input_ids: torch.LongTensor,                  # concepts
        token_type_ids: torch.LongTensor = None,      # segments
        position_ids: dict = None, # age and abspos
        inputs_embeds: torch.Tensor = None,
        values: torch.LongTensor = None,
        units: torch.LongTensor = None,
        **kwargs
    ):
        if inputs_embeds is not None:
            return inputs_embeds

        embeddings = self.a * self.concept_embeddings(input_ids)
        
        if token_type_ids is not None:
            print(token_type_ids.shape)
            segments_embedded = self.segment_embeddings(token_type_ids)
            print(segments_embedded.shape)
            embeddings += self.b * segments_embedded

        if position_ids is not None:
            if 'age' in position_ids:
                ages_embedded = self.age_embeddings(position_ids['age'])
                embeddings += self.c * ages_embedded
            if 'abspos' in position_ids:
                abspos_embedded = self.abspos_embeddings(position_ids['abspos'])
                embeddings += self.d * abspos_embedded
            # if 'value' in position_ids:
            #     value_embedded = self.value_embeddings(position_ids['value'])
            #     embeddings += value_embedded
            # if 'unit' in position_ids:
            #     unit_embedded = self.unit_embeddings(position_ids['unit'])
            #     embeddings += unit_embedded
        if values is not None:
            value_embedded = self.value_embeddings(values)
            # print("value_embedded: ", value_embedded.shape)
            embeddings += self.e * value_embedded
        if units is not None:
            unit_embedded = self.unit_embeddings(units)
            # print("unit_embedded: ", unit_embedded.shape)
            embeddings += self.f * unit_embedded
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings