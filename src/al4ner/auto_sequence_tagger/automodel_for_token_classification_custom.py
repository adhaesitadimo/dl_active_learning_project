from transformers import AutoModelForTokenClassification, AutoModel

import torch
from collections import OrderedDict
from torch.nn import Module, CrossEntropyLoss, Linear
from torch.nn.functional import dropout
from transformers import (
    DistilBertForTokenClassification, CamembertForTokenClassification,
XLMForTokenClassification, XLMRobertaForTokenClassification, RobertaForTokenClassification,
    BertForTokenClassification, 
XLNetForTokenClassification, AlbertForTokenClassification, ElectraForTokenClassification
)
from transformers.configuration_auto import (
    AlbertConfig,
    AutoConfig,
    BartConfig,
    BertConfig,
    CamembertConfig,
    DistilBertConfig,
    ElectraConfig,
    RobertaConfig,
    XLMConfig,
    XLMRobertaConfig,
    XLNetConfig,
)

MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING = OrderedDict(
    [
        (DistilBertConfig, DistilBertForTokenClassification),
        (CamembertConfig, CamembertForTokenClassification),
        (XLMConfig, XLMForTokenClassification),
        (XLMRobertaConfig, XLMRobertaForTokenClassification),
        (RobertaConfig, RobertaForTokenClassification),
        (BertConfig, BertForTokenClassification),
        (XLNetConfig, XLNetForTokenClassification),
        (AlbertConfig, AlbertForTokenClassification),
        (ElectraConfig, ElectraForTokenClassification),
    ]
)

class DropoutMC(Module):
    def __init__(self, p: float, activate=False):
        super().__init__()
        self.activate = activate
        self.p = p

    def forward(self, x):
        return dropout(x, self.p, training=self.training or self.activate)


def activate_mc_dropout(model, activate: bool = True, random: bool = True, verbose: bool = False):
    for layer in model.children():
        if isinstance(layer, DropoutMC):
            if verbose:
                print(f"Current DO state : {layer.activate}")
                print(f"Switching state to: {activate}")
            layer.activate = activate
            layer.p = layer.p if not random else torch.clamp(torch.rand(1), 0, 0.5).item()
        else:
            activate_mc_dropout(layer, activate=activate, random=random, verbose=verbose)


class AutoModelForTokenClassificationCustom(AutoModelForTokenClassification):
    def __init__(self, config):

        super(AutoModelForTokenClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.model = AutoModel(config)
        self.dropout = DropoutMC(config.hidden_dropout_prob)
        self.classifier = Linear(config.hidden_size, config.num_labels)

        self.init_weights()
        
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, num_labels, *model_args, **kwargs):
        
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        for config_class, model_class in MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING.items():
            if isinstance(config, config_class):
                config.num_labels = num_labels
                print(config_class)
                return model_class.from_pretrained(pretrained_model_name_or_path, *model_args, config=config, **kwargs)


    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None, loss_mask=None):
        outputs = self.model(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = (attention_mask.view(-1) == 1)
                if loss_mask is not None:
                    active_loss &= loss_mask.view(-1)

                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)
