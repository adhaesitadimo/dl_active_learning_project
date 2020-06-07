from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertEmbeddings, BertEncoder, BertPooler
import torch


def prepare_attention_mask(attention_mask):
    attention_mask = attention_mask.unsqueeze(1)
    attention_mask = (1.0 - attention_mask) * -10000.0
    return attention_mask


class BertWithCustomAttentionMask(BertPreTrainedModel):
    def __init__(self, config, gen_attention_mask):
        super(BertWithCustomAttentionMask, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)
        self.gen_attention_mask = gen_attention_mask
    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True):
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device
        
        mask = self.gen_attention_mask(input_ids.size())
        mask = mask.to(dtype=dtype, device=device)
        if attention_mask is not None:
            batch_size = attention_mask.shape[0]
            seq_len = attention_mask.shape[1]
            attention_mask = attention_mask.to(dtype=dtype, device=device)
            attention_mask = attention_mask.unsqueeze(1)
            attention_mask = attention_mask.expand((batch_size, seq_len, seq_len)).clone()
            mask = mask.expand((batch_size, seq_len, seq_len)).clone()
            mask *= attention_mask
        mask = prepare_attention_mask(mask)
        
        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers = self.encoder(embedding_output,
                                      mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output
    
    
def gen_delexicalization_mask(size):
    return (1 - torch.eye(size[1])).unsqueeze(0)
    
    
def delexicalize(bert):
    old_model = bert.bert
    new_model = BertWithCustomAttentionMask(old_model.config, gen_delexicalization_mask)
    new_model.load_state_dict(old_model.state_dict())
    bert.bert = new_model
