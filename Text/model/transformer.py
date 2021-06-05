from transformers import BertModel, BertConfig
import torch.nn as nn
import torch
import os


def norm(x):
    norm = torch.norm(x, p=2, dim=1)
    x = x / (norm.expand(1, -1).t() + .0001)
    return x

class BaseLine(nn.Module):

    def __init__(self, args, n_class):
        super(BaseLine, self).__init__()
        self.args = args
        self.config = BertConfig.from_pretrained(os.path.join("data","bert_config"), output_hidden_states=True)
        self.transformer = BertModel(config=self.config)
        self.transformer.init_weights()
        hidden_size = self.config.hidden_size
        self.classifier = nn.Linear(hidden_size, n_class)
        self.__init_classifier_weight()

    def __init_classifier_weight(self):
        nn.init.kaiming_normal_(self.classifier.weight.data)

    def forward(self, inps, inp_embeds=None):
        """
        inps (N,seq_lens)
        attention_mask (N,seq_lens)
        return (N, hidden_size)
        """

        if inp_embeds is not None:
            out = self.transformer.forward(inputs_embeds=inp_embeds)
        else:
            out = self.transformer.forward(inps)

        hidden = out[0]
        all_hidden_states = out[2]

        return hidden[:, 0], [lh.mean(1) for lh in all_hidden_states][1:], hidden.mean(1)

    def feature_list(self, inps):
        in_feature, all_hidden, _ = self.forward(inps, None)

        return in_feature, all_hidden

    def intermediate_forward(self, inps):
        out = self.transformer(inps, )
        hidden = out[0]

        return hidden[:, 0]

