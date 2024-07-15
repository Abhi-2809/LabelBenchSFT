from transformers import (
    T5EncoderModel,
    AutoTokenizer,
)
import torch
import torch.nn as nn
from LabelTrain.model.model_skeleton import register_model


class T5EncoderForSequenceClassification(nn.Module):
    def __init__(self, num_output, ret_emb, model_name, device_map="cuda"):
        super(T5EncoderForSequenceClassification, self).__init__()
        model = T5EncoderModel.from_pretrained(model_name, device_map=device_map)
        model.config.use_cache = False
        self.encoder = model
        self.embed_dim = model.config.hidden_size
        self.ret_emb = ret_emb
        self.num_output = num_output

        # Set num_output to 0 to return the embedding.
        if num_output != 0:
            self.classifier = nn.Linear(self.embed_dim, num_output).cuda()
        else:
            self.classifier = nn.Identity().cuda()

    def forward(self, input_ids, ret_features=False, freeze=False):
        if freeze:
            with torch.no_grad():
                features = self.encoder(input_ids=input_ids)[0]
        else:
            features = self.encoder(input_ids=input_ids)[0]
        features = features[:, 0, :]  # take <s> token (equiv. to [CLS])

        if ret_features:
            return self.classifier(features), features.data
        elif self.ret_emb:
            return self.classifier(features), features
        else:
            return self.classifier(features)

    def get_embedding_dim(self):
        return self.embed_dim

    def get_preprocess(self):
        return None


def init_transformers(model_config, model_name):
    model = T5EncoderForSequenceClassification(
        num_output=model_config["num_output"],
        ret_emb=model_config["ret_emb"] if "ret_emb" in model_config else False,
        model_name=model_name)
    return model


@register_model("t5-small", "t5-small")
def init_t5_small(model_config):
    return init_transformers(model_config, "t5-small")


@register_model("t5-base", "t5-base")
def init_t5_small(model_config):
    return init_transformers(model_config, "t5-base")


@register_model("t5-large", "t5-large")
def init_t5_small(model_config):
    return init_transformers(model_config, "t5-large")


@register_model("t5-3b", "t5-3b")
def init_t5_small(model_config):
    return init_transformers(model_config, "t5-3b")


@register_model("t5-11b", "t5-11b")
def init_t5_small(model_config):
    return init_transformers(model_config, "t5-11b")
