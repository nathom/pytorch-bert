from torch import nn
from transformers import BertModel


class IntentModel(nn.Module):
    def __init__(self, args, tokenizer, target_size):
        super().__init__()
        self.tokenizer = tokenizer
        self.model_setup(args)
        self.target_size = target_size

        # task1: add necessary class variables as you wish.

        # task2: initilize the dropout and classify layers
        self.dropout = nn.Dropout(p=0.1)  # Some random value I put
        self.classify = Classifier(args, target_size)

    def model_setup(self, args):
        print(f"Setting up {args.model} model")

        # task1: get a pretrained model of 'bert-base-uncased'
        self.encoder = BertModel.from_pretrained("bert-base-uncased")

        self.encoder.resize_token_embeddings(len(self.tokenizer))  # transformer_check

    def forward(self, inputs, labels):
        """
        task1:
            feeding the input to the encoder,
        task2:
            take the last_hidden_state's <CLS> token as output of the
            encoder, feed it to a drop_out layer with the preset dropout rate in the argparse argument,
        task3:
            feed the output of the dropout layer to the Classifier which is provided for you.
        """
        # Task 1:
        outputs = self.encoder(**inputs)

        # Task 2:
        cls_output = outputs.last_hidden_state[:, 0]  # Extract <CLS> token
        cls_output = self.dropout(cls_output)

        # Task 3:
        logits = self.classify(cls_output)

        return logits


class Classifier(nn.Module):
    def __init__(self, args, target_size):
        super().__init__()
        input_dim = args.embed_dim
        self.top = nn.Linear(input_dim, args.hidden_dim)
        self.relu = nn.ReLU()
        self.bottom = nn.Linear(args.hidden_dim, target_size)

    def forward(self, hidden):
        middle = self.relu(self.top(hidden))
        logit = self.bottom(middle)
        return logit


class CustomModel(nn.Module):
    def __init__(self, args, tokenizer, target_size):
        super().__init__()
        self.tokenizer = tokenizer
        self.encoder = BertModel.from_pretrained("bert-base-uncased")
        self.encoder.resize_token_embeddings(len(self.tokenizer))  # transformer_check
        self.target_size = target_size

        self.dropout = nn.Dropout(p=0.1)  # Some random value I put
        self.fc = nn.Linear(self.encoder.config.hidden_size, target_size)

    def forward(self, inputs, labels):
        out = self.encoder(**inputs)
        out = out.last_hidden_state[:, 0]  # Extract <CLS> token
        out = self.dropout(out)
        out = self.fc(out)
        return out


class SupConModel(nn.Module):
    def __init__(self, args, tokenizer, target_size):
        super().__init__()
        self.tokenizer = tokenizer
        self.encoder = BertModel.from_pretrained("bert-base-uncased")
        self.encoder.resize_token_embeddings(len(self.tokenizer))  # transformer_check
        self.target_size = target_size

        self.dropout1 = nn.Dropout(p=0.1)  # Some random value I put
        self.dropout2 = nn.Dropout(p=0.1)  # Some random value I put
        hidden_size = self.encoder.config.hidden_size
        self.norm = nn.BatchNorm1d(hidden_size)
        self.fc = nn.Linear(hidden_size, target_size)

        # self.softmax = nn.Softmax()

    def forward(self, inputs, labels):
        out1 = self.encoder(**inputs)
        out1 = out1.last_hidden_state[:, 0]  # Extract <CLS> token
        out1 = self.dropout1(out1)
        out = self.fc(out1)
        # out = self.softmax(out)
        return out
