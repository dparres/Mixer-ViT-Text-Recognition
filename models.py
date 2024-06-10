import torch
from torch import nn
from torch.nn import functional as F
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class ViT_Simple(nn.Module):

    def __init__(self, len_vocab, E, weights_path = "microsoft/trocr-base-stage1"):
        super().__init__()

        self.processor = TrOCRProcessor.from_pretrained(weights_path)
        self.trocr_encoder = VisionEncoderDecoderModel.from_pretrained(weights_path).encoder
        self.trocr_encoder.pooler = Identity()

        self.gelu = torch.nn.GELU()
        self.projection_E = nn.Linear(768,E)
        self.projection_V = nn.Linear(E, len_vocab + 1) # classes + blank token

    def forward(self, x, targets=None, target_lengths=None):

        x = self.trocr_encoder(x).last_hidden_state # (b, 577, 768)

        x = self.projection_E(x) # (b, 577, 768) to (b, 577,E)
        x = self.gelu(x)

        x = self.projection_V(x) # (b, 577,E) to (b, 577, V)

        if targets is not None:
            x = x.permute(1, 0, 2)
            loss = self.ctc_loss(x,targets, target_lengths)
            return x, loss

        return x, None

    @staticmethod
    def ctc_loss(x, targets, target_lengths):
        batch_size = x.size(1)
       
        log_probs = F.log_softmax(x, 2)
    
        input_lengths = torch.full(
            size=(batch_size,), fill_value=log_probs.size(0), dtype=torch.int32
        )

        loss = nn.CTCLoss(blank=0)(
            log_probs, targets, input_lengths, target_lengths
        )
        return loss

class Mixer_ViT(nn.Module):

    def __init__(self, len_vocab, E, weights_path = "microsoft/trocr-base-stage1"):
        super().__init__()

        self.processor = TrOCRProcessor.from_pretrained(weights_path)
        self.trocr_encoder = VisionEncoderDecoderModel.from_pretrained(weights_path).encoder
        self.trocr_encoder.pooler = Identity()

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.projection_E = nn.Linear(577, E)
        self.projection_V = nn.Linear(768, len_vocab + 1) # classes + blank token

    def forward(self, x, targets=None, target_lengths=None):

        x = self.trocr_encoder(x).last_hidden_state # (b, 577, 768)

        x = x.permute(0,2,1) # (b, 577, 768) to (b, 768, 577)
        x = self.dropout1(x)
        x = self.projection_E(x) # (b, 768, 577) to (b, 768,E)

        x = x.permute(0,2,1) # (b, 768,E) to (b,E, 768)
        x = self.dropout2(x)
        x = self.projection_V(x) # (b,E, 768) to (b,E, V)

        if targets is not None:
            x = x.permute(1, 0, 2)
            loss = self.ctc_loss(x,targets, target_lengths)
            return x, loss

        return x, None

    @staticmethod
    def ctc_loss(x, targets, target_lengths):
        batch_size = x.size(1)
       
        log_probs = F.log_softmax(x, 2)
    
        input_lengths = torch.full(
            size=(batch_size,), fill_value=log_probs.size(0), dtype=torch.int32
        )

        loss = nn.CTCLoss(blank=0)(
            log_probs, targets, input_lengths, target_lengths
        )
        return loss

class Mixer_GELU_ViT(nn.Module):

    def __init__(self, len_vocab, E, weights_path = "microsoft/trocr-base-stage1"):
        super().__init__()

        self.processor = TrOCRProcessor.from_pretrained(weights_path)
        self.trocr_encoder = VisionEncoderDecoderModel.from_pretrained(weights_path).encoder
        self.trocr_encoder.pooler = Identity()

        self.gelu = torch.nn.GELU()
        self.dropout1 = nn.Dropout(0.5)
        self.projection_E = nn.Linear(577, E)
        self.dropout2 = nn.Dropout(0.5)
        self.projection_V = nn.Linear(768, len_vocab + 1) # classes + blank token

    def forward(self, x, targets=None, target_lengths=None):
        
        x = self.trocr_encoder(x).last_hidden_state # (b, 577, 768)

        x = x.permute(0,2,1) # (b, 577, 768) to (b, 768, 577)
        x = self.dropout1(x)
        x = self.projection_E(x) # (b, 768, 577) to (b, 768,E)
        x = self.gelu(x)

        x = x.permute(0,2,1) # (b, 768,E) to (b,E, 768)
        x = self.dropout2(x)
        x = self.projection_V(x) # (b,E, 768) to (b,E, V)

        if targets is not None:
            x = x.permute(1, 0, 2)
            loss = self.ctc_loss(x,targets, target_lengths)
            return x, loss

        return x, None
    
    @staticmethod
    def ctc_loss(x, targets, target_lengths):
        batch_size = x.size(1)
       
        log_probs = F.log_softmax(x, 2)
    
        input_lengths = torch.full(
            size=(batch_size,), fill_value=log_probs.size(0), dtype=torch.int32
        )

        loss = nn.CTCLoss(blank=0)(
            log_probs, targets, input_lengths, target_lengths
        )
        return loss