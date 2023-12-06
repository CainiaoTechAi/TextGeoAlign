import torch
import torch.nn as nn
from src.M3GAL.tokenization_bert import BertTokenizer
from transformers.models.bert.configuration_bert import BertConfig
from torch.nn.parallel import DistributedDataParallel as DDP
from src.M3GAL.xbert import BertModel


class AddressEncoder(nn.Module):
    def __init__(
        self,
        tokenizer,
        text_encoder,
    ):
        super(AddressEncoder, self).__init__()

        bert_config = BertConfig.from_pretrained(text_encoder)
        bert_config.gis_embedding = 0
        self.text_encoder = BertModel.from_pretrained(
            text_encoder, config=bert_config, add_pooling_layer=True
        )
        self.tokenizer = tokenizer

    def forward(self, input):
        token = self.tokenizer(input, padding="longest", return_tensors="pt")
        device = next(self.parameters()).device
        input_ids = token.input_ids.to(device)
        attention_mask = token.attention_mask.to(device)
        embedding_output = self.text_encoder.embeddings(input_ids=input_ids)

        text_output = self.text_encoder(
            attention_mask=attention_mask,
            encoder_embeds=embedding_output,
            return_dict=True,
            mode="text",
        )

        pooled_output = text_output[1]

        return pooled_output


class AddressEncoder2(nn.Module):
    def __init__(self, encoder):
        super(AddressEncoder2, self).__init__()
        self.text_encoder = encoder

    def forward(self, input_ids, attention_mask):
        output = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooler_output = output.pooler_output
        return pooler_output


class M3GAL(nn.Module):
    """
    Code is hacked from https://github.com/facebookresearch/moco/blob/main/moco/builder.py
    https://arxiv.org/abs/1911.05722
    """

    def __init__(
        self,
        encoder_q,
        encoder_k,
        gc_encoder,
        gc_encoder_shadow,
        dim=128,
        K=65536,
        m=0.999,
        T=0.07,
        gc_m=0.999,
        normalize=True,
    ):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco menmentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """

        super(M3GAL, self).__init__()

        # setup attributes
        self.K = K
        self.m = m
        self.T = T
        self.gc_m = gc_m
        self.normalize = normalize

        # setup encoders
        # query address encoder
        self.encoder_q = encoder_q
        # poi address encoder
        self.encoder_k = encoder_k
        # gc_encoder encodes geolocation
        self.gc_encoder = gc_encoder
        # shadow encoder won't be updated by gradient descent
        self.gc_encoder_shadow = gc_encoder_shadow

        # encoder_k and gc_encoder_shadow won't be optimized by gradient descent
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        for gc_param, gc_param_shadow in zip(
            self.gc_encoder.parameters(), self.gc_encoder_shadow.parameters()
        ):
            gc_param.data.copy_(gc_param_shadow.data)
            gc_param_shadow.requires_grad = False

        # queues
        self.register_buffer("queue", torch.randn(dim, K))
        self.register_buffer("gc_queue", torch.randn(dim, K))
        if self.normalize:
            self.queue = nn.functional.normalize(self.queue, dim=0)
            self.gc_queue = nn.functional.normalize(self.gc_queue, dim=0)

        # queue pointer
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

        for gc_param, gc_param_shadow in zip(
            self.gc_encoder.parameters(), self.gc_encoder_shadow.parameters()
        ):
            gc_param_shadow.data.copy_(
                gc_param_shadow.data * self.gc_m + gc_param.data * (1.0 - self.gc_m)
            )

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, gc_keys):
        batch_size = keys.shape[0]
        assert batch_size == gc_keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size] = keys.T
        self.gc_queue[:, ptr : ptr + batch_size] = gc_keys.T
        ptr = (ptr + batch_size) % self.K

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def compute_embedding(self, input, type):
        """
        Compute embedding of different modal
        """
        device = next(self.parameters()).device
        if type == "q":
            embedding = self.encoder_q(input)
        elif type == "k":
            embedding = self.encoder_k(input)
        elif type == "gc_shadow":
            embedding = self.gc_encoder_shadow(torch.tensor(input).to(device).float())
        elif type == "gc":
            embedding = self.gc_encoder(torch.tensor(input).to(device).float())
        else:
            raise NotImplementedError(f"type should be q or k but is {type}")

        if self.normalize:
            embedding = nn.functional.normalize(embedding, dim=1)

        return embedding

    def forward(self, addr_q, addr_k, geolocation):
        """
        Input:
            addr_q: a batch of query addresses
            addr_k: a batch of key addresses
            geolocation: a batch of geo_location points

        Output:
            logits, targets
        """

        # if the model is on cuda, data should load to cuda
        device = next(self.parameters()).device
        geolocation = geolocation.to(device)

        # compute query features
        q = self.encoder_q(addr_q)
        if self.normalize:
            q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()

            # compute gc embedding before momentum update
            gc_shadow = self.gc_encoder_shadow(geolocation)

            k = self.encoder_k(addr_k)
            if self.normalize:
                gc_shadow = nn.functional.normalize(gc_shadow, dim=1)
                k = nn.functional.normalize(k, dim=1)

        gc = self.gc_encoder(geolocation)
        if self.normalize:
            gc = nn.functional.normalize(gc, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        gc_l_pos = torch.einsum("nc,nc->n", [q, gc]).unsqueeze(-1)

        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])
        gc_l_neg = torch.einsum("nc,ck->nk", [q, self.gc_queue.clone().detach()])

        # logits Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        gc_logits = torch.cat([gc_l_pos, gc_l_neg], dim=1)

        # apply temperature
        logits /= self.T
        gc_logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=device)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k, gc_shadow)

        return logits, gc_logits, labels


def create_M3GAL(args):
    print("=> creating model '{}'".format(args.text_encoder))
    if args.bert_from_local:
        tokenizer = BertTokenizer.from_pretrained(
            "checkpoints/bert-base-chinese-tokenizer"
        )
        encoder_q_base_model = AddressEncoder(
            tokenizer, "checkpoints/bert-base-chinese-model"
        )
        encoder_k_base_model = AddressEncoder(
            tokenizer, "checkpoints/bert-base-chinese-model"
        )
    else:
        tokenizer = BertTokenizer.from_pretrained(args.text_encoder)
        encoder_q_base_model = AddressEncoder(tokenizer, args.text_encoder)
        encoder_k_base_model = AddressEncoder(tokenizer, args.text_encoder)

    gc_encoder = nn.Sequential(
        nn.Linear(2, 64),
        nn.ReLU(),
        nn.Linear(64, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 768),
    )
    gc_encoder_shadow = nn.Sequential(
        nn.Linear(2, 64),
        nn.ReLU(),
        nn.Linear(64, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 768),
    )
    model = M3GAL(
        encoder_q=encoder_q_base_model,
        encoder_k=encoder_k_base_model,
        gc_encoder=gc_encoder,
        gc_encoder_shadow=gc_encoder_shadow,
        dim=args.moco_dim,
        K=args.moco_k,
        m=args.moco_m,
        T=args.moco_t,
        gc_m=0,
        normalize=args.normalize,
    )
    print(model)
    if args.gpu is not None:
        model = model.cuda()
    if args.multiprocessing_distributed:
        model = DDP(model, device_ids=[args.gpu])

    return model
