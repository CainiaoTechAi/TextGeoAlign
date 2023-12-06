import torch
import torch.nn as nn
from src.M3GAL.tokenization_bert import BertTokenizer
from torch.nn.parallel import DistributedDataParallel as DDP
from src.M3GAL.model import AddressEncoder


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

        # queues
        self.register_buffer("queue", torch.randn(dim, K))
        self.register_buffer("gc_queue", torch.randn(dim, K))
        if self.normalize:
            self.queue = nn.functional.normalize(self.queue, dim=0)
            self.gc_queue = nn.functional.normalize(self.gc_queue, dim=0)

        # queue pointer
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

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

    def forward(self, query, docs, gls):
        """
        Input:
            query: a batch of query addresses,
            docs: a batch x c poi_addr, c is the number of candidates
            gls: batchsize x c x 2 longitude and latitude
        """
        bs = len(query)
        assert len(docs) % bs == 0, (len(docs), bs)
        doc_nums = len(docs) // bs

        device = next(self.parameters()).device

        query_embedding = self.encoder_q(query)
        docs_embedding = self.encoder_k(docs)
        gls_embedding = self.gc_encoder_shadow(torch.tensor(gls).to(device).float())

        if self.normalize:
            query_embedding = nn.functional.normalize(query_embedding)
            docs_embedding = nn.functional.normalize(docs_embedding)
            gls_embedding = nn.functional.normalize(gls_embedding)

        docs_score = torch.einsum(
            "bik,bjk->bij",
            query_embedding.view(bs, 1, -1),
            docs_embedding.view(bs, doc_nums, -1),
        ).squeeze(1)

        gls_score = torch.einsum(
            "bik,bjk->bij",
            query_embedding.view(bs, 1, -1),
            gls_embedding.view(bs, doc_nums, -1),
        ).squeeze(1)

        docs_score /= self.T
        gls_score /= self.T
        labels = torch.zeros(bs, dtype=torch.long, device=device)
        return docs_score, gls_score, labels


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
