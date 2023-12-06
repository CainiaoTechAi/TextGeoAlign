import torch
import torch.nn as nn


# This model is used for fine tune with GeoETS trainset
# The only difference is the forward method
class MultiModal_MoCo(nn.Module):
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
        geohash_encoder,
        geohash_encoder_shadow,
        tokenizer,
        dim=128,
        K=65536,
        m=0.999,
        T=0.07,
        gc_m=None,
        geohash_m=None,
        mlp=False,
        normalize=True,
    ):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco menmentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """

        super(MultiModal_MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.tokenizer = tokenizer

        self.gc_m = m if gc_m is None else gc_m
        self.geohash_m = m if geohash_m is None else geohash_m

        # create the encoders
        # number_class is the output fc dimension
        self.encoder_q = encoder_q
        self.encoder_k = encoder_k
        # gc_encoder encodes geolocations, shadow encoder won't be
        # updated by gradient descent
        self.gc_encoder = gc_encoder
        self.gc_encoder_shadow = gc_encoder_shadow
        self.geohash_encoder = geohash_encoder
        self.geohash_encoder_shadow = geohash_encoder_shadow

        self.normalize = normalize

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc
            )
            self.encoder_k.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc
            )

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        if self.normalize:
            self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("gc_queue", torch.randn(dim, K))
        if self.normalize:
            self.gc_queue = nn.functional.normalize(self.gc_queue, dim=0)
        self.register_buffer("geohash_queue", torch.randn(dim, K))
        if self.normalize:
            self.geohash_queue = nn.functional.normalize(self.geohash_queue, dim=0)

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

        for geohash_param, geohash_param_shadow in zip(
            self.geohash_encoder.parameters(), self.geohash_encoder_shadow.parameters()
        ):
            geohash_param_shadow.data.copy_(
                geohash_param_shadow.data * self.geohash_m
                + geohash_param.data * (1.0 - self.geohash_m)
            )

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, gc_keys, geohash_keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity
        assert batch_size == gc_keys.shape[0]

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size] = keys.T
        self.gc_queue[:, ptr : ptr + batch_size] = gc_keys.T
        self.geohash_queue[:, ptr : ptr + batch_size] = geohash_keys.T
        ptr = (ptr + batch_size) % self.K

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def compute_embedding(self, input, type="q"):
        """
        Compute embedding of different modal
        """
        if type == "q" or type == "k":
            addr_token = self.tokenizer(input, padding="longest", return_tensors="pt")

            # if model is on cuda, data should load to cuda
            device = next(self.parameters()).device
            addr_input_ids = addr_token.input_ids.to(device)
            addr_attention_mask = addr_token.attention_mask.to(device)
            if type == "q":
                embedding = self.encoder_q(addr_input_ids, addr_attention_mask)
            elif type == "k":
                embedding = self.encoder_k(addr_input_ids, addr_attention_mask)
        elif type == "gc_shadow":
            device = next(self.parameters()).device
            embedding = self.gc_encoder_shadow(torch.tensor(input).to(device).float())
        elif type == "geohash_shadow":
            device = next(self.parameters()).device
            o, [h, c] = self.geohash_encoder_shadow(
                torch.tensor(input).to(device).float()
            )
            return o[:, -1, :]
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

        query_token = self.tokenizer(query, padding="longest", return_tensors="pt").to(
            device
        )
        query_input_ids = query_token.input_ids.to(device)
        query_attention_mask = query_token.attention_mask.to(device)
        query_embedding = self.encoder_q(query_input_ids, query_attention_mask)

        docs_token = self.tokenizer(docs, padding="longest", return_tensors="pt").to(
            device
        )
        docs_input_ids = docs_token.input_ids.to(device)
        docs_attention_mask = docs_token.attention_mask.to(device)
        docs_embedding = self.encoder_k(docs_input_ids, docs_attention_mask)

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
