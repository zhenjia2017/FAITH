import torch
import torch.nn.functional as F

from faith.heterogeneous_answering.graph_neural_network.answering.answering_factory import (
    AnsweringFactory,
)
from faith.heterogeneous_answering.graph_neural_network.encoders.encoder_factory import (
    EncoderFactory,
)


class NoStructureModel(torch.nn.Module):
    """Model that does not leverage any graph structure."""

    def __init__(self, config):
        super(NoStructureModel, self).__init__()
        self.config = config

        # load parameters
        self.num_layers = config["gnn_num_layers"]
        self.emb_dimension = config["gnn_emb_dimension"]
        self.dropout = config["gnn_dropout"] if "gnn_dropout" in config else 0.0

        # encoder
        self.encoder = EncoderFactory.get_encoder(config)

        # NN layers
        for i in range(self.num_layers):
            # updating entities
            setattr(
                self,
                "w_ent_ent_" + str(i),
                torch.nn.Linear(in_features=self.emb_dimension, out_features=self.emb_dimension),
            )

            # updating evidences
            setattr(
                self,
                "w_ev_ev_" + str(i),
                torch.nn.Linear(in_features=self.emb_dimension, out_features=self.emb_dimension),
            )
        # answering
        self.answering = AnsweringFactory.get_answering(config)

        # move layers to cuda
        if torch.cuda.is_available():
            self.cuda()

    def forward(self, batch, train=False):
        """
        Forward step of model.
        """
        # get data
        srs = batch["sr"]
        entities = batch["entities"]
        evidences = batch["evidences"]
        ev_to_ent = batch["ev_to_ent"]

        # encoding
        sr_vec = self.encoder.encode_srs_batch(srs)
        evidences_mat = self.encoder.encode_evidences_batch(
            evidences, srs
        )  # size: batch_size x num_ev x emb
        entities_mat = self.encoder.encode_entities_batch(
            entities, srs, evidences_mat, ev_to_ent, sr_vec
        )  # size: batch_size x num_ent x emb

        # apply graph neural updates
        for i in range(self.num_layers):
            # UPDATE ENTITIES
            w_ent_ent = getattr(self, "w_ent_ent_" + str(i))  # size: emb x emb
            ent_messages_ent = w_ent_ent(entities_mat)  # batch_size x num_ent x emb
            entities_mat = F.relu(ent_messages_ent)  # batch_size x num_ent x emb

            # UPDATE EVIDENCES
            w_ev_ev = getattr(self, "w_ev_ev_" + str(i))  # size: emb x emb
            ev_messages_ev = w_ev_ev(evidences_mat)  # batch_size x num_ev x emb
            evidences_mat = F.relu(ev_messages_ev)  # batch_size x num_ev x emb

            # PREPARE FOR NEXT LAYER
            entities_mat = F.dropout(entities_mat, self.dropout, training=train)
            evidences_mat = F.dropout(evidences_mat, self.dropout, training=train)

        # obtain answer probabilities, loss, qa-metrics
        res = self.answering(batch, train, entities_mat, sr_vec, evidences_mat)
        return res
