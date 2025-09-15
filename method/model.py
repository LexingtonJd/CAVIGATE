import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from easydict import EasyDict as edict

'''2025/2/26-start'''
# from method.model_components import BertAttention, LinearLayer, TrainablePositionalEncoding, GMMBlock
from method.DyGMMBlock import BertAttention, LinearLayer, \
    TrainablePositionalEncoding, DyGMMBlock, Dyaggregate, Dyquery, Dyscore

'''2025/2/26-end'''
from method.model_components import clip_nce, clip_kl_only_pos, frame_nce, BertSelfAttention

'''增加llama_encoder'''
from method.llama import LLaMATransformer
import time


class DLDKD(nn.Module):
    def __init__(self, config, opt):
        super(DLDKD, self).__init__()
        self.config = config
        self.epoch = 0
        self.use_clip = opt.use_clip
        self.double_branch = opt.double_branch
        self.video_level_branch = opt.video_level_branch

        if self.double_branch:

            self.frame_s_gmm_aggregate = Dyaggregate(
                edict(hidden_size=config.A_hidden_size, intermediate_size=config.A_hidden_size,
                      hidden_dropout_prob=config.drop, num_attention_heads=config.n_heads,
                      attention_probs_dropout_prob=config.drop, frame_len=128, sft_factor=0.8))
            self.frame_s_weight_token = nn.Parameter(torch.randn(1, 1, config.A_hidden_size))

            self.frame_a_gmm_aggregate = Dyaggregate(
                edict(hidden_size=config.A_hidden_size, intermediate_size=config.A_hidden_size,
                      hidden_dropout_prob=config.drop, num_attention_heads=config.n_heads,
                      attention_probs_dropout_prob=config.drop, frame_len=128, sft_factor=0.8))
            self.frame_a_weight_token = nn.Parameter(torch.randn(1, 1, config.A_hidden_size))

            #model components for video-level score
            # self.frame_s_Dyscore = Dyscore(
            #     edict(hidden_size=128, intermediate_size=128,
            #           hidden_dropout_prob=config.drop, num_attention_heads=config.n_heads,
            #           attention_probs_dropout_prob=config.drop, frame_len=128, sft_factor=1.0))
            # self.frame_s_weight_token = nn.Parameter(torch.randn(1, 1, 128))
            #
            # self.frame_a_Dyscore = Dyscore(
            #     edict(hidden_size=128, intermediate_size=128,
            #           hidden_dropout_prob=config.drop, num_attention_heads=config.n_heads,
            #           attention_probs_dropout_prob=config.drop, frame_len=128, sft_factor=1.0))
            # self.frame_a_weight_token = nn.Parameter(torch.randn(1, 1, 128))

            #model components for query weight
            # self.query_weight = Dyquery(
            #     edict(hidden_size=config.A_hidden_size, intermediate_size=config.A_hidden_size,
            #           hidden_dropout_prob=config.drop, num_attention_heads=config.n_heads,
            #           attention_probs_dropout_prob=config.drop, sft_factor=1.0))
            # self.sub_query_weight_token = nn.Parameter(torch.randn(1, 1, config.A_hidden_size))
            # self.aud_query_weight_token = nn.Parameter(torch.randn(1, 1, config.A_hidden_size))

            #model components for sub-video query encoding
            self.sub_query_input_proj = LinearLayer(config.query_input_size, config.A_hidden_size, layer_norm=True,
                                                  dropout=config.input_drop, relu=True)

            self.sub_query_encoder = BertAttention(
                edict(hidden_size=config.A_hidden_size, intermediate_size=config.A_hidden_size,
                      hidden_dropout_prob=config.drop, num_attention_heads=config.n_heads,
                      attention_probs_dropout_prob=config.drop))

            self.sub_query_encoder_2 = BertAttention(
                edict(hidden_size=config.A_hidden_size, intermediate_size=config.A_hidden_size,
                      hidden_dropout_prob=config.drop, num_attention_heads=config.n_heads,
                      attention_probs_dropout_prob=config.drop))

            self.sub_query_pos_embed = TrainablePositionalEncoding(max_position_embeddings=config.max_desc_l,
                                                                 hidden_size=config.A_hidden_size,
                                                                 dropout=config.input_drop)

            self.sub_modular_vector_mapping = nn.Linear(in_features=config.A_hidden_size, out_features=1, bias=False)

            #model components for aud-video query encoding
            self.aud_query_input_proj = LinearLayer(config.query_input_size, config.A_hidden_size, layer_norm=True,
                                                  dropout=config.input_drop, relu=True)

            self.aud_query_encoder = BertAttention(
                edict(hidden_size=config.A_hidden_size, intermediate_size=config.A_hidden_size,
                      hidden_dropout_prob=config.drop, num_attention_heads=config.n_heads,
                      attention_probs_dropout_prob=config.drop))

            self.aud_query_encoder_2 = BertAttention(
                edict(hidden_size=config.A_hidden_size, intermediate_size=config.A_hidden_size,
                      hidden_dropout_prob=config.drop, num_attention_heads=config.n_heads,
                      attention_probs_dropout_prob=config.drop))

            self.aud_query_pos_embed = TrainablePositionalEncoding(max_position_embeddings=config.max_desc_l,
                                                                 hidden_size=config.A_hidden_size,
                                                                 dropout=config.input_drop)

            self.aud_modular_vector_mapping = nn.Linear(in_features=config.A_hidden_size, out_features=1, bias=False)

            #model components for sub-frame feat
            self.frame_s_video_input_proj = LinearLayer(config.visual_input_size, config.A_hidden_size, layer_norm=True,
                                                        dropout=config.input_drop, relu=True)
            self.frame_s_sub_input_proj = LinearLayer(512, config.A_hidden_size, layer_norm=True,
                                                      dropout=config.input_drop, relu=True)

            self.frame_s_video_pos_embed = TrainablePositionalEncoding(max_position_embeddings=config.max_ctx_l,
                                                                       hidden_size=config.A_hidden_size,
                                                                       dropout=config.input_drop)
            self.frame_s_sub_pos_embed = TrainablePositionalEncoding(max_position_embeddings=config.max_ctx_l,
                                                                         hidden_size=config.A_hidden_size,
                                                                         dropout=config.input_drop)

            self.frame_s_video_encoder = BertAttention(
                edict(hidden_size=config.A_hidden_size, intermediate_size=config.A_hidden_size,
                      hidden_dropout_prob=config.drop, num_attention_heads=config.n_heads,
                      attention_probs_dropout_prob=config.drop))
            self.frame_s_sub_encoder = BertAttention(
                edict(hidden_size=config.A_hidden_size, intermediate_size=config.A_hidden_size,
                      hidden_dropout_prob=config.drop, num_attention_heads=config.n_heads,
                      attention_probs_dropout_prob=config.drop))

            self.frame_s_video_encoder_2 = BertAttention(
                edict(hidden_size=config.A_hidden_size, intermediate_size=config.A_hidden_size,
                      hidden_dropout_prob=config.drop, num_attention_heads=config.n_heads,
                      attention_probs_dropout_prob=config.drop))
            # self.frame_s_gmm_encoder = DyGMMBlock(
            #     edict(hidden_size=config.A_hidden_size, intermediate_size=config.A_hidden_size,
            #           hidden_dropout_prob=config.drop, num_attention_heads=config.n_heads,
            #           attention_probs_dropout_prob=config.drop, frame_len=128, sft_factor=0.6))
            # self.frame_s_weight_token = nn.Parameter(torch.randn(1, 1, config.A_hidden_size))

            self.frame_s_cross_att = BertSelfAttention(
                edict(hidden_size=config.A_hidden_size, num_attention_heads=config.n_heads,
                      attention_probs_dropout_prob=config.drop))

            self.frame_s_cross_layernorm = nn.LayerNorm(config.A_hidden_size)

            self.frame_s_mapping_linear = nn.Linear(config.A_hidden_size, out_features=config.A_hidden_size)

            # model components for aud-frame feat
            self.frame_a_video_input_proj = LinearLayer(config.visual_input_size, config.A_hidden_size, layer_norm=True,
                                                        dropout=config.input_drop, relu=True)
            self.frame_a_aud_input_proj = LinearLayer(768, config.A_hidden_size, layer_norm=True,
                                                      dropout=config.input_drop, relu=True)

            self.frame_a_video_pos_embed = TrainablePositionalEncoding(max_position_embeddings=config.max_ctx_l,
                                                                       hidden_size=config.A_hidden_size,
                                                                       dropout=config.input_drop)
            self.frame_a_aud_pos_embed = TrainablePositionalEncoding(max_position_embeddings=config.max_ctx_l,
                                                                     hidden_size=config.A_hidden_size,
                                                                     dropout=config.input_drop)

            self.frame_a_video_encoder = BertAttention(
                edict(hidden_size=config.A_hidden_size, intermediate_size=config.A_hidden_size,
                      hidden_dropout_prob=config.drop, num_attention_heads=config.n_heads,
                      attention_probs_dropout_prob=config.drop))
            self.frame_a_aud_encoder = BertAttention(
                edict(hidden_size=config.A_hidden_size, intermediate_size=config.A_hidden_size,
                      hidden_dropout_prob=config.drop, num_attention_heads=config.n_heads,
                      attention_probs_dropout_prob=config.drop))

            self.frame_a_video_encoder_2 = BertAttention(
                edict(hidden_size=config.A_hidden_size, intermediate_size=config.A_hidden_size,
                      hidden_dropout_prob=config.drop, num_attention_heads=config.n_heads,
                      attention_probs_dropout_prob=config.drop))
            # self.frame_a_gmm_encoder = DyGMMBlock(
            #     edict(hidden_size=config.A_hidden_size, intermediate_size=config.A_hidden_size,
            #           hidden_dropout_prob=config.drop, num_attention_heads=config.n_heads,
            #           attention_probs_dropout_prob=config.drop, frame_len=128, sft_factor=0.6))
            # self.frame_a_weight_token = nn.Parameter(torch.randn(1, 1, config.A_hidden_size))

            self.frame_a_cross_att = BertSelfAttention(
                edict(hidden_size=config.A_hidden_size, num_attention_heads=config.n_heads,
                      attention_probs_dropout_prob=config.drop))

            self.frame_a_cross_layernorm = nn.LayerNorm(config.A_hidden_size)

            self.frame_a_mapping_linear = nn.Linear(config.A_hidden_size, out_features=config.A_hidden_size)


        # if self.video_level_branch:
        #     #models for sub-clip feat
        #     self.clip_s_video_input_proj = LinearLayer(config.visual_input_size, config.C_hidden_size, layer_norm=True,
        #                                                 dropout=config.input_drop, relu=True)
        #     self.clip_s_sub_input_proj = LinearLayer(1024, config.C_hidden_size, layer_norm=True,
        #                                               dropout=config.input_drop, relu=True)
        #
        #     self.clip_s_video_pos_embed = TrainablePositionalEncoding(max_position_embeddings=config.max_ctx_l,
        #                                                                hidden_size=config.C_hidden_size,
        #                                                                dropout=config.input_drop)
        #     self.clip_s_sub_pos_embed = TrainablePositionalEncoding(max_position_embeddings=config.max_ctx_l,
        #                                                              hidden_size=config.C_hidden_size,
        #                                                              dropout=config.input_drop)
        #
        #     self.clip_s_video_encoder = BertAttention(
        #         edict(hidden_size=config.C_hidden_size, intermediate_size=config.C_hidden_size,
        #               hidden_dropout_prob=config.drop, num_attention_heads=config.n_heads,
        #               attention_probs_dropout_prob=config.drop))
        #     self.clip_s_sub_encoder = BertAttention(
        #         edict(hidden_size=config.C_hidden_size, intermediate_size=config.C_hidden_size,
        #               hidden_dropout_prob=config.drop, num_attention_heads=config.n_heads,
        #               attention_probs_dropout_prob=config.drop))
        #
        #     self.clip_s_gmm_encoder = DyGMMBlock(
        #         edict(hidden_size=config.C_hidden_size, intermediate_size=config.C_hidden_size,
        #               hidden_dropout_prob=config.drop, num_attention_heads=config.n_heads,
        #               attention_probs_dropout_prob=config.drop, frame_len=32, sft_factor=0.6))
        #     self.clip_s_weight_token = nn.Parameter(torch.randn(1, 1, config.C_hidden_size))
        #
        #     self.clip_s_cross_att = BertSelfAttention(
        #         edict(hidden_size=config.C_hidden_size, num_attention_heads=config.n_heads,
        #               attention_probs_dropout_prob=config.drop))
        #
        #     self.clip_s_cross_layernorm = nn.LayerNorm(config.C_hidden_size)
        #
        #     self.clip_s_mapping_linear = nn.Linear(config.C_hidden_size, out_features=config.C_hidden_size)
        #
        #     #models for aud-clip feat
        #     self.clip_a_video_input_proj = LinearLayer(config.visual_input_size, config.C_hidden_size, layer_norm=True,
        #                                                 dropout=config.input_drop, relu=True)
        #     self.clip_a_aud_input_proj = LinearLayer(768, config.C_hidden_size, layer_norm=True,
        #                                               dropout=config.input_drop, relu=True)
        #
        #     self.clip_a_video_pos_embed = TrainablePositionalEncoding(max_position_embeddings=config.max_ctx_l,
        #                                                                hidden_size=config.C_hidden_size,
        #                                                                dropout=config.input_drop)
        #     self.clip_a_aud_pos_embed = TrainablePositionalEncoding(max_position_embeddings=config.max_ctx_l,
        #                                                              hidden_size=config.C_hidden_size,
        #                                                              dropout=config.input_drop)
        #
        #     self.clip_a_video_encoder = BertAttention(
        #         edict(hidden_size=config.C_hidden_size, intermediate_size=config.C_hidden_size,
        #               hidden_dropout_prob=config.drop, num_attention_heads=config.n_heads,
        #               attention_probs_dropout_prob=config.drop))
        #     self.clip_a_aud_encoder = BertAttention(
        #         edict(hidden_size=config.C_hidden_size, intermediate_size=config.C_hidden_size,
        #               hidden_dropout_prob=config.drop, num_attention_heads=config.n_heads,
        #               attention_probs_dropout_prob=config.drop))
        #
        #     self.clip_a_gmm_encoder = DyGMMBlock(
        #         edict(hidden_size=config.C_hidden_size, intermediate_size=config.C_hidden_size,
        #               hidden_dropout_prob=config.drop, num_attention_heads=config.n_heads,
        #               attention_probs_dropout_prob=config.drop, frame_len=32, sft_factor=0.6))
        #     self.clip_a_weight_token = nn.Parameter(torch.randn(1, 1, config.C_hidden_size))
        #
        #     self.clip_a_cross_att = BertSelfAttention(
        #         edict(hidden_size=config.C_hidden_size, num_attention_heads=config.n_heads,
        #               attention_probs_dropout_prob=config.drop))
        #
        #     self.clip_a_cross_layernorm = nn.LayerNorm(config.C_hidden_size)
        #
        #     self.clip_a_mapping_linear = nn.Linear(config.C_hidden_size, out_features=config.C_hidden_size)

        self.nce_criterion = clip_nce(reduction='mean')
        self.video_nce_criterion = frame_nce(reduction='mean')

        self.reset_parameters()
        self.use_clip = opt.use_clip

        self.clip_loss = clip_kl_only_pos()

        '''查询多样化损失的超参数'''
        self.mrg = 0.2
        self.alpha = 32
        self.lamda = 1


        self.scale_weight = opt.loss_scale_weight
        if opt.decay_way > 3 and opt.decay_way < 7:
            self.init_weight = opt.loss_init_weight
        else:
            self.init_weight = 0

        self.weight = 1


    def reset_parameters(self):
        """ Initialize the weights."""

        def re_init(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            elif isinstance(module, nn.Conv1d):
                module.reset_parameters()
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

        self.apply(re_init)

    def set_hard_negative(self, use_hard_negative, hard_pool_size):
        """use_hard_negative: bool; hard_pool_size: int, """
        self.config.use_hard_negative = use_hard_negative
        self.config.hard_pool_size = hard_pool_size

    def forward(self, frame_video_feat, frame_sub_feat, frame_aud_feat, frame_video_mask, frame_sub_mask, frame_aud_mask, query_feat, query_mask, query_labels, cap_ids):

        sub_frame_feat, aud_frame_feat = self.encode_context(frame_video_feat, frame_sub_feat, frame_aud_feat, frame_video_mask, frame_sub_mask, frame_aud_mask)

        sub_query, aud_query, sub_weight, aud_weight = self.encode_query(query_feat, query_mask)

        sub_frame_score, sub_frame_score_, aud_frame_score, aud_frame_score_ = self.get_pred_from_raw_query(sub_query, aud_query, query_labels, sub_frame_feat, aud_frame_feat,
                                                                                                            frame_video_mask, sub_weight, sub_weight, return_query_feats=True)

        sub_query = sub_query.squeeze(1)
        aud_query = aud_query.squeeze(1)

        label_dict = {}
        for index, label in enumerate(query_labels):
            if label in label_dict:
                label_dict[label].append(index)
            else:
                label_dict[label] = []
                label_dict[label].append(index)

        # frame_nce_loss = 0
        # frame_trip_loss = 0
        #
        # clip_nce_loss = 0
        # clip_trip_loss = 0

        sub_frame_nce_loss = 0
        sub_frame_trip_loss = 0

        sub_video_nce_loss = 0
        sub_video_trip_loss = 0

        aud_frame_nce_loss = 0
        aud_frame_trip_loss = 0

        aud_video_nce_loss = 0
        aud_video_trip_loss = 0

        sub_neg_term_loss = 0
        aud_neg_term_loss = 0
        # sub_neg_term_loss += 0.014 * (self.scale_weight * self.weight + self.init_weight) * self.group_soft_aggregate_loss_singlefn(sub_query, label_dict)
        # aud_neg_term_loss += 0.014 * (self.scale_weight * self.weight + self.init_weight) * self.group_soft_aggregate_loss_singlefn(aud_query, label_dict)
        sub_neg_term_loss += 2.0 * (self.scale_weight * self.weight + self.init_weight) * self.group_soft_aggregate_loss_singlefn(sub_query, label_dict)
        aud_neg_term_loss += 2.0 * (self.scale_weight * self.weight + self.init_weight) * self.group_soft_aggregate_loss_singlefn(aud_query, label_dict)

        if self.double_branch:
            # frame_nce_loss = 0.04 * self.nce_criterion(query_labels, label_dict, frame_score_)
            # frame_trip_loss = self.get_clip_triplet_loss(frame_score, query_labels)

            sub_frame_nce_loss = 0.04 * self.nce_criterion(query_labels, label_dict, sub_frame_score_)
            sub_frame_trip_loss = self.get_clip_triplet_loss(sub_frame_score, query_labels)

            aud_frame_nce_loss = 0.04 * self.nce_criterion(query_labels, label_dict, aud_frame_score_)
            aud_frame_trip_loss = self.get_clip_triplet_loss(aud_frame_score, query_labels)

        # if self.video_level_branch:
        #     # clip_nce_loss = 0.02 * self.nce_criterion(query_labels, label_dict, clip_score_)
        #     # clip_trip_loss = self.get_clip_triplet_loss(clip_score, query_labels)
        #
        #     sub_clip_nce_loss = 0.02 * self.nce_criterion(query_labels, label_dict, sub_clip_score_)
        #     sub_clip_trip_loss = self.get_clip_triplet_loss(sub_clip_score, query_labels)
        #
        #     aud_clip_nce_loss = 0.02 * self.nce_criterion(query_labels, label_dict, aud_clip_score_)
        #     aud_clip_trip_loss = self.get_clip_triplet_loss(aud_clip_score, query_labels)

        loss = 0

        loss += sub_neg_term_loss + aud_neg_term_loss

        if self.double_branch:
            # loss += frame_nce_loss + frame_trip_loss
            loss += sub_frame_nce_loss + sub_frame_trip_loss
            loss += aud_frame_nce_loss + aud_frame_trip_loss

        if self.video_level_branch:
            # loss += clip_nce_loss + clip_trip_loss
            loss += sub_video_nce_loss + sub_video_trip_loss
            loss += aud_video_nce_loss + aud_video_trip_loss

        return loss, {"loss_overall": float(loss),
                      'sub_frame_nce_loss': sub_frame_nce_loss, 'sub_frame_trip_loss': sub_frame_trip_loss,
                      'aud_frame_nce_loss': aud_frame_nce_loss, 'aud_frame_trip_loss': aud_frame_trip_loss,
                      'sub_neg_term_loss': sub_neg_term_loss, 'aud_neg_term_loss': aud_neg_term_loss
                     }

    def encode_query(self, query_feat, query_mask):
        sub_encoded_query = self.encode_input(query_feat, query_mask, self.sub_query_input_proj, self.sub_query_encoder,
                                          self.sub_query_pos_embed, is_query=True, encoder_layer_2=self.sub_query_encoder_2)  # (N, Lq, D)

        aud_encoded_query = self.encode_input(query_feat, query_mask, self.aud_query_input_proj, self.aud_query_encoder,
                                          self.aud_query_pos_embed, is_query=True, encoder_layer_2=self.aud_query_encoder_2)  # (N, Lq, D)

        # sub_weight, aud_weight = self.query_weight(sub_encoded_query, aud_encoded_query, query_mask.unsqueeze(1),
        #                                            self.sub_query_weight_token, self.aud_query_weight_token)

        sub_query = self.re_get_modularized_queries(sub_encoded_query, query_mask, "sub")  # (N, 1, D)

        aud_query = self.re_get_modularized_queries(aud_encoded_query, query_mask, "aud")  # (N, 1, D)

        return sub_query, aud_query, None, None

    def encode_context(self, frame_video_feat, frame_sub_feat, frame_aud_feat, video_mask=None, sub_mask=None, aud_mask=None):

        if self.double_branch:
            #frame level sub-video feat
            s_encoded_frame_video_feat = self.encode_input(frame_video_feat, video_mask, self.frame_s_video_input_proj,
                                                           self.frame_s_video_encoder, self.frame_s_video_pos_embed)
            s_encoded_frame_sub_feat = self.encode_input(frame_sub_feat, sub_mask, self.frame_s_sub_input_proj,
                                                         self.frame_s_sub_encoder, self.frame_s_sub_pos_embed)

            # s_x_encoded_video_feat = self.cross_context_encoder(s_encoded_frame_video_feat, video_mask, s_encoded_frame_sub_feat, sub_mask,
            #                                                     self.frame_s_cross_att, self.frame_s_cross_layernorm, self.frame_s_gmm_aggregate, self.frame_s_weight_token)  # (N, L, D)
            s_x_encoded_video_feat = self.cross_context_encoder(s_encoded_frame_video_feat, video_mask, s_encoded_frame_sub_feat, sub_mask,
                                                                self.frame_s_cross_att, self.frame_s_cross_layernorm, self.frame_s_gmm_aggregate, self.frame_s_weight_token)  # (N, L, D)


            s_x_encoded_video_feat = self.frame_s_video_encoder_2(s_x_encoded_video_feat, video_mask.unsqueeze(1))
            # s_x_encoded_video_feat = self.gmm_context_encoder(s_x_encoded_video_feat, video_mask, self.frame_s_gmm_encoder,
            #                                                 self.frame_s_weight_token, "frame")

            s_encoded_frame_feat = self.frame_s_mapping_linear(s_x_encoded_video_feat)

            # frame level aud-video feat
            a_encoded_frame_video_feat = self.encode_input(frame_video_feat, video_mask, self.frame_a_video_input_proj,
                                                           self.frame_a_video_encoder, self.frame_a_video_pos_embed)
            a_encoded_frame_aud_feat = self.encode_input(frame_aud_feat, aud_mask, self.frame_a_aud_input_proj,
                                                         self.frame_a_aud_encoder, self.frame_a_aud_pos_embed)

            # a_x_encoded_video_feat = self.cross_context_encoder(a_encoded_frame_video_feat, video_mask, a_encoded_frame_aud_feat, aud_mask,
            #                                                     self.frame_a_cross_att, self.frame_a_cross_layernorm, self.frame_a_gmm_aggregate, self.frame_a_weight_token)  # (N, L, D)
            a_x_encoded_video_feat = self.cross_context_encoder(a_encoded_frame_video_feat, video_mask, a_encoded_frame_aud_feat, aud_mask,
                                                                self.frame_a_cross_att, self.frame_a_cross_layernorm, self.frame_a_gmm_aggregate, self.frame_a_weight_token)  # (N, L, D)


            a_x_encoded_video_feat = self.frame_a_video_encoder_2(a_x_encoded_video_feat, video_mask.unsqueeze(1))
            # a_x_encoded_video_feat = self.gmm_context_encoder(a_x_encoded_video_feat, video_mask, self.frame_a_gmm_encoder,
            #                                                 self.frame_a_weight_token, "frame")

            a_encoded_frame_feat = self.frame_a_mapping_linear(a_x_encoded_video_feat)

            if self.video_level_branch:
                # #clip level sub-video feat
                # s_encoded_clip_video_feat = self.encode_input(pool_video_feat, None, self.clip_s_video_input_proj,
                #                                               self.clip_s_video_encoder, self.clip_s_video_pos_embed)
                # s_encoded_clip_sub_feat = self.encode_input(pool_sub_feat, None, self.clip_s_sub_input_proj,
                #                                             self.clip_s_sub_encoder, self.clip_s_sub_pos_embed)
                #
                # s_x_encoded_pool_video_feat = self.cross_context_encoder(s_encoded_clip_video_feat, None,
                #                                                          s_encoded_clip_sub_feat,None,
                #                                                          self.clip_s_cross_att, self.clip_s_cross_layernorm)  # (N, L, D)
                #
                # s_x_encoded_pool_video_feat = self.clip_s_gmm_encoder(s_x_encoded_pool_video_feat, None, weight_token=self.clip_s_weight_token)
                #
                # s_encoded_clip_feat = self.clip_s_mapping_linear(s_x_encoded_pool_video_feat)
                #
                # #clip level aud-video feat
                # a_encoded_clip_video_feat = self.encode_input(pool_video_feat, None, self.clip_a_video_input_proj,
                #                                               self.clip_a_video_encoder, self.clip_a_video_pos_embed)
                # a_encoded_clip_aud_feat = self.encode_input(pool_aud_feat, None, self.clip_a_aud_input_proj,
                #                                             self.clip_a_aud_encoder, self.clip_a_aud_pos_embed)
                #
                # a_x_encoded_pool_video_feat = self.cross_context_encoder(a_encoded_clip_video_feat, None,
                #                                                          a_encoded_clip_aud_feat,None,
                #                                                          self.clip_a_cross_att, self.clip_a_cross_layernorm)  # (N, L, D)
                #
                # a_x_encoded_pool_video_feat = self.clip_a_gmm_encoder(a_x_encoded_pool_video_feat, None, weight_token=self.clip_a_weight_token)
                #
                # a_encoded_clip_feat = self.clip_a_mapping_linear(a_x_encoded_pool_video_feat)

                return s_encoded_frame_feat, a_encoded_frame_feat
            else:
                return None, None, None
        return None, None, None


    @staticmethod
    def encode_input(feat, mask, input_proj_layer, encoder_layer, pos_embed_layer, is_query=False, encoder_layer_2=None,
                     weight_token=None):
        """
        Args:
            feat: (N, L, D_input), torch.float32
            mask: (N, L), torch.float32, with 1 indicates valid query, 0 indicates mask
            input_proj_layer: down project input
            encoder_layer: encoder layer
            pos_embed_layer: positional embedding layer
        """
        feat = input_proj_layer(feat)
        feat = pos_embed_layer(feat)

        if mask is not None:
            mask = mask.unsqueeze(1)  # (N, 1, L), torch.FloatTensor

        if is_query:
            feat = encoder_layer(feat, mask)
            feat = encoder_layer_2(feat, mask)
            return feat
        else:
            if weight_token is not None:
                return encoder_layer(feat, mask, weight_token)  # (N, L, D_hidden)
            return encoder_layer(feat, mask)  # (N, L, D_hidden)


    def get_modularized_queries(self, encoded_query, query_mask, t=1):
        """
        Args:
            encoded_query: (N, L, D)
            query_mask: (N, L)
            return_modular_att: bool
            for the query feat, employ an attention layer to generate the sentence-level feature, t is two branch
        """
        if t == 1:
            modular_attention_scores = self.A_modular_vector_mapping(encoded_query)  # (N, L, 2 or 1)
        if t == 2:
            modular_attention_scores = self.B_modular_vector_mapping(encoded_query)  # (N, L, 2 or 1)
        if t == 3:
            modular_attention_scores = self.C_modular_vector_mapping(encoded_query)  # (N, L, 2 or 1)
        modular_attention_scores = F.softmax(mask_logits(modular_attention_scores, query_mask.unsqueeze(2)), dim=1)
        modular_queries = torch.einsum("blm,bld->bmd", modular_attention_scores, encoded_query)  # (N, 2 or 1, D)
        return modular_queries

    def re_get_modularized_queries(self, encoded_query, query_mask, mark=None, return_modular_att=False):
        """
        Args:
            encoded_query: (N, L, D)
            query_mask: (N, L)
            return_modular_att: bool
        """
        if mark == "sub":
            modular_attention_scores = self.sub_modular_vector_mapping(encoded_query)  # (N, L, 2 or 1)
        else:
            modular_attention_scores = self.aud_modular_vector_mapping(encoded_query)
        modular_attention_scores = F.softmax(mask_logits(modular_attention_scores, query_mask.unsqueeze(2)), dim=1)
        modular_queries = torch.einsum("blm,bld->bmd", modular_attention_scores, encoded_query)  # (N, 2 or 1, D)
        # print("modular_queries:", modular_queries.shape)
        return modular_queries
        # if return_modular_att:
        #     assert modular_queries.shape[1] == 2
        #     return modular_queries[:, 0], modular_queries[:, 1], modular_attention_scores
        # else:
        #     assert modular_queries.shape[1] == 2
        #     return modular_queries[:, 0], modular_queries[:, 1]  # (N, D) * 2

    @staticmethod
    def get_clip_scale_scores(modularied_query, context_feat, mask=None):
        """ Calculate video2query scores for each pair of video and query inside the batch.
        Args:
            modularied_query: (N, D)
            context_feat: (N, L, D), output of the first transformer encoder layer
            context_mask: (N, L)
        Returns:
            context_query_scores: (N, N)  score of each query w.r.t. each video inside the batch,
                diagonal positions are positive. used to get negative samples.
        """
        modularied_query = F.normalize(modularied_query, dim=-1)  # (N, D)
        context_feat = F.normalize(context_feat, dim=-1)  # (N, L, D)
        # print(context_feat.shape)
        if mask is None:
            clip_level_query_context_scores = torch.einsum("md,nld->mln", modularied_query, context_feat)  # (N, L, N)
        else:
            clip_level_query_context_scores = torch.einsum("md,nld->mln", modularied_query, context_feat)
            mask = mask.transpose(0, 1).unsqueeze(0)
            clip_level_query_context_scores = mask_logits(clip_level_query_context_scores, mask)
        # print("mask:", mask.shape, mask) mask: torch.Size([1, 123, 2179])
        # print(clip_level_query_context_scores.shape)
        query_context_scores, indices = torch.max(clip_level_query_context_scores,
                                                  dim=1)  # (N, N) diagonal positions are positive pairs
        # clip_len = clip_level_query_context_scores.shape[1]
        # vals, indices = clip_level_query_context_scores.topk(k=5, dim=1, largest=True, sorted=True)#(N, topK, N)
        # query_context_scores = torch.mean(vals, dim=1)#(N, N)
        return query_context_scores, clip_level_query_context_scores, indices

    @staticmethod
    def get_unnormalized_clip_scale_scores(modularied_query, context_feat, mask=None):
        """ Calculate video2query scores for each pair of video and query inside the batch.
        Args:
            modularied_query: (N, D)
            context_feat: (N, L, D), output of the first transformer encoder layer
            context_mask: (N, L)
        Returns:
            context_query_scores: (N, N)  score of each query w.r.t. each video inside the batch,
                diagonal positions are positive. used to get negative samples.
        """

        if mask is None:
            query_context_scores = torch.einsum("md,nld->mln", modularied_query, context_feat)
        else:
            query_context_scores = torch.einsum("md,nld->mln", modularied_query, context_feat)
            mask = mask.transpose(0, 1).unsqueeze(0)
            query_context_scores = mask_logits(query_context_scores, mask)
        query_context_scores, _ = torch.max(query_context_scores, dim=1)
        return query_context_scores

    def key_clip_guided_attention(self, frame_feat, proposal_feat, feat_mask, max_index, query_labels):
        selected_max_index = max_index[[i for i in range(max_index.shape[0])], query_labels]
        # print("selected_max_index", selected_max_index.shape) [640]
        expand_frame_feat = frame_feat[query_labels]

        expand_proposal_feat = proposal_feat[query_labels]
        # print("expand_proposal_feat", expand_proposal_feat.shape) [640,384]
        key = self.mapping_linear[0](expand_frame_feat)
        query = expand_proposal_feat[[i for i in range(key.shape[0])], selected_max_index, :].unsqueeze(-1)
        value = self.mapping_linear[1](expand_frame_feat)

        if feat_mask is not None:
            expand_feat_mask = feat_mask[query_labels]
            scores = torch.bmm(key, query).squeeze()
            masked_scores = scores.masked_fill(expand_feat_mask.eq(0), -1e9).unsqueeze(1)
            masked_scores = nn.Softmax(dim=-1)(masked_scores)
            attention_feat = torch.bmm(masked_scores, value).squeeze()
        else:
            scores = nn.Softmax(dim=-1)(torch.bmm(key, query).transpose(1, 2))
            attention_feat = torch.bmm(scores, value).squeeze()

        return attention_feat

    def key_clip_guided_attention_in_inference(self, frame_feat, proposal_feat, feat_mask, max_index):
        key = self.mapping_linear[0](frame_feat)
        value = self.mapping_linear[1](frame_feat)
        num_vid = frame_feat.shape[0]

        index = torch.arange(num_vid).unsqueeze(1)
        query = proposal_feat[index, max_index.t()]
        if feat_mask is not None:
            scores = torch.bmm(key, query.transpose(2, 1))
            masked_scores = scores.masked_fill(feat_mask.unsqueeze(-1).eq(0), -1e9)
            masked_scores = nn.Softmax(dim=1)(masked_scores)
            attention_feat = torch.bmm(masked_scores.transpose(1, 2), value)
        else:
            scores = torch.bmm(key, query.transpose(2, 1))
            scores = nn.Softmax(dim=1)(scores)
            attention_feat = torch.bmm(scores.transpose(1, 2), value)

        return attention_feat

    def get_pred_from_raw_query(self, sub_query, aud_query, query_labels=None, sub_frame_feat=None, aud_frame_feat=None, frame_feat_mask=None,
                                sub_weight=None, aud_weight=None, return_query_feats=False):
        #use when you are in eval mode
        if not return_query_feats:
            sub_query, aud_query, sub_weight, aud_weight = self.encode_query(sub_query, aud_query)

        sub_query = sub_query.squeeze(1) #(Nq, d)
        aud_query = aud_query.squeeze(1) #(Nq, d)

        # sub_weight = sub_weight.unsqueeze(-1) #(Nq,)
        # aud_weight = aud_weight.unsqueeze(-1) #(Nq,)

        sub_frame_score, _, _ = self.get_clip_scale_scores(sub_query, sub_frame_feat, frame_feat_mask)
        aud_frame_score, _, _ = self.get_clip_scale_scores(aud_query, aud_frame_feat, frame_feat_mask)
        # raw_frame_score = (sub_frame_score + aud_frame_score) / 2.0
        # frame_score = (sub_frame_score * sub_weight + aud_frame_score * aud_weight)


        if return_query_feats:
            sub_frame_score_ = self.get_unnormalized_clip_scale_scores(sub_query, sub_frame_feat, frame_feat_mask)
            aud_frame_score_ = self.get_unnormalized_clip_scale_scores(aud_query, aud_frame_feat, frame_feat_mask)

            return sub_frame_score, sub_frame_score_, aud_frame_score, aud_frame_score_
        else:
            return sub_frame_score, aud_frame_score

    def get_modularized_videos(self, encoded_video_C, video_mask):
        """
        Args:
            encoded_query: (N, L, D)
            query_mask: (N, L)
            return_modular_att: bool
        """
        # print("encoded_video_A", encoded_video_A.shape)
        modular_attention_scores_C = self.C_video_vector_mapping(encoded_video_C)
        modular_attention_scores_C = F.softmax(mask_logits(modular_attention_scores_C, video_mask.unsqueeze(-1)), dim=1)
        modular_videos_C = torch.squeeze(
            torch.einsum("blm,bld->bmd", modular_attention_scores_C, encoded_video_C))  # (N, D)
        return modular_videos_C

    # '''2025/2/27-start'''
    # def get_neg_scores_loss(self, query_context_scores, labels):
    #     v2t_scores = query_context_scores.t()
    #     t2v_scores = query_context_scores
    #     labels = np.array(labels)
    #
    #     # cal_v2t_loss
    #     v2t_loss = 0
    #     for i in range(v2t_scores.shape[0]):
    #         # pos_pair_scores = torch.mean(v2t_scores[i][np.where(labels == i)])
    #         neg_pair_scores, _ = torch.sort(v2t_scores[i][np.where(labels != i)[0]], descending=True)
    #         if self.config.use_hard_negative:
    #             sample_neg_pair_scores = neg_pair_scores[0]
    #         else:
    #             v2t_sample_max_idx = neg_pair_scores.shape[0]
    #             sample_neg_pair_scores = neg_pair_scores[
    #                 torch.randint(0, v2t_sample_max_idx, size=(1,)).to(v2t_scores.device)]
    #         v2t_loss += (- torch.log(1. - torch.sigmoid(sample_neg_pair_scores))).sum()
    #
    #     # cal_t2v_loss
    #     text_indices = torch.arange(t2v_scores.shape[0]).to(t2v_scores.device)
    #     # t2v_pos_scores = t2v_scores[text_indices, labels]
    #     mask_score = copy.deepcopy(t2v_scores.data)
    #     mask_score[text_indices, labels] = 999
    #     _, sorted_scores_indices = torch.sort(mask_score, descending=True, dim=1)
    #     t2v_sample_max_idx = min(1 + self.config.hard_pool_size,
    #                              t2v_scores.shape[1]) if self.config.use_hard_negative else t2v_scores.shape[1]
    #     sample_indices = sorted_scores_indices[
    #         text_indices, torch.randint(1, t2v_sample_max_idx, size=(t2v_scores.shape[0],)).to(t2v_scores.device)]
    #     # sample_indices = sorted_scores_indices[
    #     #     text_indices, torch.randint(1, 2, size=(t2v_scores.shape[0],)).to(t2v_scores.device)]
    #     t2v_neg_scores = t2v_scores[text_indices, sample_indices]
    #     t2v_loss = (- torch.log(1. - torch.sigmoid(t2v_neg_scores))).sum()
    #
    #     return t2v_loss / len(t2v_scores) + v2t_loss / len(v2t_scores)
    # '''2025/2/27-end'''

    '''2025/2/28-start'''

    def query_diverse_loss(self, x, label_dict):

        bs = x.shape[0]
        x = F.normalize(x, dim=-1)
        cos = torch.matmul(x, x.t())

        N_one_hot = torch.zeros((bs, bs))
        for i, label in label_dict.items():
            N_one_hot[label[0]:(label[-1] + 1), label[0]:(label[-1] + 1)] = torch.ones((len(label), len(label)))
        N_one_hot = N_one_hot - torch.eye(bs)
        N_one_hot = N_one_hot.cuda()

        neg_exp = torch.exp(self.alpha * (cos + self.mrg))

        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp))
        focal = torch.where(N_one_hot == 1, cos, torch.zeros_like(cos))

        neg_term = (((1 + focal) ** self.lamda) * torch.log(1 + N_sim_sum)).sum(dim=0).sum() / bs

        return neg_term

    '''2025/2/28-end'''

    def get_clip_triplet_loss(self, query_context_scores, labels):
        v2t_scores = query_context_scores.t()
        t2v_scores = query_context_scores
        labels = np.array(labels)

        # cal_v2t_loss
        v2t_loss = 0
        for i in range(v2t_scores.shape[0]):
            pos_pair_scores = torch.mean(v2t_scores[i][np.where(labels == i)])
            neg_pair_scores, _ = torch.sort(v2t_scores[i][np.where(labels != i)[0]], descending=True)
            if self.config.use_hard_negative:
                sample_neg_pair_scores = neg_pair_scores[0]
            else:
                v2t_sample_max_idx = neg_pair_scores.shape[0]
                sample_neg_pair_scores = neg_pair_scores[
                    torch.randint(0, v2t_sample_max_idx, size=(1,)).to(v2t_scores.device)]
            v2t_loss += (self.config.margin + sample_neg_pair_scores - pos_pair_scores).clamp(min=0).sum()

        # cal_t2v_loss
        text_indices = torch.arange(t2v_scores.shape[0]).to(t2v_scores.device)
        t2v_pos_scores = t2v_scores[text_indices, labels]
        mask_score = copy.deepcopy(t2v_scores.data)
        mask_score[text_indices, labels] = 999
        _, sorted_scores_indices = torch.sort(mask_score, descending=True, dim=1)
        t2v_sample_max_idx = min(1 + self.config.hard_pool_size,
                                 t2v_scores.shape[1]) if self.config.use_hard_negative else t2v_scores.shape[1]
        sample_indices = sorted_scores_indices[
            text_indices, torch.randint(1, t2v_sample_max_idx, size=(t2v_scores.shape[0],)).to(t2v_scores.device)]
        # sample_indices = sorted_scores_indices[
        #     text_indices, torch.randint(1, 2, size=(t2v_scores.shape[0],)).to(t2v_scores.device)]
        t2v_neg_scores = t2v_scores[text_indices, sample_indices]

        t2v_loss = (self.config.margin + t2v_neg_scores - t2v_pos_scores).clamp(min=0)

        # print((t2v_loss.sum() / len(t2v_scores) + v2t_loss / len(v2t_scores)).shape)
        return t2v_loss.sum() / len(t2v_scores) + v2t_loss / len(v2t_scores)

    def get_frame_trip_loss(self, query_context_scores):
        """ ranking loss between (pos. query + pos. video) and (pos. query + neg. video) or (neg. query + pos. video)
        Args:
            query_context_scores: (N, N), cosine similarity [-1, 1],
                Each row contains the scores between the query to each of the videos inside the batch.
        """

        bsz = len(query_context_scores)

        diagonal_indices = torch.arange(bsz).to(query_context_scores.device)
        pos_scores = query_context_scores[diagonal_indices, diagonal_indices]  # (N, )
        query_context_scores_masked = copy.deepcopy(query_context_scores.data)
        # impossibly large for cosine similarity, the copy is created as modifying the original will cause error
        query_context_scores_masked[diagonal_indices, diagonal_indices] = 999
        pos_query_neg_context_scores = self.get_neg_scores(query_context_scores, query_context_scores_masked)
        neg_query_pos_context_scores = self.get_neg_scores(query_context_scores.transpose(0, 1),
                                                           query_context_scores_masked.transpose(0, 1))
        loss_neg_ctx = self.get_ranking_loss(pos_scores, pos_query_neg_context_scores)
        loss_neg_q = self.get_ranking_loss(pos_scores, neg_query_pos_context_scores)
        return loss_neg_ctx + loss_neg_q

    def get_neg_scores(self, scores, scores_masked):
        """
        scores: (N, N), cosine similarity [-1, 1],
            Each row are scores: query --> all videos. Transposed version: video --> all queries.
        scores_masked: (N, N) the same as scores, except that the diagonal (positive) positions
            are masked with a large value.
        """

        bsz = len(scores)
        batch_indices = torch.arange(bsz).to(scores.device)

        _, sorted_scores_indices = torch.sort(scores_masked, descending=True, dim=1)

        sample_min_idx = 1  # skip the masked positive

        sample_max_idx = min(sample_min_idx + self.config.hard_pool_size, bsz) if self.config.use_hard_negative else bsz

        # sample_max_idx = 2

        # (N, )
        sampled_neg_score_indices = sorted_scores_indices[batch_indices, torch.randint(sample_min_idx, sample_max_idx,
                                                                                       size=(bsz,)).to(scores.device)]

        sampled_neg_scores = scores[batch_indices, sampled_neg_score_indices]  # (N, )
        return sampled_neg_scores

    def get_ranking_loss(self, pos_score, neg_score):
        """ Note here we encourage positive scores to be larger than negative scores.
        Args:
            pos_score: (N, ), torch.float32
            neg_score: (N, ), torch.float32
        """
        return torch.clamp(self.config.margin + neg_score - pos_score, min=0).sum() / len(pos_score)

    @staticmethod
    def threshold_topk_nd(score: torch.Tensor, d: float) -> torch.Tensor:
        """
        对输入形状为 (m, n, lv) 的 score 张量的每个 (lv,) 向量：
        - 从大到小排序
        - 累加直到超过阈值 d
        - 保留这些最大值，其余置零

        Args:
            score: Tensor of shape (m, n, lv)，最后一维每个向量之和为 1
            d: float, 累积阈值，0 < d <= 1

        Returns:
            Tensor of shape (m, n, lv)，仅保留 top-k 和大于 d 的值，其余为 0
        """
        m, n, lv = score.shape
        score_flat = score.view(-1, lv)  # 变成 (bs, lv)，其中 bs = m * n

        # 排序并获取索引
        sorted_vals, sorted_idx = torch.sort(score_flat, dim=1, descending=True)
        cumsum = sorted_vals.cumsum(dim=1)

        # 找每行第一个累积和大于 d 的位置
        mask_pos = (cumsum >= d).float().argmax(dim=1)  # shape: (bs,)

        bs = score_flat.shape[0]
        row_idx = torch.arange(bs, device=score.device).unsqueeze(1)
        col_idx = torch.arange(lv, device=score.device).unsqueeze(0)

        sorted_mask = col_idx <= mask_pos.unsqueeze(1)

        # 用排序索引恢复到原始顺序
        mask = torch.zeros_like(score_flat, dtype=torch.bool)
        mask.scatter_(1, sorted_idx, sorted_mask)

        # 应用掩码并 reshape 回 (m, n, lv)
        out = (score_flat * mask.to(score.dtype)).view(m, n, lv)
        return out

    def batch_generate_gauss_masks(X: torch.Tensor, width: float) -> torch.Tensor:
        """
        向量化生成形状 (Nq, Nv, lv) 的高斯掩码：
        对于 X[i,j]，以其最大值位置 K 为中心，生成长度 lv 的高斯掩码。
        """
        Nq, Nv, lv = X.shape
        device = X.device
        dtype = X.dtype

        # 1) 找到每条子序列的最大值索引 K，形状 (Nq, Nv)
        Ks = X.argmax(dim=-1).to(dtype)  # float, 方便后面算

        # 2) 生成归一化后的位置向量 positions，形状 (lv,)
        positions = torch.arange(lv, device=device, dtype=dtype) / (lv - 1)

        # 3) 计算每条子序列的归一化中心 center，形状 (Nq, Nv)
        centers = Ks / (lv - 1)  # 0 ~ 1

        # 4) 计算 sigma
        sigma = max(width, 1e-2) / 9.0

        # 5) 计算 diff = positions - centers[..., None]，形状 (Nq, Nv, lv)
        #    利用广播： centers[...,None] -> (Nq,Nv,1)，positions -> (1,1,lv)
        diff = positions.view(1, 1, lv) - centers.unsqueeze(-1)

        # 6) 生成高斯权重（未归一化），形状 (Nq, Nv, lv)
        gauss = torch.exp(- (diff ** 2) / (2 * sigma ** 2))

        # 7) 按每条子序列最大值归一化
        max_per_row = gauss.amax(dim=-1, keepdim=True)  # (Nq, Nv, 1)
        masks = gauss / max_per_row

        return masks

    @staticmethod
    def get_TIB_scores(modularied_query, context_feat, mask=None):
        """ Calculate video2query scores for each pair of video and query inside the batch.
        Args:
            modularied_query: (N, D)
            context_feat: (N, L, D), output of the first transformer encoder layer
            context_mask: (N, L)
        Returns:
            context_query_scores: (N, N)  score of each query w.r.t. each video inside the batch,
                diagonal positions are positive. used to get negative samples.
        """
        if mask is None:
            weight = torch.einsum("md,nld->mln", modularied_query, context_feat)  # (m, l, n)
        else:
            weight = torch.einsum("md,nld->mln", modularied_query, context_feat)
            mask = mask.transpose(0, 1).unsqueeze(0)
            weight = mask_logits(weight, mask)

        weight = weight.permute(0, 2, 1)  # （m， n， l）

        weight_mask = DLDKD.batch_generate_gauss_masks(weight, width=3.0)
        weight = torch.einsum("mnl,mnl->mnl", weight_mask, weight)

        weight_softmax = F.softmax(weight / 0.6, dim=-1)
        weighted_context_feat = torch.einsum("mnl,nld->mnd", weight_softmax, context_feat)
        weighted_context_feat = F.normalize(weighted_context_feat, dim=-1)
        modularied_query = F.normalize(modularied_query, dim=-1)
        TIB_level_scores = torch.einsum("md,mnd->mn", modularied_query, weighted_context_feat)
        return TIB_level_scores

    '''2025/3/17-end'''

    '''2025/3/17-start'''

    @staticmethod
    def get_unnormalized_TIB_scores(modularied_query, context_feat, mask=None):
        """ Calculate video2query scores for each pair of video and query inside the batch.
        Args:
            modularied_query: (N, D)
            context_feat: (N, L, D), output of the first transformer encoder layer
            context_mask: (N, L)
        Returns:
            context_query_scores: (N, N)  score of each query w.r.t. each video inside the batch,
                diagonal positions are positive. used to get negative samples.
        """
        # modularied_query = F.normalize(modularied_query, dim=-1)  # (N, D)
        # context_feat = F.normalize(context_feat, dim=-1)  # (N, L, D)
        # # print(context_feat.shape)
        if mask is None:
            weight = torch.einsum("md,nld->mln", modularied_query, context_feat)  # (m, l, n)
        else:
            weight = torch.einsum("md,nld->mln", modularied_query, context_feat)
            mask = mask.transpose(0, 1).unsqueeze(0)
            weight = mask_logits(weight, mask)

        weight = weight.permute(0, 2, 1)  # （m， n， l）

        weight_mask = DLDKD.batch_generate_gauss_masks(weight, width=3.0)
        weight = torch.einsum("mnl,mnl->mnl", weight_mask, weight)

        weight_softmax = F.softmax(weight / 0.6, dim=-1)
        weighted_context_feat = torch.einsum("mnl,nld->mnd", weight_softmax, context_feat)
        TIB_level_scores_ = torch.einsum("md,mnd->mn", modularied_query, weighted_context_feat)
        return TIB_level_scores_

    '''2025/3/17-end'''

    @staticmethod
    def cross_context_encoder(main_context_feat, main_context_mask, side_context_feat, side_context_mask,
                              cross_att_layer, norm_layer, aggregate=None, weight_token=None):
        """
        Args:
            main_context_feat: (N, Lq, D)
            main_context_mask: (N, Lq)
            side_context_feat: (N, Lk, D)
            side_context_mask: (N, Lk)
            cross_att_layer: cross attention layer
            norm_layer: layer norm layer
            aggregate:
            weight_token:
        """
        cross_mask = None
        if main_context_mask is not None:
            cross_mask = torch.einsum("bm,bn->bmn", main_context_mask, side_context_mask)  # (N, Lq, Lk)
        cross_out = cross_att_layer(main_context_feat, side_context_feat, side_context_feat, cross_mask)  # (N, Lq, D)
        if aggregate is not None:
            cross_weight, main_weight = aggregate(cross_out, main_context_feat, main_context_mask.unsqueeze(1), weight_token) #(N, Lq, 1) * 2
            residual_out = norm_layer(cross_out * cross_weight + main_context_feat * main_weight) #(N, Lq, D)
        else:
            residual_out = norm_layer(cross_out + main_context_feat) #(N, Lq, D)
        return residual_out

    # @staticmethod
    # def cross_context_encoder(main_context_feat, main_context_mask, side_context_feat, side_context_mask,
    #                           cross_att_layer, norm_layer, Dyaggregate=None, weight_token=None, t=None):
    #     """
    #     Args:
    #         main_context_feat: (N, Lq, D)
    #         main_context_mask: (N, Lq)
    #         side_context_feat: (N, Lk, D)
    #         side_context_mask: (N, Lk)
    #         cross_att_layer: cross attention layer
    #         norm_layer: layer norm layer
    #         mha_mapping: fusion score
    #     """
    #     cross_mask = None
    #     if main_context_mask is not None:
    #         cross_mask = torch.einsum("bm,bn->bmn", main_context_mask, side_context_mask)  # (N, Lq, Lk)
    #
    #     cross_out = cross_att_layer(main_context_feat, side_context_feat, side_context_feat, cross_mask)  # (N, Lq, D)
    #     # print("mha_score.shape:", mha_score.shape)
    #
    #     if Dyaggregate is None:
    #         residual_out = norm_layer(cross_out + main_context_feat)
    #         return residual_out
    #
    #
    #     if t == "frame" and cross_out.shape[1] != 128:
    #         vid_len = cross_out.shape[1]
    #         vid_fix = 128 - cross_out.shape[1]
    #         temp_feat = 0.0 * main_context_feat.mean(dim=1, keepdim=True).repeat(1, vid_fix, 1)
    #         cross_temp_feat = 0.0 * cross_out.mean(dim=1, keepdim=True).repeat(1, vid_fix, 1)
    #         main_context_feat = torch.cat([main_context_feat, temp_feat], dim=1)
    #         cross_out = torch.cat([cross_out, cross_temp_feat], dim=1)
    #
    #         temp_mask = 0.0 * main_context_mask.mean(dim=1, keepdim=True).repeat(1, vid_fix).type_as(main_context_mask)
    #         video_mask = torch.cat([main_context_mask, temp_mask], dim=1)
    #         residual_out = Dyaggregate(cross_out, main_context_feat, video_mask.unsqueeze(1), weight_token=weight_token)
    #         residual_out = torch.where(
    #             video_mask.unsqueeze(-1).repeat(1, 1, residual_out.shape[-1]) == 1.0, \
    #             residual_out, 0. * residual_out)
    #         residual_out = residual_out[:, : vid_len, :]
    #     else:
    #         if main_context_mask is not None:
    #             residual_out = Dyaggregate(cross_out, main_context_feat, main_context_mask.unsqueeze(1), weight_token=weight_token)
    #         else:
    #             residual_out = Dyaggregate(cross_out, main_context_feat, None, weight_token=weight_token)
    #
    #
    #     residual_out = norm_layer(residual_out)
    #     return residual_out


    @staticmethod
    def gmm_context_encoder(context_feat, context_mask, gmm_encoder, weight_token, t=None):
        if t == "frame":
            if context_feat.shape[1] != 128:
                context_len = context_feat.shape[1]
                context_fix = 128 - context_feat.shape[1]

                temp_feat = 0.0 * context_feat.mean(dim=1, keepdim=True).repeat(1, context_fix, 1)
                context_feat = torch.cat([context_feat, temp_feat], dim=1)

                temp_mask = 0.0 * context_mask.mean(dim=1, keepdim=True).repeat(1, context_fix).type_as(context_mask)
                context_mask = torch.cat([context_mask, temp_mask], dim=1)
                encoded_context_feat = gmm_encoder(context_feat, context_mask.unsqueeze(1), weight_token)
                encoded_context_feat = torch.where(
                    context_mask.unsqueeze(-1).repeat(1, 1, encoded_context_feat.shape[-1]) == 1.0, \
                    encoded_context_feat, 0. * encoded_context_feat)
                encoded_context_feat = encoded_context_feat[:, : context_len, :]
            else:
                encoded_context_feat = gmm_encoder(context_feat, context_mask.unsqueeze(1), weight_token)
        else:
            encoded_context_feat = gmm_encoder(context_feat, None, weight_token)

        return encoded_context_feat


def mask_logits(target, mask):
    return target * mask + (1 - mask) * (-1e10)