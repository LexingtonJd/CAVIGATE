import json
import torch
import torch.utils.data as data
import numpy as np
import re
import h5py


def getVideoId(cap_id):
    vid_id = cap_id.split('#')[0]
    return vid_id

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9]", " ", string)
    return string.strip().lower().split()

def read_video_ids(cap_file):
    video_ids_list = []
    with open(cap_file, 'r') as cap_reader:
        for line in cap_reader.readlines():
            cap_id, caption = line.strip().split(' ', 1)
            video_id = getVideoId(cap_id)
            if video_id not in video_ids_list:
                video_ids_list.append(video_id)
    return video_ids_list

def average_to_fixed_length(visual_input, map_size):
    visual_input = torch.from_numpy(visual_input)
    num_sample_clips = map_size
    num_clips = visual_input.shape[0]
    idxs = torch.arange(0, num_sample_clips + 1, 1.0) / num_sample_clips * num_clips

    idxs = torch.min(torch.round(idxs).long(), torch.tensor(num_clips - 1))

    new_visual_input = []

    for i in range(num_sample_clips):

        s_idx, e_idx = idxs[i].item(), idxs[i + 1].item()
        if s_idx < e_idx:
            new_visual_input.append(torch.mean(visual_input[s_idx:e_idx], dim=0))
        else:
            new_visual_input.append(visual_input[s_idx])
    new_visual_input = torch.stack(new_visual_input, dim=0).numpy()


    return new_visual_input

def uniform_feature_sampling(features, max_len):
    num_clips = features.shape[0]

    if max_len is None or num_clips <= max_len:
        return features
    idxs = np.arange(0, max_len + 1, 1.0) / max_len * num_clips
    idxs = np.round(idxs).astype(np.int32)
    idxs[idxs > num_clips - 1] = num_clips - 1
    new_features = []
    for i in range(max_len):
        s_idx, e_idx = idxs[i], idxs[i + 1]
        if s_idx < e_idx:
            new_features.append(np.mean(features[s_idx:e_idx], axis=0))
        else:
            new_features.append(features[s_idx])
    new_features = np.asarray(new_features)
    return new_features


def l2_normalize_np_array(np_array, eps=1e-5):
    """np_array: np.ndarray, (*, D), where the last dim will be normalized"""
    return np_array / (np.linalg.norm(np_array, axis=-1, keepdims=True) + eps)



#用于将一个batch（列表）的原始样本组合成训练时所需的张量格式。每个样本是一个tuple，包含视频特征、对应的多条caption特征、以及其他辅助信息。
def collate_train(data):
    """
    Build mini-batch tensors from a list of (video, caption) tuples.
    """
    # Sort a data list by caption length
    #首先检查第一个样本的 “caption 列表” (data[0][1]) 是否存在。
    if data[0][1] is not None:
        #如果存在，就按照每个样本中所有caption序列的长度总和进行降序排序（越长的排在越前面），以便后续对齐时效率更高。
        data.sort(key=lambda x: len(x[1]), reverse=True)
    # pool_video_features, pool_sub_features, pool_aud_features, frame_video_features, frame_sub_features, frame_aud_features, captions, idxs, cap_ids, video_ids = zip(*data)
    frame_video_features, frame_sub_features, frame_aud_features, captions, idxs, cap_ids, video_ids = zip(*data)
    kong=[]
    for i in cap_ids:
        kong = np.concatenate((kong,i))
    cap_ids=kong
    # #videos
    # #将每个样本的 pool_video_features（如果使用 video-level branch，每个都是形如 [1, D] 的张量）在第 0 维拼接成形状 [batch_size, D] 的张量。
    # pool_videos = torch.cat(pool_video_features, dim=0).float()
    # pool_subs = torch.cat(pool_sub_features, dim=0).float()
    # pool_auds = torch.cat(pool_aud_features, dim=0).float()

    #video_lengths：记录每个样本帧序列的长度（即每个视频有多少帧被抽取出来）。
    video_lengths = [len(frame) for frame in frame_video_features]
    sub_lengths = [len(sub) for sub in frame_sub_features]
    aud_lengths = [len(aud) for aud in frame_aud_features]

    #frame_vec_len：每一帧向量的维度大小。
    frame_vec_len = len(frame_video_features[0][0])
    sub_vec_len = len(frame_sub_features[0][0])
    aud_vec_len = len(frame_aud_features[0][0])
    #初始化两个张量：
        #frame_videos，形状 [batch_size, max_num_frames, frame_vec_len]，存放对齐后的帧特征；
        #videos_mask，形状 [batch_size, max_num_frames]，存放每帧是否存在的掩码（padding 部分为 0）。
    frame_videos = torch.zeros(len(frame_video_features), max(video_lengths), frame_vec_len)
    frame_subs = torch.zeros(len(frame_sub_features), max(sub_lengths), sub_vec_len)
    frame_auds = torch.zeros(len(frame_aud_features), max(aud_lengths), aud_vec_len)
    videos_mask = torch.zeros(len(frame_video_features), max(video_lengths))
    subs_mask = torch.zeros(len(frame_sub_features), max(sub_lengths))
    auds_mask = torch.zeros(len(frame_aud_features), max(aud_lengths))
    #遍历每个样本，将其真实帧特征复制到对应位置，并将 mask 对应位置置为 1。
    for i, frames in enumerate(frame_video_features):
        end = video_lengths[i]
        frame_videos[i, :end, :] = frames[:end, :]
        videos_mask[i, :end] = 1.0

    for i, subs in enumerate(frame_sub_features):
        end = sub_lengths[i]
        frame_subs[i, :end, :] = subs[:end, :]
        subs_mask[i, :end] = 1.0

    for i, auds in enumerate(frame_aud_features):
        end = aud_lengths[i]
        frame_auds[i, :end, :] = auds[:end, :]
        auds_mask[i, :end] = 1.0

    #captions
    feat_dim = captions[0][0].shape[-1]

    merge_captions = []
    all_lengths = []
    labels = []

    for index, caps in enumerate(captions):
        labels.extend(index for _ in range(len(caps)))
        all_lengths.extend(len(cap) for cap in caps)
        merge_captions.extend(cap for cap in caps)

    target = torch.zeros(len(all_lengths), max(all_lengths), feat_dim)
    words_mask = torch.zeros(len(all_lengths), max(all_lengths))

    for index, cap in enumerate(merge_captions):
        end = all_lengths[index]
        target[index, :end, :] = cap[:end, :]
        words_mask[index, :end] = 1.0

    # if clip_captions[0] is not None:
    #
    #     clip_vec_len = len(clip_video_features[0][0])
    #     clip_videos = torch.zeros(len(clip_video_features), max(video_lengths), clip_vec_len)
    #     for i, frames in enumerate(clip_video_features):
    #         end = video_lengths[i]
    #         clip_videos[i, :end, :] = frames[:end, :]
    #
    #     feat_dim = clip_captions[0][0].shape[-1]
    #     merge_captions = []
    #     clip_labels = []
    #
    #     for index, caps in enumerate(clip_captions):
    #         clip_labels.extend(index for _ in range(len(caps)))
    #         merge_captions.extend(cap for cap in caps)
    #
    #     clip_target = torch.zeros(len(all_lengths), feat_dim)
    #
    #     for index, cap in enumerate(merge_captions):
    #         clip_target[index, :] = cap
    # else:
    #     clip_target=None
    #     clip_videos=None

    # return dict(pool_video_features=pool_videos,
    #             pool_sub_features=pool_subs,
    #             frame_video_features=frame_videos,
    #             frame_sub_features=frame_subs,
    #             clip_video_features=clip_videos,
    #             videos_mask=videos_mask,
    #             subs_mask=subs_mask,
    #             text_feat=target,
    #             text_mask=words_mask,
    #             text_labels=labels,
    #             clip_text_feat=clip_target,
    #             cap_ids=cap_ids
    #             )
    return dict(frame_video_features=frame_videos,
                frame_sub_features=frame_subs,
                frame_aud_features=frame_auds,
                videos_mask=videos_mask,
                subs_mask=subs_mask,
                auds_mask=auds_mask,
                text_feat=target,
                text_mask=words_mask,
                text_labels=labels,
                cap_ids=cap_ids
                )


def collate_frame_val(data):
    # pool_video_features, pool_sub_features, pool_aud_features, frame_video_features, frame_sub_features, frame_aud_features, idxs, video_ids = zip(*data)
    frame_video_features, frame_sub_features, frame_aud_features, idxs, video_ids = zip(*data)

    # Merge videos (convert tuple of 1D tensor to 4D tensor)
    # # videos
    # pool_videos = torch.cat(pool_video_features, dim=0).float()
    # pool_subs = torch.cat(pool_sub_features, dim=0).float()
    # pool_auds = torch.cat(pool_aud_features, dim=0).float()
    video_lengths = [len(frame) for frame in frame_video_features]
    sub_lengths = [len(sub) for sub in frame_sub_features]
    aud_lengths = [len(aud) for aud in frame_aud_features]
    frame_vec_len = len(frame_video_features[0][0])
    sub_vec_len = len(frame_sub_features[0][0])
    aud_vec_len = len(frame_aud_features[0][0])
    frame_videos = torch.zeros(len(frame_video_features), max(video_lengths), frame_vec_len)
    frame_subs = torch.zeros(len(frame_sub_features), max(sub_lengths), sub_vec_len)
    frame_auds = torch.zeros(len(frame_aud_features), max(aud_lengths), aud_vec_len)
    videos_mask = torch.zeros(len(frame_video_features), max(video_lengths))
    subs_mask = torch.zeros(len(frame_sub_features), max(sub_lengths))
    auds_mask = torch.zeros(len(frame_aud_features), max(aud_lengths))
    for i, frames in enumerate(frame_video_features):
        end = video_lengths[i]
        frame_videos[i, :end, :] = frames[:end, :]
        videos_mask[i, :end] = 1.0

    for i, subs in enumerate(frame_sub_features):
        end = sub_lengths[i]
        frame_subs[i, :end, :] = subs[:end, :]
        subs_mask[i, :end] = 1.0

    for i, auds in enumerate(frame_aud_features):
        end = aud_lengths[i]
        frame_auds[i, :end, :] = auds[:end, :]
        auds_mask[i, :end] = 1.0

    return frame_videos, frame_subs, frame_auds, videos_mask, subs_mask, auds_mask, idxs, video_ids


def collate_text_val(data):
    if data[0][0] is not None:
        data.sort(key=lambda x: len(x[0]), reverse=True)
    captions, idxs, cap_ids = zip(*data)

    if captions[0] is not None:
        # Merge captions (convert tuple of 1D tensor to 2D tensor)
        lengths = [len(cap) for cap in captions]
        target = torch.zeros(len(captions), max(lengths), captions[0].shape[-1])
        words_mask = torch.zeros(len(captions), max(lengths))
        for i, cap in enumerate(captions):
            end = lengths[i]
            target[i, :end] = cap[:end]
            words_mask[i, :end] = 1.0
    else:
        target = None
        words_mask = None

    return target, words_mask, idxs, cap_ids
import os
class Dataset4DLDKD(data.Dataset):
    """
    Load captions and video frame features by pre-trained CNN model.
    """

    def __init__(self, cap_file, clip_vid_feat_path, sub_feat_path, aud_feat_path, text_feat_path, opt, video2frames=None):
        # Captions
        self.captions = {}
        self.cap_ids = []
        self.video_ids = []
        self.vid_caps = {}
        self.video2frames = video2frames

        with open(cap_file, 'r') as cap_reader:
            for line in cap_reader.readlines():
                cap_id, caption = line.strip().split(' ', 1)
                video_id = getVideoId(cap_id)
                self.captions[cap_id] = caption
                self.cap_ids.append(cap_id)
                if video_id not in self.video_ids:
                    self.video_ids.append(video_id)
                if video_id in self.vid_caps:
                    self.vid_caps[video_id].append(cap_id)
                else:
                    self.vid_caps[video_id] = []
                    self.vid_caps[video_id].append(cap_id)
        self.text_feat_path = text_feat_path


        self.map_size = opt.map_size
        self.max_ctx_len = opt.max_ctx_l
        self.max_desc_len = opt.max_desc_l

        self.length = len(self.vid_caps)
        self.text_feat = h5py.File(self.text_feat_path, 'r')

        self.use_clip = opt.use_clip
        self.video_level_branch = opt.video_level_branch
        # print(self.use_clip)
        # print(type(self.use_clip))
        # exit()
        if self.use_clip:
            # self.clip_text_feat_path = clip_text_feat_path
            self.clip_vid_feat_path = clip_vid_feat_path
            # self.clip_text_feat = h5py.File(self.clip_text_feat_path, 'r')
            self.clip_vid_feat = h5py.File(self.clip_vid_feat_path, 'r')
            #
            # self.clip_text_feat = h5py.File(os.path.join("/home/zms/cxk/TCL/ac_tcl_text_feat.hdf5"), 'r')
            # self.clip_vid_feat = h5py.File(os.path.join("/home/zms/cxk/TCL/ac_tcl_vid_feat.hdf5"), 'r')
        self.sub_feat_path = sub_feat_path
        self.sub_feat = h5py.File(self.sub_feat_path, 'r')

        self.aud_feat_path = aud_feat_path
        self.aud_feat = h5py.File(self.aud_feat_path, 'r')

    def __getitem__(self, index):
        video_id = self.video_ids[index]
        cap_ids = self.vid_caps[video_id]

        clip_vecs = []
        video_vecs = self.clip_vid_feat[video_id]
        for i in video_vecs:
            clip_vecs.append(i)


        clip_video_feature = uniform_feature_sampling(np.array(clip_vecs), self.max_ctx_len)
        clip_video_feature = l2_normalize_np_array(clip_video_feature)
        clip_video_feature = torch.from_numpy(clip_video_feature)

        # if self.video_level_branch:
        #     pool_video_features = average_to_fixed_length(np.array(frame_vecs), self.map_size)
        #     pool_video_features = l2_normalize_np_array(pool_video_features)
        #     pool_video_features = torch.from_numpy(pool_video_features).unsqueeze(0)
        # else:
        #     pool_video_features = None
        # text
        cap_tensors = []
        # clip_cap_tensors = []
        for cap_id in cap_ids:

            cap_feat = self.text_feat[cap_id][...]
            cap_tensor = torch.from_numpy(l2_normalize_np_array(cap_feat))[:self.max_desc_len]
            cap_tensors.append(cap_tensor)
            # if self.use_clip:
            #     clip_cap_feat = self.clip_text_feat[cap_id][...]
            #     clip_cap_feat = torch.from_numpy(clip_cap_feat)
            #     clip_cap_tensors.append(clip_cap_feat)
            # else:
            #     clip_cap_tensors=None

        frame_sub_feature = uniform_feature_sampling(self.sub_feat[video_id][:], self.max_ctx_len)
        frame_sub_feature = l2_normalize_np_array(frame_sub_feature)
        frame_sub_feature = torch.from_numpy(frame_sub_feature)

        # pool_sub_features = average_to_fixed_length(self.sub_feat[video_id][:], self.map_size)
        # pool_sub_features = l2_normalize_np_array(pool_sub_features)
        # pool_sub_features = torch.from_numpy(pool_sub_features).unsqueeze(0)

        frame_aud_feature = uniform_feature_sampling(self.aud_feat[video_id][:], self.max_ctx_len)
        frame_aud_feature = l2_normalize_np_array(frame_aud_feature)
        frame_aud_feature = torch.from_numpy(frame_aud_feature)

        # pool_aud_features = average_to_fixed_length(self.aud_feat[video_id][:], self.map_size)
        # pool_aud_features = l2_normalize_np_array(pool_aud_features)
        # pool_aud_features = torch.from_numpy(pool_aud_features).unsqueeze(0)

        return clip_video_feature, frame_sub_feature, frame_aud_feature, cap_tensors, index, cap_ids, video_id

    def __len__(self):
        return self.length

class VisDataSet4DLDKD(data.Dataset):

    def __init__(self, clip_vid_feat_path, sub_feat_path, aud_feat_path, video2frames, opt, video_ids=None):
        self.video2frames = video2frames
        if video_ids is not None:
            self.video_ids = video_ids
        else:
            self.video_ids = video2frames.keys()
        self.length = len(self.video_ids)
        self.map_size = opt.map_size
        self.max_ctx_len = opt.max_ctx_l
        self.video_level_branch = opt.video_level_branch
        self.sub_feat_path = sub_feat_path
        self.sub_feat = h5py.File(self.sub_feat_path, 'r')
        self.aud_feat_path = aud_feat_path
        self.aud_feat = h5py.File(self.aud_feat_path, 'r')


        self.clip_vid_feat_path = clip_vid_feat_path
        # self.clip_text_feat = h5py.File(self.clip_text_feat_path, 'r')
        self.clip_vid_feat = h5py.File(self.clip_vid_feat_path, 'r')


    def __getitem__(self, index):
        video_id = self.video_ids[index]


        # video
        clip_vecs = []
        video_vecs = self.clip_vid_feat[video_id]
        for i in video_vecs:
            clip_vecs.append(i)

        clip_video_feature = uniform_feature_sampling(np.array(clip_vecs), self.max_ctx_len)
        clip_video_feature = l2_normalize_np_array(clip_video_feature)
        clip_video_feature = torch.from_numpy(clip_video_feature)


        # if self.video_level_branch:
        #     pool_video_features = average_to_fixed_length(np.array(frame_vecs), self.map_size)
        #     pool_video_features = l2_normalize_np_array(pool_video_features)
        #     pool_video_features = torch.from_numpy(pool_video_features).unsqueeze(0)
        # else:
        #     pool_video_features = None

        frame_sub_feature = uniform_feature_sampling(self.sub_feat[video_id][:], self.max_ctx_len)
        frame_sub_feature = l2_normalize_np_array(frame_sub_feature)
        frame_sub_feature = torch.from_numpy(frame_sub_feature)

        # pool_sub_features = average_to_fixed_length(self.sub_feat[video_id][:], self.map_size)
        # pool_sub_features = l2_normalize_np_array(pool_sub_features)
        # pool_sub_features = torch.from_numpy(pool_sub_features).unsqueeze(0)

        frame_aud_feature = uniform_feature_sampling(self.aud_feat[video_id][:], self.max_ctx_len)
        frame_aud_feature = l2_normalize_np_array(frame_aud_feature)
        frame_aud_feature = torch.from_numpy(frame_aud_feature)

        # pool_aud_features = average_to_fixed_length(self.aud_feat[video_id][:], self.map_size)
        # pool_aud_features = l2_normalize_np_array(pool_aud_features)
        # pool_aud_features = torch.from_numpy(pool_aud_features).unsqueeze(0)

        return clip_video_feature, frame_sub_feature, frame_aud_feature, index, video_id

    def __len__(self):
        return self.length


class TxtDataSet4DLDKD(data.Dataset):
    """
    Load captions
    """

    def __init__(self, cap_file, text_feat_path,opt):
        # Captions
        self.captions = {}
        self.cap_ids = []
        with open(cap_file, 'r') as cap_reader:
            for line in cap_reader.readlines():
                cap_id, caption = line.strip().split(' ', 1)
                self.captions[cap_id] = caption
                self.cap_ids.append(cap_id)
        self.text_feat_path = text_feat_path
        self.max_desc_len = opt.max_desc_l
        self.length = len(self.cap_ids)
        self.text_feat = h5py.File(self.text_feat_path, 'r')


    def __getitem__(self, index):
        cap_id = self.cap_ids[index]

        cap_feat = self.text_feat[cap_id][...]
        cap_tensor = torch.from_numpy(l2_normalize_np_array(cap_feat))[:self.max_desc_len]

        return cap_tensor, index, cap_id

    def __len__(self):
        return self.length

if __name__ == '__main__':
    pass


