# Defines the H2O dataset loader and dataloader factory.


from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import os
import pickle
import cv2
import numpy as np


H2O_STATS = {'2d': {'min': [0, 0],
                    'max': [1280,  720],
                    'mean': [680.92605918, 349.150914]},
             '3dc': {'min': [-0.33688032, -0.3860082 ,  0.13619463],
                     'max': [0.45182842, 0.34188776, 0.76754248],
                     'mean': [ 0.02773121, -0.00888447,  0.4658583 ]},
             '3dw': {'min': [-0.32412812, -0.27855986,  0.16916465],
                     'max': [0.37120016, 0.28456741, 0.74655667],
                     'mean': [0.01147821, 0.01590901, 0.4859054 ]}}

INTRINSICS = {'fx': 636.6593017578125, 'fy': 636.251953125,
              'cx': 635.283881879317, 'cy': 366.8740353496978,
              'width': 1280.0, 'height': 720.0}


def normalize_traj(traj, target='3d', use_global=True):
    if target == '3d':

        if use_global:
            minvals = np.array([H2O_STATS['3dw']['min']])
            maxvals = np.array([H2O_STATS['3dw']['max']])
        else:
            minvals = np.array([H2O_STATS['3dc']['min']])
            maxvals = np.array([H2O_STATS['3dc']['max']])
        traj_norm = (traj - minvals) / (maxvals - minvals + 1e-6)
    else:

        traj_norm = np.copy(traj).astype(np.float32)
        traj_norm[:, 0] = traj[:, 0] / H2O_STATS['2d']['max'][0]
        traj_norm[:, 1] = traj[:, 1] / H2O_STATS['2d']['max'][1]
    return traj_norm


def denormalize_traj(traj_norm, target='3d', use_global=True):
    if target == '3d':

        if use_global:
            minvals = np.array([H2O_STATS['3dw']['min']])
            maxvals = np.array([H2O_STATS['3dw']['max']])
        else:
            minvals = np.array([H2O_STATS['3dc']['min']])
            maxvals = np.array([H2O_STATS['3dc']['max']])
        traj = traj_norm * (maxvals - minvals + 1e-6) + minvals
    else:

        traj = np.copy(traj_norm)
        traj[:, 0] = traj_norm[:, 0] * H2O_STATS['2d']['max'][0]
        traj[:, 1] = traj_norm[:, 1] * H2O_STATS['2d']['max'][1]
    return traj


def world_to_camera(traj3d_global, cam2world):
    traj3d_local = []
    for xyz, Rc2w in zip(traj3d_global, cam2world):

        extrinsics = np.linalg.inv(Rc2w)
        xyz_homo = np.concatenate([xyz[:, np.newaxis], np.ones((1, 1))], axis=0)
        xyz_local = extrinsics.dot(xyz_homo)[:3, 0]
        traj3d_local.append(xyz_local)
    traj3d_local = np.array(traj3d_local)

    return traj3d_local


def XYZ_to_uv(traj3d, intrinsics):


    traj2d = np.zeros((traj3d.shape[0], 2), dtype=np.float32)
    traj2d[:, 0] = traj3d[:, 0] * intrinsics['fx'] / (traj3d[:, 2] + 1e-6) + intrinsics['cx']
    traj2d[:, 1] = traj3d[:, 1] * intrinsics['fy'] / (traj3d[:, 2] + 1e-6) + intrinsics['cy']


    traj2d[:, 0] = np.clip(traj2d[:, 0], 0, intrinsics['width']-1)
    traj2d[:, 1] = np.clip(traj2d[:, 1], 0, intrinsics['height']-1)
    traj2d = np.floor(traj2d).astype(np.int32)

    return traj2d


class H2O(Dataset):
    def __init__(self, root_dir, phase='train', transform=None, data_cfg=None, model_cfg=None):
        self.root_dir = os.path.join(root_dir, 'Ego3DTraj')
        self.phase = phase
        self.transform = transform


        self._MAX_FRAMES = getattr(data_cfg, 'max_frames', -1)
        self._MAX_DEPTH = getattr(data_cfg, 'max_depth', 1.0)
        self.load_all = getattr(data_cfg, 'load_all', False)


        self.target = getattr(model_cfg, 'target', '3d')
        self.modalities = getattr(model_cfg, 'modalities', ['loc'])
        self.use_global = getattr(model_cfg, 'use_global', False)
        self.centralize = getattr(model_cfg, 'centralize', False)
        self.normalize = getattr(model_cfg, 'normalize', True)


        self.samples = self.get_split()


        self.traj_data, self.video_data = self.read_all_data()

        if isinstance(phase, list):
            self.check_intrinsics()
            self.traj_stats = self.get_stats()


    def get_split(self):
        def read_split(filepath):
            assert os.path.exists(filepath), "File path does not exist! {}".format(filepath)
            samples = []
            with open(filepath, 'r') as f:
                for line in f.readlines():
                    samples.append(line.strip())
            return samples

        split_dir = os.path.join(self.root_dir, 'splits')
        if isinstance(self.phase, str):
            samples = read_split(os.path.join(split_dir, self.phase + '.txt'))
        else:
            samples = []
            for phase in self.phase:
                samples.extend(read_split(os.path.join(split_dir, phase + '.txt')))
        return samples


    def get_time_stamps(self, total_frames, clip_len, stepsize=15):
        time_stamps = []
        ps, pe = 0, clip_len-1
        while pe < total_frames:
            time_stamps.append([ps, pe])
            ps += stepsize
            pe += stepsize
        time_stamps = np.array(time_stamps)
        return time_stamps


    def preprocess_trajdata(self, trajdata, intrinsics):

        traj_2d_all, traj_3dc_all, traj_3dw_all, timestamps_all, campose_call = [], [], [], [], []
        for data in trajdata:
            start, end = data['start'], data['end']
            traj3d_world = data['traj3d']
            cam2world = data['cam2world']
            num_frames = end - start + 1


            traj3d_camera = world_to_camera(traj3d_world, cam2world)

            traj2d = XYZ_to_uv(traj3d_camera, intrinsics)

            if self.normalize:

                traj2d = normalize_traj(traj2d, target='2d')
                traj3d_world = normalize_traj(traj3d_world, target='3d', use_global=True)
                traj3d_camera = normalize_traj(traj3d_camera, target='3d', use_global=False)


            max_frames = num_frames if self._MAX_FRAMES < 0 else self._MAX_FRAMES
            time_stamps = self.get_time_stamps(num_frames, max_frames)

            if time_stamps.shape[0] > 0:
                for ps, pe in time_stamps:
                    traj_2d_all.append(traj2d[ps: pe+1])
                    traj_3dc_all.append(traj3d_camera[ps: pe+1])
                    traj_3dw_all.append(traj3d_world[ps: pe+1])
                    timestamps_all.append(np.array([start + ps, start + pe]))
                    campose_call.append(cam2world[ps: pe+1])
            else:

                traj_2d_all.append(traj2d)
                traj_3dc_all.append(traj3d_camera)
                traj_3dw_all.append(traj3d_world)
                timestamps_all.append(np.array([start, end]))
                campose_call.append(cam2world)

        return traj_2d_all, traj_3dc_all, traj_3dw_all, timestamps_all, campose_call


    def read_rgb(self, video_file):
        assert os.path.exists(video_file), "Video file does not exist! {}".format(video_file)
        cap = cv2.VideoCapture(video_file)
        success, frame = cap.read()
        videos = []
        while success:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            videos.append(frame)
            success, frame = cap.read()
        cap.release()
        cv2.destroyAllWindows()
        videos = np.array(videos)
        return videos


    def read_all_data(self):
        traj_data = {'2d': [], '3dc': [], '3dw': []}
        video_data = {'data': [], 'file': [], 'timestamps': [], 'campose': [], 'glip_feats': [], 'motion_feats': [], }
        for sample in self.samples:
            traj_file = os.path.join(self.root_dir, 'traj', sample + '.pkl')
            assert os.path.exists(traj_file), 'Trajectory file does not exist! {}'.format(traj_file)
            with open(traj_file, 'rb') as f:
                data = pickle.load(f)
            traj2d_l, traj3d_lc, traj3d_lw, timestamps_l, campose_l = self.preprocess_trajdata(data['left_hand'], data['intrinsics'])
            traj2d_r, traj3d_rc, traj3d_rw, timestamps_r, campose_r = self.preprocess_trajdata(data['right_hand'], data['intrinsics'])


            traj_data['2d'].extend(traj2d_l + traj2d_r)
            traj_data['3dc'].extend(traj3d_lc + traj3d_rc)
            traj_data['3dw'].extend(traj3d_lw + traj3d_rw)
            timestamps = timestamps_l + timestamps_r
            campose = campose_l + campose_r


            vid_file = os.path.join(self.root_dir, 'video', sample + '.mp4')
            video_data['file'].extend([vid_file]*len(timestamps))
            video_data['timestamps'].extend(timestamps)
            video_data['campose'].extend(campose)
            if self.load_all:
                video = self.read_rgb(vid_file)
                video = self.transform(video) if self.transform is not None else torch.from_numpy(np.transpose(video, (3, 0, 1, 2)))
                video_data['data'].extend([video[:, ps: pe+1] for ps, pe in timestamps])


            glip_feats_all = np.load(os.path.join('/data/glip_feats_h2o', '__mnt__share_disk__mjy__DIFFIP_egopat_mamba_h2o__h2o_dataset__Ego3DTraj__video__' + sample + '.npy'))

            video_data['glip_feats'].extend([glip_feats_all[ps: pe+1] for ps, pe in timestamps])


            motion_feats_all = np.load(os.path.join('/data/motion_feats_h2o', '__mnt__share_disk__mjy__DIFFIP_egopat_mamba_h2o__h2o_dataset__Ego3DTraj__video__' + sample + '.npy'))


            for ps, pe in timestamps:
                start_homo_inv = np.linalg.inv(motion_feats_all[ps])

                current_homo_group = motion_feats_all[ps: pe+1]
                current_homo_group = np.einsum('ij,njk->nik', start_homo_inv, current_homo_group)


                video_data['motion_feats'].extend([current_homo_group])


        return traj_data, video_data


    def check_intrinsics(self):
        all_intrinsics = []
        for sample in self.samples:
            traj_file = os.path.join(self.root_dir, 'traj', sample + '.pkl')
            with open(traj_file, 'rb') as f:
                data = pickle.load(f)
            all_intrinsics.append(data['intrinsics'])

        for i in range(len(all_intrinsics)-1):
            if all_intrinsics[i] != all_intrinsics[i+1]:
                print("Not all intrinsics are equal!")


    def get_stats(self):
        stats = {k: dict() for k in self.traj_data.keys()}
        for k, traj_list in self.traj_data.items():
            dim = 2 if k == '2d' else 3
            sumvals = np.zeros((dim), dtype=np.float32)
            num_pt = 0
            minvals, maxvals = [], []
            for traj in traj_list:
                minvals.append(np.min(traj, axis=0))
                maxvals.append(np.max(traj, axis=0))
                sumvals += np.sum(traj, axis=0)
                num_pt += traj.shape[0]

            stats[k]['min'] = np.min(np.array(minvals), axis=0)
            stats[k]['max'] = np.max(np.array(maxvals), axis=0)
            stats[k]['mean'] = sumvals / num_pt
        return stats


    def __len__(self):
        return len(self.traj_data['3dw'])


    def __getitem__(self, index):

        input_data = torch.tensor([])
        vid_file = self.video_data['file'][index]
        ps, pe = self.video_data['timestamps'][index]
        len_valid = pe - ps + 1
        cam_pose = np.copy(self.video_data['campose'][index])


        if self.target == '3d':
            output_traj = np.copy(self.traj_data['3dw'][index]) if self.use_global else np.copy(self.traj_data['3dc'][index])
            dim_traj = 3
        else:
            output_traj = np.copy(self.traj_data['2d'][index])
            dim_traj = 2
        if self.centralize:
            output_traj = output_traj * 2.0 - 1.0


        glip_feats =  self.video_data['glip_feats'][index]

        motion_feats = self.video_data['motion_feats'][index]


        if self.phase != 'train':
            num_obs = np.floor(len_valid * 0.6).astype(int)
            motion_feats[num_obs:len_valid] = motion_feats[num_obs-1]

        if len_valid < self._MAX_FRAMES:
            output_traj = np.concatenate([output_traj, np.zeros((self._MAX_FRAMES-len_valid, dim_traj))], axis=0)
            cam_pose = np.concatenate([cam_pose, np.tile(np.eye(4), (self._MAX_FRAMES-len_valid, 1, 1))], axis=0)

            glip_feats = np.concatenate([glip_feats, np.zeros((self._MAX_FRAMES-len_valid, 256, 7, 12))], axis=0)

            motion_feats = np.concatenate([motion_feats, np.zeros((self._MAX_FRAMES-len_valid, 3, 3))], axis=0)


        output_traj = torch.from_numpy(output_traj).to(torch.float32)
        input_pose = torch.from_numpy(cam_pose).to(torch.float32)

        glip_feats = torch.from_numpy(glip_feats).to(torch.float32)

        motion_feats = torch.from_numpy(motion_feats).to(torch.float32)


        if 'rgb' in self.modalities:
            if self.load_all:
                input_data = self.video_data['data'][index].clone()
            else:
                rgb = self.read_rgb(vid_file)
                rgb = self.transform(rgb) if self.transform is not None else torch.from_numpy(np.transpose(rgb, (3, 0, 1, 2)))
                input_data = rgb[:, ps: pe+1]
            assert input_data.size(1) == len_valid

            if len_valid < self._MAX_FRAMES:
                c, t, h, w = input_data.size()
                input_data = torch.concat([input_data, torch.zeros([c, self._MAX_FRAMES-len_valid, h, w])], axis=1)

        return vid_file, input_data, input_pose, len_valid, output_traj, glip_feats, motion_feats, (ps, pe)


def build_dataloaders(cfg, phase='trainval'):


    data_root = os.path.join(cfg.DATA.data_path, cfg.DATA.dataset)
    if phase == 'trainval':

        trainset = H2O(data_root, phase='train', transform=None, data_cfg=cfg.DATA, model_cfg=cfg.MODEL)
        train_loader = DataLoader(trainset, batch_size=cfg.TRAIN.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)

        valset = H2O(data_root, phase='val', transform=None, data_cfg=cfg.DATA, model_cfg=cfg.MODEL)
        val_loader = DataLoader(valset, batch_size=cfg.TEST.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

        print("Number of train/val: {}/{}".format(len(trainset), len(valset)))
        return train_loader, val_loader
    else:

        testset = H2O(data_root, phase='test', transform=None, data_cfg=cfg.DATA, model_cfg=cfg.MODEL)
        test_loader = DataLoader(testset, batch_size=cfg.TEST.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

        print("Number of test samples: {}".format(len(testset)))
        return test_loader


if __name__ == '__main__':

    from gluoncv.torch.data.transforms.videotransforms.video_transforms import Compose, Resize, Normalize
    from gluoncv.torch.data.transforms.videotransforms.volume_transforms import ClipToTensor
    from tqdm import tqdm

    class data_cfg: pass
    data_cfg.max_frames = 64
    data_cfg.load_all = True

    class model_cfg: pass
    model_cfg.target = '2d'
    model_cfg.modalities = ['rgb', 'loc']
    model_cfg.use_global = True
    model_cfg.centralize = True
    model_cfg.normalize = True

    rgb_transform = Compose([Resize([64, 64]), ClipToTensor(), Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])


    trainset = H2O('data/H2O', phase='train', transform=rgb_transform, data_cfg=data_cfg, model_cfg=model_cfg)
    train_loader = DataLoader(trainset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)

    valset = H2O('data/H2O', phase='val', transform=rgb_transform, data_cfg=data_cfg, model_cfg=model_cfg)
    val_loader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

    testset = H2O('data/H2O', phase='val', transform=rgb_transform, data_cfg=data_cfg, model_cfg=model_cfg)
    test_loader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

    print("# train: {}, # val: {}, # test: {}".format(len(trainset), len(valset), len(testset)))

    for i, (filename, clip, odometry, nf, traj) in tqdm(enumerate(train_loader), total=len(train_loader), desc='Train'):

        pass

    for i, data in tqdm(enumerate(val_loader), total=len(val_loader), desc='Val'):

        pass

    for i, data in tqdm(enumerate(test_loader), total=len(test_loader), desc='Test'):

        pass
