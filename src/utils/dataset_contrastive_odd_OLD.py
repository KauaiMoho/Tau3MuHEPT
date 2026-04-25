# -*- coding: utf-8 -*-

"""
Created on 2021/4/14
@author: Siqi Miao
"""

import os
import os.path as osp
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import permutations, product
from .root2df import Root2Df

import torch
import torch_geometric
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader, DataListLoader
import pickle
import multiprocessing

class Tau3MuDataset(Dataset):
    def __init__(self, setting, data_config, endcap, debug=False):
        self.setting = setting
        self.data_dir = Path(data_config['data_dir'])
        self.conditions = data_config.get('conditions', False)
        self.node_feature_names = data_config['node_feature_names']
        self.only_one_tau = data_config['only_one_tau']
        self.splits = data_config['splits']
        self.pos_neg_ratio = data_config['pos_neg_ratio']
        self.endcap = endcap

        self.n_hits_min = data_config.get('n_hits_min', 1)
        self.n_hits_max = data_config.get('n_hits_max', np.inf)

        self.coords = data_config.get('coords')

        global signal_folder
        global bkg_folder
        global far_station

        signal_dataset = data_config['signal_dataset']
        bkg_dataset = data_config['bkg_dataset']
        signal_folder = data_config['signal_folder']
        bkg_folder = data_config['bkg_folder']
        far_station = data_config.get('far_station', False)

        self.eta_thresh = data_config.get('eta_thresh', False)
        self.cut = data_config.get('cut', False)
        self.debug = debug

        print(f'[INFO] Debug mode: {self.debug}')
        print(self.setting)

        super(Tau3MuDataset, self).__init__(root=self.data_dir)

        self.idx_split = torch.load(osp.join(self.processed_dir, 'idx_split.pt'))

        sample = self.get(0)
        self.x_dim = sample.x.shape[-1]

        print_splits(self)

    def len(self):
        return len([f for f in os.listdir(self.processed_dir) if f.startswith('data_')])

    def get(self, idx):
        return torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
    
    @property
    def raw_file_names(self):
        return ['DsTau3muPU0_MTD.pkl', 'DSTau3Mu_pCut1GeV_DF.pkl', 'MinBiasPU200_MTD.pkl'] if 'mix' in self.setting else [signal_dataset, bkg_dataset]


    @property
    def processed_file_names(self):
        return ['done.pt']

    @property
    def processed_dir(self):
        if 'half' in self.setting:
            return osp.join(self.root, f'processed/endcap_{self.endcap}')
        return osp.join(self.root, 'processed')

    def download(self):

        print('Please put .pkl or .csv files ino $PROJECT_DIR/data/raw!')
        raise KeyboardInterrupt

    def process(self):
        os.makedirs(self.processed_dir, exist_ok=True)
        df = self.get_df()
        
        df = df.sample(frac=1, random_state=42).reset_index(drop=False) # Shuffle the dataset. Set seed to make results reproducible
        df = df.rename(columns={'index': 'og_index'}) # Store the indices of the original dataset for prediction analysis
        
        if self.debug:
            df = df.iloc[:100]
        
        global eta
        global phi
        global r
        global z
        global theta
        
        if 'mu_hit_global_eta' in df.keys():
            eta = 'mu_hit_global_eta'
            phi = 'mu_hit_global_phi'
            r = 'mu_hit_global_r'
            z = 'mu_hit_global_z'
            theta = 'mu_hit_global_theta'
        else:
            eta = 'mu_hit_sim_eta'
            phi = 'mu_hit_sim_phi'
            r = 'mu_hit_sim_r'
            z = 'mu_hit_sim_z'
            theta = 'mu_hit_sim_theta'
        
        if 'EMTF' in self.node_feature_names:
            assert 'full' in self.setting, 'half detector setting is currently not supported'
            pt = 'EMTF_mu_pt'
            eta = 'EMTF_mu_eta'
            phi = 'EMTF_mu_phi'
        
        self.feature_names = self.node_feature_names
        
        if 'mu_hit_sim_cosphi' in self.feature_names or 'mu_hit_sim_sinphi' in self.feature_names:
            # Add phi transformations
            r_copy = df[r].to_numpy()
            phi_copy = df[phi].to_numpy()
            cos = []
            sin = []
            x = []
            y = []
            
            for i in range(len(r_copy)):
                
                hit_r = r_copy[i]
                hit_phi = phi_copy[i]
                
                cos.append(np.cos(hit_phi))
                sin.append(np.sin(hit_phi))
                
                x.append(hit_r*np.cos(hit_phi))
                y.append(hit_r*np.sin(hit_phi))
                
            df['mu_hit_sim_cosphi'] = cos
            df['mu_hit_sim_sinphi'] = sin
            df['mu_hit_sim_x'] = x
            df['mu_hit_sim_y'] = y
        
        # TODO: Figure out how to not require these to be global.
        
        if 'half' in self.setting:
            
            for name in ['mu_hit_nlog_eta', 'mu_hit_nlog_phi', 'mu_hit_dist', 'mu_hit_dot']:
                if name in self.feature_names:
                    self.feature_names.remove(name)
    
        print('[INFO] Processing entries...')
        valid_idx_half = [0, 0]
        valid_idx = 0
        data_list = []
    
        for entry in tqdm(df.itertuples(), total=len(df)):
            
            masked_entry = self.mask_hits(entry, self.conditions, n_hits_min=self.n_hits_min)
            if masked_entry == None: # Don't use events that don't have any hits left after the mask is applied
                continue
            
            if 'half' not in self.setting:
                data = self._process_one_entry(masked_entry)
                torch.save(data, osp.join(self.processed_dir, f'data_{valid_idx}.pt'))
                valid_idx += 1
                data_list.append(data)
            else:
                for i, endcap in enumerate([1,-1]):
                    if 'half' in self.setting:
                        entry = Tau3MuDataset.split_endcap(masked_entry, endcap)
                        
                        if entry == None: # Don't use an endcap that is empty
                            continue
                        else:
                            # half-detector, tau and non-tau endcap
                            data = self._process_one_entry(entry, endcap=endcap)
                            endcap_dir = osp.join(self.root, f'processed/endcap_{i}')
                            os.makedirs(endcap_dir, exist_ok=True)
                            torch.save(data, osp.join(endcap_dir, f'data_{valid_idx_half[i]}.pt'))
                            valid_idx_half[i] += 1
                            data_list[i].append(data)
                    
        if 'half' in self.setting:
            for i in range(2):
                idx_split = Tau3MuDataset.get_idx_split(data_list[i], self.splits, self.pos_neg_ratio)
                endcap_dir = osp.join(self.root, f'processed/endcap_{i}')
                torch.save(idx_split, osp.join(endcap_dir, 'idx_split.pt'))
                print(f'[INFO] Saved endcap_{i} with {len(data_list[i])} graphs.')
        else:
            idx_split = Tau3MuDataset.get_idx_split(data_list, self.splits, self.pos_neg_ratio)
            torch.save(idx_split, osp.join(self.processed_dir, 'idx_split.pt'))
            print(f'[INFO] Saved {len(data_list)} graphs.')

        torch.save(True, osp.join(self.processed_dir, 'done.pt'))

    def _process_one_entry(self, entry, endcap=0, only_eval=False):
    
        if 'half' in self.setting:
            if entry['n_gen_tau']==1: # If signal event, only return hits on tau endcap
                if ((entry['gen_tau_eta'] * entry[eta]) > 0).sum() == entry['n_mu_hit']: 
                    only_eval=False
                else:
                    only_eval=True
        else:
            only_eval = False
        
        x = Tau3MuDataset.get_node_features(entry, self.node_feature_names)
        coords = self.get_coors_for_hits(entry)
        
        gen_mu = Tau3MuDataset.get_gen_mu_kinematics(entry)
        gen_tau = Tau3MuDataset.get_gen_tau_kinematics(entry)
        y = torch.tensor(entry['y']).float().view(-1, 1)
        
        if 'mu_hit_truth' in entry.keys():
            hit_truth = self.get_hit_truth(entry)
            return Data(x=x, y=y, coords=coords, sample_idx=entry['og_index'], endcap=endcap, only_eval=only_eval, gen_mu=gen_mu, gen_tau=gen_tau, hit_truth=hit_truth)

        else:
            return Data(x=x, y=y, coords=coords, sample_idx=entry['og_index'], endcap=endcap, only_eval=only_eval, gen_mu=gen_mu, gen_tau=gen_tau)

    def pt_to_event_row(pt_file_path):
        """Converts a single .pt file into a dictionary representing one event."""
        data = torch.load(pt_file_path)
        x = data.x.numpy()
        
        # Map the indices from your PointCloudBuilder DEFAULT_FEATURES
        return {
            'mu_hit_global_r':   x[:, 0],
            'mu_hit_global_phi': x[:, 1],
            'mu_hit_global_z':   x[:, 2],
            'mu_hit_global_eta': x[:, 3],
            'mu_hit_bend':       x[:, 4],
            # Extract Gen-level truth stored in the Data object attributes
            'gen_tau_pt':        data.gen_tau_pt.item() if hasattr(data, 'gen_tau_pt') else 0,
            'n_gen_tau':         1 if (hasattr(data, 'y') and data.y == 1) else 0,
            'event_id':          int(Path(pt_file_path).stem.split('_')[0].replace('data', ''))
        }

    
    def get_df_save_path(self):
        save_name = ''
        save_name += 'mix_' if 'mix' in self.setting else 'raw_'
        
        df_dir = self.data_dir / 'scores' / f'{save_name}'
        df_dir.mkdir(parents=True, exist_ok=True)
        return df_dir / f'{save_name}.pkl'

    def get_df(self):
        df_save_path = self.get_df_save_path()
        if df_save_path.exists():
            print(f'[INFO] Loading {df_save_path}...')
            with open(df_save_path, 'rb') as handle: 
                return pickle.load(handle)
    
        # Assuming self.data_dir contains 'pos_pt_files' and 'neg_pt_files' folders
        pos_path = Path(self.data_dir) / self.signal_folder
        neg_path = Path(self.data_dir) / self.bkg_folder

        print(f'[INFO] Processing .pt files from {pos_path} and {neg_path}...')

        pos_list = [pt_to_event_row(f) for f in pos_path.glob("*.pt")]
        pos200 = pd.DataFrame(pos_list)
        pos200['y'] = 1

        neg_list = [pt_to_event_row(f) for f in neg_path.glob("*.pt")]
        neg200 = pd.DataFrame(neg_list)
        neg200['y'] = 0
        
        #assert self.only_one_tau # Only one-tau is supported
        if self.only_one_tau:
            pos200 = pos200[pos200.n_gen_tau == 1].reset_index(drop=True)
            if pos0 is not None:
                pos0 = pos0[pos0.n_gen_tau == 1].reset_index(drop=True)
            
            #try:
            #    neg200 = neg200[neg200.n_gen_tau == 1].reset_index(drop=True)
            #except:
            #    pass
            
        if self.cut:
            pos200 = pos200[pos200.apply(lambda x: self.filter_samples(x), axis=1)].reset_index(drop=True)
            if pos0 is not None:
                pos0 = pos0[pos0.apply(lambda x: self.filter_samples(x), axis=1)].reset_index(drop=True)

        if pos0 is not None and len(pos0) > 100000:
            print('[INFO] Sampling from pos0 to fasten processing & training...')
            pos0 = pos0.sample(100000).reset_index(drop=True)

        if 'mix' in self.setting:
            pos, neg = self.mix(pos0, neg200, pos200, self.setting)
        else:
            pos, neg = pos200, neg200
        
        if 'half' in self.setting:
            min_pos_neg_ratio = len(pos) / (len(neg) * 2)
        else:
            min_pos_neg_ratio = len(pos) / len(neg)
        print(f'[INFO] min_pos_neg_ratio: {min_pos_neg_ratio}')

        #assert self.pos_neg_ratio >= min_pos_neg_ratio, f'min_pos_neg_ratio = {min_pos_neg_ratio}! Now pos_neg_ratio = {self.pos_neg_ratio}!'
        
        print(f'[INFO] Concatenating pos & neg, saving to {df_save_path}...')
        df = pd.concat((pos, neg), join='outer', ignore_index=True)
        df.to_pickle(df_save_path)
        return df

        #Should return a df with all needed cols from this file.


    @staticmethod
    def get_node_features(entry, feature_names):

        # Directly index the entry using features = entry[feature_names] is extremely slow!
        features = np.stack([entry[feature] for feature in feature_names], axis=1)
        
        return torch.tensor(features, dtype=torch.float)

    @staticmethod
    def get_gen_mu_kinematics(entry):
        
        if entry['y'] == 1:
            
            pt = entry['gen_mu_pt']
            eta = entry['gen_mu_eta']
            phi = entry['gen_mu_phi']
            
            lead, sublead, soft = np.argsort(pt)
            
            features = [np.array([pt[i],eta[i],phi[i]]) for i in (lead,sublead,soft)]
            
            features = np.concatenate(features).reshape(1,9)
        else:
            return torch.empty(1,9)
        
        return torch.tensor(features, dtype=torch.float)
    
    @staticmethod
    def get_gen_tau_kinematics(entry):
        
        if entry['y'] == 1:
            features = np.stack([entry[feature] for feature in ['gen_tau_pt', 'gen_tau_eta', 'gen_tau_phi']], axis=1).reshape(1,3)
        else:
            return torch.empty(1,3)
        
        return torch.tensor(features, dtype=torch.float)

    @staticmethod
    def get_idx_split(data, splits, pos2neg):
        np.random.seed(42)
        assert sum(splits.values()) == 1.0
        y_dist = np.array([event.y[0][0] for event in data])
        only_eval = np.array([event.only_eval for event in data])
        
        pos_idx = np.argwhere( np.logical_and(y_dist == 1,~only_eval) ).reshape(-1)  # if is half-detector model, do not use non-tau endcap for training
        neg_idx = np.argwhere( np.logical_and(y_dist == 0,~only_eval) ).reshape(-1)
        only_eval_idx = np.argwhere(only_eval).reshape(-1)
        
        print('Number of Positive Samples: ', len(pos_idx))
        print('Number of Negative Samples: ', len(neg_idx))
        print('Minimum pos2neg:', len(pos_idx)/len(neg_idx))
        
        try:
            assert len(pos_idx) <= len(neg_idx) * pos2neg, 'The number of negative samples is not enough given the pos_neg_ratio!'
        except:
            print('Number of negative samples not enough. Using minimum possible')
            pos2neg = len(pos_idx)/len(neg_idx)
        
        n_train_pos, n_valid_pos = int(splits['train'] * len(pos_idx)), int(splits['valid'] * len(pos_idx))
        if pos2neg > 0:
            n_train_neg, n_valid_neg = int(n_train_pos / pos2neg), int(n_valid_pos / pos2neg)
        else:
            n_train_neg, n_valid_neg = int(splits['train']*len(neg_idx)), int(splits['valid']*len(neg_idx))

        pos_train_idx = pos_idx[:n_train_pos]
        pos_valid_idx = pos_idx[n_train_pos:n_train_pos + n_valid_pos]
        pos_test_idx = pos_idx[n_train_pos + n_valid_pos:]

        neg_train_idx = neg_idx[:n_train_neg]
        neg_valid_idx = neg_idx[n_train_neg:n_train_neg + n_valid_neg]
        neg_test_idx = neg_idx[n_train_neg + n_valid_neg:]

        return {'pos_train': pos_train_idx.tolist(),
                'pos_valid': pos_valid_idx.tolist(),
                'pos_test':  pos_test_idx.tolist(),
                'neg_train': neg_train_idx.tolist(),
                'neg_valid': neg_valid_idx.tolist(),
                'neg_test':  neg_test_idx.tolist()
                
                }
    
    def get_coors_for_hits(self, entry):
            
        coors = torch.tensor(np.stack([entry[feature] for feature in self.coords]).T)
        return coors
        
    def get_hit_truth(self, entry):
        if entry['y'] == 1:

            return torch.tensor(entry['mu_hit_truth'])
        else:
            return torch.zeros(entry[eta].shape)
        
        
    @staticmethod
    def mix(pos0, neg200, pos200, setting):
        neg_idx = np.arange(len(neg200))
        np.random.shuffle(neg_idx)

        # first len(pos0) neg data will be used as noise in pos0
        noise_in_pos0 = neg200.loc[neg_idx[:len(pos0)]].reset_index(drop=True)
        # remaining neg data will remain negative
        neg200 = neg200.loc[neg_idx[len(pos0):]].reset_index(drop=True)

        print('[INFO] Mixing data...')
        # mixed_pos = noise_in_pos0
        mixed_pos = []
        for idx, entry in tqdm(pos0.iterrows(), total=len(pos0)):
            for k, v in entry.items():
                if 'gen' in k:  # directly keep gen variables
                    continue
                elif isinstance(v, int):  # accumulate n_mu_hit
                    assert k == 'n_mu_hit'
                    entry['node_label'] = np.concatenate((np.zeros(noise_in_pos0.iloc[idx][k]), np.ones(v)))
                    entry[k] += noise_in_pos0.iloc[idx][k]
                else:  # concat hit features
                    assert isinstance(v, np.ndarray)
                    mixed_hits = np.concatenate((noise_in_pos0.iloc[idx][k], v))
                    entry[k] = mixed_hits
            mixed_pos.append(entry.values)
        mixed_pos = pd.DataFrame(data=mixed_pos, columns=entry.index)

        if 'check' in setting:
            return mixed_pos, pos200
        elif 'sanity' in setting:
            return noise_in_pos0, neg200
        else:
            return mixed_pos, neg200

    @staticmethod
    def split_endcap(masked_entry, endcap):
        

        entry = {}
        endcap_idx = np.sign(masked_entry[z]) == endcap

        for k, v in masked_entry.items():
            if isinstance(v, np.ndarray) and 'gen' not in k and k != 'y' and 'L1' not in k and 'EMTF' not in k:
                assert v.shape[0] == masked_entry['n_mu_hit']
                entry[k] = v[endcap_idx]
            else:
                entry[k] = v
        
        entry['n_mu_hit'] = endcap_idx.sum().item()
        
        if entry['n_mu_hit'] < 1: # Only return an entry if it has hits left
            return None
        
        if masked_entry['y'] == 1: # If signal event, only return hits on tau endcap
            if ((masked_entry['gen_tau_eta'] * entry[eta]) > 0).sum() == entry['n_mu_hit']: 
                entry['y'] = 1
            else:
                entry['y'] = 0
        
        return entry

    def mask_hits(self, entry, conditions, n_hits_min=1):
        n_mu_hit = eval(f'len(entry.{self.node_feature_names[0]})')
        
        if n_mu_hit == 0: return None
        
        if conditions == False:
            masked_entry = {'n_mu_hit': n_mu_hit}
            for k in entry._fields:
                value = getattr(entry, k)
                if isinstance(value, np.ndarray):
                    masked_entry[k] = value.reshape(-1)
                else:
                    if k != 'n_mu_hit':
                        masked_entry[k] = value
            
            return masked_entry
        
        mask = np.ones(n_mu_hit, dtype=bool)
        for k, v in conditions.items():
            k = k.split('-')[1]
            assert isinstance(getattr(entry, k), np.ndarray)
            mask *= eval('entry.' + k + v)
        
        
        n_mu_hit = mask.sum()
        masked_entry = {'n_mu_hit': n_mu_hit}
        
        if not(mask.sum() >= n_hits_min): # Only return an entry if it has hits left
            return None
        
        for k in entry._fields:
            value = getattr(entry, k)
            if isinstance(value, np.ndarray) and 'gen' not in k and k != 'y' and 'L1' not in k and k != 'n_' not in k:
                try: masked_entry[k] = value[mask].reshape(-1)
                except:
                    continue
            else:
                if k != 'n_mu_hit':
                    masked_entry[k] = value
        return masked_entry



def get_data_loaders_contrastive(setting, data_config, batch_size, endcap=1):
    
    if endcap == 1 or endcap==0:
        idx = 0
    elif endcap == -1:
        idx = 1
    
    dataset = Tau3MuDataset(setting, data_config, idx)
    print('Retrieving Data Loaders from:'+dataset.processed_paths[idx])
    train_loader = [DataLoader(dataset[dataset.idx_split['pos_train']], batch_size=batch_size, shuffle=True,drop_last=True),DataLoader(dataset[dataset.idx_split['neg_train']], batch_size=batch_size, shuffle=True,drop_last=True)]
    valid_loader = [DataLoader(dataset[dataset.idx_split['pos_valid']], batch_size=batch_size, shuffle=True,drop_last=True),DataLoader(dataset[dataset.idx_split['neg_valid']], batch_size=batch_size, shuffle=True,drop_last=True)]
    test_loader = [DataLoader(dataset[dataset.idx_split['pos_test']], batch_size=batch_size, shuffle=True,drop_last=True),DataLoader(dataset[dataset.idx_split['neg_test']], batch_size=batch_size, shuffle=True,drop_last=True)]
    return {'train': train_loader, 'valid': valid_loader, 'test': test_loader}, dataset.x_dim, dataset


def print_splits(dataset):
    def get_pos_neg_count(y):
        pos_y = (y == 1).sum()
        neg_y = (y == 0).sum()
        pos2neg_ratio = pos_y / neg_y
        return pos_y, neg_y, pos2neg_ratio

    print('[Splits]')
    for k, v in dataset.idx_split.items():
        y = torch.cat([dataset[i].y for i in v])
        pos_y, neg_y, pos2neg_ratio = get_pos_neg_count(y)
        print(f'    {k}: {len(v)}. # pos: {pos_y}, # neg: {neg_y}. Pos:Neg: {pos2neg_ratio:.3f}')
