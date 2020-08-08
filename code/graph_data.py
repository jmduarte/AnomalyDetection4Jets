import os.path as osp
import torch
from torch_geometric.data import Dataset, Data
import itertools
import tables
import numpy as np
import pandas as pd
from pyjet import cluster,DTYPE_PTEPM

class GraphDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, start=0, stop=-1, n_particles=-1):
        self.start = start
        self.stop = stop
        self.n_particles = n_particles
        super(GraphDataset, self).__init__(root, transform, pre_transform) 


    @property
    def raw_file_names(self):
        return ['events_LHCO2020_backgroundMC_Pythia.h5']

    @property
    def processed_file_names(self):
        njets = 24043
        if self.start!=0 and self.stop!=-1:
            njets = self.stop-self.start
        return ['data_{}.pt'.format(i) for i in range(njets)]

    def __len__(self):
        return len(self.processed_file_names)

    def download(self):
        # Download to `self.raw_dir`.
        pass


    def process(self):
        data = []
        nonzero_particles = []
        event_indices = []
        masses = []
        px = []
        py = []
        pz = []
        e = []
        for raw_path in self.raw_paths:
            df = pd.read_hdf(raw_path,stop=10000) # just read first 10000 events
            all_events = df.values
            rows = all_events.shape[0]
            cols = all_events.shape[1]
            event_idx = 0
            for i in range(rows):
                pseudojets_input = np.zeros(len([x for x in all_events[i][::3] if x > 0]), dtype=DTYPE_PTEPM)
                for j in range(cols // 3):
                    if (all_events[i][j*3]>0):
                        pseudojets_input[j]['pT'] = all_events[i][j*3]
                        pseudojets_input[j]['eta'] = all_events[i][j*3+1]
                        pseudojets_input[j]['phi'] = all_events[i][j*3+2]
                    pass
                
                # cluster jets from the particles in one observation
                sequence = cluster(pseudojets_input, R=1.0, p=-1)
                jets = sequence.inclusive_jets()
                for jet in jets: # for each jet get (px, py, pz, e)
                    if jet.pt < 200: continue
                    if self.n_particles > -1: 
                        n_particles = self.n_particles
                    else:
                        n_particles = len(jet)
                    particles = np.zeros((n_particles, 4))
                    
                    # store all the particles of this jet
                    for p, part in enumerate(jet):
                        if n_particles > -1 and p >= n_particles: break
                        particles[p,:] = np.array([part.px,
                                                   part.py,
                                                   part.pz,
                                                   part.e])
                    data.append(particles)
                    nonzero_particles.append(len(jet))
                    event_indices.append(event_idx)
                    masses.append(jet.mass)
                    px.append(jet.px)
                    py.append(jet.py)
                    pz.append(jet.pz)
                    e.append(jet.e)
                
                event_idx += 1

        ijet = 0
        for data_idx, d in enumerate(data):
            n_particles = nonzero_particles[ijet]
            pairs = [[i, j] for (i, j) in itertools.product(range(n_particles),range(n_particles)) if i!=j]
            edge_index = torch.tensor(pairs, dtype=torch.long)
            edge_index=edge_index.t().contiguous()
            x = torch.tensor(d, dtype=torch.float)
            y = torch.tensor(d, dtype=torch.float)
            data = Data(x=x, edge_index=edge_index, y=y)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            
            # save data in format (jet_Data, event_of_jet, mass_of_jet, px, py, pz, e)
            torch.save((data, event_indices[data_idx], masses[data_idx], px[data_idx], py[data_idx], pz[data_idx], e[data_idx]), osp.join(self.processed_dir, 'data_{}.pt'.format(ijet)))
            ijet += 1

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        return data


