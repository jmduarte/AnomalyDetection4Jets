from graph_data import GraphDataset
print("generating full dataset with padding")
bb1 = GraphDataset(root='/anomalyvol/data/gnn_geom/pad/', bb=1, n_particles=100, full=True)
print("bb1 done")
bb2 = GraphDataset(root='/anomalyvol/data/gnn_geom/pad/', bb=2, n_particles=100, full=True)
print("bb2 done")
bb3 = GraphDataset(root='/anomalyvol/data/gnn_geom/pad/', bb=3, n_particles=100, full=True)
print("bb3 done")