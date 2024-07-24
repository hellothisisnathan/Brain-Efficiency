import pandas as pd

proj_vol = pd.read_csv('../hypothalamus/hy exclusive edge list 100.csv')
dists = pd.read_csv('../edge_withDistancesAndRadii.csv')

new = pd.merge(dists, proj_vol[['origin_id', 'target_id', 'projection_volume']], how='left', left_on=['n1', 'n2'], right_on=['origin_id', 'target_id'])
new[['n1', 'n2', 'intensity', 'distance', 'R1+R2', 'projection_volume']].to_csv('./hy edge list intensities and projection volumes.csv', index=None)