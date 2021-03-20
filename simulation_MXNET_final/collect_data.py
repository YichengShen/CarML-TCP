import ruamel.yaml
import yaml
import os



for AGGRE in ['cgc', 'simplemean', 'krum', 'median']:
    for I in range(1,6):
        with open('config.yml', 'r') as file:
            cfg = yaml.load(file, Loader=yaml.FullLoader)
            cfg['aggregation_method'] = AGGRE

        with open('config.yml', 'w') as fp:
            yaml.dump(cfg, fp)
        
        os.system(f"python3 main.py --num-round {I}")