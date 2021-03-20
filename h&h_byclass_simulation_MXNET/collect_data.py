from main import main

for AGGRE in ['cgc', 'simplemean', 'krum', 'median']:
    for I in range(1,6):
        main(AGGRE, I)