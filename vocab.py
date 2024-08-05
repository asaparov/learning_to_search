
NAMES = [
    'Alex', 'Bob', 'Charlie', 'David', 'Eve', 'Fred', 'Gina', 'Hank', 'Ivy', 'Jack', 
    'Kyle', 'Lily', 'Mia', 'Nate', 'Olivia', 'Pam', 'Quinn', 'Ryan', 'Sam', 'Tara', 
    'Uma', 'Victor', 'Wendy', 'Xavier', 'Yara', 'Zara'
    ]

NOUNS = [
    'qumpus', 'shumpus', 'grumpus', 'plumpus', 'clumpus', 'kumpus', 'sumpus', 'slumpus', 'umpus', 
    'flumpus', 'lumpus', 'rumpus', 'numpus', 'glumpus', 'mumpus', 'tumpus', 'humpus', 'bumpus', 
    'pumpus', 'xumpus', 'wumpus', 'jumpus', 'yumpus', 'zumpus', 'blumpus', 'dumpus', 'frumpus', 'vumpus'
    ]


CONNECTORS = {
    "is a": 'singular',
    "has": 'plural',
    "wants": 'plural',
    "likes": 'plural',
    "cares for a": "singular",
    "is friends with a": "singular",
}


VOCAB = NAMES + \
        NOUNS + \
        [noun + 'es' for noun in NOUNS] + \
        ['a', 'is', 'has', 'wants', 'likes', 'cares', 'for', 'friends', 'with', 'then', 'Given', 'If', 'prove' ] + \
        ['.', ' ', ',', '\n', ":"] +\
        ['Query', 'Prefix', 'Statements'] + \
        ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]