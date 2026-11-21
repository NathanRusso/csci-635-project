from enum import Enum
class CardType(Enum):
    """Card Type Enum for Magic, Pokemon, and Yu-Gi-Oh."""
    MTG = 0     # Magic: The Gathering
    PKM = 1     # Pokemon
    YGO = 2     # Yu-Gi-Oh

# Map of card types to their name
card_type_map = {
    0: "Magic: The Gathering",
    1: "Pokemon",
    2: "Yu-Gi-Oh"
}

# Data Model Paths
PATH_MODEL_CNN = 'data/635-CNN.keras'
PATH_MODEL_VIT = 'data/635-ViT.keras'
PATH_MODEL_MOBILEVIT = 'data/635-MobileVit_v2.keras'

# Original Data Paths
PATH_MTG = 'data/mtg'
PATH_PKM = 'data/pokemon'
PATH_YGO = 'data/yugioh'

# Train Paths
PATH_TRAIN = 'data/train'
PATH_TRAIN_MTG = 'data/train/train_mtg.dat'
PATH_TRAIN_PKM = 'data/train/train_pkm.dat'
PATH_TRAIN_YGO = 'data/train/train_ygo.dat'

# Test Paths
PATH_TEST = 'data/test'
PATH_TEST_MTG = 'data/test/test_mtg.dat'
PATH_TEST_PKM = 'data/test/test_pkm.dat'
PATH_TEST_YGO = 'data/test/test_ygo.dat'

# Validation Paths
PATH_VALIDATION = 'data/validation'
PATH_VALIDATION_MTG = 'data/validation/validation_mtg.dat'
PATH_VALIDATION_PKM = 'data/validation/validation_pkm.dat'
PATH_VALIDATION_YGO = 'data/validation/validation_ygo.dat'

# Card Feature Sizes
SIZE_FEATURE = 350_000
SIZE_FEATURE_WIDTH = 500
SIZE_FEATURE_HEIGHT = 700

# Card Counts
COUNT_MTG = 16_990
COUNT_PKM = 21_199
COUNT_YGO = 14_599
COUNT_TRAIN = 9343
COUNT_TEST = 2919
COUNT_VALIDATION = 2335
