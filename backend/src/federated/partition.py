"""Split training cases into 3 hospital shards. Same validation set for everyone."""
from src.data.transforms import get_file_lists

NUM_HOSPITALS = 3


def get_partitions(max_train=60, max_val=12):
    train, val = get_file_lists()
    train = train[:max_train]
    val = val[:max_val]
    shards = [train[i::NUM_HOSPITALS] for i in range(NUM_HOSPITALS)]
    return shards, val


if __name__ == "__main__":
    shards, val = get_partitions()
    for i, s in enumerate(shards):
        print(f"Hospital {i}: {len(s)} cases")
    print(f"Shared validation: {len(val)} cases")
