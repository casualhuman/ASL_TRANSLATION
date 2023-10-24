

def class_name():
    """
    Generate a list of lowercase English alphabet letters from 'a' to 'z'.
    """

    values = [chr(i) for i in range(ord('a'), ord('z') + 1)]

    return values

def class_weight():

    data_dict = {
    "a": 4161,
    "b": 5419,
    "c": 5265,
    "d": 6040,
    "e": 2658,
    "f": 6357,
    "g": 5941,
    "h": 5960,
    "i": 6146,
    "j": 5017,
    "k": 5686,
    "l": 6221,
    "m": 2369,
    "n": 1961,
    "o": 3170,
    "p": 1744,
    "q": 1142,
    "r": 3008,
    "s": 2247,
    "t": 2720,
    "u": 2763,
    "v": 4027,
    "w": 2342,
    "x": 2030,
    "y": 2456,
    "z": 1906
    }
    
    total_samples = sum(data_dict.values())

    class_dict = {}

    for key, value in data_dict.items():
        weight = total_samples / (len(data_dict.keys()) * value)
        class_dict[key] = round(weight, 2)

    return class_dict
