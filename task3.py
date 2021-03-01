import random

"""Task3: Generate a list of 100'000 tuples (x,id), where x is a value between 0 and 1000 and id is a unique
identifier. Return the id of the 500 elements with the smallest x. You may return more than 500 elements,
if the 500th element has the same value as the 501st, 502nd, etc., however, you should not return fewer than 500
elements. Make sure you do this efficiently. Hint: this can be done in O(n) """


# For this question we consider values of x are integers.

# tuples generator
def random_tuples_gen(min, max, n):
    """
    The generator generate a list of n tuples (x,id), where x is a value between 0 and 1000 and id is a unique
identifier.
    :param n: length of the list
    :param min: min value of x
    :param max: max value of x
    :return: a tuple (x, id) generator
    """
    for i in range(n):
        yield random.randint(min, max), i


# Solution
"""
1, We firstly create a hashmap. The key of the map takes all possible values from the tuple, which range from 1 to 
1000 and the value of a key corresponds to a list which contain all the ids of the tuples which have value this 
key.

2. We iterate the map as their keys correspond to the value of tuples, so we will get the smallest values firstly, and 
we stop the loop when we have more than 500 ids.
"""


def build_map_from_tuples(tuples_gen):
    """
    We create the hashmap from a generator of a list of tuples (x, id)
    :param tuples_gen: a generator of a list of tuples
    :return: a hashmap (value, [ids])
    """
    value_map = {}
    for i in range(1001):
        value_map[i] = []
    for tuple in tuples_gen:
        value_map[tuple[0]].append(tuple[1])
    for key in list(value_map):
        if len(value_map[key]) == 0:
            value_map.pop(key)
    return value_map


def get_ids_from_map(value_map, top_n):
    """
    Get the top_n ids from the maps. These ids are the top_n smallest values' ids.
    :param value_map: hashmap achieve from build_map_from_tuples
    :param top_n: top_n smallest number
    :return: a list of ids
    """
    ids = []
    for value in value_map.values():
        ids = ids + value
        if len(ids) >= top_n:
            break
    return ids


def main():
    list_tuple = random_tuples_gen(0, 1000, 100000)
    value_map = build_map_from_tuples(list_tuple)
    ids = get_ids_from_map(value_map, 500)
    print(f"The id of the 500 elements with the smallest x: {ids}")


if __name__ == "__main__":
    main()