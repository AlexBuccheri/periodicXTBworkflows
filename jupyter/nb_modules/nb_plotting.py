""" Migrate plotting and data analysis functions from NBs to here
"""


def bin_list(input: list, bin_size: int) -> list:
    """ Bin a list.

    Could be split into an indexing routine and list comprehension.
    Example
    bin_list([1, 2, 3, 4, 5], 2)
    Return [[1, 2], [3, 4], [5]]
    """

    n = len(input)
    n_bins = int((n + (n % bin_size)) / 2)

    indices = []
    for i in range(0, n_bins):
        start = 0 + i * bin_size
        end = start + bin_size
        indices.append((start, end))

    # If the data is not evenly divisible by the bin size
    # fix the end value of the final indices
    if n % bin_size > 0:
        last_element = indices.pop()
        indices.append((last_element[0], n))

    binned_list = [input[start: end] for start, end in indices]
    return binned_list
