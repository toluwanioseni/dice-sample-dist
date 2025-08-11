import logging
from typing import List, Tuple, Union, Optional
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger("dice_sample")

# In function definitions for this class you will see types after each argument
# and the return type after the list of arguments with the syntax shown below. We
# expect all functions you write to also have types specified. The mypy extension
# to vscode will check that types are correct and report inconsistencies in the
# "problems" tab below the editor. If you are interested to learn more, go here:
# https://docs.python.org/3/library/typing.html

def generate_sample(die_type_counts: Tuple[int],
                    die_type_face_probs: Tuple,
                    num_draws: int,
                    rolls_per_draw: int, 
                    seed: Optional[int] = 63108,
                    ) -> NDArray[np.integer]:
    """Randomly selects a die from the bag num_draws time. Each die is rolled
    num_rolls times and the results are returned.

    Args:
        die_type_counts (Tuple[int]): The number of each type of die present
        in the bag. 
        die_type_face_probs (Tuple): Contains one Tuple for each die type 
        specifying the probability with which that die type rolls each face.
        Example with 2 3-sided dice- [[0.5, 0.2, 0.3]], [0, .5, .5]
        num_draws (int): The number of times to pull a die from the bag
        rolls_per_draw (int): The number of times a selected die is rolled.

    Returns:
        Tuple: A tuple of draws, each draw being a tuple of faces rolled.
        Example: 
    """
    die_type_counts_array = np.array(die_type_counts)
    die_type_probs = die_type_counts_array / sum(die_type_counts_array)
    # A tuple containing the number of faces on each dice of each type.
    face_counts_tuple = tuple(map(len, die_type_face_probs))
    # Set die_types_draw to a numpy ndarray of indices of randomly selected 
    # dice by using np.random.choice with the optional probabilities p = ... 
    np.random.seed(seed)
    die_types_drawn = None # YOUR CODE HERE
    # Define roll with the argument structure below, where draw_type is the
    # zero-based index of the type of die to be rolled. Use np.random.choice
    # to produce rolls_per_draw random rolls according to the die_type_face_probs
    # for the die type indicated. The easy way to do this is to use draw_type
    # as an index into die_type_face_probs and feed the result into
    # np.random.choice using the optional p = ... argument.
    def roll(draw_type: int) -> NDArray[np.integer]:
        None # YOUR CODE HERE
    # In python, map returns a map object which can be coerced into a tuple or
    # list, which we then coerce again into an np.array. The final result is an
    # array of num_draws arrays each containing rolls_per_draw rolls.
    return np.array(tuple(map(roll, die_types_drawn)))

