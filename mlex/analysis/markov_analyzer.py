import pandas as pd
from mlex.utils.utils import ensure_directory_exists


class MarkovAnalyzer:
    def __init__(self):
        pass

    def analyze(self, array_data, column_name, path_save, verbose=True):
        states = sorted(array_data.unique())

        states_map = {state: i for i, state in enumerate(states)}

        frequencies = pd.DataFrame(
            0,
            index=states,
            columns=states,
            dtype=int
        )

        for i in range(len(array_data) - 1):
            current_state = array_data.iloc[i]
            next_state = array_data.iloc[i+1]
            frequencies.loc[current_state, next_state] += 1

        if verbose:
            print(f"\n--- Transition Frequency Matrix for the algorithm: {column_name.upper()} ---")
            print(frequencies.to_string())
            print("\n" + "="*80)

        sum_row = frequencies.sum(axis=1)

        sum_row[sum_row == 0] = 1 

        probability_matrix = frequencies.div(sum_row, axis=0)

        if verbose:
            print(f"\n--- Transition Probability Matrix for the algorithm: {column_name.upper()} ---")
            print(probability_matrix.to_string())
            print("\n" + "="*80)

        ensure_directory_exists(path_save)
        frequencies.to_csv(f"{path_save}/frequencies_{column_name.lower()}.csv", index=False)
        probability_matrix.to_csv(f"{path_save}/probability_matrix_{column_name.lower()}.csv", index=False)

        return frequencies, probability_matrix
