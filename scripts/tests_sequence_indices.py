def _get_gru4rec_sequence_indices_fn(
    complete_sequence, max_context_len: int, sliding_window_step_size: int
):
    sequence_len = len(complete_sequence)
    end_indexes = list(range(1, sequence_len - 1, sliding_window_step_size))
    start_indexes = [max(0, idx - max_context_len) for idx in end_indexes]
    return start_indexes, end_indexes


if __name__ == "__main__":
    complete_sequence = list(range(10))
    max_context_len = 15
    sliding_window_step_size = 1

    start_indexes, end_indexes = _get_gru4rec_sequence_indices_fn(
        complete_sequence, max_context_len, sliding_window_step_size
    )
    print(start_indexes)
    print(end_indexes)
    for start_idx, end_idx in zip(start_indexes, end_indexes):
        print(complete_sequence[start_idx:end_idx])
