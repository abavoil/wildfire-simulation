def get_frames(nb_records, nb_frames):
    """
    Generator of the indices to display if we have nb_records things and want to display nb_frames, and show the last thing
    """
    show_every = max(1, nb_records // (nb_frames + 1))
    yield from range(0, nb_records - 1, show_every)
    yield nb_records - 1
