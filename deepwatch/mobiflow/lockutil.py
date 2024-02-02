import fcntl

def acquire_lock(f):
    if f is None:
        return

    while True:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            # Wait until lock can be acquired
            continue


def release_lock(f):
    if f is None:
        return

    while True:
        try:
            fcntl.flock(f, fcntl.LOCK_UN)
            return
        except BlockingIOError:
            # Wait until lock can be acquired
            continue
