"""
progress.py — one logging style for every stage.

    from progress import stage, step, done, bar

    with stage('build dataset'):
        for f in bar(files, 'aggregating'):
            ...
        step('sanitised 30k non-finite cells')
    done('wrote X', path)

Prints elapsed time per stage and flushes, so a long run shows movement
instead of silence.
"""
import contextlib
import sys
import time

try:
    from tqdm.auto import tqdm
except ImportError:                                  # tqdm is in requirements
    tqdm = None

_t0 = time.time()
_depth = 0


def _elapsed():
    s = time.time() - _t0
    return f'{int(s)//60:d}:{int(s) % 60:02d}'


def log(msg, indent=0):
    pad = '  ' * (_depth + indent)
    print(f'[{_elapsed()}] {pad}{msg}', flush=True)


def step(msg):
    """A line inside the current stage."""
    log(msg, indent=1)


def done(msg, path=None):
    log(f'{msg}{"  ->  " + str(path) if path else ""}', indent=1)


@contextlib.contextmanager
def stage(name):
    """A named block, timed, with a banner either side."""
    global _depth
    log(f'>>> {name}')
    _depth += 1
    t = time.time()
    try:
        yield
    finally:
        _depth -= 1
        log(f'<<< {name}  ({time.time() - t:.0f}s)')


def bar(it, desc, total=None, unit='it'):
    """tqdm when available, a plain counter when not."""
    if tqdm is not None:
        return tqdm(it, desc='  ' * (_depth + 1) + desc, total=total,
                    unit=unit, file=sys.stdout, dynamic_ncols=True, leave=True)
    return it
