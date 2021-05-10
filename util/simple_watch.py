import time


# commentいれるとか工夫しろよ
def simple_watch():
    if hasattr(simple_watch, "start"):
        start = simple_watch.start
        end = time.time()
        elapsed = end - start
        simple_watch.start = end
        return elapsed
    else:
        simple_watch.start = time.time()
        return 0


def to_str(t):
    t = int(t)
    m, s = divmod(t, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return "{}h{}m".format(h, m)
    if m > 0:
        return"{}m{}s".format(m, s)
    return "{}s".format(s)


def watch_as_str(cmt=""):
    elapsed = simple_watch()
    return cmt + ": " + to_str(elapsed)

