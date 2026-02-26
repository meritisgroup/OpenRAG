import threading

def get_executor_threads():
    threads = [t for t in threading.enumerate() if t.name.startswith('ThreadPoolExecutor')]
    return len(threads)