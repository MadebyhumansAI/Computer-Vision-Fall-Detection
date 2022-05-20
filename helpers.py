import urllib.request
import sys
import time

def download_progress_hook(count, block_size, total_size):
    """
    A hook to report the progress of a download. Displays a progress bar.
    """
    global start_time

    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time

    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = min(int(count * block_size * 100 / total_size), 100)
    
    sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                    (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()