from datetime import datetime


class Timer:
    def __init__(self):
        self.start_time = {"epoch": [], "weight": [], "geo": [], "knn": []}
        self.end_time = {"epoch": [], "weight": [], "geo": [],"knn": []}
        self.opts = ["epoch", "weight", "geo", "knn"]

    def start(self, opt="epoch"):
        """Start the timer."""
        self.start_time[opt] += [datetime.now()]

    def stop(self, opt="epoch"):
        """Stop the timer."""
        self.end_time[opt] += [datetime.now()]

    def summary(self):
        """Return the elapsed time in seconds."""
        elapsed_time = {}
        summary_time = {}
        for opt in self.opts:
            start_time = self.start_time[opt]
            end_time = self.end_time[opt]
            total = len(start_time)
            elapsed_time[opt] = [
                (e - s).total_seconds() for s, e in zip(start_time, end_time)
            ]
            summary_time[opt] = sum(elapsed_time[opt]) / total
        print(f"Timer Summary:{summary_time} \n elapsed Summary: {elapsed_time}")
        return elapsed_time, summary_time


timer_epoch = Timer()
