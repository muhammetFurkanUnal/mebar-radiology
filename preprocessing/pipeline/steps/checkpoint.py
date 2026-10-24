from ..context import sample_metrics


class Checkpoint:
    def __init__(self, records: list, stage: str):
        self.records = records
        self.stage = stage

    def step(self, ctx):
        self.records.append(sample_metrics(ctx, self.stage))
        return ctx
