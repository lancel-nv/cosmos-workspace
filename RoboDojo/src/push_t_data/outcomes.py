from dataclasses import dataclass


@dataclass(frozen=True)
class OutcomeQuota:
    min_episodes: int
    min_successes: int = 0
    min_failures: int = 0

    def __post_init__(self):
        for value in (self.min_episodes, self.min_successes, self.min_failures):
            if type(value) is not int or value < 0:
                raise ValueError("Outcome minima must be nonnegative integers")

    def met(self, episodes, successes):
        if not 0 <= successes <= episodes:
            raise ValueError("Invalid observed outcome counts")
        return (
            episodes >= self.min_episodes
            and successes >= self.min_successes
            and episodes - successes >= self.min_failures
        )

    def require(self, episodes, successes):
        if not self.met(episodes, successes):
            raise RuntimeError(
                f"Outcome quota unmet: episodes={episodes}/{self.min_episodes}, "
                f"successes={successes}/{self.min_successes}, failures={episodes - successes}/{self.min_failures}"
            )
