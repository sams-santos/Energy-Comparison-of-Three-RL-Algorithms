def reward_function(r_type: str, distance: float) -> float:
    """Return reward according to selected reward type."""
    if r_type == "R1":
        return 1.0 / distance if distance != 0 else 0.0
    if r_type == "R2":
        return -distance
    if r_type == "R3":
        return -(distance ** 2)
    raise ValueError(f"Unknown reward type: {r_type}")


def epsilon_decay(e_type: str, episode: int, total_episodes: int,epsilon_init: float) -> float:
    """Return decayed epsilon according to chosen schedule."""
    if e_type == "fixed":
        return epsilon_init
    if e_type == "linear":
        return epsilon_init * (1.0 - episode / total_episodes)
    if e_type == "concave":
        return epsilon_init * (0.999 ** episode)
    if e_type == "convex":
        return epsilon_init * (1.0 - (episode / total_episodes) ** 6)
    if e_type == "step":
        return epsilon_init * (1.0 - 0.1 * (episode // 1000))
    raise ValueError(f"Unknown epsilon decay type: {e_type}")