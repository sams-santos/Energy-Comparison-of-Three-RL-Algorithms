def reward_function(r_type: str, distance: float) -> float:
    """Return reward according to selected reward type."""
    if r_type == "R1":
        return 1.0 / distance if distance != 0 else 0.0
    if r_type == "R2":
        return -distance
    if r_type == "R3":
        return -(distance ** 2)
    raise ValueError(f"Unknown reward type: {r_type}")
