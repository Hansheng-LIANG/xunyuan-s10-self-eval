"""Read and validate physics samples recorded during a route."""

from .io import FormatError, boolean, contained, loads, number, vector


def validate_sample(row):
    if not isinstance(row, dict):
        raise FormatError("Telemetry row must be an object")
    number(row.get("time_s"), "time_s", 0)
    vector(row.get("position_m"), 3, "position_m")
    for key in ["yaw_rad", "roll_rad", "pitch_rad", "yaw_rate_rps"]:
        number(row.get(key), key)
    number(row.get("speed_mps"), "speed_mps", 0)
    points = row.get("envelope_points_m")
    if not isinstance(points, list) or len(points) < 4:
        raise FormatError("envelope_points_m needs at least four full-envelope support points")
    for point in points:
        vector(point, 3, "envelope point")
    count = row.get("forbidden_contacts")
    if type(count) is not int or count < 0:
        raise FormatError("forbidden_contacts must be a nonnegative integer")
    for key in ["out_of_bounds", "action_valid", "intervention"]:
        boolean(row.get(key), key)


def load_trace(root, name):
    path = contained(root, name)
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise FormatError(f"{name}:{line_number}: blank telemetry row")
            row = loads(line)
            validate_sample(row)
            if rows and row["time_s"] <= rows[-1]["time_s"]:
                raise FormatError(f"{name}:{line_number}: timestamps must increase")
            if rows and len(row["envelope_points_m"]) != len(rows[0]["envelope_points_m"]):
                raise FormatError(f"{name}:{line_number}: collision envelope point count changed")
            rows.append(row)
    if not rows or abs(rows[0]["time_s"]) > 1e-9:
        raise FormatError(f"{name}: trace must start at time 0 with at least one row")
    return rows
