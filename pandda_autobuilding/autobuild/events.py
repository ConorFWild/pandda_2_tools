from typing import NamedTuple


class Event(NamedTuple):
    dtag: str
    event_idx: int
    site: int
    bdc: float
    resolution: float
    x: float
    y: float
    z: float


def get_events(events_table):
    events = {}

    for idx, row in events_table.iterrows():
        events[(row["dtag"], row["event_idx"])] = Event(dtag=row["dtag"],
                                                        event_idx=row["event_idx"],
                                                        site=row["site_idx"],
                                                        bdc=row["1-BDC"],
                                                        resolution=row["analysed_resolution"],
                                                        x=row["x"],
                                                        y=row["y"],
                                                        z=row["z"],
                                                        )


    return events
