"""
K2 AeroSim — CalculiX .frd result reader
==========================================
Reads the node block, the element block and the nodal result blocks
(DISP, STRESS, …) of an ASCII .frd file.

Layout (CalculiX "short" format): a data line is " -1", a 10-column id and
12-column values, which may run together ("-1.16E-03-2.2E-09"), so the
fields are cut by column, never split on whitespace. Element connectivity
follows each " -1" element line on one or more " -2" lines.

Shell elements are written EXPANDED: an S4 becomes an 8-node hexahedron
(C3D8I) through the wall, with the same element number; its first four
nodes are the inner (bottom) face and the last four the outer (top) face,
node-for-node in the order of the shell element's own connectivity.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FrdResult:
    nodes: dict = field(default_factory=dict)      # id -> (x, y, z)
    elements: dict = field(default_factory=dict)   # id -> (type code, [node ids])
    blocks: dict = field(default_factory=dict)     # name -> {node id: [values]} (last step)


def read_frd(path) -> FrdResult:
    res = FrdResult()
    lines = Path(path).read_text(encoding="ascii", errors="replace").splitlines()
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        head = line[:6].strip()
        if head == "2C":                                    # nodes
            i += 1
            while i < n and not lines[i].startswith(" -3"):
                l = lines[i]
                if l.startswith(" -1"):
                    res.nodes[int(l[3:13])] = tuple(_values(l, 3))
                i += 1
        elif head == "3C":                                  # elements
            i += 1
            eid = etype = None
            while i < n and not lines[i].startswith(" -3"):
                l = lines[i]
                if l.startswith(" -1"):
                    eid, etype = int(l[3:13]), int(l[13:18])
                    res.elements[eid] = (etype, [])
                elif l.startswith(" -2") and eid is not None:
                    res.elements[eid][1].extend(
                        int(l[k:k + 10]) for k in range(3, len(l.rstrip()), 10))
                i += 1
        elif line.startswith(" -4"):                        # nodal result block
            name = line[5:13].strip()
            data = {}
            i += 1
            while i < n and not lines[i].startswith(" -3"):
                l = lines[i]
                if l.startswith(" -1"):
                    data[int(l[3:13])] = _values(l)
                i += 1
            res.blocks[name] = data         # later steps overwrite earlier ones
        i += 1
    return res


def _values(line, count=None):
    """12-column floats after the " -1" + 10-column id."""
    body = line[13:].rstrip()
    vals = [float(body[k:k + 12]) for k in range(0, len(body), 12)]
    return vals[:count] if count else vals
