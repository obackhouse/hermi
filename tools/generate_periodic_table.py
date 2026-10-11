"""Generate the packaged periodic table from PubChem's periodic table data."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib.request import Request, urlopen

SOURCE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/periodictable/JSON"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "hermi" / "periodic_table.json"
CONFIGURATION_TERM = re.compile(r"(\d+)([spdf])([0-9]+)")
NOBLE_GASES = {"He", "Ne", "Ar", "Kr", "Xe", "Rn"}


def shell_electrons(configuration: str, known_shells: dict[str, list[int]]) -> list[int]:
    """Convert an electron configuration to electron counts by principal shell."""
    configuration = configuration.replace("(predicted)", "").strip()
    core_match = re.match(r"^\[([A-Z][a-z]?)\]", configuration)
    if core_match is None:
        shells: list[int] = []
    else:
        core_symbol = core_match.group(1)
        if core_symbol not in known_shells:
            raise ValueError(f"unknown noble-gas core: {core_symbol}")
        shells = known_shells[core_symbol].copy()
        configuration = configuration[core_match.end() :]

    terms = list(CONFIGURATION_TERM.finditer(configuration))
    if not terms:
        raise ValueError(f"cannot parse electron configuration: {configuration!r}")

    for term in terms:
        principal_shell = int(term.group(1))
        occupancy = int(term.group(3))
        shells.extend([0] * (principal_shell - len(shells)))
        shells[principal_shell - 1] += occupancy

    return shells


def periodic_table() -> list[dict[str, str | int | list[int]]]:
    """Fetch and reduce the periodic table to the fields used by hermi."""
    request = Request(SOURCE, headers={"User-Agent": "hermi periodic table generator"})
    with urlopen(request, timeout=30) as response:
        data = json.load(response)

    columns = data["Table"]["Columns"]["Column"]
    column_index = {name: index for index, name in enumerate(columns)}
    elements = []
    known_shells: dict[str, list[int]] = {}

    for row in data["Table"]["Row"]:
        cells = row["Cell"]
        atomic_charge = int(cells[column_index["AtomicNumber"]])
        symbol = cells[column_index["Symbol"]]
        configuration = cells[column_index["ElectronConfiguration"]]
        shells = shell_electrons(configuration, known_shells)
        if sum(shells) != atomic_charge:
            raise ValueError(f"electron count for {symbol} does not match atomic charge")

        if symbol in NOBLE_GASES:
            known_shells[symbol] = shells
        elements.append(
            {
                "symbol": symbol,
                "atomic_charge": atomic_charge,
                "name": cells[column_index["Name"]],
                "shells": shells,
            }
        )

    return elements


def main() -> None:
    """Write the periodic table JSON file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    elements = periodic_table()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(elements, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
