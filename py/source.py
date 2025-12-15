from dataclasses import dataclass
from time import time
from printing import error, warning
from collections.abc import Iterable,Iterator
# Source ADT
@dataclass(frozen=True)
class SerialSource:
    port: str


@dataclass(frozen=True)
class MicrophoneSource:
    default: bool = False
    index: int | None = None
    substring: str | None = None


@dataclass(frozen=True)
class FileSource:
    path: str


Source = SerialSource | MicrophoneSource | FileSource
def source_parser(source: str) -> Source:
    stream = iter(source.split(":"))
    first = next(stream, None)
    match first:
        case None:
            error("empty source")
        case "serial":
            port = next(stream, None)
            match port:
                case None:
                    error("serial: requires a COMPORT, eg. --source serial:COM0")
                case _:
                    return SerialSource(port)
        case "file":
            path = next(stream, None)
            match path:
                case None:
                    error("file: requires a PATH, eg. --source file:./data.csv")
                case _:
                    return FileSource(path)
        case "microphone":
            submethod = next(stream, None)
            match submethod:
                case None:
                    error(
                        "microphone: requires a submethod eg. --source microphone:default"
                    )
                case "default":
                    return MicrophoneSource(default=True)
                case "index":
                    i = next(stream, None)
                    match i:
                        case None:
                            error(
                                "microphone:index requires a number, eg. --source microphone:index:0"
                            )
                        case _:
                            if not i.isdigit():
                                error(
                                    f'{i} is not a digit in "microphone:index:{i}"'
                                )
                            return MicrophoneSource(index=int(i))
                case "name":
                    name = next(stream, None)
                    match name:
                        case None:
                            error(
                                "microphone:index requires a name which may be a substring of the full system name, eg. --source microphone:name:built-in"
                            )
                        case _:
                            return MicrophoneSource(substring=name)
                case _:
                    error(f"Unknown method {submethod}")
        case _:
            error(f"Unknown method {first}")

@dataclass
class DataEntry:
    time: float
    microphone: int
    quantity: int

    @staticmethod
    def from_csv_entry(s: str) -> "DataEntry|None":
        parsed = s.strip().split(",")[:3]
        raw_time, raw_microphone, raw_quantity = parsed
        # if any arent digits
        if any(map(lambda a: not a.isdigit(), parsed)):
            warning(f"could not parse mangled CSV entry {parsed} non-digit present.")
            return None
        else:
            time = float(raw_time)
            microphone = int(raw_microphone)
            quantity = int(raw_quantity)
            return DataEntry(time, microphone, quantity)

    def to_csv_entry(self) -> str:
        return f"{self.time},{self.microphone},{self.quantity}"

    @staticmethod
    def from_mic_iterable(microphone_values: Iterable[int]) -> "Iterator[DataEntry]":
        start = time()
        return map(
            lambda mic: DataEntry(time() - start, int(mic), QUANTITY), microphone_values
        )
        # filter(lambda mic: not mic.isdigit(), microphone_values),

    # @staticmethod
    # def from_iterable(xs: Iterable) -> "Iterator[DataEntry]":
    #     return cast(  # needed because pyright is unaware of the type narrowing in the filter
    #         Iterator[DataEntry],
    #         filter(
    #             lambda x: isinstance(x, DataEntry),
    #             map(lambda entry: DataEntry.from_csv_entry(entry), xs),
    #         ),
    #     )

    # @staticmethod


# returns an iterator over the serial connection
#
