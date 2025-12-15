import argparse
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from source import Source

@dataclass(frozen=True)
class Record:
    seconds:int

@dataclass(frozen=True)
class Run:
    window_size:int
    feature_count:int
Command = Run|Record

@dataclass(frozen=True)
class Args:
    source: Source
    output:str
    command:Command
    @staticmethod
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

    @staticmethod
    def from_parsed_args(raw:Namespace) -> "Args":
        # subcommand = raw.run if raw.run is not None else raw.record if raw.record else None
        # if raw.source is None:
        #     # error("missing data source")
        print(raw)
        source = Args.source_parser(raw.source)
        output = raw.output if raw.output is not None else raw.model
        command : Command
        match raw.command:
            case "run":
                command = Run(raw.window_size,raw.feature_count)
            case "record":
                command = Record(raw.seconds)
            case _:
                error(f"unknown sub-option {raw.command}")
        return Args(source,output,command)
    def open_source(self) -> Iterator[DataEntry]:
        pass # TODO

def parse_command_line() -> Args:
    def source_parser(source: str) -> Source:
        stream = iter(source.split(":"))
        first = next(stream, None)
        match first:
            case None:
                error("empty source")
            case "serial":
                port = next(stream, None)
                match port:
                    case Non9
                    very ugly
                    Send feedback
e:
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

    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("-v", "--verbose", action="store_true")
    shared.add_argument(
        "-s",
        "--source",
        help="choose source of sound data, either serial:COMPORT, microphone:[default | index:N | name:STR] | file:PATH",
        metavar="SOURCE",
        default="microphone:default"
        # required=True,
    )

    parser = argparse.ArgumentParser(
        prog="echosafe",
        description="Arduino to TensorFlowLite Interface bridge",  # whatever that means
        epilog=f"for more information see {REPO_URL}",
    )
    subparsers = parser.add_subparsers(dest="command")
    record = subparsers.add_parser("record",parents=[shared])
    record.add_argument(
        "-t", "--time", metavar="SECONDS", help="time in seconds to record"
    )
    record.add_argument("-o", "--output", metavar="FILE", help="file to write data to",default="recordings/recording.csv")

    run = subparsers.add_parser("run",parents=[shared])
    run.add_argument("-f","--feature-count", type=int, default=FEATURE_COUNT)
    run.add_argument("-w", "--window-size", type=int, default=WINDOW_SIZE)
    run.add_argument(
        "-o",
        "--output",
        default="./models/model.tflite",
        metavar="FILE",
        help="file path to write model",
    )

    parsed = parser.parse_args()
    return Args.from_parsed_args(parsed)
