import jsonlines


def read_jsonl(file):
    with jsonlines.open(file) as rdr:
        yield from handle_jsonl(rdr)


def handle_jsonl(jsonl_reader):
    for ob in jsonl_reader:
        yield ob
