from shrink.extractive_shrink import ExtractiveShrink
from shrink.shrink_method import ShrinkMethod
from shrink.vector_avg_shrink import VectorAverage


def get_shrink_class(args, logger) -> ShrinkMethod:
    meta = {
        "body_shrink_extractive_last_sentences": -1,
        "paper_id": None,
        "processed_count": 0,
        "failed_count": 0
    }
    if args.body_shrink_method == 'vector_avg':
        return VectorAverage(meta, logger)
    elif args.body_shrink_method == 'extractive':
        return ExtractiveShrink(meta, logger)
