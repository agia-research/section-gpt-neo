from shrink.extractive_shrink import ExtractiveShrink
from shrink.shrink_method import ShrinkMethod
from shrink.vector_avg_shrink import VectorAverage


def get_shrink_class(args) -> ShrinkMethod:
    if args.body_shrink_method == 'vector_avg':
        return VectorAverage()
    elif args.body_shrink_method == 'extractive':
        return ExtractiveShrink()
