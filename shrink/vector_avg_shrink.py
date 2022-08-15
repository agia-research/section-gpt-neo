import math

import numpy as np

from shrink.shrink_method import ShrinkMethod


class VectorAverage(ShrinkMethod):

    def tokenize(self, args, text, body_size, encoder, pad, encoded_pad):
        text_list = text.split() if text else []
        chunk_size = math.ceil(len(text_list) / body_size)
        if len(text_list) > 0 and chunk_size > 0:
            text_arrays = np.array_split(text_list, chunk_size)
            encoded_list = []
            encoder.max_length = body_size
            lowest_chunk_size = body_size
            for tl in text_arrays:
                if len(tl) < lowest_chunk_size:
                    lowest_chunk_size = len(tl)
            for tl in text_arrays:
                body = " ".join(tl)
                encoded_list.append(encoder.encode(body, max_length=lowest_chunk_size))
            np_array = np.array(encoded_list)
            avg = np.mean(np_array, axis=0, dtype=np.int32)




            avg_list = avg.tolist()
            if len(avg_list) < body_size:
                remainder = body_size - len(avg_list)
                while remainder > 0:
                    avg_list = avg_list + encoded_pad
                    remainder = remainder - len(encoded_pad)

            # start + avg_body
            return avg_list
        else:
            return []
