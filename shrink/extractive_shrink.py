import json

from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.kl import KLSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer

from shrink.shrink_method import ShrinkMethod


def get_sentence_list_word_count(sentence_list):
    words_count = 0
    for s in sentence_list:
        words_count += len(s.words)
    return words_count


class ExtractiveShrink(ShrinkMethod):

    def tokenize(self, args, article, body_size, encoder, pad, encoded_pad):
        if article:
            try:
                rel_matrix = json.loads(args.relavance_matrix)
                body = ''
                sentences = []
                not_adding = []
                for key in article.keys():

                    text = article[key]

                    max_sentences = self.find_max_sentences(rel_matrix, key)
                    if max_sentences:
                        parser = PlaintextParser.from_string(text, Tokenizer('english'))
                        sentences = sentences + list(self.get_extractive_summary(parser, args.body_shrink_extractive_method,
                                                    max_sentences))
                    else:
                        not_adding.append(key)

                if len(sentences) < 100 and len(not_adding) > 0:
                    text = article[not_adding[0]]
                    parser = PlaintextParser.from_string(text, Tokenizer('english'))
                    sentences = sentences + list(self.get_extractive_summary(parser, args.body_shrink_extractive_method, 100 - len(sentences)))

                for s in sentences:
                    body += " ".join(s.words)
                encoded_text = encoder.encode(body, max_length=body_size)
                if len(encoded_text) < body_size:
                    remainder = body_size - len(encoded_text)
                    while remainder > 0:
                        encoded_text = encoded_text + encoded_pad
                        remainder = remainder - len(encoded_pad)
                return encoded_text
            except BaseException as e:
                self.logger.error(e)
                return []
        else:
            return []

    def get_summarizer(self, method):
        if method == 'text_rank':
            return TextRankSummarizer()
        elif method == 'lex_rank':
            return LexRankSummarizer()
        elif method == 'lsa':
            return LsaSummarizer()
        elif method == 'luhn':
            return LuhnSummarizer()
        elif method == 'kl':
            return KLSummarizer()
        else:
            return None

    def get_extractive_summary(self, parser, method, max_sentences):
        summarizer = self.get_summarizer(method)
        summary_sentences = summarizer(parser.document, sentences_count=max_sentences)
        return summary_sentences

    def summarize_for_max_words(self, parser, method, max_words, num_of_sentences=100, offset=10,
                                last_iteration_words_count=-1, optimize=False):
        sentence_list = self.get_extractive_summary(parser, method, num_of_sentences)
        if optimize:
            words_count = get_sentence_list_word_count(sentence_list)

            # higher word count > reduce sentence size
            if words_count > max_words:
                return self.summarize_for_max_words(parser, method, max_words, num_of_sentences - 1, offset,
                                                    words_count)

            # lower words counts
            elif words_count + offset < max_words:
                # last time it was higher than max > stop here
                if last_iteration_words_count > max_words:
                    self.setup_last_sentence_count(num_of_sentences)
                    return sentence_list

                # last time it was same words count :> text is over smaller > stop here
                elif last_iteration_words_count == words_count:
                    return sentence_list

                # increase sentence size
                else:
                    return self.summarize_for_max_words(parser, method, max_words, num_of_sentences + 1, offset,
                                                        words_count)

            # less than max, withing offset
            else:
                self.setup_last_sentence_count(num_of_sentences)
                return sentence_list
        else:
            self.setup_last_sentence_count(num_of_sentences)
            return sentence_list

    def setup_last_sentence_count(self, num_of_sentences):
        self.meta["body_shrink_extractive_last_sentences"] = num_of_sentences

    def find_max_sentences(self, rel_matrix, key):
        if key in rel_matrix:
            return rel_matrix[key]
        else:
            return None
