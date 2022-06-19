import argparse


def get_arg_parser():
    parser = argparse.ArgumentParser()

    # from main.py
    parser.add_argument("--tpu", type=str, help="Name of TPU to train on, if any.")
    parser.add_argument("--gpu_ids", nargs="+", type=str, default=["device:GPU:0"],
                        help="If training on GPU, can specify your GPU names in a list - i.e 'device:GPU:0 device:GPU:1'")
    parser.add_argument("--model", type=str, default=None, help="JSON file that contains model parameters.")
    parser.add_argument("--steps_per_checkpoint", type=int, default=5000, help="Save a model checkpoint every X steps.")
    parser.add_argument("--auto_layout", action="store_true", help="If set, generates and prints the most memory "
                                                                   "efficient layout according to MTF auto layout.")
    parser.add_argument("--auto_layout_and_mesh_shape", action="store_true",
                        help="If set, generates and prints the most memory efficient layout and mesh shape according to"
                             " MTF auto layout.")
    parser.add_argument("--new", action="store_true", help="If set, deletes previous checkpoint, if it exists, and "
                                                           "starts a new training run")
    parser.add_argument("--predict", action="store_true", help="If set, uses the model to predict rather than train.")
    parser.add_argument("--eval", action="store_true", help="If set, run model in evaluation mode.")
    parser.add_argument("--prompt", type=str, help="path to .txt file containing a prompt for prediction. If empty, "
                                                   "defaults to unicorns.",
                        default="")
    parser.add_argument("--prompt_text", type=str, help="prompt for prediction. If empty, "
                                                        "defaults to unicorns.",
                        default="")
    parser.add_argument("--check_dataset", action="store_true",
                        help="If set, outputs sample from the dataset and quits.")
    parser.add_argument("--sacred_id", type=str, default="nosacred", help="Sacred run id.")
    parser.add_argument("--entmax_sampling", action="store_true", help="(experimental) use entmax sampling")
    parser.add_argument("--export", action="store_true", help="If set, will export the model.")
    parser.add_argument("--predict_out_dir", type=str, default="./", help="Directory if saving prediction outputs.")
    parser.add_argument("--chunk_size", type=int, default=2048, help="How big a chunk should be in chunk mode. "
                                                                     "Should equal your model's context size")
    parser.add_argument("--summary_tag", type=str, default=">Summary:", help="Tag of summary")
    parser.add_argument("--start_tag", type=str, default="<|startoftext|>Text:", help="Tag of start")
    parser.add_argument("--end_tag", type=str, default="<|endoftext|>", help="Tag of end")
    parser.add_argument("--pad_tag", type=str, default="<|pad|>", help="Tag of pad")
    parser.add_argument("--summary_size", type=int, default=1000, help="Number of words to allocate for summary")


    # from create_tfrecords.py
    parser.add_argument("--input_dir", type=str, default="./data", help="Path to where your files are located. Files ending in .zst are "
                                                      "treated as archives, all others as raw text.")
    parser.add_argument("--files_per", type=int, default=100000, help="Text files per tfrecord")
    parser.add_argument("--name", type=str, default="openwebtext",
                        help="Name of output files will be name_i.tfrecords where i is the number of the file")
    parser.add_argument("--output_dir", type=str, default="./tfrecords", help="Where to put tfrecords")
    parser.add_argument("--encoder_path", type=str,
                        help="Path to encoder files, or leave unspecified to use GPT2 tokenizer")
    parser.add_argument("--minimum_size", type=int, default=100,
                        help="Minimum size a document has to be to be included")
    parser.add_argument("--ftfy", action="store_false", help="normalize with ftfy")
    parser.add_argument("--wikitext-detokenize", action="store_false", help="use wikitext detokenizer")
    parser.add_argument("--separator", nargs="+", type=int, default=[50256],
                        help="separator to place between files in chunk mode")
    parser.add_argument("--write_dataset_config", action="store_true",
                        help="Write the dataset config file on completion")
    parser.add_argument("--processes", type=int, default=0, help="Number of processes to use. Defaults to cpu count.")
    parser.add_argument("--arxiv_papers", action="store_true",
                        help="Json file processor and vector avg for long documents")
    parser.add_argument("--add_abstract", action="store_true", help="Add paper abstract to text")
    parser.add_argument("--add_introduction", action="store_true", help="Add paper introduction to text")
    return parser