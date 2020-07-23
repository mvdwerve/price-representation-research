from optparse import OptionGroup


def parse_GIM_args(parser):
    group = OptionGroup(parser, "GIM training options")
    group.add_option(
        "--learning_rate", type="float", default=1.5e-4, help="Learning rate"
    )
    group.add_option(
        "--prediction_step",
        type="int",
        default=12,
        help="Time steps k to predict into future",
    )
    group.add_option(
        "--negative_samples",
        type="int",
        default=256,
        help="Number of negative samples to be used for training",
    )
    group.add_option(
        "--sampling_method",
        type="int",
        default=1,
        help="Which type of method to use for negative sampling: \n"
        "0 - inside the loop for the prediction time-steps. Slow, but samples from all but the current pos sample \n"
        "1 - outside the loop for prediction time-steps, "
        "Low probability (<0.1%) of sampling the positive sample as well. \n"
        "2 - outside the loop for prediction time-steps. Sampling only within the current sequence"
        "Low probability of sampling the positive sample as well. \n",
    )
    parser.add_option(
        "--nosubsampledata",
        action="store_true",
        default=False,
        help="Boolean flag to disable subsample in the dataset per file, or rather expose all samples.",
    )
    parser.add_option_group(group)
    return parser
