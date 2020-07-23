from optparse import OptionGroup


def parse_architecture_args(parser):
    group = OptionGroup(parser, "Architecture options")

    group.add_option(
        "--loss",
        default="InfoNCE",
        help="Choose between different loss functions to be used for training.",
        choices=[
            "InfoNCE",
            "VAE",
            "BCE-Movement",
            "BCE-Up-Movement",
            "BCE-Anomaly",
            "BCE-Future-Anomaly",
        ],
    )
    parser.add_option(
        "--enc_hidden", type="int", default=32, help="Hidden size of encoder layers.",
    )
    parser.add_option(
        "--reg_hidden",
        type="int",
        default=4,
        help="Hidden size of autoregressive layers.",
    )
    parser.add_option_group(group)
    return parser
