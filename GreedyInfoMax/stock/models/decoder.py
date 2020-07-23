import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, input_dim, encoder):
        super(Decoder, self).__init__()

        self.model = nn.Sequential()

        # iterate over the encoder layers in reverse
        for idx in reversed(range(len(encoder.kernel_sizes))):
            self.model.add_module(
                "layer {}".format(idx),
                self.new_block(
                    input_dim,
                    encoder.hidden if idx != 0 else encoder.input_dim,
                    encoder.kernel_sizes[idx],
                    encoder.strides[idx],
                    encoder.padding[idx],
                    add_relu=(idx != 0),  # only the last layer has no ReLU any more
                ),
            )
            input_dim = encoder.hidden

    def new_block(self, in_dim, out_dim, kernel_size, stride, padding, add_relu=True):
        # create the new block
        new_block = nn.Sequential(
            *(
                [
                    nn.ConvTranspose1d(
                        in_dim,
                        out_dim,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                    )
                ]
                + ([nn.ReLU()] if add_relu else [])
            )
        )

        # return the new block
        return new_block

    def forward(self, x):
        return self.model(x)
