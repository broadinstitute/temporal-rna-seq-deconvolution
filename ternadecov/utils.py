import pandas as pd
import torch


def melt_tensor_to_pandas(
    input_tensor: torch.Tensor, dimnames, *dimlabels
) -> pd.DataFrame:
    """Like pandas.melt() but for torch tensors.
    Creates a long form table with values and their indices"""

    input_tensor_dim = input_tensor.dim()
    input_shape = input_tensor.shape

    # Check dimlabels have been provided for all dimensions
    assert len(dimlabels) == input_tensor_dim

    # check dim labels same size as tensor dims
    for i in range(0, len(dimlabels) - 1):
        assert len(dimlabels[i]) == input_shape[i]

    # prepare categorical columns
    output_label_idx = []
    for n in range(input_tensor_dim):
        e = torch.prod(torch.tensor(input_shape)[(n + 1) :])
        a = torch.prod(torch.tensor(input_shape)[:(n)])
        if n == 0:
            a = 1
        elif n == len(input_shape) - 1:
            e = 1
        output_label_idx.append(
            torch.tile(torch.arange(len(dimlabels[n])).repeat_interleave(e), (a,))
        )

    # Generate output dataframe
    output = {}
    for j in range(len(output_label_idx)):
        output[dimnames[j]] = pd.Categorical(
            output_label_idx[j].numpy()
        ).rename_categories(dimlabels[j])
    output["values"] = input_tensor.flatten().numpy()

    return pd.DataFrame(output)
