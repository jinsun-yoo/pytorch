import torch


@torch.library.register_fake("_c10d_functional::recv")
def recv_abstract(input_list, src, tag, group_name):
    return [torch.empty_like(tensor) for tensor in input_list]

@torch.library.register_fake("_c10d_functional::send")
def recv_abstract(input_list, dst, tag, group_name):
    return [torch.empty_like(tensor) for tensor in input_list]