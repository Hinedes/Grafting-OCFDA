import torch


def csr_to_dense(values, row_offsets, col_indices, rows, cols):
    row_counts = torch.diff(row_offsets.to(torch.int64))
    row_indices = torch.repeat_interleave(
        torch.arange(rows, device=col_indices.device, dtype=torch.int64),
        row_counts,
    )
    indices = torch.stack([row_indices, col_indices.to(torch.int64)])
    return torch.sparse_coo_tensor(
        indices,
        values,
        size=(rows, cols),
        dtype=values.dtype,
        device=values.device,
    ).to_dense()


def dense_to_csr(dense):
    row_indices, col_indices = dense.nonzero(as_tuple=True)
    values = dense[row_indices, col_indices]
    row_counts = torch.bincount(row_indices, minlength=dense.shape[0])
    row_offsets = torch.empty(dense.shape[0] + 1, dtype=torch.int32, device=dense.device)
    row_offsets[0] = 0
    row_offsets[1:] = torch.cumsum(row_counts, dim=0).to(torch.int32)
    return values, row_offsets, col_indices.to(torch.int16)


def spmm(values, row_offsets, row_indices, col_indices, rhs, rows):
    del row_indices
    sparse = csr_to_dense(values, row_offsets, col_indices, rows, rhs.shape[0])
    return sparse.to(rhs.dtype).matmul(rhs)


def sddmm(row_offsets, row_indices, col_indices, lhs, rhs):
    del row_indices
    dense_product = lhs.matmul(rhs.T)
    row_counts = torch.diff(row_offsets.to(torch.int64))
    rows = torch.repeat_interleave(
        torch.arange(row_offsets.numel() - 1, device=col_indices.device, dtype=torch.int64),
        row_counts,
    )
    return dense_product[rows, col_indices.to(torch.int64)]


def csr_transpose(values, row_offsets, col_indices, rows, cols):
    dense = csr_to_dense(values, row_offsets, col_indices, rows, cols)
    return dense_to_csr(dense.T.contiguous())


def csr_add(values, row_offsets, row_indices, col_indices, dense):
    del row_indices
    sparse = csr_to_dense(values, row_offsets, col_indices, dense.shape[0], dense.shape[1])
    return dense + sparse.to(dense.dtype)
