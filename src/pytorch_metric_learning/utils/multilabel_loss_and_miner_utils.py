import torch
from . import loss_and_miner_utils as lmu

def check_shapes_multilabels(embeddings, labels):
    if labels is not None and embeddings.shape[0] != len(labels):
        raise ValueError("Number of embeddings must equal number of labels")
    if labels is not None:
        if isinstance(labels[0], list) or isinstance(labels[0], torch.Tensor):
            pass
        else:
            raise ValueError("labels must be a list of 1d tensors or a list of lists")

def set_ref_emb(embeddings, labels, ref_emb, ref_labels):
    if ref_emb is None:
        ref_emb, ref_labels = embeddings, labels
    check_shapes_multilabels(ref_emb, ref_labels)
    return ref_emb, ref_labels

def convert_to_pairs(indices_tuple, labels, num_classes, ref_labels=None, device=None, threshold=0.3):
    """
    This returns anchor-positive and anchor-negative indices,
    regardless of what the input indices_tuple is
    Args:
        indices_tuple: tuple of tensors. Each tensor is 1d and specifies indices
                        within a batch
        labels: a tensor which has the label for each element in a batch
    """
    if indices_tuple is None:
        return get_all_pairs_indices(labels, num_classes, ref_labels, device=device, threshold=threshold)
    elif len(indices_tuple) == 4:
        return indices_tuple
    else:
        a, p, n = indices_tuple
        return a, p, a, n
    
def get_matches_and_diffs(labels, num_classes, ref_labels=None, device=None, threshold=0.3):
    jaccard_matrix = jaccard(num_classes, labels, ref_labels, device=device, threshold=threshold)
    matches = torch.where(jaccard_matrix > threshold, 1, 0).to(device)
    diffs = matches ^ 1
    if ref_labels is labels:
        matches.fill_diagonal_(0)
    return matches, diffs, jaccard_matrix


def get_all_pairs_indices(labels, num_classes, ref_labels=None, device=None, threshold=0.3):
    """
    Given a tensor of labels, this will return 4 tensors.
    The first 2 tensors are the indices which form all positive pairs
    The second 2 tensors are the indices which form all negative pairs
    """
    matches, diffs, multi_val = get_matches_and_diffs(labels, num_classes, ref_labels, device, threshold=threshold)
    a1_idx, p_idx = torch.where(matches)
    a2_idx, n_idx = torch.where(diffs)
    return a1_idx, p_idx, a2_idx, n_idx, multi_val

def jaccard(n_classes, labels, ref_labels=None, threshold=0.3, device=torch.device("cpu")):
    if ref_labels is None:
        ref_labels = labels
        
    labels1 = labels.float()
    labels2 = ref_labels.float()

    # compute jaccard similarity
    # jaccard = intersection / union 
    labels1_union = labels1.sum(-1)
    labels2_union = labels2.sum(-1)
    union = labels1_union.unsqueeze(1) + labels2_union.unsqueeze(0)
    intersection = torch.mm(labels1, labels2.T)
    jaccard_matrix = intersection / (union - intersection)
    
    # return indices of jaccard similarity above threshold
    return jaccard_matrix

def convert_to_triplets(indices_tuple, labels, ref_labels=None, t_per_anchor=100):
    """
    This returns anchor-positive-negative triplets
    regardless of what the input indices_tuple is
    """
    if indices_tuple is None:
        if t_per_anchor == "all":
            return get_all_triplets_indices(labels, ref_labels)
        else:
            return lmu.get_random_triplet_indices(
                labels, ref_labels, t_per_anchor=t_per_anchor
            )
    elif len(indices_tuple) == 3:
        return indices_tuple
    else:
        a1, p, a2, n = indices_tuple
        p_idx, n_idx = torch.where(a1.unsqueeze(1) == a2)
        return a1[p_idx], p[p_idx], n[n_idx]
    

def get_all_triplets_indices(labels, ref_labels=None):
    matches, diffs = get_matches_and_diffs(labels, ref_labels)
    triplets = matches.unsqueeze(2) * diffs.unsqueeze(1)
    return torch.where(triplets)


def remove_self_comparisons(
    indices_tuple, curr_batch_idx, ref_size, ref_is_subset=False
):
    # remove self-comparisons
    assert len(indices_tuple) in [4, 5]
    s, e = curr_batch_idx[0], curr_batch_idx[-1]
    if len(indices_tuple) == 4:
        a, p, n = indices_tuple
        keep_mask = lmu.not_self_comparisons(
            a, p, s, e, curr_batch_idx, ref_size, ref_is_subset
        )
        a = a[keep_mask]
        p = p[keep_mask]
        n = n[keep_mask]
        assert len(a) == len(p) == len(n)
        return a, p, n
    elif len(indices_tuple) == 5:
        a1, p, a2, n = indices_tuple
        keep_mask = lmu.not_self_comparisons(
            a1, p, s, e, curr_batch_idx, ref_size, ref_is_subset
        )
        a1 = a1[keep_mask]
        p = p[keep_mask]
        assert len(a1) == len(p)
        assert len(a2) == len(n)
        return a1, p, a2, n

