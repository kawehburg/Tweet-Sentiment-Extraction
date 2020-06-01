import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def _get_score(start, end, max_len):
    start_score = []
    start = start.item()
    end = end.item()
    for i in range(max_len):
        start_score.append(min(end + 1 - start, max(end + 1 - i, 0)) / max(end + 1 - start, end + 1 - i))
    end_score = []
    for j in range(max_len):
        end_score.append(min(end + 1 - start, max(j + 1 - start, 0)) / max(end + 1 - start, j + 1 - start))
    return start_score, end_score


def get_jcd_score(start_positions, end_positions):
    bc_size = start_positions.size(0)
    start_score = []
    end_score = []
    for i in range(bc_size):
        start, end = _get_score(start_positions[i], end_positions[i], 200)
        start_score.append(start)
        end_score.append(end)
    return start_score, end_score


def distance_loss(start_logits, end_logits, start_positions, end_positions, temperature=0):
    start_score, end_score = get_jcd_score(start_positions, end_positions)
    start_score = torch.tensor(start_score).to(start_logits.device)
    end_score = torch.tensor(end_score).to(end_logits.device)
    start_loss = -(F.log_softmax(start_logits, dim=-1) * start_score[:, :start_logits.size(1)]).mean()
    end_loss = -(F.log_softmax(end_logits, dim=-1) * end_score[:, :end_logits.size(1)]).mean()
    total_loss = (start_loss + end_loss) / 2
    return total_loss


print(
    distance_loss(torch.rand((4, 192)), torch.rand((4, 192)), torch.tensor([2, 5, 7, 3]), torch.tensor([8, 6, 10, 19])))

