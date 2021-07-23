import models
import util
import torch


p = models.position_encoding.PositionEmbeddingSine(9, normalize=True)
a = torch.zeros(2, 5, 9)
m = util.misc.nested_tensor_from_tensor_list([a])
encode = p(m)
print(p(m).size())
