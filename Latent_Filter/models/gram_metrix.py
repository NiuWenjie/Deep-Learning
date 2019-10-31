######################### PyTorch version ############################
# Given an activated filter maps of any particular layer, return its respected gram matrix
class convert_to_gram(nn.Module):
    def __init__(self):
        super(convert_to_gram, self).__init__()

    def forward(self, feature_map):
        # Get the dimensions of the filter maps to reshape them into two dimenions
        dimension = [*feature_map.size()]
        reshaped_maps = torch.reshape(feature_map,[dimension[1] * dimension[2], dimension[3]])

        # Compute the inner product to get the gram matrix
        if dimension[1] * dimension[2] > dimension[3]:
            return torch.mm(reshaped_maps.t(),reshaped_maps)
        else:
            return torch.mm(reshaped_maps, reshaped_maps.t())



######################## Tensorflow version ########################
def convert_to_gram(filter_maps):
    # Get the dimensions of the filter maps to reshape them into two dimenions
    dimension = filter_maps.get_shape().as_list()
    reshaped_maps = tf.reshape(filter_maps, [dimension[1] * dimension[2], dimension[3]])

    # Compute the inner product to get the gram matrix
    if dimension[1] * dimension[2] > dimension[3]:
        return tf.matmul(reshaped_maps, reshaped_maps, transpose_a=True)
    else:
        return tf.matmul(reshaped_maps, reshaped_maps, transpose_b=True)
