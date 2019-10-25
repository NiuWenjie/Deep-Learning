z_linear = z.view(z.size(0), -1)  # [1, 512*7*7]
z_ = z_linear.clone()
mean = torch.mean(z_[z_!=0]) # 求非0均值
# mean = torch.mean(z_, dim=1, keepdim=True) # 求所有元素均值
z_[z_ < mean] = 0 # 重点操作，将z_中小于均值的元素置0
z_threshold = z_.view(z.size(0), z.size(1), z.size(2), z.size(3))  # [1, 512, 7, 7]
