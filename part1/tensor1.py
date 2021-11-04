import torch

# Tensor 操作
if __name__ == '__main__':

    # 创建tensor

    x = torch.empty(5,3)
    print(x)

    print(" ")

    x = torch.rand(5,3)
    print(x)

    print("")

    x = torch.zeros(5, 3, dtype=torch.long)
    print(x)

    print("")

    x = torch.tensor([5.5,3])
    print(x)

    print("")

    x = x.new_ones(5, 3, dtype=torch.float64) # 返回的tensor默认具有相同的torch.dtype和torch.device
    print(x)

    x = torch.randn_like(x, dtype=torch.float) # 指定新的数据类型
    print(x)

    print("")

    print(x.size())
    print(x.shape)

    print("")

    # 算数操作
    y = torch.rand(5, 3)
    print(x)
    print(y)
    print(x + y)

    print("")

    print(torch.add(x, y))

    print("")

    result = torch.empty(5,3)
    torch.add(x, y, out=result)
    print(result)

    print("")

    y.add_(x)
    print(y)

    print()

    # 索引
    print(y)
    print(x)
    y = x[0, :]
    y += 1
    print(y)
    print(x[0, :]) # 源tensor也被改了

    print()

    # 改变形状
    y = x.view(15)
    print(y)
    print(x)
    z = x.view(-1, 5) # -1所指的维度可以根据其他维度的值推出来
    print(z)
    print(x.size(), y.size(), z.size())

    print()

    x += 1
    print(x)
    print(y) # y也加了1

    print()

    x_cp = x.clone().view(15) # 深拷贝
    x -= 1
    print(x)
    print(x_cp)

    print()

    x = torch.randn(1)
    print(x)
    print(x.item())

    # 线性代数