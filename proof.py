import torch
from over9000 import Over9000


def Loss(net, batch):
    return (net(batch[0]) - batch[1]).sum() ** 2


def Check(load_optimizer_state_dict=True):
    """
    This function verifies if loading the state_dict of the LookAHead
    optimizer allows a neural net to continue training starting
    from a given checkpoint.

    The test is design as following: there are two batches, two neural nets and
    two LookAHead optimizers. The net2 will start training from a checkpoint
    of net1 after the backward pass of the first batch, net2 will only train on
    the second batch. At the and, net2 and net1 must have the same weights.

        net1(batch1) -> checkpoint -> net1(batch2) -> final1
                            |                           ||
                            v                           ||
                          net2     -> net2(batch2) -> final2

    This check must fail if the we don't load the optimizer state_dict.
    """
    net1 = torch.nn.Linear(2, 1)
    opt1 = Over9000(net1.parameters())

    net2 = torch.nn.Linear(2, 1)
    opt2 = Over9000(net2.parameters())

    batch1 = torch.rand(16, 2), torch.rand(16)
    batch2 = torch.rand(16, 2), torch.rand(16)

    loss = Loss(net1, batch1)
    loss.backward()
    opt1.step()

    # This is where the checkpoint is made
    net2.load_state_dict(net1.state_dict())
    if load_optimizer_state_dict:
        opt2.load_state_dict(opt1.state_dict())

    loss = Loss(net1, batch2)

    opt1.zero_grad()
    loss.backward()
    opt1.step()

    loss = Loss(net2, batch2)
    loss.backward()
    opt2.step()

    return net1.weight.equal(net2.weight) and net1.bias.equal(net2.bias)


print("Loading LookAHead optimizer state_dict, same nets ?", Check(True))
print("NOT Loading LookAHead optimizer state_dict, same nets ?", Check(False))
