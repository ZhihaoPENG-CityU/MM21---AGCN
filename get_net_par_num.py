
def num_net_parameter(net):
    
    all_num = sum(i.numel() for i in net.parameters())
    grad_num = sum(j.numel() for j in net.parameters() if j.requires_grad)
            
    print ('[The network parameters]', all_num , '[para_requires_grad]', grad_num)