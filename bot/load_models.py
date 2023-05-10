import torch

def load_models(generator_template, path_gen, only_gen=True, 
                discriminator_template=None, path_disc=None):
    # Do not switch generator to eval mode
    generator = generator_template()
    generator.load_state_dict(torch.load(path_gen, map_location=torch.device('cpu')))
    for param in generator.parameters():
        param.requires_grad = False

    if not only_gen:
        discriminator = discriminator_template()
        discriminator.load_state_dict(torch.load(path_disc, map_location=torch.device('cpu')))
        discriminator.eval()
        for param in discriminator.parameters():
            param.requires_grad = False
        return generator, discriminator
    
    return generator