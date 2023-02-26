
import torch
from torchvision.utils import make_grid

def training(generator, discriminator, device, num_epochs, dataloader, criterion, gen_opt, disc_opt):
    G_losses = []
    D_losses = []
    img_list = []
    iters = 0
    fixed_noise = torch.randn(64, 100, 1, 1).to(device)
    for epoch in range(num_epochs):
        print(f"Current on Epoch {epoch + 1}")
        for i, data in enumerate(dataloader, ):
            # Train discriminator
            discriminator.zero_grad()
            real_batch = data[0].to(device)
            b_size = real_batch.size(0)
            label = torch.full((b_size,), 1, dtype=torch.float32, device=device)
            output = discriminator(real_batch).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()
            noise = torch.randn(b_size, 100, 1, 1, device=device)
            fake = generator(noise)
            label.fill_(0)
            output = discriminator(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            disc_opt.step()

            # Train Generator
            generator.zero_grad()
            label.fill_(1)
            output = discriminator(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            gen_opt.step()

            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, num_epochs, i, len(dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            if (iters % 250 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = generator(fixed_noise).detach().cpu()
                img_list.append(make_grid(fake, padding=2, normalize=True))
            iters += 1
    return G_losses, D_losses, img_list