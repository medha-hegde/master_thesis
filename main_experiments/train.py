# Imports
import torch
import numpy as np
from torch.optim import Adam

from tqdm.notebook import trange
from torch.optim.lr_scheduler import LambdaLR

import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import wandb

from main_experiments.sampler import Euler_Maruyama_sampler
from main_experiments.unet_setup import marginal_prob_std_fn, loss_fn_cond, diffusion_coeff_fn


def sample_text_caption(text_prompt, tokenizer, text_encoder, textmodel_maxtokens):
    sample_text = text_prompt + " \"%s\"" % "HELLO"
    sample_text_y = [str(sample_text)]
    sample_inputs = tokenizer(sample_text_y, max_length=textmodel_maxtokens, padding="max_length",
                              truncation=True, return_tensors="pt")
    with torch.no_grad():
        sample_encoder_hidden_states = text_encoder(sample_inputs.input_ids.cuda())[0]
        sample_encoder_hidden_states = sample_encoder_hidden_states.cpu()

    return sample_encoder_hidden_states

# Training function
def train_diffusion_model(config,
                          dataloader,
                          score_model,
                          tokenizer,
                          text_encoder,
                          tokenizer_2=None,
                          text_encoder_2=None,
                          n_epochs=100,
                          lr=10e-4,
                          model_name="transformer",
                          model_save_path=None,
                          checkpoint=None,
                          dummy_run=False,
                          combined_embedding=False
                          ):

    device = config["device"]
    # optimizer
    start_epoch = 0  # last checkpoint epoch for saving total no. of epochs run
    optimizer = Adam(score_model.parameters(), lr=lr)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: max(0.2, 0.98 ** epoch))

    if checkpoint is not None:
        score_model.load_state_dict(checkpoint['model'])
        score_model.train()
        print("loading UNet model..")

        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        print("loading optimizer..")

        scheduler.load_state_dict(checkpoint['scheduler'])
        print("loading scheduler..")

    tqdm_epoch = trange(n_epochs)

    sample_encoder_hidden_states = sample_text_caption(config["sample_text_prompt"], tokenizer, text_encoder,
                                                       config["textmodel_maxtokens"])
    if combined_embedding is not None:
        sample_encoder_hidden_states_2 = sample_text_caption(config["sample_text_prompt"], tokenizer_2, text_encoder_2,
                                                           config["textmodel_maxtokens"])
        sample_encoder_hidden_states = torch.cat((sample_encoder_hidden_states, sample_encoder_hidden_states_2),
                                                        dim=2)

    for epoch in tqdm_epoch:
        avg_loss = 0.
        num_items = 0
        for step, batch in enumerate(dataloader):

            if dummy_run:
                if step > 0:
                    break

            x = batch['pixel_values'].to(device)

            encoder_hidden_states = text_encoder(batch["input_ids"].cuda())[0]
            encoder_hidden_states = encoder_hidden_states.cpu()
            if combined_embedding is not None:
                encoder_hidden_states_2 = text_encoder_2(batch["input_ids_2"].cuda())[0]
                encoder_hidden_states_2 = encoder_hidden_states_2.cpu()
                encoder_hidden_states = torch.cat((encoder_hidden_states, encoder_hidden_states_2), dim=2)

            loss = loss_fn_cond(score_model, x, encoder_hidden_states, marginal_prob_std_fn)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]

        score_model.eval()

        sample_samples = Euler_Maruyama_sampler(score_model,
                                                marginal_prob_std_fn,
                                                diffusion_coeff_fn,
                                                config["sample_batch_size"],
                                                num_steps=250,
                                                device=device,
                                                y=sample_encoder_hidden_states)

        score_model.train()

        # Sample visualization
        sample_samples = sample_samples.clamp(0.0, 1.0)
        sample_grid = make_grid(sample_samples, nrow=int(np.sqrt(config['sample_batch_size'])))

        plt.figure(figsize=(6, 6))
        plt.axis('off')
        plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
        plt.show()

        scheduler.step()
        lr_current = scheduler.get_last_lr()[0]
        print('{} Average Loss: {:5f} lr {:.1e}'.format(start_epoch + epoch, avg_loss / num_items, lr_current))

        # Print the averaged training loss so far.
        tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))

        # Update the checkpoint after each epoch of training.
        state = {'epoch': start_epoch + epoch + 1,
                 'model': score_model.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'loss': avg_loss / num_items,
                 'scheduler': scheduler.state_dict()}

        wandb.log({'epoch': start_epoch + epoch + 1, 'loss': avg_loss / num_items, 'lr': lr_current})
        torch.save(state, f'{model_save_path}ckpt_{model_name}.pt')
