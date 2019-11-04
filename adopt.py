import os

import torch 
from torch import nn, optim

import params
from utils import make_variable


def train_target(source_encoder, target_encoder, critic,
                 source_dataloader, target_dataloader):
    
    target_encoder.train()
    critic.train()
    source_encoder.eval()

    criterion = nn.CrossEntropyLoss()
    optimizer_target = optim.Adam(target_encoder.parameters() ,
                               lr=params.c_learning_rate,
                               betas=(params.beta1, params.beta2))
    optimizer_critic = optim.Adam(critic.parameters() ,
                                  lr=params.d_learning_rate,
                                  betas=(params.beta1, params.beta2))
    len_data_loader = min(len(source_dataloader), len(target_dataloader))
    data_zip = enumerate( zip( source_dataloader, target_dataloader ) )

    for epoch in range(params.num_epochs):
        # print("................................here...............")        
        for step, ( (source_image, _), (target_image, _) ) in data_zip:
            source_image = make_variable( source_image )
            target_image = make_variable( target_image )

            optimizer_critic.zero_grad()
            
            source_feature = source_encoder( source_image )
            target_feature = target_encoder( target_image )
            concat_feature = torch.cat( ( source_feature, target_feature ), 0 )

            pred_concat = make_variable( critic( concat_feature ) )

            source_label = make_variable( torch.ones( source_feature.size(0) ).long() )
            target_label = make_variable( torch.zeros( target_feature.size(0) ).long() )
            concat_label = torch.cat( ( source_label, target_label ) )

            loss_critic = criterion( concat_feature, concat_label )
            loss_critic.backward()

            optimizer_critic.step()

            pred_cls = torch.squeeze(pred_concat.max(1)[1])
            acc = (pred_cls == concat_label).float().mean()

            optimizer_critic.zero_grad()
            optimizer_target.zero_grad()

            target_feature = target_encoder( target_image )

            pred_target = critic( target_feature )

            label_target = make_variable( torch.ones( pred_target.size(0) ).long() )

            loss_target = criterion( pred_target, label_target )
            loss_target.backward()

            optimizer_target.step()

            # if ((step + 1) % params.log_step == 0):
            print("Epoch [{}/{}] Step [{}/{}]:"
                    "d_loss={:.5f} g_loss={:.5f} acc={:.5f}"
                    .format(epoch + 1,
                            params.num_epochs,
                            step + 1,
                            len_data_loader,
                            loss_critic.item(),
                            loss_target.item(),
                            acc.item()))

        #############################
        # 2.4 save model parameters #
        #############################
        if ((epoch + 1) % params.save_step == 0):
            torch.save(critic.state_dict(), os.path.join(
                params.model_root,
                "ADDA-critic-{}.pt".format(epoch + 1)))
            torch.save(target_encoder.state_dict(), os.path.join(
                params.model_root,
                "ADDA-target-encoder-{}.pt".format(epoch + 1)))

    torch.save(critic.state_dict(), os.path.join(
        params.model_root,
        "ADDA-critic-final.pt"))
    torch.save(target_encoder.state_dict(), os.path.join(
        params.model_root,
        "ADDA-target-encoder-final.pt"))
    return target_encoder

