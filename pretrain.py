from torch import nn
from torch import optim

import params
from utils import make_variable, save_model

def train_src( source_encoder, source_classifier, data_loader ):

    source_encoder.train()
    source_classifier.train()

    optimizer = optim.Adam(
                            list(source_encoder.parameters()) + list(source_classifier.parameters()))#,
                            # lr = params.c_learning_rate,
                            # betas = ( params.beta1, params.beta2 )
                            # )
    
    criterion = nn.CrossEntropyLoss()

    for epoch in range( params.num_epochs_pre ):

        for step, ( images, lables ) in enumerate( data_loader ):

            images = make_variable(images)
            lables = make_variable( lables.squeeze_() )

            optimizer.zero_grad()

            preds = source_classifier( source_encoder( images ) )

            loss = criterion( preds, lables )

            loss.backward()
            optimizer.step()

            # print step info
            if ((step + 1) % params.log_step_pre == 0):
                print("Epoch [{}/{}] Step [{}/{}]: loss={}"
                      .format(epoch + 1,
                              params.num_epochs_pre,
                              step + 1,
                              len(data_loader),
                              loss.data.item()))

        # eval model on test set
        if ((epoch + 1) % params.eval_step_pre == 0):
            eval_src(source_encoder, source_classifier, data_loader)

        # save model parameters
        if ((epoch + 1) % params.save_step_pre == 0):
            save_model(source_encoder, "ADDA-source-encoder-{}.pt".format(epoch + 1))
            save_model(
                source_classifier, "ADDA-source-classifier-{}.pt".format(epoch + 1))

    # # save final model
    save_model(source_encoder, "ADDA-source-encoder-final.pt")
    save_model(source_classifier, "ADDA-source-classifier-final.pt")

    return source_encoder, source_classifier



def eval_src( source_encoder, source_classifier, data_loader ):

    loss = 0
    accuracy = 0 

    source_encoder.eval()
    source_classifier.eval()

    criterion = nn.CrossEntropyLoss()

    for (images, labels) in data_loader:
        images = make_variable( images, volatile = True )
        labels = make_variable( labels )

        preds = source_classifier( source_encoder( images ) )
        loss += criterion( preds, labels ).item()

        pred_cls = preds.data.max(1)[1]
        # print(pred_cls.eq(labels.data).cpu().sum())
        accuracy += pred_cls.eq(labels.data).cpu().sum() / len(labels)

    
    loss /= len(data_loader)
    accuracy /= len( data_loader )

    print("Avg Loss = {}, Avg Accuracy = {:2%}".format(loss, accuracy))

