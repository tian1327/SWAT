from utils.models import build_classifier_head, save_model_ckpt
from testing import validate, calculate_scores
import copy
import torch

def train_flyp(args, logger, loss_logger, model, tokenizer,
               train_dataloader, val_dataloader, test_dataloader, text_prompts):
    """ 
    Finetune like you pretrain
    Train the model with contrastive loss, using the text descriptions from labels.
    Can be modified to lock the text encoder.
    """

    assert (args.loss_name == 'CE' or args.loss_name == 'WeightedCE'), 'FLYP use CE loss for contrastive loss calculation.'


    logit_scale = args.logit_scale
    loss = args.loss
    optimizer = args.optimizer
    scheduler = args.scheduler

    val_loss = -1
    val_acc = -1
    test_acc = -1
    best_val_acc = -1
    best_epoch = -1
    best_iter = -1
    best_model = model
    best_head = None
    best_logit_scale = logit_scale
    best_records = {}
    num_iter = 0
    # if args.model_path:
    #     model, _ = load_model(args, logger, model, test_dataloader, logit_scale, classifier_head)

    logger.info(f"Start Training FLYP ......")

    model.train()
    for epoch in range(1, args.epochs+1):
        
        train_loss_sum = 0
        # train for 1 epoch
        for inputs, labels, tokenized_text, source in train_dataloader:
            optimizer.zero_grad()            
            num_iter += 1
            images = inputs.to(args.device)
            labels = labels.to(args.device).long()  
            tokenized_text = tokenized_text.to(args.device) # currently we use 1 template for semi-aves as in prompt_maker(), can be updated to randomly sample 1 from the 80 prompts

            image_features = model.encode_image(images)
            image_feature = image_features / image_features.norm(dim=-1, keepdim=True)

            prompts = tokenized_text.squeeze()
            text_features = model.encode_text(prompts)
            text_feature = text_features / text_features.norm(dim=-1, keepdim=True) # Normalization
            
            scale = logit_scale.exp()            
            logits_per_image = scale * image_feature @ text_feature.t()
            logits_per_text = logits_per_image.t()
            labels = torch.arange(logits_per_image.shape[0], dtype=torch.long).to(args.device)
            
            if args.loss_name == 'CE':
                total_loss = (loss(logits_per_image, labels) + loss(logits_per_text, labels)) / 2
            elif args.loss_name == 'WeightedCE':
                total_loss = (loss(logits_per_image, labels, source) + loss(logits_per_text, labels, source)) / 2
            else:
                raise ValueError(f'Loss {args.loss_name} not supported for FLYP training.')
            
            # total_loss = (loss(logits_per_image, labels) + loss(logits_per_text, labels)) / 2
            train_loss = total_loss.item()
            train_loss_sum += train_loss             
                
            total_loss.backward()
            optimizer.step()
            scheduler.step() # update learning rate for each iteration
        
        # validate after 1 epoch
        # rebuild the classifier head using the updated text encoder
        if args.early_stop or epoch == args.epochs:   
            new_head = build_classifier_head(args, model, text_prompts, tokenizer)               
            val_acc, val_loss, confusion_matrix = validate(args,data_loader=val_dataloader, model=model, logger=logger, 
                                                            loss=loss, logit_scale=logit_scale,
                                                            classifier_head=new_head, show_confusion_matrix=True,
                                                            dataset=args.dataset, 
                                                            output_dir=args.output_dir, device=args.device,
                                                            pre_extracted=False, 
                                                            )
            scores = calculate_scores(confusion_matrix)        
                            
        # check if val acc has improved, here i use the val_acc rather than val_loss,
        # as we might modify the loss function for confusing pairs
        if (args.early_stop or epoch == args.epochs) and val_acc >= best_val_acc:
            # logger.info(f'Val Acc improved from {round(best_val_acc, 3)} to {round(val_acc, 3)}')
            best_val_acc = val_acc
            best_logit_scale = logit_scale
            best_epoch = epoch
            best_iter = num_iter
            best_scores = copy.deepcopy(scores)
            best_confusion_matrix = copy.deepcopy(confusion_matrix)
            best_head = copy.deepcopy(new_head)
            best_model = copy.deepcopy(model)

            # save into the best_records
            best_records['best_val_acc'] = best_val_acc
            best_records['best_logit_scale'] = best_logit_scale
            best_records['best_epoch'] = best_epoch
            best_records['best_iter'] = best_iter
            best_records['best_scores'] = best_scores
            best_records['best_confusion_matrix'] = best_confusion_matrix

        # test after 1 epoch
        if args.early_stop or epoch == args.epochs:
            test_acc, _, _ = validate(args,data_loader=test_dataloader, model=model, logger=logger, 
                                loss=loss, logit_scale=logit_scale,
                                classifier_head=new_head,
                                dataset=args.dataset, 
                                output_dir=args.output_dir, device=args.device,
                                pre_extracted=False, 
                                )        

        train_loss_avg = train_loss_sum / len(train_dataloader)
        loss_logger.write(f'{epoch},{num_iter},{round(train_loss_avg, 6)},{round(val_loss, 6)},{round(val_acc, 6)}, {round(test_acc, 6)}\n')
        loss_logger.flush()
        logger.info(f"Epoch {int(epoch)}, Iter {num_iter}, Trn Loss: {round(train_loss_avg, 6)}, Val Loss: {round(val_loss, 6)}, Val Acc: {round(val_acc, 3)}, Test Acc: {round(test_acc, 3)}")          
        
        # sometime train_loss goes to nan due to numerical instability, 
        # here to log the logit_scale and stops the training
        if test_acc == 0.5:
            logger.info(f'logit_scale: {logit_scale.item()}, scale: {scale.item()}')
            logger.info('Test Acc is 0.5, stop training.')
            exit()           
        
        ## save model checkpoints every X epochs
        if args.save_ckpt and (num_iter % args.save_freq == 0 or epoch == args.epochs):            
            model_path = save_model_ckpt(args, best_records,
                                        model, new_head, optimizer, scheduler, logit_scale,
                                         val_acc, epoch, num_iter)
            logger.info(f'Model ckpt saved to: {model_path}')      
    
    logger.info(f'Training done.')
    logger.info(f'Best val Acc: {round(best_val_acc, 3)} at epoch {best_epoch}, iter {best_iter}')

    return best_model, best_head, best_records, best_logit_scale 


def train_supervised_contrastive(args, logger, loss_logger, model, classifier_head,
                                logit_scale, loss, optimizer, scheduler,
                                train_dataloader, val_dataloader, test_dataloader):
    """ train CLIP visual encoder with supervised contrastive loss, then linear prob to evaluate learned representations """
    assert args.loss == 'SupCon' or args.loss == 'FASupCon',  \
        'Supervised Contrastive Loss is used for training.'
    
    