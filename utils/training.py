import random
import torch
import numpy as np
import copy
from .models import build_classifier_head, save_model_ckpt, save_test_scores
from .dataloader import extract_dataloader, extract_train_dataloader
from testing import validate, calculate_scores, validate_dataset, load_model
import time

def set_training_seed(args):

    # set the seed for training
    random.seed(args.training_seed)
    torch.manual_seed(args.training_seed)
    np.random.seed(args.training_seed)
    torch.cuda.manual_seed_all(args.training_seed)

    # this is critical for reproducibility for ResNet50 models
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_zeroshot(args, test_loader, model, logger, loss, logit_scale, classifier_head):
    if args.method == 'dataset-cls':
        zs_test_acc, zs_loss, zs_confusion_matrix = validate_dataset(args,data_loader=test_loader, model=model, logger=logger,
                                                            loss=loss, logit_scale=logit_scale,
                                                            classifier_head=classifier_head, show_confusion_matrix=True,
                                                            dataset=args.dataset,
                                                            output_dir=args.output_dir, device=args.device,
                                                            pre_extracted=args.pre_extracted,
                                                            )
    else:
        zs_test_acc, zs_loss, zs_confusion_matrix = validate(args,data_loader=test_loader, model=model, logger=logger,
                                                            loss=loss, logit_scale=logit_scale,
                                                            classifier_head=classifier_head, show_confusion_matrix=True,
                                                            dataset=args.dataset,
                                                            output_dir=args.output_dir, device=args.device,
                                                            pre_extracted=args.pre_extracted,
                                                            )
    logger.info(f"+++++ Zero-shot Test Acc: {round(zs_test_acc, 3)}")
    # zs_scores = calculate_scores(zs_confusion_matrix)
    # save_test_scores(zs_scores, zs_confusion_matrix, output_dir, 'zeroshot_test')

    return zs_test_acc


def train_probing(args, logger, loss_logger, model, classifier_head,
                  tokenized_text_prompts, preprocess,
                  train_loader, val_loader, test_loader, reload_model=False):
    """ Train the model with Cross-Entropy Loss, linear probing"""

    if reload_model:
        # load_model(args, logger, model, test_loader, classifier_head)
        load_model(args, logger, model, None, classifier_head)


        # Here we reextract the test dataloader for fast testing after training, tau normalization, and WiSE-FT
        new_train_fea_path = f'{args.dataset_root}/pre_extracted/{args.dataset}_{args.model_cfg}_{args.seed}_train_features_new.pth'
        new_val_fea_path = f'{args.dataset_root}/pre_extracted/{args.dataset}_{args.model_cfg}_{args.seed}_val_features_new.pth'
        new_test_fea_path = f'{args.dataset_root}/pre_extracted/{args.dataset}_{args.model_cfg}_{args.seed}_test_features_new.pth'

        train_loader = extract_train_dataloader(args, model, args.train_split, new_train_fea_path,
                                                preprocess, tokenized_text_prompts, bsz=args.bsz)
        val_loader = extract_dataloader(args, model, args.val_split, new_val_fea_path, preprocess, tokenized_text_prompts)
        test_loader = extract_dataloader(args, model, args.test_split, new_test_fea_path, preprocess, tokenized_text_prompts)
        logger.info(f'Extracted train, val, test dataloader for probing.')
        # reset the pre_extracted flag
        args.pre_extracted = True
        logger.info(f'Reset args.pre_extracted: {args.pre_extracted}')
        time.sleep(0.5)

    logger.info(f"Start Training (linear probing) ......")

    logit_scale = args.logit_scale
    loss = args.loss
    optimizer = args.optimizer
    scheduler = args.scheduler

    model.eval()
    classifier_head.train()

    best_records = {}
    best_val_acc = -1
    num_iter = 0
    for epoch in range(1, args.epochs+1):

        train_loss_sum = 0
        for inputs, labels, text, source in train_loader:
            num_iter += 1
            images = inputs.to(args.device)
            labels = labels.to(args.device).long()

            if not args.pre_extracted:
                image_features = model.encode_image(images)
                image_feature = image_features / image_features.norm(dim=-1, keepdim=True)
            else:
                image_feature = images

            logits = classifier_head(image_feature)
            logits = logits * logit_scale.exp()
            total_loss = loss(logits, labels)
            train_loss = total_loss.item()
            train_loss_sum += train_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step() # update learning rate for each iteration

        # validate after 1 epoch
        val_acc, val_loss, confusion_matrix = validate(args, data_loader=val_loader, model=model, logger=logger,
                                                        loss=args.loss, logit_scale=logit_scale,
                                                        classifier_head=classifier_head, show_confusion_matrix=True,
                                                        dataset=args.dataset,
                                                        output_dir=args.output_dir, device=args.device,
                                                        pre_extracted=args.pre_extracted,
                                                        )
        scores = calculate_scores(confusion_matrix)

        if (args.early_stop or epoch == args.epochs) and val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_logit_scale = copy.deepcopy(logit_scale)
            best_epoch = epoch
            best_iter = num_iter
            best_scores = copy.deepcopy(scores)
            best_confusion_matrix = copy.deepcopy(confusion_matrix)
            best_head = copy.deepcopy(classifier_head)
            best_model = copy.deepcopy(model)

            # save into the best_records
            best_records['best_val_acc'] = best_val_acc
            best_records['best_logit_scale'] = best_logit_scale
            best_records['best_epoch'] = best_epoch
            best_records['best_iter'] = best_iter
            best_records['best_scores'] = best_scores
            best_records['best_confusion_matrix'] = best_confusion_matrix

        # test after 1 epoch
        test_acc, _, _ = validate(args,data_loader=test_loader, model=model, logger=logger,
                                loss=args.loss, logit_scale=logit_scale,
                                classifier_head=classifier_head,
                                dataset=args.dataset,
                                output_dir=args.output_dir, device=args.device,
                                pre_extracted=args.pre_extracted,
                                )

        train_loss_avg = train_loss_sum / len(train_loader)
        loss_logger.write(f'{epoch},{num_iter},{round(train_loss_avg, 6)},{round(val_loss, 6)},{round(val_acc, 6)},{round(test_acc, 6)}\n')
        loss_logger.flush()
        logger.info(f"Epoch {int(epoch)}, Iter {num_iter}, Trn Loss: {round(train_loss_avg, 6)}, Val Loss: {round(val_loss, 6)}, Val Acc: {round(val_acc, 3)}, Test Acc: {round(test_acc, 3)}")

        ## save model checkpoints every X epochs
        if args.save_ckpt and (num_iter % args.save_freq == 0 or epoch == args.epochs):
            model_path = save_model_ckpt(args, best_records,
                                        model, classifier_head, optimizer, scheduler, logit_scale,
                                         val_acc, epoch, num_iter)
            logger.info(f'Model ckpt saved to: {model_path}')

    logger.info(f'Probing done.')

    return best_model, best_head, best_records, best_logit_scale, val_loader, test_loader


def train_CMLP(args, logger, loss_logger, model, classifier_head, preprocess, \
               tokenized_text_prompts,train_loader, val_loader, test_loader, \
                reload_model=False, text_dataloader=None):
    """ Train the model with Cross-Entropy Loss, linear probing"""

    if reload_model:
        load_model(args, logger, model, test_loader, classifier_head)
        # Here we reextract the test dataloader for fast testing after training, tau normalization, and WiSE-FT
        new_train_fea_path = f'{args.dataset_root}/pre_extracted/{args.dataset}_{args.model_cfg}_{args.seed}_train_features_new.pth'
        new_val_fea_path = f'{args.dataset_root}/pre_extracted/{args.dataset}_{args.model_cfg}_{args.seed}_val_features_new.pth'
        new_test_fea_path = f'{args.dataset_root}/pre_extracted/{args.dataset}_{args.model_cfg}_{args.seed}_test_features_new.pth'

        train_loader = extract_train_dataloader(args, logger, model, args.train_split, new_train_fea_path,
                                                preprocess, tokenized_text_prompts, args.bsz)

        val_loader = extract_dataloader(args, model, args.val_split, new_val_fea_path, preprocess, tokenized_text_prompts)
        test_loader = extract_dataloader(args, model, args.test_split, new_test_fea_path, preprocess, tokenized_text_prompts)
        logger.info(f'Extracted train, val, test dataloader for stage 2 training.')
        # reset the pre_extracted flag
        args.pre_extracted = True
        logger.info(f'Reset args.pre_extracted: {args.pre_extracted}')

    logger.info(f"Start Training (cross-modal linear probing) ......")

    logit_scale = args.logit_scale
    loss = args.loss
    optimizer = args.optimizer
    scheduler = args.scheduler

    assert text_dataloader is not None, 'text_dataloader is None.'
    assert args.pre_extracted, 'args.pre_extracted is False.'

    model.eval()
    classifier_head.train()

    best_records = {}
    best_val_acc = -1
    test_acc = -1
    num_iter = 0
    val_loss = -1
    val_acc = -1

    text_loader = text_dataloader
    text_loader_iter = iter(text_loader)

    for epoch in range(1, args.epochs+1):

        train_loss_sum = 0
        for inputs, labels, text, source in train_loader:
            num_iter += 1
            images = inputs.to(args.device)
            labels = labels.to(args.device).long()

            image_feature = images

            try:
                text, text_label = next(text_loader_iter)
            except StopIteration:
                text_loader_iter = iter(text_loader)
                text, text_label = next(text_loader_iter)

            # concatenate image and text features
            combined_feature = torch.cat([image_feature, text], dim=0)
            combined_labels = torch.cat([labels, text_label], dim=0)

            logits = classifier_head(combined_feature)
            logits = logits * logit_scale.exp()
            total_loss = loss(logits, combined_labels)
            train_loss = total_loss.item()
            train_loss_sum += train_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step() # update learning rate for each iteration

        # validate after 1 epoch
        if args.early_stop or epoch == args.epochs:
            val_acc, val_loss, confusion_matrix = validate(args, data_loader=val_loader, model=model, logger=logger,
                                                        loss=args.loss, logit_scale=logit_scale,
                                                        classifier_head=classifier_head, show_confusion_matrix=True,
                                                        dataset=args.dataset,
                                                        output_dir=args.output_dir, device=args.device,
                                                        pre_extracted=args.pre_extracted,
                                                        )
            scores = calculate_scores(confusion_matrix)

        if (args.early_stop or epoch == args.epochs) and val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_logit_scale = copy.deepcopy(logit_scale)
            best_epoch = epoch
            best_iter = num_iter
            best_scores = copy.deepcopy(scores)
            best_confusion_matrix = copy.deepcopy(confusion_matrix)
            best_head = copy.deepcopy(classifier_head)
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
            test_acc, _, _ = validate(args,data_loader=test_loader, model=model, logger=logger,
                                    loss=args.loss, logit_scale=logit_scale,
                                    classifier_head=classifier_head,
                                    dataset=args.dataset,
                                    output_dir=args.output_dir, device=args.device,
                                    pre_extracted=args.pre_extracted,
                                    )

        train_loss_avg = train_loss_sum / len(train_loader)
        loss_logger.write(f'{epoch},{num_iter},{round(train_loss_avg, 6)},{round(val_loss, 6)},{round(val_acc, 6)},{round(test_acc, 6)}\n')
        loss_logger.flush()
        logger.info(f"Epoch {int(epoch)}, Iter {num_iter}, Trn Loss: {round(train_loss_avg, 6)}, Val Loss: {round(val_loss, 6)}, Val Acc: {round(val_acc, 3)}, Test Acc: {round(test_acc, 3)}")

        ## save model checkpoints every X epochs
        if args.save_ckpt and (num_iter % args.save_freq == 0 or epoch == args.epochs):
            model_path = save_model_ckpt(args, best_records,
                                        model, classifier_head, optimizer, scheduler, logit_scale,
                                         val_acc, epoch, num_iter)
            logger.info(f'Model ckpt saved to: {model_path}')

    logger.info(f'Probing done.')

    return best_model, best_head, best_records, best_logit_scale, val_loader, test_loader


def train_ce(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader, reload_model):
    """ Train the model with Cross-Entropy Loss, finetuning visual encoder and classifier"""

    logger.info(f"Start standard finetuning ......")

    if reload_model:
        load_model(args, logger, model, None, classifier_head)

    model.eval() if args.freeze_visual else model.train()
    classifier_head.train()

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
    best_head = classifier_head
    best_logit_scale = logit_scale
    best_records = {}
    num_iter = 0

    for epoch in range(1, args.epochs+1):
        train_loss_sum = 0
        for inputs, labels, text, source in train_loader:
            num_iter += 1
            images = inputs.to(args.device)
            labels = labels.to(args.device).long()
            source = source.to(args.device)

            image_features = model.encode_image(images)
            image_feature = image_features / image_features.norm(dim=-1, keepdim=True)

            logits = classifier_head(image_feature)
            logits = logits * logit_scale.exp()
            if args.loss_name == 'WeightedCE':
                total_loss = loss(logits, labels, source) # for WeightedCE, needs to input the source
            else:
                total_loss = loss(logits, labels)
            train_loss = total_loss.item()
            train_loss_sum += train_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step() # update learning rate for each iteration

        # validate after 1 epoch
        # if (args.early_stop and num_iter >= args.start_validation) or \
        #     epoch == args.epochs:
        if args.early_stop or epoch == args.epochs:
            val_acc, val_loss, confusion_matrix = validate(args, data_loader=val_loader, model=model, logger=logger,
                                                        loss=args.loss, logit_scale=logit_scale,
                                                        classifier_head=classifier_head, show_confusion_matrix=True,
                                                        dataset=args.dataset,
                                                        output_dir=args.output_dir, device=args.device,
                                                        pre_extracted=args.pre_extracted,
                                                        )
            scores = calculate_scores(confusion_matrix)

        # check if val acc has improved, here i use the val_acc rather than val_loss,
        # as we might modify the loss function for confusing pairs
        if (args.early_stop or epoch == args.epochs) and val_acc >= best_val_acc:
            # logger.info(f'Val Acc improved from {round(best_val_acc, 3)} to {round(val_acc, 3)}')
            best_val_acc = val_acc
            best_logit_scale = copy.deepcopy(logit_scale)
            best_epoch = epoch
            best_iter = num_iter
            best_scores = copy.deepcopy(scores)
            best_confusion_matrix = copy.deepcopy(confusion_matrix)
            best_head = copy.deepcopy(classifier_head)
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
            test_acc, _, _ = validate(args,data_loader=test_loader, model=model, logger=logger,
                                loss=args.loss, logit_scale=logit_scale,
                                classifier_head=classifier_head,
                                dataset=args.dataset,
                                output_dir=args.output_dir, device=args.device,
                                pre_extracted=args.pre_extracted,
                                )

        train_loss_avg = train_loss_sum / len(train_loader)
        loss_logger.write(f'{epoch},{num_iter},{round(train_loss_avg, 6)},{round(val_loss, 6)},{round(val_acc, 6)},{round(test_acc, 6)}\n')
        loss_logger.flush()
        logger.info(f"Epoch {int(epoch)}, Iter {num_iter}, Trn Loss: {round(train_loss_avg, 6)}, Val Loss: {round(val_loss, 6)}, "
                    f"logit_scale: {round(logit_scale.item(), 4)} Val Acc: {round(val_acc, 3)}, Test Acc: {round(test_acc, 3)}")

        ## save model checkpoints every X epochs
        if args.save_ckpt and (num_iter % args.save_freq == 0 or epoch == args.epochs):
            model_path = save_model_ckpt(args, best_records,
                                        model, classifier_head, optimizer, scheduler, logit_scale,
                                         val_acc, epoch, num_iter)
            logger.info(f'Model ckpt saved to: {model_path}')

        if epoch == args.stop_epochs:
            break

    logger.info(f'Training done.')
    logger.info(f'Best val Acc: {round(best_val_acc, 3)} at epoch {best_epoch}, iter {best_iter}')

    return best_model, best_head, best_records, best_logit_scale


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


def train_balanced_contrastive(args, logger, loss_logger, model, classifier_head,
                                logit_scale, loss, optimizer, scheduler,
                                train_dataloader, val_dataloader, test_dataloader):
    """ train CLIP visual encoder with supervised contrastive loss, then linear prob to evaluate learned representations """
    exit()




def train_dataset_cls(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader):
    """
    Train the model with Cross-Entropy Loss, finetuning visual encoder and classifier
    For dataset classification
    """

    logger.info(f"Start standard finetuning ......")

    model.eval() if args.freeze_visual else model.train()
    classifier_head.train()

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
    best_head = classifier_head
    best_logit_scale = logit_scale
    best_records = {}
    num_iter = 0

    for epoch in range(1, args.epochs+1):
        train_loss_sum = 0
        for inputs, labels, text, source in train_loader:
            num_iter += 1
            images = inputs.to(args.device)
            # labels = labels.to(args.device).long()
            labels = source.to(args.device).long() # use the source as labels
            # source = source.to(args.device)

            image_features = model.encode_image(images)
            image_feature = image_features / image_features.norm(dim=-1, keepdim=True)

            logits = classifier_head(image_feature)
            logits = logits * logit_scale.exp()
            total_loss = loss(logits, labels)
            train_loss = total_loss.item()
            train_loss_sum += train_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step() # update learning rate for each iteration

        # validate after 1 epoch
        # if (args.early_stop and num_iter >= args.start_validation) or \
        #     epoch == args.epochs:
        if args.early_stop or epoch == args.epochs:
            val_acc, val_loss, confusion_matrix = validate_dataset(args, data_loader=val_loader, model=model, logger=logger,
                                                        loss=args.loss, logit_scale=logit_scale,
                                                        classifier_head=classifier_head, show_confusion_matrix=True,
                                                        dataset=args.dataset,
                                                        output_dir=args.output_dir, device=args.device,
                                                        pre_extracted=args.pre_extracted,
                                                        )
            scores = calculate_scores(confusion_matrix)

        # check if val acc has improved, here i use the val_acc rather than val_loss,
        # as we might modify the loss function for confusing pairs
        if (args.early_stop or epoch == args.epochs) and val_acc >= best_val_acc:
            # logger.info(f'Val Acc improved from {round(best_val_acc, 3)} to {round(val_acc, 3)}')
            best_val_acc = val_acc
            best_logit_scale = copy.deepcopy(logit_scale)
            best_epoch = epoch
            best_iter = num_iter
            best_scores = copy.deepcopy(scores)
            best_confusion_matrix = copy.deepcopy(confusion_matrix)
            best_head = copy.deepcopy(classifier_head)
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
            test_acc, _, _ = validate_dataset(args,data_loader=test_loader, model=model, logger=logger,
                                loss=args.loss, logit_scale=logit_scale,
                                classifier_head=classifier_head,
                                dataset=args.dataset,
                                output_dir=args.output_dir, device=args.device,
                                pre_extracted=args.pre_extracted,
                                )

        train_loss_avg = train_loss_sum / len(train_loader)
        loss_logger.write(f'{epoch},{num_iter},{round(train_loss_avg, 6)},{round(val_loss, 6)},{round(val_acc, 6)},{round(test_acc, 6)}\n')
        loss_logger.flush()
        logger.info(f"Epoch {int(epoch)}, Iter {num_iter}, Trn Loss: {round(train_loss_avg, 6)}, Val Loss: {round(val_loss, 6)}, Val Acc: {round(val_acc, 3)}, Test Acc: {round(test_acc, 3)}")

        ## save model checkpoints every X epochs
        if args.save_ckpt and (num_iter % args.save_freq == 0 or epoch == args.epochs):
            model_path = save_model_ckpt(args, best_records,
                                        model, classifier_head, optimizer, scheduler, logit_scale,
                                         val_acc, epoch, num_iter)
            logger.info(f'Model ckpt saved to: {model_path}')

        if epoch == args.stop_epochs:
            break

    logger.info(f'Training done.')
    logger.info(f'Best val Acc: {round(best_val_acc, 3)} at epoch {best_epoch}, iter {best_iter}')

    return best_model, best_head, best_records, best_logit_scale






def train_ce_mixed(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader):
    """
    Train the model with Cross-Entropy Loss, finetuning visual encoder and classifier.
    Half batch from retrieved data, half batch from few-shot data.
    """

    train_loader, train_dataloader_fs = train_loader
    train_loader_fs = iter(train_dataloader_fs)

    logger.info(f"Start Training ......")

    model.eval() if args.freeze_visual else model.train()
    classifier_head.train()

    best_val_acc = -1
    best_epoch = -1
    best_iter = -1
    best_model = model
    best_head = classifier_head
    best_logit_scale = logit_scale
    best_records = {}
    num_iter = 0
    for epoch in range(1, args.epochs+1):

        train_loss_sum = 0
        # train for 1 epoch
        for inputs, labels, text, source in train_loader:
            num_iter += 1
            images = inputs.to(args.device)
            labels = labels.to(args.device).long()
            source = source.to(args.device)

            # get a batch of few-shot data, handle the case when the few-shot data is exhausted, just loop back
            try:
                inputs_fs, labels_fs, text_fs, source_fs = next(train_loader_fs)
            except StopIteration:
                train_loader_fs = iter(train_dataloader_fs)
                inputs_fs, labels_fs, text_fs, source_fs = next(train_loader_fs)
            images_fs = inputs_fs.to(args.device)
            labels_fs = labels_fs.to(args.device).long()
            source_fs = source_fs.to(args.device)

            # concatenate the retrieved data and few-shot data
            images = torch.cat([images, images_fs], dim=0)
            labels = torch.cat([labels, labels_fs], dim=0)
            source = torch.cat([source, source_fs], dim=0)

            image_features = model.encode_image(images)
            image_feature = image_features / image_features.norm(dim=-1, keepdim=True)

            logits = classifier_head(image_feature)
            logits = logits * logit_scale.exp()
            if args.loss_name == 'WeightedCE':
                total_loss = loss(logits, labels, source) # for WeightedCE, needs to input the source
            else:
                total_loss = loss(logits, labels)
            train_loss = total_loss.item()
            train_loss_sum += train_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step() # update learning rate for each iteration

        # validate after 1 epoch
        # if (args.early_stop and num_iter >= args.start_validation) or \
        #     epoch == args.epochs:
        val_acc, val_loss, confusion_matrix = validate(args, data_loader=val_loader, model=model, logger=logger,
                                                        loss=args.loss, logit_scale=logit_scale,
                                                        classifier_head=classifier_head, show_confusion_matrix=True,
                                                        dataset=args.dataset,
                                                        output_dir=args.output_dir, device=args.device,
                                                        pre_extracted=args.pre_extracted,
                                                        )
        scores = calculate_scores(confusion_matrix)

        # check if val acc has improved, here i use the val_acc rather than val_loss,
        # as we might modify the loss function for confusing pairs
        if (args.early_stop or epoch == args.epochs) and val_acc >= best_val_acc:
            # logger.info(f'Val Acc improved from {round(best_val_acc, 3)} to {round(val_acc, 3)}')
            best_val_acc = val_acc
            best_logit_scale = copy.deepcopy(logit_scale)
            best_epoch = epoch
            best_iter = num_iter
            best_scores = copy.deepcopy(scores)
            best_confusion_matrix = copy.deepcopy(confusion_matrix)
            best_head = copy.deepcopy(classifier_head)
            best_model = copy.deepcopy(model)

            # save into the best_records
            best_records['best_val_acc'] = best_val_acc
            best_records['best_logit_scale'] = best_logit_scale
            best_records['best_epoch'] = best_epoch
            best_records['best_iter'] = best_iter
            best_records['best_scores'] = best_scores
            best_records['best_confusion_matrix'] = best_confusion_matrix

        # test after 1 epoch
        test_acc, _, _ = validate(args,data_loader=test_loader, model=model, logger=logger,
                                loss=args.loss, logit_scale=logit_scale,
                                classifier_head=classifier_head,
                                dataset=args.dataset,
                                output_dir=args.output_dir, device=args.device,
                                pre_extracted=args.pre_extracted,
                                )

        train_loss_avg = train_loss_sum / len(train_loader)
        loss_logger.write(f'{epoch},{num_iter},{round(train_loss_avg, 6)},{round(val_loss, 6)},{round(val_acc, 6)},{round(test_acc, 6)}\n')
        loss_logger.flush()
        logger.info(f"Epoch {int(epoch)}, Iter {num_iter}, Trn Loss: {round(train_loss_avg, 6)}, Val Loss: {round(val_loss, 6)}, Val Acc: {round(val_acc, 3)}, Test Acc: {round(test_acc, 3)}")

        ## save model checkpoints every X epochs
        if args.save_ckpt and (num_iter % args.save_freq == 0 or epoch == args.epochs):
            model_path = save_model_ckpt(args, best_records,
                                        model, classifier_head, optimizer, scheduler, logit_scale,
                                         val_acc, epoch, num_iter)
            logger.info(f'Model ckpt saved to: {model_path}')

        if epoch == args.stop_epochs:
            break

    logger.info(f'Training done.')
    logger.info(f'Best val Acc: {round(best_val_acc, 3)} at epoch {best_epoch}, iter {best_iter}')

    return best_model, best_head, best_records, best_logit_scale



def train_fixmatch(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader):
    """
    Train the model with fixmatch SSL method.
    Part of the batch from labeled data, part from unlabeled data
    """

    train_dataloader, u_train_dataloader = train_loader
    # print(f'train_dataloader: {len(train_dataloader)}')
    # print(f'u_train_dataloader: {len(u_train_dataloader)}')
    train_loader = iter(train_dataloader)
    # u_train_loader = iter(u_train_dataloader)

    logger.info(f"Start FixMatch Training ......")

    model.eval() if args.freeze_visual else model.train()
    classifier_head.train()

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
    best_head = classifier_head
    best_logit_scale = logit_scale
    best_records = {}
    num_iter = 0

    for epoch in range(1, args.epochs+1):
        train_loss_sum = 0

        for u_inputs, _, _, _ in u_train_dataloader: # get a batch of weakly augmented data and strongly augmented data

            num_iter += 1
            # print(f'len(u_inputs): {len(u_inputs)}')
            # print type of u_inputs
            # print(f'type(u_inputs): {type(u_inputs)}')
            # print(f'u_inputs: {u_inputs}')
            # print(f'num_iter: {num_iter}')
            u_input_w, u_input_s = u_inputs
            # print(f'u_input_w: {u_input_w.size()}, u_input_s: {u_input_s.size()}')

            # load a batch of labeled data
            try:
                inputs, labels, text, source = next(train_loader)
            except StopIteration:
                train_loader = iter(train_dataloader)
                inputs, labels, text, source = next(train_loader)

            # concate the labeled data and unlabeled data
            inputs_all = torch.cat([inputs, u_input_w, u_input_s], dim=0)

            images = inputs_all.to(args.device)
            labels = labels.to(args.device).long()

            image_features = model.encode_image(images)
            image_feature = image_features / image_features.norm(dim=-1, keepdim=True)

            logits = classifier_head(image_feature)

            # get the logits for labeled data
            logits_l = logits[:inputs.size(0)]

            # get the logits for weakly augmented data and strongly augmented data
            logits_u_w, logits_u_s = logits[inputs.size(0):].chunk(2)

            # calculate the loss for labeled data
            loss_l = F.cross_entropy(logits_l, labels, reduction='mean')

            # get the pseudo labels for weakly augmented data
            # pseudo_labels_w = torch.softmax(logits_u_w * logit_scale.exp(), dim=-1)
            pseudo_labels_w = torch.softmax(logits_u_w, dim=-1)


            # get the max probability and prediction based on the pseudo_labels_w
            max_probs_w, targets_u_w = torch.max(pseudo_labels_w, dim=-1)
            mask_w = max_probs_w.ge(args.threshold).float()

            # calculate the consistency loss
            loss_u = (F.cross_entropy(logits_u_s, targets_u_w, reduction='none') * mask_w).mean()

            # add up to the total loss
            total_loss = loss_l + args.lambda_u * loss_u

            train_loss = total_loss.item()
            train_loss_sum += train_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step() # update learning rate for each iteration

        # validate after 1 epoch
        # if (args.early_stop and num_iter >= args.start_validation) or \
        #     epoch == args.epochs:
        if args.early_stop or epoch == args.epochs:
            val_acc, val_loss, confusion_matrix = validate(args,
                                                        #    data_loader=val_loader,
                                                           data_loader=test_loader, # note that here i used test loader for validation
                                                           model=model, logger=logger,
                                                        loss=args.loss, logit_scale=logit_scale,
                                                        classifier_head=classifier_head, show_confusion_matrix=True,
                                                        dataset=args.dataset,
                                                        output_dir=args.output_dir, device=args.device,
                                                        pre_extracted=args.pre_extracted,
                                                        )
            scores = calculate_scores(confusion_matrix)

        # check if val acc has improved, here i use the val_acc rather than val_loss,
        # as we might modify the loss function for confusing pairs
        if (args.early_stop or epoch == args.epochs) and val_acc >= best_val_acc:
            # logger.info(f'Val Acc improved from {round(best_val_acc, 3)} to {round(val_acc, 3)}')
            best_val_acc = val_acc
            best_logit_scale = copy.deepcopy(logit_scale)
            best_epoch = epoch
            best_iter = num_iter
            best_scores = copy.deepcopy(scores)
            best_confusion_matrix = copy.deepcopy(confusion_matrix)
            best_head = copy.deepcopy(classifier_head)
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
            test_acc, _, _ = validate(args,data_loader=test_loader, model=model, logger=logger,
                                loss=args.loss, logit_scale=logit_scale,
                                classifier_head=classifier_head,
                                dataset=args.dataset,
                                output_dir=args.output_dir, device=args.device,
                                pre_extracted=args.pre_extracted,
                                )

        train_loss_avg = train_loss_sum / len(train_loader)
        loss_logger.write(f'{epoch},{num_iter},{round(train_loss_avg, 6)},{round(val_loss, 6)},{round(val_acc, 6)},{round(test_acc, 6)}\n')
        loss_logger.flush()
        logger.info(f"Epoch {int(epoch)}, Iter {num_iter}, Trn Loss: {round(train_loss_avg, 6)}, Val Loss: {round(val_loss, 6)}, Val Acc: {round(val_acc, 3)}, Test Acc: {round(test_acc, 3)}")

        ## save model checkpoints every X epochs
        if args.save_ckpt and (num_iter % args.save_freq == 0 or epoch == args.epochs):
            model_path = save_model_ckpt(args, best_records,
                                        model, classifier_head, optimizer, scheduler, logit_scale,
                                         val_acc, epoch, num_iter)
            logger.info(f'Model ckpt saved to: {model_path}')

        if epoch == args.stop_epochs:
            break

    logger.info(f'Training done.')
    logger.info(f'Best val Acc: {round(best_val_acc, 3)} at epoch {best_epoch}, iter {best_iter}')

    return best_model, best_head, best_records, best_logit_scale






def train_ce_multitask(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader, dataset_classifier_head):
    """ Train the model with Cross-Entropy Loss, finetuning visual encoder and classifier"""

    logger.info(f"Start Training ......")

    model.eval() if args.freeze_visual else model.train()
    classifier_head.train()
    dataset_classifier_head.train()

    best_val_acc = -1
    best_epoch = -1
    best_iter = -1
    best_model = model
    best_head = classifier_head
    best_logit_scale = logit_scale
    best_records = {}
    num_iter = 0
    dataset_weight = 1.0
    for epoch in range(1, args.epochs+1):
        # dataset_weight *= args.dataset_wd # decay the dataset weight for each epoch
        dataset_weight = args.dataset_wd

        train_loss_sum = 0
        dataset_loss_sum = 0
        # train for 1 epoch
        for inputs, labels, text, source in train_loader:
            num_iter += 1
            images = inputs.to(args.device)
            labels = labels.to(args.device).long()
            source = source.to(args.device)

            image_features = model.encode_image(images)
            image_feature = image_features / image_features.norm(dim=-1, keepdim=True)

            logits = classifier_head(image_feature)
            logits = logits * logit_scale.exp()
            if args.loss_name == 'WeightedCE':
                total_loss = loss(logits, labels, source) # for WeightedCE, needs to input the source
            else:
                total_loss = loss(logits, labels)

            train_loss = total_loss.item()
            train_loss_sum += train_loss

            # add the dataset classification loss
            dataset_logits = dataset_classifier_head(image_feature)
            # dataset_logits = dataset_logits * logit_scale.exp()
            dataset_loss = loss(dataset_logits, source)
            dataset_loss_sum += dataset_loss.item()

            multitask_loss = total_loss + dataset_loss * dataset_weight

            optimizer.zero_grad()
            multitask_loss.backward()
            optimizer.step()
            scheduler.step() # update learning rate for each iteration

        # validate after 1 epoch
        # if (args.early_stop and num_iter >= args.start_validation) or \
        #     epoch == args.epochs:
        val_acc, val_loss, confusion_matrix, dataset_val_acc, dataset_confusion_matrix = validate_multitask(args, data_loader=val_loader, model=model, logger=logger,
                                                        loss=args.loss, logit_scale=logit_scale,
                                                        classifier_head=classifier_head, dataset_classifier_head=dataset_classifier_head,
                                                        show_confusion_matrix=True,
                                                        dataset=args.dataset,
                                                        output_dir=args.output_dir, device=args.device,
                                                        pre_extracted=args.pre_extracted,
                                                        )
        scores = calculate_scores(confusion_matrix)
        # dataset_scores = calculate_scores(dataset_confusion_matrix)

        # check if val acc has improved, here i use the val_acc rather than val_loss,
        # as we might modify the loss function for confusing pairs
        if (args.early_stop or epoch == args.epochs) and val_acc >= best_val_acc:
            # logger.info(f'Val Acc improved from {round(best_val_acc, 3)} to {round(val_acc, 3)}')
            best_val_acc = val_acc
            best_logit_scale = copy.deepcopy(logit_scale)
            best_epoch = epoch
            best_iter = num_iter
            best_scores = copy.deepcopy(scores)
            best_confusion_matrix = copy.deepcopy(confusion_matrix)
            best_head = copy.deepcopy(classifier_head)
            best_model = copy.deepcopy(model)

            # save into the best_records
            best_records['best_val_acc'] = best_val_acc
            best_records['best_logit_scale'] = best_logit_scale
            best_records['best_epoch'] = best_epoch
            best_records['best_iter'] = best_iter
            best_records['best_scores'] = best_scores
            best_records['best_confusion_matrix'] = best_confusion_matrix

        # test after 1 epoch
        test_acc, _, _, dataset_test_acc, _ = validate_multitask(args,data_loader=test_loader, model=model, logger=logger,
                                loss=args.loss, logit_scale=logit_scale,
                                classifier_head=classifier_head, dataset_classifier_head=dataset_classifier_head,
                                dataset=args.dataset,
                                output_dir=args.output_dir, device=args.device,
                                pre_extracted=args.pre_extracted,
                                )

        train_loss_avg = train_loss_sum / len(train_loader)
        dataset_loss_avg = dataset_loss_sum / len(train_loader)

        loss_logger.write(f'{epoch},{num_iter},{round(train_loss_avg, 6)},{round(val_loss, 6)},{round(val_acc, 6)},{round(test_acc, 6)}\n')
        loss_logger.flush()
        logger.info(f"Epoch {int(epoch)}, Iter {num_iter}, Trn Loss: {round(train_loss_avg, 6)}, Dataset Loss: {round(dataset_loss_avg, 6)}, weight: {round(dataset_weight, 3)}, Val Loss: {round(val_loss, 6)}, Val Acc: {round(val_acc, 3)}, Dataset Val Acc: {round(dataset_val_acc, 3)}, Test Acc: {round(test_acc, 3)}, Dataset Test Acc: {round(dataset_test_acc, 3)}")

        ## save model checkpoints every X epochs
        if args.save_ckpt and (num_iter % args.save_freq == 0 or epoch == args.epochs):
            model_path = save_model_ckpt(args, best_records,
                                        model, classifier_head, optimizer, scheduler, logit_scale,
                                         val_acc, epoch, num_iter)
            logger.info(f'Model ckpt saved to: {model_path}')

        if epoch == args.stop_epochs:
            break

    logger.info(f'Training done.')
    logger.info(f'Best val Acc: {round(best_val_acc, 3)} at epoch {best_epoch}, iter {best_iter}')

    return best_model, best_head, best_records, best_logit_scale

def mixup_data(x, y, alpha=1.0, mix_prob=0.5):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    # samply a number uniformly from [0, 1],
    # this includes the clean/unmixed images to the training process
    flag = torch.rand(1).item()
    if flag <= mix_prob: # do mixup
        lam = lam
    else: # do not mixup
        lam = 1.0

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam

def mixup_data_fs(x_retr, y_retr, x_fs, y_fs, alpha=1.0, mix_prob=0.5):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    # samply a number uniformly from [0, 1],
    # this includes the clean/unmixed images to the training process
    flag = torch.rand(1).item()
    if flag <= mix_prob:
        lam = 0.0 # set to 0.0 to use few-shot data only
    else: # do not mixup
        lam = 1.0

    mixed_x = lam * x_retr + (1.0 - lam) * x_fs
    y_a, y_b = y_retr, y_fs

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)

def mixup_criterion_lam_list(criterion, pred, y_a, y_b, lam_list):
    # each value in lam_list is the lambda value for each image in the batch
    return sum([lam_list[i] * criterion(pred[i], y_a[i]) + (1.0 - lam_list[i]) * criterion(pred[i], y_b[i]) for i in range(len(lam_list))])

def train_mixup(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader):
    """
    Train the model with Cross-Entropy Loss, finetuning visual encoder and classifier
    Use Mixup method to augment the training data
    """

    logger.info(f"Start Training mixup ......")

    model.eval() if args.freeze_visual else model.train()
    classifier_head.train()

    logit_scale = args.logit_scale
    loss = args.loss
    optimizer = args.optimizer
    scheduler = args.scheduler

    best_val_acc = -1
    best_epoch = -1
    best_iter = -1
    best_model = model
    best_head = classifier_head
    best_logit_scale = logit_scale
    best_records = {}
    num_iter = 0
    for epoch in range(1, args.epochs+1):

        train_loss_sum = 0
        # train for 1 epoch
        for inputs, labels, text, source in train_loader:
            num_iter += 1
            images = inputs.to(args.device)
            labels = labels.to(args.device).long()
            # source = source.to(args.device)

            # apply the mixup strategy
            images, targets_a, targets_b, lam = mixup_data(images, labels, alpha=args.mixup_alpha, mix_prob=args.mix_prob)

            image_features = model.encode_image(images)
            image_feature = image_features / image_features.norm(dim=-1, keepdim=True)

            logits = classifier_head(image_feature)
            logits = logits * logit_scale.exp()

            # mixup loss
            total_loss = mixup_criterion(loss, logits, targets_a, targets_b, lam)

            train_loss = total_loss.item()
            train_loss_sum += train_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step() # update learning rate for each iteration

        # validate after 1 epoch
        # if (args.early_stop and num_iter >= args.start_validation) or \
        #     epoch == args.epochs:
        val_acc, val_loss, confusion_matrix = validate(args, data_loader=val_loader, model=model, logger=logger,
                                                        loss=args.loss, logit_scale=logit_scale,
                                                        classifier_head=classifier_head, show_confusion_matrix=True,
                                                        dataset=args.dataset,
                                                        output_dir=args.output_dir, device=args.device,
                                                        pre_extracted=args.pre_extracted,
                                                        )
        scores = calculate_scores(confusion_matrix)

        # check if val acc has improved, here i use the val_acc rather than val_loss,
        # as we might modify the loss function for confusing pairs
        if (args.early_stop or epoch == args.epochs) and val_acc >= best_val_acc:
            # logger.info(f'Val Acc improved from {round(best_val_acc, 3)} to {round(val_acc, 3)}')
            best_val_acc = val_acc
            best_logit_scale = copy.deepcopy(logit_scale)
            best_epoch = epoch
            best_iter = num_iter
            best_scores = copy.deepcopy(scores)
            best_confusion_matrix = copy.deepcopy(confusion_matrix)
            best_head = copy.deepcopy(classifier_head)
            best_model = copy.deepcopy(model)

            # save into the best_records
            best_records['best_val_acc'] = best_val_acc
            best_records['best_logit_scale'] = best_logit_scale
            best_records['best_epoch'] = best_epoch
            best_records['best_iter'] = best_iter
            best_records['best_scores'] = best_scores
            best_records['best_confusion_matrix'] = best_confusion_matrix

        # test after 1 epoch
        test_acc, _, _ = validate(args,data_loader=test_loader, model=model, logger=logger,
                                loss=args.loss, logit_scale=logit_scale,
                                classifier_head=classifier_head,
                                dataset=args.dataset,
                                output_dir=args.output_dir, device=args.device,
                                pre_extracted=args.pre_extracted,
                                )

        train_loss_avg = train_loss_sum / len(train_loader)
        loss_logger.write(f'{epoch},{num_iter},{round(train_loss_avg, 6)},{round(val_loss, 6)},{round(val_acc, 6)},{round(test_acc, 6)}\n')
        loss_logger.flush()
        logger.info(f"Epoch {int(epoch)}, Iter {num_iter}, Trn Loss: {round(train_loss_avg, 6)}, Val Loss: {round(val_loss, 6)}, Val Acc: {round(val_acc, 3)}, Test Acc: {round(test_acc, 3)}")

        ## save model checkpoints every X epochs
        if args.save_ckpt and (num_iter % args.save_freq == 0 or epoch == args.epochs):
            model_path = save_model_ckpt(args, best_records,
                                        model, classifier_head, optimizer, scheduler, logit_scale,
                                         val_acc, epoch, num_iter)
            logger.info(f'Model ckpt saved to: {model_path}')

        if epoch == args.stop_epochs:
            break

    logger.info(f'Training done.')
    logger.info(f'Best val Acc: {round(best_val_acc, 3)} at epoch {best_epoch}, iter {best_iter}')

    return best_model, best_head, best_records, best_logit_scale


def train_mixup_fs(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader):
    """
    Train the model with Cross-Entropy Loss, finetuning visual encoder and classifier
    Use Mixup method to augment the training data
    """
    train_loader, train_dataloader_fs = train_loader
    train_loader_fs = iter(train_dataloader_fs)

    logger.info(f"Start Training ......")

    model.eval() if args.freeze_visual else model.train()
    classifier_head.train()

    best_val_acc = -1
    best_epoch = -1
    best_iter = -1
    best_model = model
    best_head = classifier_head
    best_logit_scale = logit_scale
    best_records = {}
    num_iter = 0
    for epoch in range(1, args.epochs+1):

        train_loss_sum = 0
        # train for 1 epoch
        for inputs, labels, text, source in train_loader: # this is still the mixed data
            num_iter += 1
            images = inputs.to(args.device)
            labels = labels.to(args.device).long()

            # get a batch of few-shot data, handle the case when the few-shot data is exhausted, just loop back
            try:
                inputs_fs, labels_fs, text_fs, source_fs = next(train_loader_fs)
            except StopIteration:
                train_loader_fs = iter(train_dataloader_fs)
                inputs_fs, labels_fs, text_fs, source_fs = next(train_loader_fs)
            images_fs = inputs_fs.to(args.device)
            labels_fs = labels_fs.to(args.device).long()

            # apply the mixup strategy
            images, targets_a, targets_b, lam = mixup_data_fs(images, labels, images_fs, labels_fs,
                                                           alpha=args.mixup_alpha, mix_prob=args.mix_prob)

            image_features = model.encode_image(images)
            image_feature = image_features / image_features.norm(dim=-1, keepdim=True)

            logits = classifier_head(image_feature)
            logits = logits * logit_scale.exp()

            # mixup loss
            total_loss = mixup_criterion(loss, logits, targets_a, targets_b, lam)

            train_loss = total_loss.item()
            train_loss_sum += train_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step() # update learning rate for each iteration

        # validate after 1 epoch
        # if (args.early_stop and num_iter >= args.start_validation) or \
        #     epoch == args.epochs:
        val_acc, val_loss, confusion_matrix = validate(args, data_loader=val_loader, model=model, logger=logger,
                                                        loss=args.loss, logit_scale=logit_scale,
                                                        classifier_head=classifier_head, show_confusion_matrix=True,
                                                        dataset=args.dataset,
                                                        output_dir=args.output_dir, device=args.device,
                                                        pre_extracted=args.pre_extracted,
                                                        )
        scores = calculate_scores(confusion_matrix)

        # check if val acc has improved, here i use the val_acc rather than val_loss,
        # as we might modify the loss function for confusing pairs
        if (args.early_stop or epoch == args.epochs) and val_acc >= best_val_acc:
            # logger.info(f'Val Acc improved from {round(best_val_acc, 3)} to {round(val_acc, 3)}')
            best_val_acc = val_acc
            best_logit_scale = copy.deepcopy(logit_scale)
            best_epoch = epoch
            best_iter = num_iter
            best_scores = copy.deepcopy(scores)
            best_confusion_matrix = copy.deepcopy(confusion_matrix)
            best_head = copy.deepcopy(classifier_head)
            best_model = copy.deepcopy(model)

            # save into the best_records
            best_records['best_val_acc'] = best_val_acc
            best_records['best_logit_scale'] = best_logit_scale
            best_records['best_epoch'] = best_epoch
            best_records['best_iter'] = best_iter
            best_records['best_scores'] = best_scores
            best_records['best_confusion_matrix'] = best_confusion_matrix

        # test after 1 epoch
        test_acc, _, _ = validate(args,data_loader=test_loader, model=model, logger=logger,
                                loss=args.loss, logit_scale=logit_scale,
                                classifier_head=classifier_head,
                                dataset=args.dataset,
                                output_dir=args.output_dir, device=args.device,
                                pre_extracted=args.pre_extracted,
                                )

        train_loss_avg = train_loss_sum / len(train_loader)
        loss_logger.write(f'{epoch},{num_iter},{round(train_loss_avg, 6)},{round(val_loss, 6)},{round(val_acc, 6)},{round(test_acc, 6)}\n')
        loss_logger.flush()
        logger.info(f"Epoch {int(epoch)}, Iter {num_iter}, Trn Loss: {round(train_loss_avg, 6)}, Val Loss: {round(val_loss, 6)}, Val Acc: {round(val_acc, 3)}, Test Acc: {round(test_acc, 3)}")

        ## save model checkpoints every X epochs
        if args.save_ckpt and (num_iter % args.save_freq == 0 or epoch == args.epochs):
            model_path = save_model_ckpt(args, best_records,
                                        model, classifier_head, optimizer, scheduler, logit_scale,
                                         val_acc, epoch, num_iter)
            logger.info(f'Model ckpt saved to: {model_path}')

        if epoch == args.stop_epochs:
            break

    logger.info(f'Training done.')
    logger.info(f'Best val Acc: {round(best_val_acc, 3)} at epoch {best_epoch}, iter {best_iter}')

    return best_model, best_head, best_records, best_logit_scale


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def train_cutmix(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader):
    """
    Train the model with Cross-Entropy Loss, finetuning visual encoder and classifier
    Use CutMix method to augment the training data
    """

    logger.info(f"Start Training cutmix ......")

    model.eval() if args.freeze_visual else model.train()
    classifier_head.train()

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
    best_head = classifier_head
    best_logit_scale = logit_scale
    best_records = {}
    num_iter = 0
    for epoch in range(1, args.epochs+1):

        train_loss_sum = 0
        for inputs, labels, text, source in train_loader:
            num_iter += 1
            images = inputs.to(args.device, non_blocking=True)
            labels = labels.to(args.device, non_blocking=True).long()

            # apply the cutmix strategy
            r = np.random.rand(1)
            if args.cutmix_beta > 0 and r < args.mix_prob:
                # generate mixed sample
                lam = np.random.beta(args.cutmix_beta, args.cutmix_beta)
                rand_index = torch.randperm(images.size()[0]).cuda()
                target_a = labels
                target_b = labels[rand_index]
                bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
                images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
                # adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
            else:
                target_a = labels
                target_b = labels
                lam = 1.0

            image_features = model.encode_image(images)
            image_feature = image_features / image_features.norm(dim=-1, keepdim=True)

            logits = classifier_head(image_feature)
            logits = logits * logit_scale.exp()

            # mixup loss
            total_loss = mixup_criterion(loss, logits, target_a, target_b, lam)

            train_loss = total_loss.item()
            train_loss_sum += train_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step() # update learning rate for each iteration

            # break # for fast debugging

        # validate after 1 epoch
        if args.early_stop or epoch == args.epochs:
            val_acc, val_loss, confusion_matrix = validate(args, data_loader=val_loader, model=model, logger=logger,
                                                        loss=args.loss, logit_scale=logit_scale,
                                                        classifier_head=classifier_head, show_confusion_matrix=True,
                                                        dataset=args.dataset,
                                                        output_dir=args.output_dir, device=args.device,
                                                        pre_extracted=args.pre_extracted,
                                                        )
            scores = calculate_scores(confusion_matrix)

        # check if val acc has improved, here i use the val_acc rather than val_loss,
        # as we might modify the loss function for confusing pairs
        if (args.early_stop or epoch == args.epochs) and val_acc >= best_val_acc:
            # logger.info(f'Val Acc improved from {round(best_val_acc, 3)} to {round(val_acc, 3)}')
            best_val_acc = val_acc
            best_logit_scale = copy.deepcopy(logit_scale)
            best_epoch = epoch
            best_iter = num_iter
            best_scores = copy.deepcopy(scores)
            best_confusion_matrix = copy.deepcopy(confusion_matrix)
            best_head = copy.deepcopy(classifier_head)
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
            test_acc, _, _ = validate(args,data_loader=test_loader, model=model, logger=logger,
                                loss=args.loss, logit_scale=logit_scale,
                                classifier_head=classifier_head,
                                dataset=args.dataset,
                                output_dir=args.output_dir, device=args.device,
                                pre_extracted=args.pre_extracted,
                                )

        train_loss_avg = train_loss_sum / len(train_loader)
        loss_logger.write(f'{epoch},{num_iter},{round(train_loss_avg, 6)},{round(val_loss, 6)},{round(val_acc, 6)},{round(test_acc, 6)}\n')
        loss_logger.flush()
        logger.info(f"Epoch {int(epoch)}, Iter {num_iter}, Trn Loss: {round(train_loss_avg, 6)}, Val Loss: {round(val_loss, 6)}, Val Acc: {round(val_acc, 3)}, Test Acc: {round(test_acc, 3)}")

        # flush the logger message out
        logger.handlers[0].flush()

        ## save model checkpoints every X epochs
        if args.save_ckpt and (num_iter % args.save_freq == 0 or epoch == args.epochs):
            model_path = save_model_ckpt(args, best_records,
                                        model, classifier_head, optimizer, scheduler, logit_scale,
                                         val_acc, epoch, num_iter)
            logger.info(f'Model ckpt saved to: {model_path}')

        if epoch == args.stop_epochs:
            break

    logger.info(f'Training done.')
    logger.info(f'Best val Acc: {round(best_val_acc, 3)} at epoch {best_epoch}, iter {best_iter}')

    return best_model, best_head, best_records, best_logit_scale


def train_cutmix_fs(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader):
    """
    Train the model with Cross-Entropy Loss, finetuning visual encoder and classifier
    Use CutMix method to augment the training data
    two dataloader, one from the mixed, one from the few-shot only
    """

    train_loader, train_dataloader_fs = train_loader
    train_loader_fs = iter(train_dataloader_fs)

    logger.info(f"Start Training cutmix-fs ......")

    model.eval() if args.freeze_visual else model.train()
    classifier_head.train()

    logit_scale = args.logit_scale
    loss = args.loss
    optimizer = args.optimizer
    scheduler = args.scheduler

    best_val_acc = -1
    best_epoch = -1
    best_iter = -1
    best_model = model
    best_head = classifier_head
    best_logit_scale = logit_scale
    best_records = {}
    num_iter = 0
    for epoch in range(1, args.epochs+1):

        train_loss_sum = 0
        # train for 1 epoch
        for inputs, labels, text, source in train_loader:
            num_iter += 1
            images = inputs.to(args.device)
            labels = labels.to(args.device).long()

            # get a batch of few-shot data, when the few-shot data is exhausted, just loop back
            try:
                inputs_fs, labels_fs, text_fs, source_fs = next(train_loader_fs)
            except StopIteration:
                train_loader_fs = iter(train_dataloader_fs)
                inputs_fs, labels_fs, text_fs, source_fs = next(train_loader_fs)
            images_fs = inputs_fs.to(args.device)
            labels_fs = labels_fs.to(args.device).long()

            # apply the cutmix strategy
            r = np.random.rand(1)
            if args.cutmix_beta > 0 and r < args.mix_prob:
                # generate mixed sample
                lam = np.random.beta(args.cutmix_beta, args.cutmix_beta)
                target_a = labels
                target_b = labels_fs
                bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
                images[:, :, bbx1:bbx2, bby1:bby2] = images_fs[:, :, bbx1:bbx2, bby1:bby2]
                # adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
            else:
                target_a = labels
                target_b = labels_fs
                lam = 1.0

            image_features = model.encode_image(images)
            image_feature = image_features / image_features.norm(dim=-1, keepdim=True)

            logits = classifier_head(image_feature)
            logits = logits * logit_scale.exp()

            # mixup loss
            total_loss = mixup_criterion(loss, logits, target_a, target_b, lam)

            train_loss = total_loss.item()
            train_loss_sum += train_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step() # update learning rate for each iteration

        # validate after 1 epoch
        # if (args.early_stop and num_iter >= args.start_validation) or \
        #     epoch == args.epochs:
        val_acc, val_loss, confusion_matrix = validate(args, data_loader=val_loader, model=model, logger=logger,
                                                        loss=args.loss, logit_scale=logit_scale,
                                                        classifier_head=classifier_head, show_confusion_matrix=True,
                                                        dataset=args.dataset,
                                                        output_dir=args.output_dir, device=args.device,
                                                        pre_extracted=args.pre_extracted,
                                                        )
        scores = calculate_scores(confusion_matrix)

        # check if val acc has improved, here i use the val_acc rather than val_loss,
        # as we might modify the loss function for confusing pairs
        if (args.early_stop or epoch == args.epochs) and val_acc >= best_val_acc:
            # logger.info(f'Val Acc improved from {round(best_val_acc, 3)} to {round(val_acc, 3)}')
            best_val_acc = val_acc
            best_logit_scale = copy.deepcopy(logit_scale)
            best_epoch = epoch
            best_iter = num_iter
            best_scores = copy.deepcopy(scores)
            best_confusion_matrix = copy.deepcopy(confusion_matrix)
            best_head = copy.deepcopy(classifier_head)
            best_model = copy.deepcopy(model)

            # save into the best_records
            best_records['best_val_acc'] = best_val_acc
            best_records['best_logit_scale'] = best_logit_scale
            best_records['best_epoch'] = best_epoch
            best_records['best_iter'] = best_iter
            best_records['best_scores'] = best_scores
            best_records['best_confusion_matrix'] = best_confusion_matrix

        # test after 1 epoch
        test_acc, _, _ = validate(args,data_loader=test_loader, model=model, logger=logger,
                                loss=args.loss, logit_scale=logit_scale,
                                classifier_head=classifier_head,
                                dataset=args.dataset,
                                output_dir=args.output_dir, device=args.device,
                                pre_extracted=args.pre_extracted,
                                )

        train_loss_avg = train_loss_sum / len(train_loader)
        loss_logger.write(f'{epoch},{num_iter},{round(train_loss_avg, 6)},{round(val_loss, 6)},{round(val_acc, 6)},{round(test_acc, 6)}\n')
        loss_logger.flush()
        logger.info(f"Epoch {int(epoch)}, Iter {num_iter}, Trn Loss: {round(train_loss_avg, 6)}, Val Loss: {round(val_loss, 6)}, Val Acc: {round(val_acc, 3)}, Test Acc: {round(test_acc, 3)}")

        ## save model checkpoints every X epochs
        if args.save_ckpt and (num_iter % args.save_freq == 0 or epoch == args.epochs):
            model_path = save_model_ckpt(args, best_records,
                                        model, classifier_head, optimizer, scheduler, logit_scale,
                                         val_acc, epoch, num_iter)
            logger.info(f'Model ckpt saved to: {model_path}')

        if epoch == args.stop_epochs:
            break

    logger.info(f'Training done.')
    logger.info(f'Best val Acc: {round(best_val_acc, 3)} at epoch {best_epoch}, iter {best_iter}')

    return best_model, best_head, best_records, best_logit_scale


def train_cutmix_fs2(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader):
    """
    CutMix implementation with few-shot ratio

    Train the model with Cross-Entropy Loss, finetuning visual encoder and classifier
    Use CutMix method to augment the training data
    two dataloader, one from the mixed, one from the few-shot only
    """

    train_loader, train_dataloader_fs = train_loader
    train_loader_fs = iter(train_dataloader_fs)

    logger.info(f"Start Training cutmix-fs ......")

    model.eval() if args.freeze_visual else model.train()
    classifier_head.train()

    logit_scale = args.logit_scale
    loss = args.loss
    optimizer = args.optimizer
    scheduler = args.scheduler

    best_val_acc = -1
    best_epoch = -1
    best_iter = -1
    best_model = model
    best_head = classifier_head
    best_logit_scale = logit_scale
    best_records = {}
    num_iter = 0
    for epoch in range(1, args.epochs+1):

        train_loss_sum = 0
        # train for 1 epoch
        for inputs, labels, text, source in train_loader:
            num_iter += 1
            images = inputs.to(args.device)
            labels = labels.to(args.device).long()

            # get a batch of few-shot data, when the few-shot data is exhausted, just loop back
            try:
                inputs_fs, labels_fs, text_fs, source_fs = next(train_loader_fs)
            except StopIteration:
                train_loader_fs = iter(train_dataloader_fs)
                inputs_fs, labels_fs, text_fs, source_fs = next(train_loader_fs)
            images_fs = inputs_fs.to(args.device)
            labels_fs = labels_fs.to(args.device).long()

            # concatenate the images and images_fs
            images = torch.cat([images, images_fs], dim=0)
            labels = torch.cat([labels, labels_fs], dim=0)

            # apply the cutmix strategy
            # r = np.random.rand(1)
            # if args.cutmix_beta > 0 and r < args.mix_prob:
            #     # generate mixed sample
            #     lam = np.random.beta(args.cutmix_beta, args.cutmix_beta)
            #     target_a = labels
            #     target_b = labels_fs
            #     bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
            #     images[:, :, bbx1:bbx2, bby1:bby2] = images[:, :, bbx1:bbx2, bby1:bby2]
            #     # adjust lambda to exactly match pixel ratio
            #     lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
            # else:
            #     target_a = labels
            #     target_b = labels_fs
            #     lam = 1.0

            r = np.random.rand(1)
            if args.cutmix_beta > 0 and r < args.mix_prob:
                # generate mixed sample
                lam = np.random.beta(args.cutmix_beta, args.cutmix_beta)
                rand_index = torch.randperm(images.size()[0]).cuda()
                target_a = labels
                target_b = labels[rand_index]
                bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
                images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
                # adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
            else:
                target_a = labels
                target_b = labels
                lam = 1.0

            image_features = model.encode_image(images)
            image_feature = image_features / image_features.norm(dim=-1, keepdim=True)

            logits = classifier_head(image_feature)
            logits = logits * logit_scale.exp()

            # mixup loss
            total_loss = mixup_criterion(loss, logits, target_a, target_b, lam)

            train_loss = total_loss.item()
            train_loss_sum += train_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step() # update learning rate for each iteration

        # validate after 1 epoch
        # if (args.early_stop and num_iter >= args.start_validation) or \
        #     epoch == args.epochs:
        val_acc, val_loss, confusion_matrix = validate(args, data_loader=val_loader, model=model, logger=logger,
                                                        loss=args.loss, logit_scale=logit_scale,
                                                        classifier_head=classifier_head, show_confusion_matrix=True,
                                                        dataset=args.dataset,
                                                        output_dir=args.output_dir, device=args.device,
                                                        pre_extracted=args.pre_extracted,
                                                        )
        scores = calculate_scores(confusion_matrix)

        # check if val acc has improved, here i use the val_acc rather than val_loss,
        # as we might modify the loss function for confusing pairs
        if (args.early_stop or epoch == args.epochs) and val_acc >= best_val_acc:
            # logger.info(f'Val Acc improved from {round(best_val_acc, 3)} to {round(val_acc, 3)}')
            best_val_acc = val_acc
            best_logit_scale = copy.deepcopy(logit_scale)
            best_epoch = epoch
            best_iter = num_iter
            best_scores = copy.deepcopy(scores)
            best_confusion_matrix = copy.deepcopy(confusion_matrix)
            best_head = copy.deepcopy(classifier_head)
            best_model = copy.deepcopy(model)

            # save into the best_records
            best_records['best_val_acc'] = best_val_acc
            best_records['best_logit_scale'] = best_logit_scale
            best_records['best_epoch'] = best_epoch
            best_records['best_iter'] = best_iter
            best_records['best_scores'] = best_scores
            best_records['best_confusion_matrix'] = best_confusion_matrix

        # test after 1 epoch
        test_acc, _, _ = validate(args,data_loader=test_loader, model=model, logger=logger,
                                loss=args.loss, logit_scale=logit_scale,
                                classifier_head=classifier_head,
                                dataset=args.dataset,
                                output_dir=args.output_dir, device=args.device,
                                pre_extracted=args.pre_extracted,
                                )

        train_loss_avg = train_loss_sum / len(train_loader)
        loss_logger.write(f'{epoch},{num_iter},{round(train_loss_avg, 6)},{round(val_loss, 6)},{round(val_acc, 6)},{round(test_acc, 6)}\n')
        loss_logger.flush()
        logger.info(f"Epoch {int(epoch)}, Iter {num_iter}, Trn Loss: {round(train_loss_avg, 6)}, Val Loss: {round(val_loss, 6)}, Val Acc: {round(val_acc, 3)}, Test Acc: {round(test_acc, 3)}")

        ## save model checkpoints every X epochs
        if args.save_ckpt and (num_iter % args.save_freq == 0 or epoch == args.epochs):
            model_path = save_model_ckpt(args, best_records,
                                        model, classifier_head, optimizer, scheduler, logit_scale,
                                         val_acc, epoch, num_iter)
            logger.info(f'Model ckpt saved to: {model_path}')

        if epoch == args.stop_epochs:
            break

    logger.info(f'Training done.')
    logger.info(f'Best val Acc: {round(best_val_acc, 3)} at epoch {best_epoch}, iter {best_iter}')

    return best_model, best_head, best_records, best_logit_scale


def train_CMO(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader):
    """
    Train the model with Cross-Entropy Loss, finetuning visual encoder and classifier
    Use CMO method to augment the training data
    """
    train_loader, weighted_train_loader = train_loader
    inverse_iter = iter(weighted_train_loader)

    logger.info(f"Start Training CMO ......")

    model.eval() if args.freeze_visual else model.train()
    classifier_head.train()

    logit_scale = args.logit_scale
    loss = args.loss
    optimizer = args.optimizer
    scheduler = args.scheduler

    best_val_acc = -1
    best_epoch = -1
    best_iter = -1
    best_model = model
    best_head = classifier_head
    best_logit_scale = logit_scale
    best_records = {}
    num_iter = 0
    for epoch in range(1, args.epochs+1):

        train_loss_sum = 0
        # train for 1 epoch
        for inputs, labels, text, source in train_loader:

            try:
                inputs2, targets2, text2, source2 = next(inverse_iter)
            except:
                inverse_iter = iter(weighted_train_loader)
                inputs2, targets2, text2, source2 = next(inverse_iter)

            inputs2 = inputs2.to(args.device)
            targets2 = targets2.to(args.device).long()

            num_iter += 1
            images = inputs.to(args.device)
            labels = labels.to(args.device).long()

            # apply the cutmix strategy
            r = np.random.rand(1)
            if args.cutmix_beta > 0 and r < args.mix_prob:
                # generate mixed sample
                lam = np.random.beta(args.cutmix_beta, args.cutmix_beta)
                bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
                images[:, :, bbx1:bbx2, bby1:bby2] = inputs2[:, :, bbx1:bbx2, bby1:bby2]

                # adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
                target_a = labels
                target_b = targets2

            else:
                target_a = labels
                target_b = labels
                lam = 1.0

            image_features = model.encode_image(images)
            image_feature = image_features / image_features.norm(dim=-1, keepdim=True)

            logits = classifier_head(image_feature)
            logits = logits * logit_scale.exp()

            # mixup loss
            total_loss = mixup_criterion(loss, logits, target_a, target_b, lam)

            train_loss = total_loss.item()
            train_loss_sum += train_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step() # update learning rate for each iteration

        # validate
        if args.early_stop or epoch == args.epochs:
            val_acc, val_loss, confusion_matrix = validate(args, data_loader=val_loader, model=model, logger=logger,
                                                        loss=args.loss, logit_scale=logit_scale,
                                                        classifier_head=classifier_head, show_confusion_matrix=True,
                                                        dataset=args.dataset,
                                                        output_dir=args.output_dir, device=args.device,
                                                        pre_extracted=args.pre_extracted,
                                                        )
            scores = calculate_scores(confusion_matrix)

        # check if val acc has improved, here i use the val_acc rather than val_loss,
        # as we might modify the loss function for confusing pairs
        if (args.early_stop or epoch == args.epochs) and val_acc >= best_val_acc:
            # logger.info(f'Val Acc improved from {round(best_val_acc, 3)} to {round(val_acc, 3)}')
            best_val_acc = val_acc
            best_logit_scale = copy.deepcopy(logit_scale)
            best_epoch = epoch
            best_iter = num_iter
            best_scores = copy.deepcopy(scores)
            best_confusion_matrix = copy.deepcopy(confusion_matrix)
            best_head = copy.deepcopy(classifier_head)
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
            test_acc, _, _ = validate(args,data_loader=test_loader, model=model, logger=logger,
                                loss=args.loss, logit_scale=logit_scale,
                                classifier_head=classifier_head,
                                dataset=args.dataset,
                                output_dir=args.output_dir, device=args.device,
                                pre_extracted=args.pre_extracted,
                                )

        train_loss_avg = train_loss_sum / len(train_loader)
        loss_logger.write(f'{epoch},{num_iter},{round(train_loss_avg, 6)},{round(val_loss, 6)},{round(val_acc, 6)},{round(test_acc, 6)}\n')
        loss_logger.flush()
        logger.info(f"Epoch {int(epoch)}, Iter {num_iter}, Trn Loss: {round(train_loss_avg, 6)}, Val Loss: {round(val_loss, 6)}, Val Acc: {round(val_acc, 3)}, Test Acc: {round(test_acc, 3)}")

        ## save model checkpoints every X epochs
        if args.save_ckpt and (num_iter % args.save_freq == 0 or epoch == args.epochs):
            model_path = save_model_ckpt(args, best_records,
                                        model, classifier_head, optimizer, scheduler, logit_scale,
                                         val_acc, epoch, num_iter)
            logger.info(f'Model ckpt saved to: {model_path}')

        if epoch == args.stop_epochs:
            break

    logger.info(f'Training done.')
    logger.info(f'Best val Acc: {round(best_val_acc, 3)} at epoch {best_epoch}, iter {best_iter}')

    return best_model, best_head, best_records, best_logit_scale

def saliency_bbox(img, lam):
    size = img.size()
    W = size[1]
    H = size[2]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # initialize OpenCV's static fine grained saliency detector and compute the saliency map
    img_arr = img.cpu().numpy().transpose(1, 2, 0)
    img_arr = (img_arr - img_arr.min()) / (img_arr.max() - img_arr.min())
    img_arr = (img_arr * 255).astype(np.uint8)
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    (success, saliencyMap) = saliency.computeSaliency(img_arr)
    saliencyMap = (saliencyMap * 255).astype("uint8")

    maximum_indices = np.unravel_index(np.argmax(saliencyMap, axis=None), saliencyMap.shape)
    x = maximum_indices[0]
    y = maximum_indices[1]

    # centered around the peak saliency
    bbx1 = np.clip(x - cut_w // 2, 0, W)
    bby1 = np.clip(y - cut_h // 2, 0, H)
    bbx2 = np.clip(x + cut_w // 2, 0, W)
    bby2 = np.clip(y + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def train_saliencymix(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader):
    """
    Train the model with Cross-Entropy Loss, finetuning visual encoder and classifier
    Use SaliencyMix method to augment the training data
    """

    logger.info(f"Start Training ......")

    model.eval() if args.freeze_visual else model.train()
    classifier_head.train()

    best_val_acc = -1
    best_epoch = -1
    best_iter = -1
    best_model = model
    best_head = classifier_head
    best_logit_scale = logit_scale
    best_records = {}
    num_iter = 0
    for epoch in range(1, args.epochs+1):

        train_loss_sum = 0
        # train for 1 epoch
        for inputs, labels, text, source in train_loader:
            num_iter += 1
            images = inputs.to(args.device)
            labels = labels.to(args.device).long()

            # apply the saliencymix strategy
            r = np.random.rand(1)
            if args.cutmix_beta > 0 and r < args.mix_prob:
                # generate mixed sample
                lam = np.random.beta(args.cutmix_beta, args.cutmix_beta)
                rand_index = torch.randperm(images.size()[0]).cuda()
                target_a = labels
                target_b = labels[rand_index]

                # old implementation
                bbx1, bby1, bbx2, bby2 = saliency_bbox(images[rand_index[0]], lam)
                images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]

                # adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
            else:
                target_a = labels
                target_b = labels
                lam = 1.0

            image_features = model.encode_image(images)
            image_feature = image_features / image_features.norm(dim=-1, keepdim=True)

            logits = classifier_head(image_feature)
            logits = logits * logit_scale.exp()

            # mixup loss
            total_loss = mixup_criterion(loss, logits, target_a, target_b, lam)

            train_loss = total_loss.item()
            train_loss_sum += train_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step() # update learning rate for each iteration

        # validate after 1 epoch
        # if (args.early_stop and num_iter >= args.start_validation) or \
        #     epoch == args.epochs:
        val_acc, val_loss, confusion_matrix = validate(args, data_loader=val_loader, model=model, logger=logger,
                                                        loss=args.loss, logit_scale=logit_scale,
                                                        classifier_head=classifier_head, show_confusion_matrix=True,
                                                        dataset=args.dataset,
                                                        output_dir=args.output_dir, device=args.device,
                                                        pre_extracted=args.pre_extracted,
                                                        )
        scores = calculate_scores(confusion_matrix)

        # check if val acc has improved, here i use the val_acc rather than val_loss,
        # as we might modify the loss function for confusing pairs
        if (args.early_stop or epoch == args.epochs) and val_acc >= best_val_acc:
            # logger.info(f'Val Acc improved from {round(best_val_acc, 3)} to {round(val_acc, 3)}')
            best_val_acc = val_acc
            best_logit_scale = copy.deepcopy(logit_scale)
            best_epoch = epoch
            best_iter = num_iter
            best_scores = copy.deepcopy(scores)
            best_confusion_matrix = copy.deepcopy(confusion_matrix)
            best_head = copy.deepcopy(classifier_head)
            best_model = copy.deepcopy(model)

            # save into the best_records
            best_records['best_val_acc'] = best_val_acc
            best_records['best_logit_scale'] = best_logit_scale
            best_records['best_epoch'] = best_epoch
            best_records['best_iter'] = best_iter
            best_records['best_scores'] = best_scores
            best_records['best_confusion_matrix'] = best_confusion_matrix

        # test after 1 epoch
        test_acc, _, _ = validate(args,data_loader=test_loader, model=model, logger=logger,
                                loss=args.loss, logit_scale=logit_scale,
                                classifier_head=classifier_head,
                                dataset=args.dataset,
                                output_dir=args.output_dir, device=args.device,
                                pre_extracted=args.pre_extracted,
                                )

        train_loss_avg = train_loss_sum / len(train_loader)
        loss_logger.write(f'{epoch},{num_iter},{round(train_loss_avg, 6)},{round(val_loss, 6)},{round(val_acc, 6)},{round(test_acc, 6)}\n')
        loss_logger.flush()
        logger.info(f"Epoch {int(epoch)}, Iter {num_iter}, Trn Loss: {round(train_loss_avg, 6)}, Val Loss: {round(val_loss, 6)}, Val Acc: {round(val_acc, 3)}, Test Acc: {round(test_acc, 3)}")

        ## save model checkpoints every X epochs
        if args.save_ckpt and (num_iter % args.save_freq == 0 or epoch == args.epochs):
            model_path = save_model_ckpt(args, best_records,
                                        model, classifier_head, optimizer, scheduler, logit_scale,
                                         val_acc, epoch, num_iter)
            logger.info(f'Model ckpt saved to: {model_path}')

        if epoch == args.stop_epochs:
            break

    logger.info(f'Training done.')
    logger.info(f'Best val Acc: {round(best_val_acc, 3)} at epoch {best_epoch}, iter {best_iter}')

    return best_model, best_head, best_records, best_logit_scale

def saliency_bbox_list(imgs, lam):
    corners_list = []
    for img in imgs:
        bbx1, bby1, bbx2, bby2 = saliency_bbox(img, lam)
        corners_list.append([bbx1, bby1, bbx2, bby2])
    return corners_list

def train_saliencymix2(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader):
    """
    Train the model with Cross-Entropy Loss, finetuning visual encoder and classifier
    Use SaliencyMix method to augment the training data
    +++++ here we compute the saliency for each image and apply the saliency to the image +++++
    """

    logger.info(f"Start Training saliencymix2 ......")

    model.eval() if args.freeze_visual else model.train()
    classifier_head.train()

    logit_scale = args.logit_scale
    loss = args.loss
    optimizer = args.optimizer
    scheduler = args.scheduler

    best_val_acc = -1
    best_epoch = -1
    best_iter = -1
    best_model = model
    best_head = classifier_head
    best_logit_scale = logit_scale
    best_records = {}
    num_iter = 0
    for epoch in range(1, args.epochs+1):

        train_loss_sum = 0
        # train for 1 epoch
        for inputs, labels, text, source in train_loader:
            num_iter += 1
            images = inputs.to(args.device)
            labels = labels.to(args.device).long()

            # apply the saliencymix strategy
            r = np.random.rand(1)
            if args.cutmix_beta > 0 and r < args.mix_prob:
                # generate mixed sample
                lam = np.random.beta(args.cutmix_beta, args.cutmix_beta)
                rand_index = torch.randperm(images.size()[0]).cuda()
                target_a = labels
                target_b = labels[rand_index]
                shuffled_images = images[rand_index]

                corners_list = saliency_bbox_list(shuffled_images, lam)
                lam_list = []

                for i in range(images.size(0)):
                    bbx1, bby1, bbx2, bby2 = corners_list[i]
                    images[i, :, bbx1:bbx2, bby1:bby2] = shuffled_images[i, :, bbx1:bbx2, bby1:bby2]
                    # adjust lambda to exactly match pixel ratio
                    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
                    lam_list.append(lam)

            else:
                target_a = labels
                target_b = labels
                lam_list = [1.0] * images.size(0)

            image_features = model.encode_image(images)
            image_feature = image_features / image_features.norm(dim=-1, keepdim=True)

            logits = classifier_head(image_feature)
            logits = logits * logit_scale.exp()

            # mixup loss
            total_loss = mixup_criterion_lam_list(loss, logits, target_a, target_b, lam_list)

            train_loss = total_loss.item()
            train_loss_sum += train_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step() # update learning rate for each iteration

        # validate after 1 epoch
        # if (args.early_stop and num_iter >= args.start_validation) or \
        #     epoch == args.epochs:
        if args.early_stop or epoch == args.epochs:
            val_acc, val_loss, confusion_matrix = validate(args, data_loader=val_loader, model=model, logger=logger,
                                                        loss=args.loss, logit_scale=logit_scale,
                                                        classifier_head=classifier_head, show_confusion_matrix=True,
                                                        dataset=args.dataset,
                                                        output_dir=args.output_dir, device=args.device,
                                                        pre_extracted=args.pre_extracted,
                                                        )
            scores = calculate_scores(confusion_matrix)

        # check if val acc has improved, here i use the val_acc rather than val_loss,
        # as we might modify the loss function for confusing pairs
        if (args.early_stop or epoch == args.epochs) and val_acc >= best_val_acc:
            # logger.info(f'Val Acc improved from {round(best_val_acc, 3)} to {round(val_acc, 3)}')
            best_val_acc = val_acc
            best_logit_scale = copy.deepcopy(logit_scale)
            best_epoch = epoch
            best_iter = num_iter
            best_scores = copy.deepcopy(scores)
            best_confusion_matrix = copy.deepcopy(confusion_matrix)
            best_head = copy.deepcopy(classifier_head)
            best_model = copy.deepcopy(model)

            # save into the best_records
            best_records['best_val_acc'] = best_val_acc
            best_records['best_logit_scale'] = best_logit_scale
            best_records['best_epoch'] = best_epoch
            best_records['best_iter'] = best_iter
            best_records['best_scores'] = best_scores
            best_records['best_confusion_matrix'] = best_confusion_matrix

        # test after 1 epoch
        test_acc, _, _ = validate(args,data_loader=test_loader, model=model, logger=logger,
                                loss=args.loss, logit_scale=logit_scale,
                                classifier_head=classifier_head,
                                dataset=args.dataset,
                                output_dir=args.output_dir, device=args.device,
                                pre_extracted=args.pre_extracted,
                                )

        train_loss_avg = train_loss_sum / len(train_loader)
        loss_logger.write(f'{epoch},{num_iter},{round(train_loss_avg, 6)},{round(val_loss, 6)},{round(val_acc, 6)},{round(test_acc, 6)}\n')
        loss_logger.flush()
        logger.info(f"Epoch {int(epoch)}, Iter {num_iter}, Trn Loss: {round(train_loss_avg, 6)}, Val Loss: {round(val_loss, 6)}, Val Acc: {round(val_acc, 3)}, Test Acc: {round(test_acc, 3)}")

        ## save model checkpoints every X epochs
        if args.save_ckpt and (num_iter % args.save_freq == 0 or epoch == args.epochs):
            model_path = save_model_ckpt(args, best_records,
                                        model, classifier_head, optimizer, scheduler, logit_scale,
                                         val_acc, epoch, num_iter)
            logger.info(f'Model ckpt saved to: {model_path}')

        if epoch == args.stop_epochs:
            break

    logger.info(f'Training done.')
    logger.info(f'Best val Acc: {round(best_val_acc, 3)} at epoch {best_epoch}, iter {best_iter}')

    return best_model, best_head, best_records, best_logit_scale


def train_resizemix(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader):
    """
    Train the model with Cross-Entropy Loss, finetuning visual encoder and classifier
    Use ResizeMix method to augment the training data
    """

    logger.info(f"Start Training resizemix ......")

    model.eval() if args.freeze_visual else model.train()
    classifier_head.train()

    logit_scale = args.logit_scale
    loss = args.loss
    optimizer = args.optimizer
    scheduler = args.scheduler

    best_val_acc = -1
    best_epoch = -1
    best_iter = -1
    best_model = model
    best_head = classifier_head
    best_logit_scale = logit_scale
    best_records = {}
    num_iter = 0
    for epoch in range(1, args.epochs+1):

        train_loss_sum = 0
        # train for 1 epoch
        for inputs, labels, text, source in train_loader:
            num_iter += 1
            images = inputs.to(args.device)
            labels = labels.to(args.device).long()

            # apply the resizemix strategy
            r = np.random.rand(1)
            if r < args.mix_prob:
                # generate mixed sample
                # uniform sampling the lambda from [0.1, 0.8]
                tau = np.random.uniform(0.1, 0.8)

                rand_index = torch.randperm(images.size()[0]).cuda()
                shuffled_images = images[rand_index]
                # print('shuffled_images.size()', shuffled_images.size())
                # resize the shuffled_images to a smaller size of the original images, with scale tau
                resized_images = F.interpolate(shuffled_images, scale_factor=tau, mode='bilinear', align_corners=False)
                # print('resized_images.size()', resized_images.size())

                # get the size of the resized_images
                resized_w = resized_images.size()[-1]
                resized_h = resized_images.size()[-2]

                # get the random position to paste the resized_images
                pos_x = np.random.randint(0, images.size()[-1] - resized_w)
                pos_y = np.random.randint(0, images.size()[-2] - resized_h)

                # paste the resized_images to the original images
                images[:, :, pos_x:pos_x+resized_w, pos_y:pos_y+resized_h] = resized_images

                # adjust lambda to exactly match pixel ratio
                lam = 1.0 - (resized_w * resized_h / (images.size()[-1] * images.size()[-2]))

                # labels
                target_a = labels
                target_b = labels[rand_index]
            else:
                target_a = labels
                target_b = labels
                lam = 1.0

            image_features = model.encode_image(images)
            image_feature = image_features / image_features.norm(dim=-1, keepdim=True)

            logits = classifier_head(image_feature)
            logits = logits * logit_scale.exp()

            # mixup loss, lam belongs to target_a
            total_loss = mixup_criterion(loss, logits, target_a, target_b, lam)

            train_loss = total_loss.item()
            train_loss_sum += train_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step() # update learning rate for each iteration

        # validate after 1 epoch
        # if (args.early_stop and num_iter >= args.start_validation) or \
        #     epoch == args.epochs:
        val_acc, val_loss, confusion_matrix = validate(args, data_loader=val_loader, model=model, logger=logger,
                                                        loss=args.loss, logit_scale=logit_scale,
                                                        classifier_head=classifier_head, show_confusion_matrix=True,
                                                        dataset=args.dataset,
                                                        output_dir=args.output_dir, device=args.device,
                                                        pre_extracted=args.pre_extracted,
                                                        )
        scores = calculate_scores(confusion_matrix)

        # check if val acc has improved, here i use the val_acc rather than val_loss,
        # as we might modify the loss function for confusing pairs
        if (args.early_stop or epoch == args.epochs) and val_acc >= best_val_acc:
            # logger.info(f'Val Acc improved from {round(best_val_acc, 3)} to {round(val_acc, 3)}')
            best_val_acc = val_acc
            best_logit_scale = copy.deepcopy(logit_scale)
            best_epoch = epoch
            best_iter = num_iter
            best_scores = copy.deepcopy(scores)
            best_confusion_matrix = copy.deepcopy(confusion_matrix)
            best_head = copy.deepcopy(classifier_head)
            best_model = copy.deepcopy(model)

            # save into the best_records
            best_records['best_val_acc'] = best_val_acc
            best_records['best_logit_scale'] = best_logit_scale
            best_records['best_epoch'] = best_epoch
            best_records['best_iter'] = best_iter
            best_records['best_scores'] = best_scores
            best_records['best_confusion_matrix'] = best_confusion_matrix

        # test after 1 epoch
        test_acc, _, _ = validate(args,data_loader=test_loader, model=model, logger=logger,
                                loss=args.loss, logit_scale=logit_scale,
                                classifier_head=classifier_head,
                                dataset=args.dataset,
                                output_dir=args.output_dir, device=args.device,
                                pre_extracted=args.pre_extracted,
                                )

        train_loss_avg = train_loss_sum / len(train_loader)
        loss_logger.write(f'{epoch},{num_iter},{round(train_loss_avg, 6)},{round(val_loss, 6)},{round(val_acc, 6)},{round(test_acc, 6)}\n')
        loss_logger.flush()
        logger.info(f"Epoch {int(epoch)}, Iter {num_iter}, Trn Loss: {round(train_loss_avg, 6)}, Val Loss: {round(val_loss, 6)}, Val Acc: {round(val_acc, 3)}, Test Acc: {round(test_acc, 3)}")

        ## save model checkpoints every X epochs
        if args.save_ckpt and (num_iter % args.save_freq == 0 or epoch == args.epochs):
            model_path = save_model_ckpt(args, best_records,
                                        model, classifier_head, optimizer, scheduler, logit_scale,
                                         val_acc, epoch, num_iter)
            logger.info(f'Model ckpt saved to: {model_path}')

        if epoch == args.stop_epochs:
            break

    logger.info(f'Training done.')
    logger.info(f'Best val Acc: {round(best_val_acc, 3)} at epoch {best_epoch}, iter {best_iter}')

    return best_model, best_head, best_records, best_logit_scale


def get_GEM_heatmap(gem_model, images, texts):
    heatmap_list = gem_model.batched_forward(images, texts)
    return heatmap_list


def saliency_bbox_gem(heatmap, lam):
    # convert heatmap from size [1, W, H] to [W, H]
    # detach heatmap to cpu first
    # print('heatmap.size()', heatmap.size())
    heatmap= heatmap.squeeze(0).cpu()

    # print('heatmap.size()', heatmap.size())
    # convert heatmap from torch tensor to numpy array
    heatmap = heatmap.detach().numpy()

    size = heatmap.shape
    W = size[0]
    H = size[1]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # use heatmap asthe saliencymap
    maximum_indices = np.unravel_index(np.argmax(heatmap, axis=None), heatmap.shape)
    x = maximum_indices[0]
    y = maximum_indices[1]

    # centered around the peak saliency
    bbx1 = np.clip(x - cut_w // 2, 0, W)
    bby1 = np.clip(y - cut_h // 2, 0, H)
    bbx2 = np.clip(x + cut_w // 2, 0, W)
    bby2 = np.clip(y + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def saliency_bbox_list_gem(shuffled_images, lam, gem_model, texts):

    corners_list = []
    heatmap_list = get_GEM_heatmap(gem_model, shuffled_images, texts)
    # print(shuffled_images[0].size())
    # print(heatmap_list[0].size())
    for heatmap in heatmap_list:
        bbx1, bby1, bbx2, bby2 = saliency_bbox_gem(heatmap, lam)
        corners_list.append([bbx1, bby1, bbx2, bby2])
    return corners_list

def train_attentivemix(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader):
    """
    Train the model with Cross-Entropy Loss, finetuning visual encoder and classifier
    Use AttentiveMix method to augment the training data
    """

    # initialize GEM model
    model_name = 'ViT-B/32'
    pretrained = 'laion400m_e32'
    gem_model = create_gem_model(model_name=model_name, pretrained=pretrained, device=args.device)
    gem_model.eval()
    logger.info(f'GEM model loaded from {model_name} {pretrained}')
    threshold = args.attentive_threshold

    # get the label names dict
    metric_fn = f'{args.dataset_root}/id_scname_dict.json'
    with open(metric_fn, 'r') as f:
        metrics = json.load(f)

    logger.info(f"Start Training ......")

    model.eval() if args.freeze_visual else model.train()
    classifier_head.train()

    best_val_acc = -1
    best_epoch = -1
    best_iter = -1
    best_model = model
    best_head = classifier_head
    best_logit_scale = logit_scale
    best_records = {}
    num_iter = 0
    for epoch in range(1, args.epochs+1):

        train_loss_sum = 0
        # train for 1 epoch
        for inputs, labels, text, source in train_loader:
            num_iter += 1
            images = inputs.to(args.device)
            labels = labels.to(args.device).long()

            # apply the attentivemix strategy
            r = np.random.rand(1)
            if r < args.mix_prob:

                # generate mixed sample
                rand_index = torch.randperm(images.size()[0]).cuda()
                target_a = labels
                target_b = labels[rand_index]
                shuffled_images = images[rand_index]

                # use general domain name here
                if args.attentive_name == 'general':
                    texts = [['bird'] for i in range(images.size()[0])]
                elif args.attentive_name == 'c-name':
                    # use c-name to get the heatmap
                    texts = []
                    for label in target_b:
                        texts.append([metrics[str(label.item())][1]])
                elif args.attentive_name == 's-name':
                    texts = []
                    for label in target_b:
                        texts.append([metrics[str(label.item())][0]])
                # print(target_b)
                # print(texts)

                heatmap_list = get_GEM_heatmap(gem_model, shuffled_images, texts)

                # get the binary_mask_list using the threshold
                binary_mask_list = []
                for heatmap in heatmap_list:
                    binary_mask = (heatmap > threshold).int()
                    binary_mask_list.append(binary_mask)

                # build the new image by attentively mixing the images usingt he binary_mask
                lam_list = []
                for i, binary_mask in enumerate(binary_mask_list):
                    images[i] = images[i] * (1 - binary_mask) + shuffled_images[i] * binary_mask
                    lam = 1.0 - binary_mask.sum() / (images.size()[-1] * images.size()[-2])
                    lam_list.append(lam)
            else:
                target_a = labels
                target_b = labels
                lam_list = [1.0] * images.size(0)

            # print('lam_list', lam_list)

            image_features = model.encode_image(images)
            image_feature = image_features / image_features.norm(dim=-1, keepdim=True)

            logits = classifier_head(image_feature)
            logits = logits * logit_scale.exp()

            # mixup loss, lam belongs to target_a
            total_loss = mixup_criterion_lam_list(loss, logits, target_a, target_b, lam_list)

            train_loss = total_loss.item()
            train_loss_sum += train_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step() # update learning rate for each iteration

        # validate after 1 epoch
        # if (args.early_stop and num_iter >= args.start_validation) or \
        #     epoch == args.epochs:
        val_acc, val_loss, confusion_matrix = validate(args, data_loader=val_loader, model=model, logger=logger,
                                                        loss=args.loss, logit_scale=logit_scale,
                                                        classifier_head=classifier_head, show_confusion_matrix=True,
                                                        dataset=args.dataset,
                                                        output_dir=args.output_dir, device=args.device,
                                                        pre_extracted=args.pre_extracted,
                                                        )
        scores = calculate_scores(confusion_matrix)

        # check if val acc has improved, here i use the val_acc rather than val_loss,
        # as we might modify the loss function for confusing pairs
        if (args.early_stop or epoch == args.epochs) and val_acc >= best_val_acc:
            # logger.info(f'Val Acc improved from {round(best_val_acc, 3)} to {round(val_acc, 3)}')
            best_val_acc = val_acc
            best_logit_scale = copy.deepcopy(logit_scale)
            best_epoch = epoch
            best_iter = num_iter
            best_scores = copy.deepcopy(scores)
            best_confusion_matrix = copy.deepcopy(confusion_matrix)
            best_head = copy.deepcopy(classifier_head)
            best_model = copy.deepcopy(model)

            # save into the best_records
            best_records['best_val_acc'] = best_val_acc
            best_records['best_logit_scale'] = best_logit_scale
            best_records['best_epoch'] = best_epoch
            best_records['best_iter'] = best_iter
            best_records['best_scores'] = best_scores
            best_records['best_confusion_matrix'] = best_confusion_matrix

        # test after 1 epoch
        test_acc, _, _ = validate(args,data_loader=test_loader, model=model, logger=logger,
                                loss=args.loss, logit_scale=logit_scale,
                                classifier_head=classifier_head,
                                dataset=args.dataset,
                                output_dir=args.output_dir, device=args.device,
                                pre_extracted=args.pre_extracted,
                                )

        train_loss_avg = train_loss_sum / len(train_loader)
        loss_logger.write(f'{epoch},{num_iter},{round(train_loss_avg, 6)},{round(val_loss, 6)},{round(val_acc, 6)},{round(test_acc, 6)}\n')
        loss_logger.flush()
        logger.info(f"Epoch {int(epoch)}, Iter {num_iter}, Trn Loss: {round(train_loss_avg, 6)}, Val Loss: {round(val_loss, 6)}, Val Acc: {round(val_acc, 3)}, Test Acc: {round(test_acc, 3)}")

        ## save model checkpoints every X epochs
        if args.save_ckpt and (num_iter % args.save_freq == 0 or epoch == args.epochs):
            model_path = save_model_ckpt(args, best_records,
                                        model, classifier_head, optimizer, scheduler, logit_scale,
                                         val_acc, epoch, num_iter)
            logger.info(f'Model ckpt saved to: {model_path}')

        if epoch == args.stop_epochs:
            break

    logger.info(f'Training done.')
    logger.info(f'Best val Acc: {round(best_val_acc, 3)} at epoch {best_epoch}, iter {best_iter}')

    return best_model, best_head, best_records, best_logit_scale


def train_attentivemix2(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader):
    """
    Train the model with Cross-Entropy Loss, finetuning visual encoder and classifier
    Use AttentiveMix method to augment the training data
    Use rectangular patches
    """

    # initialize GEM model
    model_name = 'ViT-B/32'
    pretrained = 'laion400m_e32'
    gem_model = create_gem_model(model_name=model_name, pretrained=pretrained, device=args.device)
    gem_model.eval()
    logger.info(f'GEM model loaded from {model_name} {pretrained}')
    threshold = args.attentive_threshold

    # get the label names dict
    metric_fn = f'{args.dataset_root}/id_scname_dict.json'
    with open(metric_fn, 'r') as f:
        metrics = json.load(f)

    logger.info(f"Start Training ......")

    model.eval() if args.freeze_visual else model.train()
    classifier_head.train()

    best_val_acc = -1
    best_epoch = -1
    best_iter = -1
    best_model = model
    best_head = classifier_head
    best_logit_scale = logit_scale
    best_records = {}
    num_iter = 0
    for epoch in range(1, args.epochs+1):

        train_loss_sum = 0
        # train for 1 epoch
        for inputs, labels, text, source in train_loader:
            num_iter += 1
            images = inputs.to(args.device)
            labels = labels.to(args.device).long()

            # apply the attentivemix strategy
            r = np.random.rand(1)
            if args.cutmix_beta > 0 and r < args.mix_prob:
                # generate mixed sample
                lam = np.random.beta(args.cutmix_beta, args.cutmix_beta)
                rand_index = torch.randperm(images.size()[0]).cuda()
                target_a = labels
                target_b = labels[rand_index]
                shuffled_images = images[rand_index]

                # build texts for GEM
                if args.attentive_name == 'general':
                    texts = [['bird'] for i in range(images.size()[0])]
                elif args.attentive_name == 'c-name':
                    texts = []
                    for label in target_b:
                        texts.append([metrics[str(label.item())][1]])
                elif args.attentive_name == 's-name':
                    texts = []
                    for label in target_b:
                        texts.append([metrics[str(label.item())][0]])
                else:
                    raise NotImplementedError

                corners_list = saliency_bbox_list_gem(shuffled_images, lam, gem_model, texts)
                lam_list = []

                for i in range(images.size(0)):
                    bbx1, bby1, bbx2, bby2 = corners_list[i]
                    images[i, :, bbx1:bbx2, bby1:bby2] = shuffled_images[i, :, bbx1:bbx2, bby1:bby2]
                    # adjust lambda to exactly match pixel ratio
                    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
                    lam_list.append(lam)

            else:
                target_a = labels
                target_b = labels
                lam_list = [1.0] * images.size(0)

            # print('lam_list', lam_list)

            image_features = model.encode_image(images)
            image_feature = image_features / image_features.norm(dim=-1, keepdim=True)

            logits = classifier_head(image_feature)
            logits = logits * logit_scale.exp()

            # mixup loss, lam belongs to target_a
            total_loss = mixup_criterion_lam_list(loss, logits, target_a, target_b, lam_list)

            train_loss = total_loss.item()
            train_loss_sum += train_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step() # update learning rate for each iteration

        # validate after 1 epoch
        # if (args.early_stop and num_iter >= args.start_validation) or \
        #     epoch == args.epochs:
        val_acc, val_loss, confusion_matrix = validate(args, data_loader=val_loader, model=model, logger=logger,
                                                        loss=args.loss, logit_scale=logit_scale,
                                                        classifier_head=classifier_head, show_confusion_matrix=True,
                                                        dataset=args.dataset,
                                                        output_dir=args.output_dir, device=args.device,
                                                        pre_extracted=args.pre_extracted,
                                                        )
        scores = calculate_scores(confusion_matrix)

        # check if val acc has improved, here i use the val_acc rather than val_loss,
        # as we might modify the loss function for confusing pairs
        if (args.early_stop or epoch == args.epochs) and val_acc >= best_val_acc:
            # logger.info(f'Val Acc improved from {round(best_val_acc, 3)} to {round(val_acc, 3)}')
            best_val_acc = val_acc
            best_logit_scale = copy.deepcopy(logit_scale)
            best_epoch = epoch
            best_iter = num_iter
            best_scores = copy.deepcopy(scores)
            best_confusion_matrix = copy.deepcopy(confusion_matrix)
            best_head = copy.deepcopy(classifier_head)
            best_model = copy.deepcopy(model)

            # save into the best_records
            best_records['best_val_acc'] = best_val_acc
            best_records['best_logit_scale'] = best_logit_scale
            best_records['best_epoch'] = best_epoch
            best_records['best_iter'] = best_iter
            best_records['best_scores'] = best_scores
            best_records['best_confusion_matrix'] = best_confusion_matrix

        # test after 1 epoch
        test_acc, _, _ = validate(args,data_loader=test_loader, model=model, logger=logger,
                                loss=args.loss, logit_scale=logit_scale,
                                classifier_head=classifier_head,
                                dataset=args.dataset,
                                output_dir=args.output_dir, device=args.device,
                                pre_extracted=args.pre_extracted,
                                )

        train_loss_avg = train_loss_sum / len(train_loader)
        loss_logger.write(f'{epoch},{num_iter},{round(train_loss_avg, 6)},{round(val_loss, 6)},{round(val_acc, 6)},{round(test_acc, 6)}\n')
        loss_logger.flush()
        logger.info(f"Epoch {int(epoch)}, Iter {num_iter}, Trn Loss: {round(train_loss_avg, 6)}, Val Loss: {round(val_loss, 6)}, Val Acc: {round(val_acc, 3)}, Test Acc: {round(test_acc, 3)}")

        ## save model checkpoints every X epochs
        if args.save_ckpt and (num_iter % args.save_freq == 0 or epoch == args.epochs):
            model_path = save_model_ckpt(args, best_records,
                                        model, classifier_head, optimizer, scheduler, logit_scale,
                                         val_acc, epoch, num_iter)
            logger.info(f'Model ckpt saved to: {model_path}')

        if epoch == args.stop_epochs:
            break

    logger.info(f'Training done.')
    logger.info(f'Best val Acc: {round(best_val_acc, 3)} at epoch {best_epoch}, iter {best_iter}')

    return best_model, best_head, best_records, best_logit_scale