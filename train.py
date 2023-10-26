"""
-*- coding: utf-8 -*-

@Author : Aoran,Li
@Time : 2023/10/25 13:32
@File : train.py
"""
from model import *
from dataloader import *


def train_one_epoch(epoch, model, train_loader, criterion, optimizer, scheduler, params, logger):
    logger.info(f'---------- Epoch {epoch} Training ---------')
    time1 = time.time()
    metric_monitor = MetricMonitor()
    model.train()

    # pbar = tqdm(train_loader)
    for images, labels in train_loader:
        images = images.to(params['DEVICE'])
        labels = labels.to(params['DEVICE'])

        label_pre = model(images)
        loss = criterion(label_pre, labels)

        acc, f1_macro = get_accuracy_f1score(label_pre, labels)
        metric_monitor.update('Loss', loss.item())
        metric_monitor.update('f1', f1_macro)
        metric_monitor.update('acc', acc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # pbar.set_description(f'Epoch {epoch}. Train {metric_monitor}')
    logger.info(f'Epoch {epoch}. Train {metric_monitor}. Time {time.strftime("%M:%S", time.gmtime(time.time() - time1))}s')
    scheduler.step()

    return (metric_monitor.metrics['Loss']['avg'],
            metric_monitor.metrics['acc']['avg'],
            metric_monitor.metrics['f1']['avg'])


def valid_one_epoch(epoch, model, valid_loader, criterion, params, logger):
    logger.info(f'---------- Epoch {epoch} Validating ---------')
    time1 = time.time()
    metric_monitor = MetricMonitor()
    model.eval()

    # pbar = tqdm(valid_loader)
    with torch.no_grad():
        for images, labels in valid_loader:
            images = images.to(params['DEVICE'])
            labels = labels.to(params['DEVICE'])

            label_pre = model(images)
            loss = criterion(label_pre, labels)

            acc, f1_macro = get_accuracy_f1score(label_pre, labels)
            metric_monitor.update('Loss', loss.item())
            metric_monitor.update('f1', f1_macro)
            metric_monitor.update('acc', acc)

            # pbar.set_description(f'Epoch {epoch}. Valid: {metric_monitor}')
        logger.info(
            f'Epoch {epoch}. Valid: {metric_monitor}. Time {time.strftime("%M:%S", time.gmtime(time.time() - time1))}s')
    return (metric_monitor.metrics['Loss']['avg'],
            metric_monitor.metrics['acc']['avg'],
            metric_monitor.metrics['f1']['avg'])


def train_main(settings):
    logger = get_logger(f'{settings["TRAIN"]["MODEL_NAME"]}', settings['PATH']['LOGGER'])

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    logger.info(f'Using device: {device}')

    seed_setting(settings['TRAIN']['SEED'])

    image_train, label_train = get_data(settings['DATA']['TRAIN_YEAR'],
                                        settings['DATA']['TARGET'],
                                        settings['DATA']['IMG_FILE'],
                                        settings['DATA']['IMAGE_HEIGHT'],
                                        settings['DATA']['IMAGE_WIDTH'],
                                        settings['DATA']['COLUMNS'])
    if settings['TRAIN']['OUT_FEATURES'] == 2:
        binary_ind = label_train[label_train['label'] != 2].index
        image_train = image_train[binary_ind]
        label_train = label_train.loc[binary_ind, :].reset_index(drop=True)

    logger.info('====================== Model Train ========================')
    config = settings['TRAIN']
    config['DEVICE'] = device

    kfold = StratifiedKFold(n_splits=config['KFOLD'])
    for k, (train_idx, valid_idx) in enumerate(kfold.split(image_train, label_train['label'])):
        train_loader, valid_loader = get_train_valid_loader(image_train, label_train, train_idx, valid_idx, config)
        logger.info(f'--------------------- fold_{k} training --------------------------')

        if config['MODEL_NAME'] in ['CNN20D2C', 'CNN20D3C']:
            model = CNN20DModel(config['OUT_FEATURES'])

        model.to(config['DEVICE'])
        model.apply(init_weight)

        criterion = nn.CrossEntropyLoss().to(config['DEVICE'])
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['LR'], weight_decay=config['WEIGHT_DECAY'])
        scheduler = CosineAnnealingLR(optimizer, T_max=10)

        fold_save_path = os.path.join(settings['PATH']['MODEL'], fr'{config["MODEL_NAME"]}/model_fold_{k}')
        if not os.path.exists(fold_save_path):
            os.makedirs(fold_save_path)

        early_stopping = [np.Inf]
        df_performance = pd.DataFrame()
        for epoch in range(config['EPOCHS']):
            train_loss, train_acc, train_f1 = train_one_epoch(epoch, model, train_loader, criterion, optimizer,
                                                              scheduler, config, logger)
            valid_loss, valid_acc, valid_f1 = valid_one_epoch(epoch, model, valid_loader, criterion, config, logger)
            df_performance.loc[len(df_performance),
                               ['epoch',
                                'train_loss', 'train_acc', 'train_f1',
                                'valid_loss', 'valid_acc', 'valid_f1']] = (epoch,
                                                                           train_loss, train_acc, train_f1,
                                                                           valid_loss, valid_acc, valid_f1)
            model_save_path = os.path.join(fold_save_path, f'model_fold_{k}_{epoch}_{round(valid_acc, 3)}.pth')
            torch.save(model.state_dict(), model_save_path)

            if epoch > 2:
                if (valid_loss > early_stopping[-1]) & (
                        early_stopping[-1] > early_stopping[-2]):  # Early stopping setting
                    # model_save_path = os.path.join(fold_save_path, f'model_fold_{k}_{epoch}_{round(valid_acc, 3)}.pth')
                    # torch.save(model.state_dict(), model_save_path)

                    logger.info(f'Fold {k} best model is epoch_{epoch} model!')
                    break
            early_stopping.append(valid_loss)

        df_performance.to_csv(os.path.join(fold_save_path, f'train_fold_{k}.csv'))

        del model, criterion, optimizer, scheduler
        del train_loader, valid_loader
        gc.collect()
        torch.cuda.empty_cache()

