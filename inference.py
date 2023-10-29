"""
-*- coding: utf-8 -*-

@Author : Aoran,Li
@Time : 2023/10/25 13:32
@File : inference.py
"""
from __init__ import *
from utilis import *
from model import *
from dataloader import *


def inference(model_config, data_config, path_config, logger):

    if model_config['MODEL_NAME'].__contains__('CNN'):
        model = CNN20DModel(model_config['OUT_FEATURES'])
    model.to(model_config['DEVICE'])

    image_test, label_test = get_data(data_config['TEST_YEAR'],
                                      data_config['TARGET'],
                                      data_config['IMG_FILE'],
                                      data_config['IMAGE_HEIGHT'],
                                      data_config['IMAGE_WIDTH'],
                                      data_config['COLUMNS'])

    if model_config['OUT_FEATURES'] == 2:
        binary_ind = label_test[label_test['label'] != 2].index
        image_test = image_test[binary_ind]
        label_test = label_test.loc[binary_ind, :].reset_index(drop=True)
    test_loader = get_test_loader(image_test, label_test, model_config)

    logger.info(f'========================= {model_config["MODEL_NAME"]} Inference =========================')

    model_path = os.path.join(path_config['MODEL'], f'{model_config["MODEL_NAME"]}')

    predictions_all = []
    for model_file in os.listdir(model_path):
        logger.info(f'------------------ {model_file} testing ----------------')
        model_file_ = os.path.join(model_path, model_file)
        model.load_state_dict(torch.load(model_file_))

        model.eval()
        pre_fold = []
        metrics = MetricMonitor()
        criterion = nn.CrossEntropyLoss().to(model_config['DEVICE'])
        for images, labels in test_loader:
            images = images.to(model_config['DEVICE'])
            labels = labels.to(model_config['DEVICE'])
            with torch.no_grad():
                label_pre = model(images)
                loss = criterion(label_pre, labels)

                acc, f1_macro = get_accuracy_f1score(label_pre, labels)
                metrics.update('Loss', loss.item())
                metrics.update('acc', acc)
                metrics.update('f1', f1_macro)

            label_pre = label_pre.cpu().numpy().tolist()
            pre_fold.extend(label_pre)
        predictions_all.append(pre_fold)
        logger.info(f'------------- {model_file}: {metrics} --------------')

    avg_model_pred = np.mean(predictions_all, axis=0)
    label_test['label_pre'] = np.argmax(avg_model_pred, axis=1)
    for i in range(len(avg_model_pred[0])):
        label_test[f'label_{i}_prob'] = avg_model_pred[:, i]

    res_path = os.path.join(path_config['RESULT'], f'{model_config["MODEL_NAME"]}_result.csv')
    os.makedirs(os.path.dirname(res_path), exist_ok=True)
    label_test.to_csv(res_path)

    return label_test


def inference_main(settings):
    logger = get_logger(f'{settings["TRAIN"]["MODEL_NAME"]}', settings['PATH']['LOGGER'], infer=True)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    settings['TRAIN']['DEVICE'] = device
    logger.info(f'Using device: {device}.')

    seed_setting(settings['TRAIN']['SEED'])

    inference(settings['TRAIN'], settings['DATA'], settings['PATH'], logger)

    # backtesting(label_df, settings['INFERENCE'])
    # ...





