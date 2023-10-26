"""
-*- coding: utf-8 -*-

@Author : Aoran,Li
@Time : 2023/10/25 13:32
@File : utilis.py
"""
from __init__ import *


# ============================== Data Preprocessing ===========================
def display_image(matrix):
    plt.imshow(matrix)
    plt.show()


def map_label(x):
    if np.isnan(x):
        return 2
    else:
        return int(x > 0)


def get_data(years, target, img_file_, image_height, image_width, columns):
    images_ = []
    labels_ = []

    for y in years:
        images_.append(np.memmap(os.path.join(img_file_, f'20d_month_has_vb_ma_{y}_images.dat'),
                                 dtype=np.uint8,
                                 mode='r').reshape((-1, image_height[20], image_width[20])))
        labels_.append(
            pd.read_feather(os.path.join(img_file_, f'20d_month_has_vb_ma_{y}_labels_w_delay.feather')))

    images_ = np.concatenate(images_)
    labels_ = pd.concat(labels_).reset_index(drop=True)
    labels_['label'] = labels_[target].apply(map_label)
    labels_ = labels_[columns]
    assert len(images_) == len(labels_), 'Image length do not match label length!'

    return images_, labels_


# ============================== Metrics Function ================================
def get_accuracy_f1score(output, y_true):
    y_pred = torch.softmax(output, dim=1)
    y_pred = torch.argmax(y_pred, dim=1).cpu()
    y_true = y_true.cpu()
    return (accuracy_score(y_true, y_pred),
            f1_score(y_true, y_pred, average='macro'))


class MetricMonitor:
    def __init__(self):
        self.metrics = None
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {'val': 0,
                                            'count': 0,
                                            'avg': 0})

    def update(self, metric_name, val):
        metric = self.metrics[metric_name]

        metric['val'] += val
        metric['count'] += 1
        metric['avg'] = metric['val'] / metric['count']

    def __str__(self):
        return " | ".join([f'{metric_name}: {round(metric["avg"], 3)}' for metric_name, metric in self.metrics.items()])


# ============================== Model Training ===========================
def seed_setting(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def init_weight(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)


# ========================== Logger =============================
def get_logger(log_name, file_name):
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    file_name = os.path.join(file_name, f'{log_name}_train.log')
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    file_handler = logging.FileHandler(filename=file_name, mode='w')

    standard_formatter = logging.Formatter(
        '%(asctime)s %(name)s [%(filename)s line:%(lineno)d] %(levelname)s %(message)s]')
    simple_formatter = logging.Formatter('%(levelname)s %(message)s]')

    console_handler.setFormatter(simple_formatter)
    file_handler.setFormatter(standard_formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
