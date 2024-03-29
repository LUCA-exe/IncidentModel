"""
parser.py

- Configuration of the parameters.
"""
import configargparse

CHOICES_DATA_USED = [
    "pos_only",
    "pos_and_neg"
]

CHOICES_DEVICE = [
    "cpu",
    "gpu"
]

# Updated for HuggingFace download options
CHOICES_ARCHITECTURES = [
    "resnet18",
    "resnet50",
    "google_vit_b_16",
    "vit_l_16"
]

# Just sigmoid available
CHOICES_ACTIVATION = [
    "softmax",
    "sigmoid"
]

CHOICES_LOSSES = [
    "softmax_cross_entropy",
    "sigmoid_bce_no_masking",
    "sigmoid_bce_with_masking"
]


def get_parser():
    parser = configargparse.ArgumentParser(description="Incident Model Parser")

    parser.add_argument('-c',
                        '--config',
                        required=True,
                        is_config_file=True,
                        help='Config file path.')

    # 'val'm deprecated
    parser.add_argument("--mode",
                        default="train",
                        required=True,
                        type=str,
                        choices=["train", "test"],
                        help="How to use the model and config the dataset, such as 'train' or 'test'.")

    parser.add_argument("--checkpoint_path",
                        default="pretrained_weights/",
                        type=str,
                        help="Path to checkpoints for training.")

    # Load if already available in the folder 'pretrained_weights'
    parser.add_argument("--load_arch_path",
                        default="None",
                        type=str,
                        help="Path to the architecture of the backbone model (pretrained from HuggingFace)")

    # TODO: make sure to use this
    parser.add_argument("--download_train_json",
                        default="True",
                        type=str,
                        help="Clean and download the images contained in the json train file")
    
    parser.add_argument("--download_val_json",
                        default="True",
                        type=str,
                        help="Clean and download the images contained in the json val file (to be used in the test phase)")
    
    parser.add_argument("--images_path",
                        default="data/images/",
                        help="Path of the folder to the downloaded images.")

    # Format of the file_name standardized
    parser.add_argument("--dataset_train",
                        default="data/multi_label_train.json",
                        help="Json file with the url/values of images.")

    # Val is used in the test phase
    parser.add_argument("--dataset_val",
                        default="data/multi_label_val.json",
                        help="Json file with the url/values of images.")

    # Deprecated: val is the actual "test"
    parser.add_argument("--dataset_test",
                        default="data/eccv_test.json",
                        help="Json file with the url/values of images.")

    parser.add_argument('--device',
                        default="cpu",
                        type=str,
                        choices=CHOICES_ACTIVATION,
                        help='Device to use')


    parser.add_argument('--num_gpus',
                        default=1,
                        type=int,
                        help='Number of gpus to use.')

    parser.add_argument('-b',
                        '--batch_size',
                        default=16,
                        type=int,
                        metavar='N',
                        help='mini-batch size (default: 16)')

    parser.add_argument('--loss',
                        type=str,
                        choices=CHOICES_LOSSES)

    parser.add_argument('--activation',
                        type=str,
                        choices=CHOICES_ACTIVATION,
                        required=True)

    parser.add_argument('--examples',
                        default='pos_and_neg',
                        help='Which dataset to train with.',
                        choices=CHOICES_DATA_USED)

    parser.add_argument('--trial_image',
                        required=True,
                        default='example_images/flood.jpg',
                        help='Image path to load for the model trials before the training/test phases')

    # modified for 'loading' the architecture from HuggingFace                    
    parser.add_argument('--arch',
                        '-a',
                        metavar='ARCH',
                        default='google_vit_b_16',
                        choices=CHOICES_ARCHITECTURES,
                        help='Which model architecture to use.')
                        
    parser.add_argument('--ignore_places_during_training',
                        default="False",
                        type=str)

    parser.add_argument('--percent_of_training_set',
                        default=100,
                        type=int)

    parser.add_argument('--pretrained_with_places',
                        default="True",
                        type=str)

    parser.add_argument('-j',
                        '--workers',
                        default=2,
                        type=int,
                        metavar='N',
                        help='number of data loading workers (default: 4)')

    parser.add_argument('--epochs',
                        default=4,
                        type=int,
                        metavar='N',
                        help='number of total epochs to run')

    parser.add_argument('--start-epoch',
                        default=0,
                        type=int,
                        metavar='N',
                        help='manual epoch number (useful on restarts)')

    parser.add_argument('--lr',
                        '--learning-rate',
                        default=0.0001,
                        type=float,
                        metavar='LR',
                        help='initial learning rate')

    parser.add_argument('--weight-decay',
                        '--wd',
                        default=1e-4,
                        type=float,
                        metavar='W',
                        help='weight decay (default: 1e-4)')

    parser.add_argument('--print-freq',
                        '-p',
                        default=50,
                        type=int,
                        metavar='N',
                        help='print frequency (default: 10)')

    parser.add_argument('--pretrained',
                        dest='pretrained',
                        action='store_false',
                        help='use pre-trained model')

    parser.add_argument('--num-places',
                        default=49,
                        type=int,
                        help='Num of places classes')

    parser.add_argument('--num-incidents',
                        default=43, 
                        type=int,
                        help='Num of incidents classes')

    parser.add_argument('--fc-dim',
                        default=1024,
                        type=int,
                        help='output dimension of network')
    return parser


def get_postprocessed_args(args):
    # turn the True/False strings into booleans where applicable
    for key, value in vars(args).items():
        if value == "True":
            setattr(args, key, True)
        elif value == "False":
            setattr(args, key, False)
    return args
