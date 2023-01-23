from utils.args import *
from .projector_manager import add_parser

# Best default arguments inherited from DER++
BEST_ARGS_DERPP = {
	'seq-mnist': {
		200: {'lr': 0.03, 'minibatch_size': 128, 'alpha_distill': 0.2, 'alpha_ce': 1.0, 
			'batch_size': 10, 'n_epochs': 1},
		500: {'lr': 0.03, 'minibatch_size': 10, 'alpha_distill': 1.0, 'alpha_ce': 0.5, 
			'batch_size': 10, 'n_epochs': 1},
	},
	"perm-mnist": {
		200: {'lr': 0.1, 'minibatch_size': 128, 'alpha_distill': 1.0, 'alpha_ce': 1.0, 
			  'batch_size': 128, 'n_epochs': 1},
		500: {'lr': 0.2, 'minibatch_size': 128, 'alpha_distill': 1.0, 'alpha_ce': 0.5, 
			  'batch_size': 128, 'n_epochs': 1},
	},
	'rot-mnist': {
		200: {'lr': 0.1, 'minibatch_size': 128, 'alpha_distill': 1.0, 'alpha_ce': 0.5, 'batch_size': 128, 
			  'n_epochs': 1},
		500: {'lr': 0.2, 'minibatch_size': 128, 'alpha_distill': 0.5, 'alpha_ce': 1.0, 'batch_size': 128, 
			  'n_epochs': 1},
	},
	'seq-cifar10': {
		20: {"lr": 0.03, "minibatch_size": 32, "batch_size": 32, "alpha_distill": 0.1, 
			"alpha_ce": 0.5, "n_epochs": 50},
		50: {"lr": 0.03, "minibatch_size": 32, "batch_size": 32, "alpha_distill": 0.1, 
			"alpha_ce": 0.5, "n_epochs": 50},
		200: {"lr": 0.03, "minibatch_size": 32, "batch_size": 32, "alpha_distill": 0.1, 
			"alpha_ce": 0.5, "n_epochs": 50},
		500: {"lr": 0.03, "minibatch_size": 32, "batch_size": 32, "alpha_distill": 0.2, 
			"alpha_ce": 0.5, "n_epochs": 50},
	},
	'seq-cifar100': {
		500: {"lr": 0.03, "minibatch_size": 32, "batch_size": 32, "alpha_distill": 0.1, 
			"alpha_ce": 0.5, "n_epochs": 50},
		2000: {"lr": 0.03, "minibatch_size": 32, "batch_size": 32, "alpha_distill": 0.1, 
			"alpha_ce": 0.5, "n_epochs": 50},
	},
	'seq-cifar100-50-10': {
		500: {"lr": 0.03, "minibatch_size": 32, "batch_size": 32, "alpha_distill": 0.1, 
			"alpha_ce": 0.5, "n_epochs": 50},
		2000: {"lr": 0.03, "minibatch_size": 32, "batch_size": 32, "alpha_distill": 0.1, 
			"alpha_ce": 0.5, "n_epochs": 50},
	},
	'seq-tinyimg': {
		200: {"lr": 0.03, "minibatch_size": 32, "batch_size": 32, "alpha_distill": 0.1, 
			"alpha_ce": 1.0, "n_epochs": 100},
		500: {"lr": 0.03, "minibatch_size": 32, "batch_size": 32, "alpha_distill": 0.2, 
			"alpha_ce": 0.5, "n_epochs": 100},
		4000: {"lr": 0.1, "minibatch_size": 64, "batch_size": 64, "alpha_distill": 0.3, 
			"alpha_ce": 0.8, "n_epochs": 100},
	},
	'seq-tinyimg-s32': {
		200: {"lr": 0.03, "minibatch_size": 32, "batch_size": 32, "alpha_distill": 0.1, 
			"alpha_ce": 1.0, "n_epochs": 100},
		500: {"lr": 0.03, "minibatch_size": 32, "batch_size": 32, "alpha_distill": 0.2, 
			"alpha_ce": 0.5, "n_epochs": 100},
		4000: {"lr": 0.1, "minibatch_size": 64, "batch_size": 64, "alpha_distill": 0.3, 
			"alpha_ce": 0.8, "n_epochs": 100},
	},
    'gcil-cifar100': {
        200: {
            'alpha_distill': 0.2,
            'alpha_ce': 0.5,
            'lr': 0.1,
            'minibatch_size': 32,
            'batch_size': 32,
            'n_epochs': 100,
        },
        500: {
            'alpha_distill': 0.2,
            'alpha_ce': 0.6,
            'lr': 0.1,
            'minibatch_size': 32,
            'batch_size': 32,
            'n_epochs': 100,
        },
        1000: {
            'alpha_distill': 0.3,
            'alpha_ce': 0.6,
            'lr': 0.1,
            'minibatch_size': 32,
            'batch_size': 32,
            'n_epochs': 100,
        },
    }
}

BEST_ARGS_ER = {
	'seq-mnist': {
		200: {'lr': 0.01, 'minibatch_size': 10, 'alpha_distill': 0, 'alpha_ce': 1.0, 
			'batch_size': 10, 'n_epochs': 1},
		500: {'lr': 0.1, 'minibatch_size': 10, 'alpha_distill': 0, 'alpha_ce': 1.0, 
			'batch_size': 10, 'n_epochs': 1},
	},
	'seq-cifar10': {
		200: {"lr": 0.1, "minibatch_size": 32, "batch_size": 32, "alpha_distill": 0, 
			"alpha_ce": 1.0, "n_epochs": 50},
		500: {"lr": 0.1, "minibatch_size": 32, "batch_size": 32, "alpha_distill": 0, 
			"alpha_ce": 1.0, "n_epochs": 50},
	},
	'seq-cifar100': {
		500: {"lr": 0.1, "minibatch_size": 32, "batch_size": 32, "alpha_distill": 0, 
			"alpha_ce": 1.0, "n_epochs": 50},
		2000: {"lr": 0.1, "minibatch_size": 32, "batch_size": 32, "alpha_distill": 0, 
			"alpha_ce": 1.0, "n_epochs": 50},
	},
	'seq-cifar100-50-10': {
		500: {"lr": 0.1, "minibatch_size": 32, "batch_size": 32, "alpha_distill": 0, 
			"alpha_ce": 1.0, "n_epochs": 50},
		2000: {"lr": 0.1, "minibatch_size": 32, "batch_size": 32, "alpha_distill": 0, 
			"alpha_ce": 1.0, "n_epochs": 50},
	},
	'seq-tinyimg': {
		200: {"lr": 0.1, "minibatch_size": 32, "batch_size": 32, "alpha_distill": 0, 
			"alpha_ce": 1.0, "n_epochs": 100},
		500: {"lr": 0.03, "minibatch_size": 32, "batch_size": 32, "alpha_distill": 0, 
			"alpha_ce": 1.0, "n_epochs": 100},
		4000: {"lr": 0.1, "minibatch_size": 64, "batch_size": 64, "alpha_distill": 0, 
			"alpha_ce": 1.0, "n_epochs": 100},
	},
}

BEST_ARGS = {
	'derpp': BEST_ARGS_DERPP,
	'er': BEST_ARGS_ER,
}

def set_best_args(args):
	# Set the best arguments if not given otherwise
	try:
		best_args = BEST_ARGS[args.base_method][args.dataset][args.buffer_size]
		for k, v in best_args.items():
			if getattr(args, k) is None:
				print("Setting {} to {}".format(k, v))
				setattr(args, k, v)
	except KeyError:
		print("No best arguments found for base_method {} dataset {} and buffer size {}."
			.format(args.base_method, args.dataset, args.buffer_size))


def get_parser() -> ArgumentParser:
	parser = ArgumentParser(description='Continual learning via backward feature projection')
	add_management_args(parser)
	add_experiment_args(parser)
	add_rehearsal_args(parser)

	parser.add_argument("--base_method", type=str, default='derpp', choices=['derpp', 'er'], 
				help="Base method to use, determining the default hyperparameters")

    # alpha in the original paper
	parser.add_argument('--alpha_distill', type=float, default=None,
				help='Weight of the replayed distillation loss.')
    # beta in the original paper
	parser.add_argument('--alpha_ce', type=float, default=None,
				help='Weight of the replayed CE loss.')

	parser.add_argument('--resnet_skip_relu', action="store_true",
				help="If set, the last ReLU of each block of ResNet is skipped.")

	parser.add_argument("--no_old_net", action="store_true",
				help="If set, the behavior will be like the old network is not available.")
	parser.add_argument("--use_buf_logits", action="store_true",
				help="If set, logits distillation will use the logits replayed from the buffer")
	parser.add_argument('--use_buf_feats', action="store_true",
				help="If set, BFP will use the features replayed from the buffer")

	# On which data the BFP loss is applied
	parser.add_argument("--old_only", action="store_true",
				help="If set, the BFP loss will be applied on the buffered data. ")
	parser.add_argument("--new_only", action="store_true",
				help="If set, the BFP loss will be applied on the online data. ")

	parser.add_argument("--no_resample", action="store_true",
				help="If set, the replayed data will not be resampled for each loss.")
				
	parser = add_parser(parser)
	
	return parser
