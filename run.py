#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pickle as pkl, pdb, nauka, os, sys, uuid


DATASETS = ["mnist", "cifar10", "cifar100", "svhn"]


class root(nauka.utils.Subcommand):
	class train(nauka.utils.Subcommand):
		@classmethod
		def addArgs(kls, argp):
			mtxp = argp.add_mutually_exclusive_group()
			mtxp.add_argument("-w", "--workDir",        default=None,               type=str,
			    help="Full, precise path to an experiment's working directory.")
			mtxp.add_argument("-b", "--baseDir",        action=nauka.utils.BaseDirPathAction)
			argp.add_argument("-d", "--dataDir",        action=nauka.utils.DataDirPathAction)
			argp.add_argument("-t", "--tmpDir",         action=nauka.utils.TmpDirPathAction)
			argp.add_argument("--summary",              action="store_true",
			    help="Print a summary of the network.")
			argp.add_argument("--fastdebug",            action="store_true",
			    help="For debug purposes, run very few iterations per epoch, thus exercising "
			         "all of the code quickly.")
			argp.add_argument("--pdb",                  action="store_true",
			    help="""Breakpoint before run start.""")
			argp.add_argument("-n", "--name",           action="append",            type=str,
			    help="Build a name for the experiment.")
			argp.add_argument("-s", "--seed",           default=0x6a09e667f3bcc908, type=int,
			    help="Seed for PRNGs. Default is 64-bit fractional expansion of sqrt(2).")
			argp.add_argument("--model",                default="bnn",              type=str,
			    choices=["bnn"],
			    help="Model Selection.")
			argp.add_argument("--dataset",              default="cifar10",          type=str,
			    choices=DATASETS,
			    help="Dataset Selection.")
			argp.add_argument("--dropout",              default=0,                  type=float,
			    help="Dropout probability.")
			argp.add_argument("-e", "--num-epochs",     default=200,                type=int,
			    help="Number of epochs")
			argp.add_argument("--batch-size", "--bs",   default=50,                 type=int,
			    help="Batch Size")
			argp.add_argument("--cuda",                 action=nauka.utils.CudaDeviceAction)
			argp.add_argument("-p", "--preset",         action=nauka.utils.PresetAction,
			    choices={"fig1":  ["--name=fig1", "--opt=adam", "--bs=100"],
			             "fig2":  ["--name=fig2", "--opt=sgd",  "--bs=25"],},
			    help="Experiment presets for commonly-used settings.")
			optp = argp.add_argument_group("Optimizers", "Tunables for all optimizers")
			optp.add_argument("--optimizer", "--opt",   action=nauka.utils.OptimizerAction,
			    default="adam",
			    help="Optimizer selection.")
			optp.add_argument("--clipnorm", "--cn",     default=1.0,                type=float,
			    help="The norm of the gradient will be clipped at this magnitude.")
			optp.add_argument("--clipval",  "--cv",     default=1.0,                type=float,
			    help="The values of the gradients will be individually clipped at this magnitude.")
			optp.add_argument("--l1",                   default=0,                  type=float,
			    help="L1 penalty.")
			optp.add_argument("--l2",                   default=0,                  type=float,
			    help="L2 penalty.")
			optp.add_argument("--decay",                default=0,                  type=float,
			    help="Learning rate decay for optimizers.")
		
		@classmethod
		def run(kls, a):
			from   experiment import Experiment;
			if a.pdb: pdb.set_trace()
			return Experiment(a).rollback().run().exitcode
	
	class download(nauka.utils.Subcommand):
		@classmethod
		def addArgs(kls, argp):
			argp.add_argument("-d", "--dataDir",        action=nauka.utils.DataDirPathAction)
			argp.add_argument("--dataset",              default="all",              type=str,
			    choices=DATASETS+["all"],
			    help="Dataset Selection.")
		
		@classmethod
		def run(kls, a):
			from   experiment import Experiment;
			return Experiment.download(a)


def main(argv):
	a = root.addAllArgs().parse_args(argv[1:])
	a.__argv__ = argv
	return a.__kls__.run(a)


if __name__ == "__main__":
	sys.exit(main(sys.argv))
