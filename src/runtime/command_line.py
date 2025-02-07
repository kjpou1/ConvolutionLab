import argparse

from src.models.command_line_args import CommandLineArgs
from src.utils.yaml_loader import load_supported_model_types

from .logging_argument_parser import LoggingArgumentParser


class CommandLine:
    @staticmethod
    def parse_arguments() -> CommandLineArgs:
        """
        Parse command-line arguments and return a CommandLineArgs object.

        Supports subcommands like 'ingest' and 'train'.
        """
        parser = LoggingArgumentParser(description="Frostfire Chart Sifter Application")

        # Create subparsers for subcommands
        subparsers = parser.add_subparsers(dest="command", help="Subcommands")

        # Subcommand: ingest
        ingest_parser = subparsers.add_parser(
            "ingest", help="Download and prepare datasets."
        )
        ingest_parser.add_argument(
            "--config",
            type=str,
            required=False,
            help="Path to the configuration file for ingestion.",
        )
        ingest_parser.add_argument(
            "--debug",
            action="store_true",
            help="Enable debug mode during ingestion.",
        )

        # Subcommand: train
        train_parser = subparsers.add_parser(
            "train", help="Train the model using the configured pipeline."
        )
        train_parser.add_argument(
            "--config",
            type=str,
            required=False,
            default="config/model_config.yaml",
            help="Path to the configuration file for training.",
        )
        train_parser.add_argument(
            "--debug",
            action="store_true",
            help="Enable debug mode during training.",
        )

        train_parser.add_argument(
            "--model-type",
            type=str,
            nargs="+",
            help="Specify one or more models to train (e.g., 'RandomForest DecisionTree').",
        )
        train_parser.add_argument(
            "--best-of-all",
            action="store_true",
            help="If set, overrides --model_type and trains all models to find the best.",
        )

        train_parser.add_argument(
            "--save-best",
            action="store_true",
            help="If set, saves the best-performing model after training.",
        )

        # Parse the arguments
        args = parser.parse_args()

        # ✅ Validation - Ensure either --model-type OR --best-of-all is set, but NOT both
        if args.command == "train":
            if args.model_type and args.best_of_all:
                parser.error(
                    "You cannot specify both --model-type and --best-of-all. Choose one."
                )
                sys.exit(1)
            if not args.model_type and not args.best_of_all:
                parser.error("You must specify either --model-type or --best-of-all.")
                sys.exit(1)
            supported_model_types = load_supported_model_types(args.config)

            # Validate model types
            if args.model_type:
                invalid_models = [
                    model
                    for model in args.model_type
                    if model not in supported_model_types
                ]

                if invalid_models:
                    print(
                        f"Error: Unsupported model type(s): {', '.join(invalid_models)}. "
                        f"Supported types are: [{', '.join(supported_model_types)}]."
                    )
                    exit(1)

        # Check if a subcommand was provided
        if args.command is None:
            parser.print_help()
            exit(1)

        # Return a CommandLineArgs object with parsed values
        return CommandLineArgs(
            command=args.command,
            config=args.config,
            debug=args.debug,
            model_type=args.model_type if hasattr(args, "model_type") else None,
            best_of_all=args.best_of_all if hasattr(args, "best_of_all") else False,
            save_best=args.save_best if hasattr(args, "save_best") else False,
        )
