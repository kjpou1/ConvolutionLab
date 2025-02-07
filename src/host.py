import asyncio

from src.config.config import Config
from src.exception import CustomException
from src.logger_manager import LoggerManager
from src.models.command_line_args import CommandLineArgs
from src.pipeline.train_pipeline import TrainPipeline
from src.services.data_ingestion_service import DataIngestionService

logging = LoggerManager.get_logger(__name__)


class Host:
    """
    Host class to manage the execution of the main application.

    This class handles initialization with command-line arguments and
    configuration, and runs the main asynchronous functionality.
    """

    def __init__(self, args: CommandLineArgs):
        """
        Initialize the Host class with command-line arguments and configuration.

        Parameters:
        args (CommandLineArgs): Command-line arguments passed to the script.
        """
        self.args = args
        self.config = Config()
        if args.config:
            self.config.config_path = args.config
        self.config.model_type = args.model_type
        self.config.best_of_all = args.best_of_all
        self.config.save_best = args.save_best
        logging.info("Host initialized with arguments: %s", self.args)

    def run(self):
        """
        Synchronously run the asynchronous run_async method.

        This is a blocking call that wraps the asynchronous method.
        """
        return asyncio.run(self.run_async())

    async def run_async(self):
        """
        Main asynchronous method to execute the host functionality.

        Determines the action based on the provided subcommand.
        """
        try:
            logging.info("Starting host operations.")

            if self.args.command == "ingest":
                logging.info("Executing data ingestion workflow.")
                await self.run_ingestion()
            elif self.args.command == "train":
                logging.info("Executing training workflow.")
                # if not self.args.model_type:
                #     raise ValueError(
                #         "A model type must be specified for the 'train' command."
                #     )
                await self.run_training()
            else:
                logging.error("No valid subcommand provided.")
                raise ValueError(
                    "Please specify a valid subcommand: 'ingest' or 'train'."
                )

        except CustomException as e:
            logging.error("A custom error occurred during host operations: %s", e)
            raise
        except Exception as e:
            logging.error("An unexpected error occurred: %s", e)
            raise  # CustomException("An unexpected error occured", sys) from e
        finally:
            logging.info("Shutting down host gracefully.")

    async def run_ingestion(self):
        """
        Execute the data ingestion workflow.
        """
        data_ingestion_service = DataIngestionService()
        data_ingestion_service.initiate_data_ingestion()

    async def run_training(self):
        """
        Execute the model training workflow.
        """
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
