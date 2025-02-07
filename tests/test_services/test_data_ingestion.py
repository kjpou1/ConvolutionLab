import os
import pytest
from src.services.data_ingestion_service import DataIngestionService
from src.exception import CustomException


def test_data_ingestion_with_missing_file():
    ingestion_service = DataIngestionService()
    ingestion_service.ingestion_config.input_data_path = "non_existent_file.csv"
    with pytest.raises(CustomException) as exc_info:
        ingestion_service.initiate_data_ingestion()

    # Check if the root cause is FileNotFoundError
    assert isinstance(exc_info.value.original_exception, FileNotFoundError)
    assert "non_existent_file.csv" in str(exc_info.value)


def test_data_ingestion_creates_files(tmp_path):
    # Create a temporary input file
    input_file = tmp_path / "input.csv"
    input_file.write_text("col1,col2\n1,2\n3,4\n5,6")

    ingestion_service = DataIngestionService()
    ingestion_service.ingestion_config.input_data_path = str(input_file)
    ingestion_service.ingestion_config.train_data_path = str(tmp_path / "train.csv")
    ingestion_service.ingestion_config.test_data_path = str(tmp_path / "test.csv")
    ingestion_service.ingestion_config.raw_data_path = str(tmp_path / "raw.csv")

    train_path, test_path = ingestion_service.initiate_data_ingestion(test_size=0.5)

    assert os.path.exists(train_path)
    assert os.path.exists(test_path)
