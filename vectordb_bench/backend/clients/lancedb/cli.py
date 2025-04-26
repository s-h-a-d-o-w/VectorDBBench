from typing import Unpack
from ....cli.cli import (
    CommonTypedDict,
    cli,
    click_parameter_decorators_from_typed_dict,
    run,
)
from pydantic import SecretStr
from .. import DB

class LanceDBTypedDict(CommonTypedDict):
    uri: str
    token: str | None = None

@cli.command()
@click_parameter_decorators_from_typed_dict(LanceDBTypedDict)
def LanceDB(**parameters: Unpack[LanceDBTypedDict]):
    """Run benchmark for LanceDB."""
    # from .config import LanceDBConfig, LanceDBIndexConfig
    from .config import LanceDBConfig

    run(
        db=DB.LanceDB,
        db_config=LanceDBConfig(
            db_label=parameters["db_label"],
            uri=parameters["uri"],
            token=SecretStr(parameters["token"]) if parameters.get("token") else None,
        ),
        # db_case_config=LanceDBIndexConfig(),
        **parameters,
    )