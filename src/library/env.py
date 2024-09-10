import os

from dotenv import load_dotenv


class Env:
    _is_instanced = False

    def __new__(cls) -> "Env":
        if not cls._is_instanced:
            cls._is_instanced = True
            load_dotenv(override=True)
            cls._instance = super().__new__(cls)
        return cls._instance

    def get(self, key: str, default: str | None = None) -> str:
        var = os.getenv(key, default)
        match var:
            case str():
                return var
            case None:
                raise KeyError


ENV = Env()
