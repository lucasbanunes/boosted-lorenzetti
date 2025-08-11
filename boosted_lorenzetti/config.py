from importlib import import_module
from typing import Tuple, List, Any, Dict
from pydantic import BaseModel


def split_import_name(import_name: str) -> Tuple[str, str]:
    """
    Splits an import name considering that the last name
    is an attribute defined inside the package.

    Parameters
    ----------
    import_name : str
        Import path

    Returns
    -------
    name: str
        The full package name
    attribute: str
        The attribute name
    """
    splitted = import_name.split(".")
    name = ".".join(splitted[:-1])
    attribute = splitted[-1]
    return name, attribute


class ObjectConfig(BaseModel):
    constructor: str
    args: List[Any] = []
    kwargs: Dict[str, Any] = {}

    def parse_object_dict(self,
                          parse_inner_objs: bool
                          ) -> Any:
        """
        Parses a config dict

        Parameters
        ----------
        object_dict : ObjectDict
            Python dictionary containg the configuration
            for instatiation of an object.
        parse_inner_objs : bool
            If true, tries to parse dicts inside object_dict
            as other configs

        Returns
        -------
        Any
            Instance of the object described by config
        """
        name, attribute = split_import_name(self.constructor)
        package = import_module(name)
        constructor = getattr(package, attribute)
        if parse_inner_objs:
            for i in range(len(self.args)):
                if isinstance(self.args[i], ObjectConfig):
                    self.args[i] = self.args[i].parse_object_dict(
                        parse_inner_objs=parse_inner_objs
                    )
                elif isinstance(self.args[i], dict):
                    config = ObjectConfig(**self.args[i])
                    self.args[i] = config.parse_object_dict(
                        parse_inner_objs=parse_inner_objs
                    )
        return constructor(*self.args, **self.kwargs)
