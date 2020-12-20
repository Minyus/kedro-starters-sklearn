from typing import Dict

from kedro.framework.hooks import hook_impl
from kedro.io import AbstractDataSet, DataCatalog

from .catalog import catalog_dict


class AddCatalogDictHook:
    """Hook to add data sets."""

    def __init__(
        self,
        catalog_dict: Dict[str, AbstractDataSet] = {},
    ):
        """
        Args:
            catalog_dict: catalog_dict to add.
        """
        assert isinstance(catalog_dict, dict), "{} is not a dict.".format(catalog_dict)
        self._catalog_dict = catalog_dict

    @hook_impl
    def after_catalog_created(self, catalog: DataCatalog) -> None:
        catalog.add_feed_dict(self._catalog_dict)


add_catalog_dict_hook = AddCatalogDictHook(catalog_dict)
