"""
K2 Aerospace — Plugin Architecture
"""

class PluginManager:
    """
    Manages external plugins for solvers and optimizers.
    """
    def __init__(self):
        self.plugins = {}

    def load_plugin(self, name: str, plugin_module):
        self.plugins[name] = plugin_module
        
    def get_plugin(self, name: str):
        return self.plugins.get(name)
