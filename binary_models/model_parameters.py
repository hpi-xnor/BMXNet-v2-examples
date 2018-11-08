
class ModelParameters:
    def __init__(self, name):
        self.name = name

    def add_group(self, parser, model):
        if not self._is_it_this_model(model):
            return None
        model_group = parser.add_argument_group(self.name, 'parameters for {}'.format(self.name))
        self._add_arguments(model_group)

    def set_args_for_model(self, opt, kwargs):
        if self._is_it_this_model(opt.model):
            self._map_opt_to_kwargs(opt, kwargs)

    def _is_it_this_model(self, model):
        raise NotImplementedError()

    def _map_opt_to_kwargs(self, opt, kwargs):
        raise NotImplementedError()

    def _add_arguments(self, parser):
        raise NotImplementedError()
