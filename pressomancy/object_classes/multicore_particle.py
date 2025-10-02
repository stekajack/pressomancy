import inspect
import numpy as np
from pressomancy.object_classes.object_class import ObjectConfigParams
from pressomancy.object_classes.rigid_obj import GenericRigidObj
from pressomancy.helper_functions import PartDictSafe
from pressomancy.helper_functions import MissingFeature

class MulticorePart(GenericRigidObj):

    '''
    Class that contains relevant paramaters and methods. At construction one must pass an espresso handle becaouse the class manages parameters that are both internal and external to espresso. It is assumed that in any simulation instanse there will be only one type of a Quadriplex. Therefore many relevant parameters are class specific, not instance specific.
    '''
    numInstances = 0
    part_types = PartDictSafe()
    config=ObjectConfigParams(
        alias='multicore'
    )

    def __init__(self, config: ObjectConfigParams):
        super().__init__(config)
        self.who_am_i = MulticorePart.numInstances
        MulticorePart.numInstances += 1

    
    def add_dipole_moments_to_virtuals(self, dip_moments, selection='random', anisotropy={'kind':'infinite','params': {}}):
        """
        Assign magnetic dipole moments to selected virtual particles.

        Parameters
        ----------
        dip_moments : array_like, shape (N, 3)
            Dipole vectors to assign (in simulation units). Each selected virtual particle receives one 3-vector. If more dipole vectors are provided than available virtual particles, the extras are ignored; if fewer are provided, only that many particles are selected.
        selection : {'random'}, optional
            Particle selection strategy. Currently only ``'random'`` is supported:
            choose ``N`` distinct virtual particles uniformly at random, where
            ``N`` is the number of dipole vectors provided. Default is ``'random'``.
        anisotropy : dict, optional
            Anisotropy model to apply to the selected particles.
            Must be a dict with keys:
                - ``'kind'`` : {'infinite', 'finite_egg'}
                    * ``'infinite'`` — no extra parameters required; dipoles are
                    assigned directly to particles.
                    * ``'finite_egg'`` — uses ESPResSo's Egg model; requires params
                    ``'egg_gamma'`` and ``'aniso_energy'``.
                - ``'params'`` : dict
                    For ``'finite_egg'``: ``{'egg_gamma': float, 'aniso_energy': float}``.
            Default is ``{'kind': 'infinite', 'params': {}}``.

        Returns
        -------
        None

        Raises
        ------
        MissingFeature
            If ``anisotropy['kind'] == 'finite_egg'`` but the ESPResSo build lacks
            the ``EGG_MODEL`` feature.
        AssertionError
            If an unsupported anisotropy kind is specified or required Egg-model
            parameters are missing.
        ValueError
            If ``dip_moments`` cannot be coerced to a ``(N, 3)`` array.

        Notes
        -----
        - Operates only on particles of type ``'virt'`` in ``self.type_part_dict['virt']``.
        - Selection uses NumPy's RNG for shuffling (uniform without replacement).
        - For the Egg model, selected particles are retyped to a ``'yolk'`` part
        type.
        """
        assert anisotropy['kind'] in ['infinite', 'finite_egg'],'Supporting infinite anisotropy and finite anisotropy via the Egg model'
        
        dip_moments=np.atleast_2d(dip_moments)
        part_handles = self.type_part_dict['virt']
        num_parts = len(part_handles)
        if selection == 'random':
            take_index = np.arange(num_parts)
            np.random.shuffle(take_index)
            take_index = take_index[:len(dip_moments)]
            part_handles = np.array(part_handles)[take_index]
        else:
            raise NotImplementedError('Only random subsampling of cores implemented is currently')
        
        if anisotropy['kind']=='infinite':
            for x,dip_mom_per_part in zip(part_handles, dip_moments):
                x.dip = dip_mom_per_part
        if anisotropy['kind']=='finite_egg':
            if 'EGG_MODEL' not in self.sys.features():
                name = f"{type(self).__name__}.{inspect.currentframe().f_code.co_name}"
                raise MissingFeature(f"{name} requires EGG_MODEL. Please enable it in your ESPResSo installation.") 
            MulticorePart.part_types.update({'yolk': 11})
            assert ['egg_gamma','aniso_energy'] in anisotropy['params'], "Finite anisotropy is realised with the Egg model, which requires 'egg_gamma' and 'aniso_energy' parameters to be specified"
            for x,dip_mom_per_part in zip(part_handles, dip_moments):
                x.dip = dip_mom_per_part
                x.egg_model_params = (True, anisotropy['params']['egg_gamma'], anisotropy['params']['aniso_energy'])
                self.change_part_type(x,'yolk')
    
    



