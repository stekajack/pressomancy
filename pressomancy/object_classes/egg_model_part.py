from pressomancy.object_classes.part_class import GenericPart
from pressomancy.object_classes.object_class import ObjectConfigParams
from pressomancy.helper_functions import PartDictSafe, SinglePairDict
import espressomd
if espressomd.version.major() == 5:
    import espressomd.propagation
    Propagation = espressomd.propagation.Propagation

class EGGPart(GenericPart):
    """
    Magnetic particle represented by ESPResSo's egg-model dynamics.

    An ``EGGPart`` is built as a real reference particle plus a virtual
    ``yolk`` particle carrying the dipole moment. The reference particle
    controls the yolk position through a relative virtual-site relation, while
    the yolk orientation is evolved independently by the egg-model Brownian
    rotational dynamics.

    The ``ori`` argument passed to :meth:`set_object` initializes both the
    reference frame and the initial yolk dipole direction. The easy axis is
    configured separately through ``axis_quat_body``, which is interpreted in
    the body-fixed frame of the real reference particle. By default,
    ``axis_quat_body`` is the identity quaternion, so the easy axis is aligned
    with ``ori``.
    """
    required_features = ['EGG_MODEL']
    numInstances = 0
    simulation_type = SinglePairDict('egg_part', 74)
    part_types = PartDictSafe({'yolk': 11})
    config = ObjectConfigParams(
        dipm=1,
        gamma=1.,
        anisotropy_energy=1.,
        axis_quat_body=[1, 0, 0, 0],
    )

    def __init__(self, config: ObjectConfigParams):
        """
        Initialize an egg-model particle wrapper.

        Parameters
        ----------
        config : ObjectConfigParams
            Configuration containing an ESPResSo system handle, dipole moment,
            egg friction, anisotropy energy, and body-frame easy-axis
            quaternion.
        """
        self.sys = config['espresso_handle']
        self.params = config
        self.who_am_i = EGGPart.numInstances
        EGGPart.numInstances += 1
        self.associated_objects = config['associated_objects']
        self.type_part_dict = PartDictSafe(
            {key: [] for key in EGGPart.part_types.keys()})

    def set_object(self, pos, ori):
        """
        Add the real reference particle and egg-model yolk to the system.

        Parameters
        ----------
        pos : array_like
            Initial position of the particle pair.
        ori : array_like
            Initial director for the reference particle and the initial yolk
            dipole direction.

        Returns
        -------
        EGGPart
            The configured object instance.
        """
        particl_real = self.add_particle(
            type_name='real', pos=pos, rotation=(True, True, True),
            director=ori)
        particl_virt = self.add_particle(
            type_name='yolk', pos=pos, rotation=(True, True, True),
            director=ori, dipm=self.params['dipm'])
        particl_virt.vs_auto_relate_to(particl_real)

        if espressomd.version.major() == 5:
            magnetodynamics_setup = {
                "is_enabled": True,
                "gamma": self.params['gamma'],
                "anisotropy_energy": self.params['anisotropy_energy'],
                "axis_quat_body": self.params['axis_quat_body'],
            }
            particl_virt.magnetodynamics.egg = magnetodynamics_setup
            particl_virt.propagation = (Propagation.TRANS_VS_RELATIVE |
                                        Propagation.ROT_VS_INDEPENDENT)
        elif espressomd.version.major() == 4:
            particl_virt.egg_model_params = {
                "use_egg_model": True,
                "egg_gamma": self.params['gamma'],
                "aniso_energy": self.params['anisotropy_energy'],
            }
        else:
            raise ImportError(
                f"Unsupported espressomd version: {espressomd.version.major()}. "
                "This code requires espressomd version 4 or higher.")

        return self
