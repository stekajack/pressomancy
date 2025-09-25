from pressomancy.object_classes.object_class import Simulation_Object, ObjectConfigParams
from pressomancy.helper_functions import PartDictSafe, SinglePairDict, load_coord_file
import os
import numpy as np

class GenericRigidObj(metaclass=Simulation_Object):
    """
    General-purpose rigid simulation object with one real center-of-mass (CoM)
    particle and a rigid arrangement of virtual particles. At construction, the object infers its virtual-particle geometry from a resource file selected by the provided alias. Resource file should 

    Parameters
    ----------
    config : ObjectConfigParams
        Configuration mapping for the object. Required keys include:
        - 'alias' : str
            Name used to locate the resource file in ``resources/`` (e.g.
            ``'<alias>.txt'``). Determines the rigid arrangement of virtual sites. Default value is raspberry_sphere.
        - 'espresso_handle' : Any
            Handle to the Espresso system. Stored as ``self.sys``.

        The value of ``config['n_parts']`` is computed automatically from the
        reference geometry file and should not be supplied by the caller.

    Attributes
    ----------
    required_features : ``['VIRTUAL_SITES_RELATIVE']``.
    numInstances : int
        Counter of constructed instances (used to assign ``who_am_i``).
    simulation_type : SinglePairDict
        Object type identifier (name and integer code).
    part_types : PartDictSafe instance containing ('real', 'virt').
    config : ObjectConfigParams
        Copy of the provided configuration with ``n_parts`` populated.
    params : ObjectConfigParams
        Alias for ``config``.
    sys : Any
        The Espresso system handle.
    associated_objects : Any
        External associations passed in via ``config``.
    who_am_i : int
        Per-instance ordinal ID.
    type_part_dict : PartDictSafe
        Mutable mapping of part-type names to lists of particle handles created
        by this object.

    Notes
    -----
    The alias determines the file ``<alias>.txt`` inside ``resources/`` whose
    contents (a list of 3D coordinates) define the rigid layout of virtual
    sites relative to the real CoM particle.

      Resource File Specification
    ---------------------------
    Each resource file is a plain text file where each line contains three
    floating-point numbers, separated by commas:

        x, y, z

    These triplets define the coordinates of virtual particles relative to
    the center-of-mass (CoM). For example:

    ``resources/raspberry_sphere.txt``::

        0.6304909963930333, -0.8579033210678375,  1.0629976072588312
        0.8402839358356956, -0.40687932624373957, 1.179467479850195
        0.08627129444723317, -1.4942773402386536, -0.12392477175892225
        ...

    The CoM particle itself is **not listed** in the file: it is always placed
    automatically at (0, 0, 0) and prepended to the virtual coordinates when
    the file is loaded. Thus, the total number of particles will be
    ``1 + len(<file>)`` (1 real + N virtuals).
    """

    required_features = ['VIRTUAL_SITES_RELATIVE']
    numInstances = 0
    _resources_dir = os.path.join(os.path.dirname(__file__), '..', 'resources')
    _resource_file: dict = {}
    _reference_sheet: dict = {}

    simulation_type = SinglePairDict('generic_rigid_object', 68)
    part_types = PartDictSafe({'real': 1, 'virt': 2})
    config = ObjectConfigParams(
        n_parts=None,
        alias='raspberry_sphere'
    )

    def __init__(self, config: ObjectConfigParams):
        """
        Initialize a ``GenericRigidObj``.

        The reference geometry is loaded on first use for a given alias and
        cached for subsequent instances. The number of particles
        (``config['n_parts']``) is deduced from the reference geometry.

        Parameters
        ----------
        config : ObjectConfigParams
            See the class docstring for required keys.

        Raises
        ------
        AssertionError
            If ``config['alias']`` is ``None``.
        """
        assert config['alias'] is not None, (
            'Generic rigid object must have an alias; it is used to locate the '
            'reference geometry file in resources!'
        )
        alias = config['alias']

        # Cache file path and reference coordinates per alias
        existing_path = GenericRigidObj._resource_file.get(alias)
        if existing_path is None:
            path = os.path.join(self._resources_dir, f"{alias}.txt")
            GenericRigidObj._resource_file[alias] = path
            GenericRigidObj._reference_sheet[alias] = load_coord_file(path)

        # Compute n_parts from cached reference coordinates
        config['n_parts'] = len(GenericRigidObj._reference_sheet[alias])

        self.params = config
        self.sys = config['espresso_handle']
        self.associated_objects = config['associated_objects']
        self.who_am_i = GenericRigidObj.numInstances
        GenericRigidObj.numInstances += 1
        self.type_part_dict = PartDictSafe(
            {key: [] for key in GenericRigidObj.part_types.keys()}
        )

    def set_object(self, pos, ori):
        """
        Instantiate the object's particles in Espresso.

        Creates one real CoM particle and a set of virtual particles placed at
        the reference coordinates.

        Parameters
        ----------
        pos : array_like, shape (3,) or broadcastable to (N, 3)
            Translation applied to the reference coordinates to place the rigid
            object in space. Typically a 3-vector.
        ori : array_like, shape (3,)
            Orientation/director vector to assign to the real particle. Virtual orientation is coalligned but should be taken as arbitrary!

        Returns
        -------
        GenericRigidObj
            The current instance (for chaining).

        Notes
        -----
        - The list of created particle handles (by type) is stored in
          ``self.type_part_dict``.
        - The first created particle is converted to type 'real' and used as
          the parent for all virtual sites.
        """
        positions = GenericRigidObj._reference_sheet[self.params['alias']] + pos
        particles = [self.add_particle(type_name='virt', pos=pos) for pos in positions]

        self.change_part_type(particles[0], 'real')
        particles[0].rotation = (True, True, True)
        np.vectorize(lambda real, virts: virts.vs_auto_relate_to(real))(
            particles[0], particles[1:]
        )
        particles[0].director = ori

        return self
