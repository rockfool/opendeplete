"""chain module.

This module contains information about a depletion chain.  A depletion chain is
loaded from an .xml file and all the nuclides are linked together.
"""

from io import StringIO
from itertools import chain
import math
import re
from collections import OrderedDict, defaultdict
from collections.abc import Mapping, Iterable
from numbers import Real
from warnings import warn

from openmc.checkvalue import check_type, check_greater_than
from openmc.data import gnd_name, zam

# Try to use lxml if it is available. It preserves the order of attributes and
# provides a pretty-printer by default. If not available,
# use OpenMC function to pretty print.
try:
    import lxml.etree as ET
    _have_lxml = True
except ImportError:
    import xml.etree.ElementTree as ET
    _have_lxml = False
import scipy.sparse as sp

import openmc.data
from openmc._xml import clean_indentation
from nuclide import Nuclide  
from nuclide import FissionYieldDistribution


# tuple of (reaction name, possible MT values, (dA, dZ)) where dA is the change
# in the mass number and dZ is the change in the atomic number
_REACTIONS = [
    ('(n,2n)', set(chain([16], range(875, 892))), (-1, 0)),
    ('(n,3n)', {17}, (-2, 0)),
    ('(n,4n)', {37}, (-3, 0)),
    ('(n,gamma)', {102}, (1, 0)),
    ('(n,p)', set(chain([103], range(600, 650))), (0, -1)),
    ('(n,a)', set(chain([107], range(800, 850))), (-3, -2))
]

__all__ = ["Chain"]

#
class ChangeFormat(object):
    """Full representation of a depletion chain.

    A depletion chain can be created by using the :meth:`from_endf` method which
    requires a list of ENDF incident neutron, decay, and neutron fission product
    yield sublibrary files. The depletion chain used during a depletion
    simulation is indicated by either an argument to
    :class:`openmc.deplete.Operator` or through the
    ``depletion_chain`` item in the :envvar:`OPENMC_CROSS_SECTIONS`
    environment variable.

    Attributes
    ----------
    nuclides : list of openmc.deplete.Nuclide
        Nuclides present in the chain.
    reactions : list of str
        Reactions that are tracked in the depletion chain
    nuclide_dict : OrderedDict of str to int
        Maps a nuclide name to an index in nuclides.
    fission_yields : None or iterable of dict
        List of effective fission yields for materials. Each dictionary
        should be of the form ``{parent: {product: yield}}`` with
        types ``{str: {str: float}}``, where ``yield`` is the fission product
        yield for isotope ``parent`` producing isotope ``product``.
        A single entry indicates yields are constant across all materials.
        Otherwise, an entry can be added for each material to be burned.
        Ordering should be identical to how the operator orders reaction
        rates for burnable materials.
    """

    def __init__(self):
        self.nuclides = []
        self.reactions = []
        self.nuclide_dict = OrderedDict()
        self._fission_yields = None
        

    def __contains__(self, nuclide):
        return nuclide in self.nuclide_dict

    def __getitem__(self, name):
        """Get a Nuclide by name."""
        return self.nuclides[self.nuclide_dict[name]]

    def __len__(self):
        """Number of nuclides in chain."""
        return len(self.nuclides)

    def from_xml(self, filename, fission_q=None):
        """Reads a depletion chain XML file.

        Parameters
        ----------
        filename : str
            The path to the depletion chain XML file.
        fission_q : dict, optional
            Dictionary of nuclides and their fission Q values [eV].
            If not given, values will be pulled from ``filename``

        """
        chain = self

        if fission_q is not None:
            check_type("fission_q", fission_q, Mapping)
        else:
            fission_q = {}

        # Load XML tree
        root = ET.parse(str(filename))

        for i, nuclide_elem in enumerate(root.findall('nuclide')):
            this_q = fission_q.get(nuclide_elem.get("name"))

            nuc = Nuclide.from_xml(nuclide_elem, this_q)
            chain.nuclide_dict[nuc.name] = i

            # Check for reaction paths
            for rx in nuc.reactions:
                if rx.type not in chain.reactions:
                    chain.reactions.append(rx.type)

            chain.nuclides.append(nuc)

        return chain

    def export_to_xml(self, filename):
        """Writes a depletion chain XML file.

        Parameters
        ----------
        filename : str
            The path to the depletion chain XML file.

        """

        root_elem = ET.Element('depletion')
        decay_elem = ET.SubElement(root_elem, 'decay_constants')
        for nuclide in self.nuclides:
            decay_elem.append(self.output_element(data=nuclide, data_type='decay_constants'))
        
        nfy_elem = ET.SubElement(root_elem,'neutron_fission_yields')    
        
        nuc_list = [nuclide for nuclide in self.nuclides if nuclide.yield_data is not None] 
        
        pre_elem = ET.SubElement(nfy_elem, 'precursor')
        pre_elem.text = str(len(nuc_list))
        
        ergpoint_elem = ET.SubElement(nfy_elem, 'energy_points')
        ergpoint_elem.text = str(1)
        
        prename_elem = ET.SubElement(nfy_elem, 'precursor_name')
        prename_elem.text = ' '.join(self.old_name(nuc.name) for nuc in nuc_list)

        erg_elem = ET.SubElement(nfy_elem, 'energy')
        erg_elem.text = str(0.0253)
        
        num_nfy_nuc = 0
        # get yield data for each nuclide 
        for nuclide in self.nuclides:
            from_yield = {}
            nuc_name = nuclide.name
            dat_yld = []
            for nuc_pre in nuc_list: 
                if 0.0253 in nuc_pre.yield_data.energies \
                        and nuc_name in nuc_pre.yield_data.products:
                    ind_erg = nuc_pre.yield_data.energies.index(0.0253)
                    ind_nuc = nuc_pre.yield_data.products.index(nuc_name)
                    if ind_nuc is not None and ind_erg is not None:
                        dat_yld.append(nuc_pre.yield_data.yield_matrix[ind_erg][ind_nuc])
                    else :
                        dat_yld.append(0.0)
                else:
                    dat_yld.append(0.0)
            #
            from_yield['name'] = nuc_name
            from_yield['energy'] = 0.0253
            from_yield['yields'] = dat_yld
            if any(dat_yld):
                num_nfy_nuc += 1
                nfy_elem.append(self.output_element(data=from_yield, data_type='neutron_fission_yields'))
        
        nuc_elem = ET.SubElement(nfy_elem, 'nuclides')
        nuc_elem.text = num_nfy_nuc
        
        clean_indentation(root_elem)
        tree = ET.ElementTree(root_elem)
        tree.write(filename, xml_declaration=True, encoding='utf-8')


    def output_element(self, data, data_type=None):
        """Write nuclide to XML element.

        Returns
        -------
        elem : xml.etree.ElementTree.Element
            XML element to write nuclide data to

        """
        if data_type == None:
            return None
        
        elem = ET.Element('nuclide_table')
        
        if data_type == "decay_constants":
            elem.set('name', self.old_name(data.name))
            # 
            if data.half_life is not None:
                elem.set('half_life', str(data.half_life))
                elem.set('decay_modes', str(len(data.decay_modes)))
                #elem.set('decay_energy', str(self.decay_energy))
                for mode, daughter, br in data.decay_modes:
                    mode_elem = ET.SubElement(elem, 'decay_type')
                    mode_elem.set('type', mode)
                    mode_elem.set('target', self.old_name(daughter))
                    mode_elem.set('branching_ratio', str(br))
            else:
                elem.set('decay_modes', str(0))
            #
            elem.set('reactions', str(len(data.reactions)))
            for rx, daughter, Q, br in data.reactions:
                rx_elem = ET.SubElement(elem, 'reaction_type')
                rx_elem.set('type', rx)
                #rx_elem.set('energy', str(Q))
                if rx != 'fission':
                    rx_elem.set('target', self.old_name(daughter))
                else:
                    rx_elem.set('energy', str(Q))
                if br != 1.0:
                    rx_elem.set('branching_ratio', str(br))

        elif data_type == 'neutron_fission_yields':
            elem.set('name', self.old_name(data['name']))
            fisyld_elem = ET.SubElement(elem, 'fission_yields')
            fisyld_elem.set('energy',str(data['energy']))
            fyd_elem = ET.SubElement(fisyld_elem, 'fy_data')
            fyd_elem.text = ' '.join(str(yld) for yld in data['yields'])

        return elem

    def old_name(self, gnd_name):
        """
        This function changs gnd name into old name 
        parameters:
          gnd_name
        return
          old_name
        """
        if '-' in gnd_name:
            print('The name is already in old convention.')
            return gnd_name
        else :
            if gnd_name.endswith('_m1'):
                gnd_name = gnd_name[:-3]
                return '-'.join(re.findall('^[A-Za-z]+', gnd_name) + re.findall('\d+', gnd_name)) + 'm'
            else :
                return '-'.join(re.findall('^[A-Za-z]+', gnd_name) + re.findall('\d+', gnd_name))
    
