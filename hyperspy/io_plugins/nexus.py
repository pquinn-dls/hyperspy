# -*- coding: utf-8 -*-
# Copyright 2007-2020 The HyperSpy developers
#
# This file is part of  HyperSpy.
#
#  HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  HyperSpy.  If not, see <http://www.gnu.org/licenses/>.
#
#
#    - The characteristics of the IO plugin as the following python variables:
#
#        # Plugin characteristics
#        # ----------------------
#	format_name = <String>
#        description = <String>
#        full_support = <Bool>	# Whether all the Hyperspy features are supported
#        # Recognised file extension
#        file_extensions = <Tuple of string>
#        default_extension = <Int>	# Index of the extension that will be used by default
#        # Reading capabilities
#        reads_images = <Bool>
#        reads_spectrum = <Bool>
#        reads_spectrum_image = <Bool>
#        # Writing capabilities
#        writes_images = <Bool>
#        writes_spectrum = <Bool>
#        writes_spectrum_image = <Bool>
#
#    - A function called file_reader with at least one attribute: filename
#
#    - A function called file_writer with at least two attributes: 
#        filename and object2save in that order.

import logging
import numpy as np
import dask.array as da
import os
import h5py 
import sys
import pprint
from hyperspy.io_plugins.hspy import overwrite_dataset

NX_ENCODING = sys.getfilesystemencoding()

_logger = logging.getLogger(__name__)
# Plugin characteristics

format_name = 'Nexus'
description = \
    'Read NXdata, hdf datasets and metadata from hdf5 or Nexus files'
full_support = False
# Recognised file extension
file_extensions = ['nxs','NXS','h5','hdf','hdf5']
default_extension = 0
# Writing capabilities:
writes = True

def _byte_to_string(value):
    
    """ Decode a byte string

    Parameters
    ----------
    value :  byte str
    
    Returns
    -------
    str, decoded version of input value

    """
    try:
        text = value.decode(NX_ENCODING)
    except UnicodeDecodeError:
        if NX_ENCODING == 'utf-8':
            text = value.decode('latin-1')
        else:
            text = value.decode('utf-8')
    return text.replace('\x00','').rstrip()

def _parse_from_file(value,lazy=True):
    
    """ To convert values from the hdf file to compatible formats

    When reading string arrays we convert or keep string arrays as
    byte_strings (some io_plugins only supports byte-strings arrays so this 
    ensures inter-compatibility across io_plugins)

    Arrays of length 1 - return the single value stored     
    
    Large datasets are returned as dask arrays if lazy=True 
    
    Parameters
    ----------
    value : input read from hdf file (array,list,tuple,string,int,float)
    lazy  : bool  {default: True}
    
    Returns
    -------
    parsed value
    
    """
    toreturn = value
    if isinstance(value,h5py.Dataset):    
       if value.size<2:
           toreturn = value[...].item()
       else:
           if lazy:
               if value.chunks:
                   toreturn = da.from_array(value,value.chunks)
               else:
                   chunks = _guess_chunks(value)
                   toreturn = da.from_array(value,chunks)
                    
           else:
               toreturn = np.array(value)

    if isinstance(toreturn, np.ndarray) and value.shape == (1,):
        toreturn = toreturn[0]        
    if isinstance(toreturn, bytes):
        toreturn= _byte_to_string(toreturn)
    if isinstance(toreturn, (np.int,np.float)):
        toreturn= toreturn
    if isinstance(toreturn, (np.ndarray)) and toreturn.dtype.char == "U": 
        toreturn = toreturn.astype("S")
    return toreturn


def _parse_to_file(value):
    
    """ Check that value is compatible with hdf5 and if not convert to 
    a suitable format. 

    For example unicode values are not compatible with hdf5 so conversion to 
    byte strings is required. 
    
    Parameters
    ----------
    value - input object to write to the hdf file
    
    Returns
    -------
    parsed value
    
    """
    totest = value
    toreturn = totest
    if isinstance(totest, (bytes,int,float)):
        toreturn= value
    if isinstance(totest, (list,tuple)):
        totest = np.array(value)        
    if isinstance(totest, (np.ndarray)) and totest.dtype.char == "U":
            toreturn = np.array(totest).astype("S")        
    elif isinstance(totest,(np.ndarray,da.Array)):  
        toreturn = totest
    if isinstance(totest, (str)):
        toreturn = totest.encode("utf-8")        
    return toreturn

def _text_split(s, sep):
    
    """ Split a string based of list of seperators
    
    Parameters
    ----------
    s   : str 
    sep : str  - seperator or list of seperators e.g. '.' or ['_','/']
    
    Returns
    -------
    list containing string sections split based on the seperators
    
    """
    stack = [s]
    for char in sep:
        pieces = []
        for substr in stack:
            pieces.extend(substr.split(char))
        stack = pieces
    if '' in stack:
        stack.remove('')
    return stack

def _getlink(h5group,rootkey):
    
    """ Return the link target path and filename.

    If a hdf group is a link to an external file or a soft link  
    this method will return the target path within 
    the external file, the external file name and if the external
    file name is an absolute path or not
    e.g. if /entry/data points to /raw_entry/tem/data in raw.nxs
    returns /raw_entry/tem/data, raw.nxs, False 
    
    
    Returns
    -------
    str, str, bool
        Link path, filename, and boolean that is True if an absolute file
        path is given.
    
    """
    _target, _filename, _abspath = None, None, False
    if rootkey != '/':
        if isinstance(h5group,h5py.Group):
            _link = h5group.get(rootkey, getlink=True)
            if isinstance(_link, h5py.ExternalLink):
                _target, _filename = _link.path, _link.filename
                _abspath = os.path.isabs(_filename)
            elif isinstance(_link, h5py.SoftLink):
                _target = _link.path
        if 'target' in h5group.attrs:
            _target = _parse_from_file(h5group.attrs['target'])
            if not _target.startswith('/'):
                _target = '/' + _target
            if _target == rootkey:
                _target = None
    return _target, _filename, _abspath


def _extract_hdf_dataset(group,dataset,lazy=True):
    
    """ Import data from hdf path 
    
    Parameters
    ----------
    group : hdf group 
        group from which to load the dataset
    dataset : str   
        path to the dataset within the group
    lazy    : bool {default:True} 
        If true use lazy opening, if false read into memory
    
    Returns
    -------
    data - dask or numpy array
    
    """
    data = group[dataset]
    if data.dtype.type is np.string_ or data.dtype.type is np.object_:
        return None
    if(lazy):
        if "chunks" in data.attrs.keys():
             chunks = data.attrs["chunks"]
        else:
            chunks = _guess_chunks(data)
        data_lazy = da.from_array(data, chunks=chunks)
    else:
       data_lazy = np.array(data)    
    return data_lazy

    
def _nexus_dataset_to_signal(group,nexus_dataset_path,lazy=True):
    """ Load an NXdata set as a hyperspy signal 
    
    Parameters
    ----------
    tree : hdf group 
    nexus_data_path : str
        Path to the NXdata set in the group 
    lazy : bool, default : True    
    
    Returns
    -------
    A signal dictionary which can be used to instantiate a signal class
    signal_dictionary = {'data': data,
                         'axes': axes information,
                         'metadata': metadata dictionary}
    
    """
    detector_index = 0
    dataentry = group[nexus_dataset_path]
    if "signal" in dataentry.attrs.keys():
        if _is_int(dataentry.attrs["signal"]):
            data_key  = "data"
        else:
            data_key  = dataentry.attrs["signal"]
    else:
        _logger.info("No signal attr associated with NXdata will\
                     try assume signal name is data")
        if "data" not in  dataentry.keys():
            raise ValueError("Signal attribute not found in NXdata and\
                             attempt to find a default \"data\" key failed")
        else:            
            data_key  = "data"
    data = dataentry[data_key] 
    nav_list = []
    # list indices...
    axis_index_list=[]
    if "axes" in dataentry.attrs.keys():           
        axes_key  = dataentry.attrs["axes"]
        axes_list = [_parse_from_file(num) for num in axes_key]
        named_axes=list(range(len(axes_list)))
        for i,ax in enumerate(axes_list):
            if ax != ".":
                index_name = ax + "_indices"
                ind_in_array = int(dataentry.attrs[index_name])
                axis_index_list.append(ind_in_array)
                if "units" in dataentry[ax].attrs:
                    units= _parse_from_file(dataentry[ax].attrs["units"])
                else:
                    units=""
                named_axes.remove(ind_in_array)
                if _is_numeric_data(dataentry[ax]):
                    if _is_linear_axis(dataentry[ax]):
                        nav_list.append({   
                            'size': data.shape[ind_in_array],
                            'index_in_array': ind_in_array,
                            'name': ax,
                            'scale': abs(dataentry[ax][1]-\
                                         dataentry[ax][0]),
                            'offset': min(dataentry[ax][0],\
                                          dataentry[ax][-1] ),
                            'units': units,
                            'navigate': True,
                            })
                    else:
                        nav_list.append({   
                            'size': data.shape[ind_in_array],
                            'index_in_array': ind_in_array,
                            'name': ax,
                            'scale': 1,
                            'offset': 0,
                            'navigate': True,
                            })
                         # when non-linear axes supported...
#                        nav_list.append({   
#                            'index_in_array': ind_in_array,
#                            'name': axname,
#                            'axis' : dataentry[ax],
#                            'units': units,
#                            'navigate': True,
#                            })
                    
        for i,ax in enumerate(axes_list):
            if ax == ".":
                if(len(data.shape)==len(axes_list)):
                    nav_list.append({   
                            'size': data.shape[named_axes[detector_index]],
                            'index_in_array': named_axes[detector_index],
                            'name': "detector_%d"%detector_index,
                            'scale': 1,
                            'offset': 0.0,
                            'units': '',
                            'navigate': False,
                           })
                    detector_index=detector_index+1
                            
        if(lazy):
            if "chunks" in data.attrs.keys():
                 chunks = data.attrs["chunks"]
            else:
                chunks = _guess_chunks(data)
            data_lazy = da.from_array(data, chunks=chunks)
        else:
           data_lazy = np.array(data)    
        # adjust the navigate boolean based on 
        # ifer the dataset isjudged to be 1D or 2D
        signal_type = _guess_signal_type(data_lazy)
        if signal_type == "Signal2D":
            nav_list[-1]["navigate"] = False
            nav_list[-2]["navigate"] = False
        if signal_type == "Signal1D":
            nav_list[-1]["navigate"] = False
        title = _text_split(nexus_dataset_path, '/')[-1]
        print("nexus_dataset",nexus_dataset_path,title)
        metadata = {'General': {'title': title},\
                    "Signal": {'signal_type': signal_type}}
        if nav_list:      
            dictionary = {'data': data_lazy,
                          'axes': nav_list,
                          'metadata':metadata}
        else:
            dictionary = {'data': data_lazy,
                          'metadata': metadata}                    
    return dictionary


def file_reader(filename,lazy=True, dset_search_keys=None,
                meta_search_keys=None,
                load_nxdata_only=True,
                small_metadata_only=True,
                **kwds):
    
    """ Reads NXdata class or hdf datasets from a file and returns signal
    
    Note
    ----
    Loading all datasets can result in a large number of signals
    Please review your datasets and use the dset_search_keys to target
    the datasets of interest. 
    value and keys are special keywords and prepended with fix_ in the metadata
    structure to avoid any issues.
    
    Datasets are all arrays with size>2 (arrays, lists)
    
    Parameters
    ----------
    myDict : dict or h5py.File object
    dset_search_keys  : str list or None  
        Only return items whose path contain the strings
        .e.g search_list = ["instrument","Fe"] will return
        data entries with instrument or Fe in their hdf path.
    meta_search_keys: : str list or None
        Only return items whose path contain the strings
        .e.g search_list = ["instrument","Fe"] will return
        all metadata entries with instrument or Fe in their hdf path.
    load_nxdata_only : bool  If true only NXdata will be converted into a signal
                 if false NXdata and any hdf datasets not bound within the
                 NXdata sets will be loaded as signals
    small_metadata_only : bool, default:True  If true arrays of size<2
                 will be loaded into the metadata structure. If false larger 
                 arrays will be also loaded into the metadata
    

    Returns
    -------
    signal dictionary or list of signal dictionaries of the form
    {'data': data,
     'metadata': metadata,
     'original_metadata':original_metadata,
     'mapping':mapping}
    

    See also 
    --------
    get_datasets_from_file 
                 
   
    """
    # search for NXdata sets...
    if dset_search_keys == None:
        dset_search_keys=[""]
    if meta_search_keys == None:
        meta_search_keys=[""]
    mapping  = kwds.get('mapping',{})   
    metadata={}
    original_metadata={}
    fin = h5py.File(filename,"r")
    signal_dict_list = []

    
    nexus_data_paths, hdf_data_paths = _find_data(fin,
                                                  search_keys=dset_search_keys)
    # strip out the metadata (basically everything other than NXdata)
    #
    # Be careful not to load a large number of datasets
    # Check against an upper limit and that the user has
    # sub-selected the datasets to load used dset_search_keys
    #
    all_metadata = load_dict_from_hdf(fin,small_metadata_only=small_metadata_only)
    
    stripped_metadata = _find_search_keys_in_dict(all_metadata,
                                                  search_keys=meta_search_keys)

    if "Experiment" in all_metadata.keys():
        if "metadata" in all_metadata["Experiment"].keys():
            if "original_metadata" in stripped_metadata["Experiment"]["metadata"]:
                original_metadata = stripped_metadata["Experiment"]["metadata"]["original_metadata"]
            if "hyperspy_metadata" in stripped_metadata["Experiment"]["metadata"]:
                metadata = stripped_metadata["Experiment"]["metadata"]["hyperspy_metadata"]
    else:
        original_metadata = all_metadata
    original_metadata = _find_search_keys_in_dict(original_metadata,
                                            search_keys=meta_search_keys)
        
    
    for data_path in nexus_data_paths:        
        dictionary = _nexus_dataset_to_signal(fin,data_path,lazy)
        dictionary["original_metadata"] = original_metadata
        dictionary["metadata"].update(metadata)        
        dictionary["mapping"] = mapping
        
        signal_dict_list.append(dictionary)
    if not load_nxdata_only:
        for data_path in hdf_data_paths:        
            data = _extract_hdf_dataset(fin,data_path,lazy)
            if data is not None:
                signal_type =  _guess_signal_type(data)
                title = _text_split(data_path, '/')[-1]
                basic_metadata = {'General': {'original_filename': \
                          os.path.split(filename)[1],\
                'title': title},\
                "Signal": {'signal_type': signal_type}}
                dictionary = {'data': data,
                              'metadata': basic_metadata,
                              'original_metadata':original_metadata,
                              'mapping':mapping}                
                signal_dict_list.append(dictionary)

        
    return signal_dict_list
        
def _is_linear_axis(data):
    
    """ Check if the data is linearly incrementing

    Parameter
    ---------
    data : dask or numpy array

    Returns
    -------
    True or False    
    
    """
    steps = np.diff(data)
    est_steps = np.array([steps[0]]*len(steps))
    return np.allclose(est_steps,steps,rtol=1.0e-5)

def _is_number(a):
    
    """ Check if the value is a number
    # will be True also for 'NaN'
    
    Parameter
    ---------
    a : python object to be tested
    
    Returns
    -------
    True or False    

    """
    try:
        float(a)
        return True
    except ValueError:
        return False

def _is_numeric_data(data):
    
    """ Check that data contains numeric data

    Parameter
    ---------
    data : dask or numpy array
    
    Returns
    -------
    True or False    

    """
    try:
        data.astype(float)
        return True
    except ValueError:
        return False
 
    
def _is_int(s):
    
    """ Check that s in an integer

    Parameter
    ---------
    s : python object to test
    
    Returns
    -------
    True or False    
    
    """
    try: 
        int(s)
        return True
    except ValueError:
        return False

        
def _find_data(myDict,search_keys=None):
    """ Read from a nexus or hdf file and return a list 
    of the dataset entries
    The method iterates through group attributes and returns NXdata or 
    hdf datasets of size >=2 if they're not already NXdata blocks
    and returns a list of the entries
    This is a convenience method to inspect a file to see which datasets
    are present rather than loading all the sets in the file as signals
    
    Parameters
    ----------
    myDict : dict or h5py.File object
    search_keys  : str list or None  
        Only return items which contain the strings
        .e.g search_list = ["instrument","Fe"] will return
        hdf entries with instrument or Fe in their hdf path.
    
    Returns
    -------
    nx_dataset_list, hdf_dataset_list
    
    nx_dataset_list is a list of all NXdata paths
    hdf_dataset_list is a list of all hdf_datasets not linked to an NXdata set
        
    """
    hdf_datasets = []
    nx_datasets = []
    rootname=""
    def find_data_in_tree(myDict,rootname):
        for key, value in myDict.items():
            if rootname != "":
                rootkey = rootname +"/"+ key
            else:
                rootkey = "/" + key
            if isinstance(value,h5py.Dataset):  
                if value.size >= 2:
                    target,b,c, = _getlink(value,rootkey)
                    if target is not None:
                        if search_keys:
                            if any(s in rootkey for s in search_keys):
                                hdf_datasets.append(target)
                        else:
                            hdf_datasets.append(target)
            elif isinstance(value,h5py.Group):
                if "NX_class" in value.attrs:
                    if value.attrs["NX_class"] == b"NXdata" \
                        and "signal" in value.attrs.keys(): 
                        if search_keys:
                            if any(s in rootkey for s in search_keys):
                                nx_datasets.append(rootkey)
                        else:
                            nx_datasets.append(rootkey) 
                find_data_in_tree(value,rootkey)

    # need to use custom recursive function as visititems in h5py
    # does not visit links
    find_data_in_tree(myDict,rootname)
    nx_datasets = list(dict.fromkeys(nx_datasets))
    hdf_datasets = list(dict.fromkeys(hdf_datasets))
    return nx_datasets,hdf_datasets


def _fix_exclusion_keys(key):
    
    """ Exclude hyperspy specific keys
    Signal and DictionaryBrowser break if a
    a key is a dict method - e.g. {"keys":2.0} 
    
    This method prepends the key with "_fix_" so the information is
    still present to work around this issue
    
    Parameter
    ---------
    key : str 
    
    Returns
    -------
    key or key prepended with '_fix_' if key is 
    in a set of excluded of keywords
    
    """
    exclusion_list=['_sig_','_list_empty_','_tuple_empty_',
                    '_bs_','_datetime_date',"_datetime_","value",
                    "keys"]
    
    if any([key.startswith(x) for x in exclusion_list]):
        return "fix_"+key
    else:
        return key


def _find_search_keys_in_dict(tree,search_keys=None):
    
    """ Search through a dict and return a dictionary
    whose full path contains one of the search keys

    This is a convenience method to inspect a file for a value
    rather than loading the file as a signal
    
    Parameters
    ----------
    tree         : h5py File object 
    search_keys  : str list or None  
        Only return items which contain the strings
        .e.g search_keys = ["instrument","Fe"] will return
        hdf entries with instrument or Fe in their hdf path.
    
    Returns
    -------
    Dictionary. When search_list is specified only full paths
    containing one or more search_keys will be returned 
        
    """
    if search_keys == None:
        search_keys=[""]
    metadata_dict = {}
    rootname=""
    # recursive function
    
    def find_searchkeys_in_tree(myDict,rootname):
        for key, value in myDict.items():
            if rootname != "":
                rootkey = rootname +"/"+ key
            else:
                rootkey = key            
            if any(s in rootkey for s in search_keys):
                mod_keys = _text_split(rootkey, (".","/") )                        
                # create the key, values in the dict
                p = metadata_dict
                for d in mod_keys[:-1]:
                    p = p.setdefault(d,{})
                p[mod_keys[-1]] = value
            if isinstance(value,dict):
                find_searchkeys_in_tree(value,rootkey)
    find_searchkeys_in_tree(tree,rootname)
    return metadata_dict



def load_dict_from_hdf(group,lazy=True,
                        small_metadata_only=True):
    
    """ Search through a group and return metadata
 
    Paramaters
    ----------
    group : hdf group
    lazy : bool, default : True
    small_metadata_only : bool, default : True
        If true only return items of size <2 (no arrays) as metadata

    Returns
    -------
    dictionary of group contents
    
    """
    
    return _find_metadata_in_tree(group,lazy=lazy)

# recursive function
def _find_metadata_in_tree(group,lazy=True,small_metadata_only=True):
    
    """ Search through a group identifying metadata
    Metadata is classed as data of size < 2
    which should covers all descriptive details - 
    strings, floats, ints 

    Paramaters
    ----------
    group : hdf group or dict
    small_metadata_only : bool, default : true
        If true only return items of size <2 (no arrays) as metadata

    Returns
    -------
    dictionary of group contents
    
        
    """
    
    tree={}
    for key,item in group.attrs.items():
        new_key=_fix_exclusion_keys(key)
        if "attrs" not in tree.keys():
            tree["attrs"]={}
        tree["attrs"][new_key] =  _parse_from_file(item,lazy=lazy)
    for key,item in group.items():
        new_key=key
        new_key =_fix_exclusion_keys(key)
        if type(item) is h5py._hl.dataset.Dataset:
            if (small_metadata_only and item.size < 2) or \
                (not small_metadata_only):
                if new_key not in tree.keys():
                    tree[new_key]={}
                tree[new_key]["value"] = _parse_from_file(item,lazy=lazy)
                for k,v in item.attrs.items():
                    if "attrs" not in tree[new_key].keys():
                        tree[new_key]["attrs"]={}
                    tree[new_key]["attrs"][k] =  _parse_from_file(v,lazy=lazy)
                
        elif type(item) is h5py._hl.group.Group:
            tree[new_key]=load_dict_from_hdf(item,
                small_metadata_only=small_metadata_only)
    return tree


    
def _guess_chunks(data):
    
    """ Guess an appropriate chunk layout for a dataset, given its shape and
    the size of each element in bytes.  Will allocate chunks only as large
    as MAX_SIZE.  Chunks are generally close to some power-of-2 fraction of
    each axis, slightly favoring bigger values for the last index.    
    
    Parameters
    ----------
    data  -  numpy, dask array or h5py.Dataset
    
    Returns
    -------
    tuple - Estimated chunk sizes for each axis
    
    """
    chunks= h5py._hl.filters.guess_chunk(data.shape, None, np.dtype(data.dtype).itemsize)
    return chunks


def _write_nexus_groups(dictionary, group, **kwds):
    
    """ Recursively iterate through dictionary and 
    create the groups and datasets
    
    Parameters
    ----------
    dictionary : dict
        dictionary contents to store to hdf group
    group : hdf group
        location to store dictionary
    **kwds : additional keywords
       additional keywords to pass to h5py.create_dataset method
       
    """
    for key, value in dictionary.items():
        if key == 'attrs':
            # we will handle attrs later...
            continue
        if isinstance(value, dict):
            if 'value' in value.keys() and not isinstance(value["value"],dict) \
            and len(set(list(value.keys())+["attrs","value"]))==2:
               value=value["value"]
            else:
                _write_nexus_groups(value, group.require_group(key),
                      **kwds)
        
        if isinstance(value, (list, tuple,np.ndarray, da.Array)):
            data = _parse_to_file(value)
            overwrite_dataset(group, data, key, chunks=None, **kwds)
        elif isinstance(value, (int,float,str,bytes)):
            group.create_dataset(key,
                data=_parse_to_file(value))     



def _write_nexus_attr(dictionary, group):
    """ Recursively iterate through dictionary and 
    write "attrs" dictionaries
    This step is called after the groups and datasets
    have been created
    
    Paramaters
    ----------
    dictionary : dict  
        Input dictionary to be written to the hdf group
    group : hdf group 
    
    """
    for key, value in dictionary.items():
        if key == 'attrs':
            for k, v in value.items():
                group.attrs[k] = _parse_to_file(v)        
        else:
            if isinstance(value, dict):
                _write_nexus_attr(dictionary[key], group[key])
                
def _guess_signal_type(data):
    
    """ Checks last 2 dimensions
    
    An area detector will be square, > 255 pixels each side and
    if rectangular ratio of sides is at most 4.
    An EDS detector will generally by ndetectors*2048 or ndetectors*4096
    the ratio test will be X10-1000  
    
    Paramaters
    ----------
    data : dask or numpy array
    
    Returns
    -------
    Estimate of signal type - "Signal1D", "Signal2D" or "BaseSignal"
    
    """
    if len(data.shape) == 1:
        signal_type = "Signal1D"
    elif len(data.shape) == 2:
        signal_type = "Signal2D"
    else:
        n1 = data.shape[-1]
        n2 = data.shape[-2]
        if n2>127 and n1 >127 and n1/n2 < 1.5:
            signal_type = "Signal2D"
        elif n1 > 127 and n2 < n1:
            signal_type = "Signal1D"
        else:
            signal_type = "BaseSignal"
    return signal_type
        


def get_metadata_in_file(filename,search_keys=None,
                        small_metadata_only=False,
                        verbose=True):
    
    """ Read the metadata from a nexus or hdf file       
    The method iterates through the group and returns a dictionary of 
    the entries.
    This is a convenience method to inspect a file for a value
    rather than loading the file as a signal
    
    Parameters
    ----------
    filename : str  
        path of the file to read
    search_keys  : str list or None  
        Only return items which contain the strings
        For example, search_keys = ["instrument","Fe"] will return
        hdf entries with instrument or Fe in their hdf path.
    verbose: bool, default : True  
        Print the results to screen 
    
    Returns
    -------
    Metadata dictionary. When search_list is specified only items
    matching the search keys will be returned 
        
    """
    if search_keys == None:
        search_keys=[""]
    fin = h5py.File(filename,"r")
    # search for NXdata sets...
    # strip out the metadata (basically everything other than NXdata)
    metadata = load_dict_from_hdf(fin,small_metadata_only)
    stripped_metadata = _find_search_keys_in_dict(metadata,search_keys=search_keys)

    if verbose:
        pprint.pprint(stripped_metadata)
    return stripped_metadata    


def get_datasets_in_file(filename,search_keys=None,
                          verbose=True):
    """ Read from a nexus or hdf file and return a list of the paths
    to the dataset entries. This method is used to inspect the contents
    of a Nexus file.
    The method iterates through group attributes and returns NXdata or 
    hdf datasets of size >=2 if they're not already NXdata blocks
    and returns a list of the entries
    This is a convenience method to inspect a file to see what datasets
    are present rather than loading all the sets in the file as signals
    
    Parameters
    ----------
    filename : str   
        path of the file to read
    search_keys  : str, list or None {default: None}  
        Only return items which contain the strings
        For example, search_keys = ["instrument","Fe"] will return
        hdf entries with "instrument" or "Fe" in their hdf path.
    verbose : boolean, default : True  
        Prints the results to screen 
    
    Returns
    -------
    list of paths to datasets. 
        When search_keys is specified only those paths in the 
        hdf file match the search keys will be returned 
        
    """
    if search_keys == None:
        search_keys=[""]
    fin = h5py.File(filename,"r")
    # search for NXdata sets...
    # strip out the metadata (basically everything other than NXdata)
    nexus_data_paths,hdf_dataset_paths = \
        _find_data(fin,search_keys=search_keys)
    if verbose:
        if nexus_data_paths:
            print("NXdata found")
            for nxd in nexus_data_paths:
                print(nxd)
        else:
            print("NXdata not found")
        if hdf_dataset_paths:    
            print("HDF datasets found")
            for hdfd in hdf_dataset_paths:
                print(hdfd,fin[hdfd].shape)
        else:
            print("No HDF datasets not found or data is captured by NXdata")
    fin.close() 
    return nexus_data_paths,hdf_dataset_paths


def write_signal(signal, nxgroup, signal_name,**kwds):
    
    """ Store the signal data as an NXdata dataset 
    
    Parameters
    ----------
    signal : Hyperspy signal
    nxgroup : HDF group entry at which to save signal data   
    signal_name : str    
        Name  under which to store the signal entry in the file
            
    
    """
    smd = signal.metadata.Signal
    if signal.axes_manager.signal_dimension == 1:
        smd.record_by = "spectrum"
    elif signal.axes_manager.signal_dimension == 2:
        smd.record_by = "image"
    else:
        smd.record_by = ""

    nxdata = nxgroup.require_group(signal_name)
    nxdata.attrs["NX_class"] = b"NXdata"    
    nxdata.attrs[u"signal"] = b"data"
    overwrite_dataset(nxdata, signal.data, u"data", chunks=None, **kwds)
    axis_names=[b"."]*len(signal.axes_manager.shape)
    for i,axis in enumerate(signal.axes_manager._axes):
        try:
            axname  = axis.name + "_indices"
        except:
            axname  = "axis"+str(i)            
        axindex = axis.index_in_array
        indices = axname+"_indices"
        nxdata.attrs[bytes(indices, 'utf-8')] = axindex
        nxdata.require_dataset(axname,data=axis.axis,shape=axis.axis.shape,
                               dtype=axis.axis.dtype)
        axis_names[axis.index_in_array]=bytes(axname, 'utf-8')
    nxdata.attrs[u"axes"] = axis_names
    

def file_writer(filename,
                signal,
                *args, **kwds):
    """ Write the signal and metadata as a nexus file
    This will save the signal in NXdata format in the file.
    As the form of the metadata is not validated it will be stored as
    an NXcollection.

    To implement NX standard or datasets with values and attributes 
    the metadata structure should have a value and attrs key to define the 
    structure to save. 
    For example to save data to an NX standards you can structure the metadata
    entry as:
    
    monochromator 
    |- value : 12.0
    |- attrs
    |--- units   : keV
    |----NXclass : NXmonochromator
    
    When read back in it's accessed in the same way which is consistent with
    the hdf storage.
    
    monochromator.value
    monochromator.attrs.units
    
    Currently only metadata (hyperspy and original), learning_results and the
    signal are stored. Analysis models are not currently supported.

    Parameters
    ----------
    filename      : str    
        Path of the file to write
    signal        : hyperspy signal
    
        
    
    """
        
    with h5py.File(filename, mode='w') as f:
        f.attrs['file_format'] = "Nexus"
        nxexpt = f.create_group('Experiment')
        nxexpt.attrs["NX_class"] = b"NXentry"                

        try:
            #
            # write the signal
            #
            signal_name = signal.metadata.General.title if \
            signal.metadata.General.title else '__unnamed__'
            if "/" in signal_name:
                signal_name = signal_name.replace("/", "_")
            nxentry = nxexpt.create_group('signal')
            nxentry.attrs[u"NX_class"] = u"NXentry"
            write_signal(signal, nxentry, signal_name,**kwds)
            #
            # write metadata 
            #
            nxentry = nxexpt.create_group('metadata')
            nxentry.attrs["NX_class"] = b"NXentry"                
            if signal.original_metadata:
                nxometa = nxentry.create_group('original_metadata')
                nxometa.attrs["NX_class"] = b"NXcollection"    
                orig_meta = signal.original_metadata.as_dictionary()
                # write the groups and structure
                _write_nexus_groups(orig_meta,nxometa)
                _write_nexus_attr(orig_meta,nxometa)
            if signal.metadata:
                nxmeta = nxentry.create_group('hyperspy_metadata')
                nxmeta.attrs["NX_class"] = b"NXcollection"    
                meta =signal.metadata.as_dictionary()
                _write_nexus_groups(meta,nxmeta)
                _write_nexus_attr(meta,nxmeta)

            if signal.learning_results:
                nxlearn = nxexpt.create_group('learning_results')
                nxlearn.attrs["NX_class"] = b"NXcollection"    
                learn = signal.learning_results.__dict__
                _write_nexus_groups(learn,nxlearn)
                _write_nexus_attr(learn,nxlearn)
                
        except BaseException:
            raise
        finally:
            f.close()

