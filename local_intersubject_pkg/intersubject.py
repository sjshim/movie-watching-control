# intersubject.py: Module contains an Intersubject class that provides
# methods for computing one-sample and within-between 
# intersubject correlation and intersubject functional correlation

import numpy as np

from .tools import save_data
from .basic_stats import compute_r, r_average

# class for computing whole sample and witihin-between group ISFC and ISC.
class Intersubject:
    """
    Perform Intersubject analysis on a group of subjects. 
    Use method group_isfc() to perform either entire- or within-between group
    ISFC/ISC analysis. Retrieve results from dicts stored in the object instance
    attributes .isfc and .isc.

    Example:
    # compute isfc and isc
    entire_isfc = Intersubject(files, (4, 4, 4, 10))
    entire_isfc.group_isfc([3, 14, 17, 28, 29], 'entire')
    
    # get results
    entire_isfc.isfc
    entire_isfc.isc

    """


    def __init__(self, data_path, datasize):
        self.data_path = data_path
        self.datasize = datasize
        self.dims = (datasize[0] * datasize[1] * datasize[2], datasize[3]) # for quick matrix size
        self.subject_ids = {} # can accomodate either one or multiple ID groups
        self.group_mask = None
        self.isfc = {}
        self.isc = {}
        self._data_sum = {}
        self._voxel_sum = {}

    def _get_sum(self, label):
        # Calculate's a group's summed data in preparation for computing the 
        # average (for one-to-average/leave-one-out ISC method)
        
        # Make empty summed data output arrays
        sum_data = np.zeros(shape=(self.dims))
        sum_vox = np.zeros(shape=(self.dims[0]))
        _group_mask = np.zeros(shape=(self.dims[0]))

        # Save each subject's data and voxel mask
        sub_list = self.subject_ids[label]
        for sub in sub_list:
            # Load subject data from npz dict object
            subject_dict = np.load(self.data_path.format(sub))
            mask = subject_dict['mask']

            # Get subject's masked data
            data = np.zeros(shape=(len(mask), self.dims[1])) # matrix of zeros
            data[mask, :] = subject_dict['data'] 

            # Add subject data to the sample-wide sum array
            sum_data = sum_data + data
            sum_vox = sum_vox + mask
        
        assert sum_data.shape == self.dims, f"sum data shape is {sum_data.shape} but should be {self.dims}"
        assert sum_vox.shape == (self.dims[0], ), f"sum vox shape is {sum_vox.shape} but should be {self.dims[0]}"

        # Create boolean mask of sample-wide surviving voxels
        # NOTE: 0.7 is magic number carry-over from YC
        _group_mask = sum_vox.copy() > (0.7 * len(sub_list)) # True if 70% participants voxels survive; magic number from YC's!
        
        # Find set union between group masks if already defined
        if self.group_mask is not None:
            self.group_mask = np.logical_or(self.group_mask, _group_mask)
        else:
            self.group_mask = _group_mask

        # Reduce summed data and voxels with mask
        self._data_sum[label] = sum_data[_group_mask, :]
        self._voxel_sum[label] = sum_vox[_group_mask]


    def _isfc_oneavg(self, sub_id, label, compare_method = 'entire', minus_one = True):
        # Calculates one subject's ISFC and ISC

        # Recursively calculate subject's isfc for within and between group method
        if compare_method == 'within_between':

            # Save dicts of within and between data separately from arguments
            label_list = list(self.subject_ids)
            
            # treat provided label as "within group" label
            within_isfc = self._isfc_oneavg(sub_id, label)

            # use remaining label as "between group"
            label_list.remove(label)
            label = label_list[0]

            between_isfc = self._isfc_oneavg(sub_id, label, minus_one=False)
            
            return within_isfc, between_isfc
        
        elif compare_method == 'entire':
            # Retrieve subject data with their mask
            subject_dict = np.load(self.data_path.format(sub_id))
            mask = subject_dict['mask']
            data = np.full(shape=(mask.shape[0], self.dims[1]), fill_value=np.nan)
            data[mask, :] = subject_dict['data']

            # Mask the subject's data and voxel mask with sample-wide voxel mask (potentially more restrictive)
            data = data[self.group_mask, :]
            mask = mask[self.group_mask]

            assert data.shape == self._data_sum[label].shape, f"subject data {data.shape} and data sum {self._data_sum[label].shape} have mismatched shapes"
            assert mask.shape == self._voxel_sum[label].shape, f"subject mask {mask.shape} and voxel sum {self._voxel_sum[label].shape} have mismatched shapes"
            
            # Check whether to subtract of subject data from summed data
            if minus_one==False:
                temp_avg = self._data_sum[label] / self._voxel_sum[label].reshape((self._voxel_sum[label].shape[0], 1))
            else:
                # Create sample-wide average timecourse minus subject's timecourse for all voxels
                numer = self._data_sum[label] - data # removes subject's data
                denom = self._voxel_sum[label] - mask # removes subject's voxels
                temp_avg = numer / denom.reshape((denom.shape[0], 1))

            # Compute ISFC correlation between subject and average-minus-one
            isfc_matrix = compute_r(data, temp_avg)
            
            return isfc_matrix

    def group_isfc(self, group_ids, compare_method = "entire", keep_isfc=True, keep_isc=True):
        """
        Calculate isfc and isc for a group of subjects.

        Note: currently only uses leave-one-out ISC calculation method.

        Parameters
        ----------
        group_ids : dict or list
            The subject IDs and their association group/label/condition.
            If only one group will be examined, the IDs can be passed as a list
            or as a dict with one key. ID lists of two groups must be passed as a 
            dict with two keys.

        compare_method : str
            Choose the group analysis method.
            -'entire' for analysis on one group of subjects only.
            -'within-between' for within-between analysis between two groups
            of subjects.

        (keep_isfc and keep_isc do not currently do anything and are placeholders 
        to selectively return isfc or isc only)

        Warning: A new Intersubject object should be created whenever a different
        analysis is performed (eg., 'within-between', then 'entire') to avoid 
        accidental carryover of the previously performed analysis' object state 
        and affecting your results.  

        """
        # Calculate isfc and isc for a group of subjects

        assert keep_isfc or keep_isc, "Either keep_isfc or keep_isc must be True"

        # Treat one list as a dict with one group; otherwise, return error
        if type(group_ids) is list:
            # try:
            if any(isinstance(item, list) for item in group_ids):
                raise TypeError("'subject_id' cannot be nested lists; provide multiple lists as dict with simple keys (eg., 0, 1; 'a', 'b') instead")
            else:
                self.subject_ids['sample'] = group_ids
                assert type(self.subject_ids) is dict
            # except TypeError:
            #     print()
        else:
            self.subject_ids = group_ids

        label_list = list(self.subject_ids)

        def get_container(this_label=None, last_dim=None):
            # Function to generate an "empty" array to be filled afterward.
            # Note: currently only works for the Intersubject class.
            if this_label is not None:
                last_dim = len(self.subject_ids[this_label])

            isfc_container = np.full((self.dims[0], self.dims[0], last_dim), np.nan)
            isc_container = np.full((self.dims[0], last_dim), np.nan)
            return isfc_container, isc_container

        # Whole-sample ISFC/ISC
        if compare_method == "entire":
            # treat first group as the primary label
            label = label_list[0]

            # get containers and summed data
            self.isfc['entire'], self.isc['entire'] = get_container(label)
            self._get_sum(label)

            # Get isfc/isc
            for i, sub in enumerate(self.subject_ids[label]):
                this_isfc = self._isfc_oneavg(sub, label, compare_method=compare_method)
                assert this_isfc.shape == self.isfc['entire'][:,:,0].shape, f"subject isfc {this_isfc.shape} and isfc container {self.isfc['entire'][:,:,0].shape} are mismatched"
                                    
                # if keep_isfc:
                self.isfc['entire'][:, :, i] = this_isfc
                # self._isfc = subject_isfc

                # if keep_isc
                self.isc['entire'][:, i] = this_isfc.diagonal() 
                # self._isc = subject_isfc.diagonal()

        # Within-between ISFC/ISC
        elif compare_method == "within_between":

            label_left, label_right = label_list[0], label_list[1]
            all_ids = self.subject_ids[label_left] + self.subject_ids[label_right]

            # Save array containers to attribute's dict
            self.isfc['within'], self.isc['within'] = get_container(last_dim = len(all_ids))
            self.isfc['between'], self.isc['between'] = get_container(last_dim = len(all_ids))

            # Get sums for both subject groups
            self._get_sum(label_left)
            self._get_sum(label_right)

            # Get within and betweeen isfc/isc
            for i, sub in enumerate(all_ids):
                # Treat label as 'within' if it matches the group
                if sub in self.subject_ids[label_left]:
                    wb_isfc = self._isfc_oneavg(sub, label_left, compare_method=compare_method)

                elif sub in self.subject_ids[label_right]:
                    wb_isfc = self._isfc_oneavg(sub, label_right, compare_method=compare_method)
                
                self.isfc['within'][:,:,i] = wb_isfc[0]
                self.isfc['between'][:,:,i] = wb_isfc[1]
                self.isc['within'][:,i] = wb_isfc[0].diagonal()
                self.isc['between'][:,i] = wb_isfc[1].diagonal()